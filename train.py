#coding=utf-8
"""
Training for the segmentation model.
reference:
	https://github.com/zhanghang1989/PyTorch-Encoding
	https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import setproctitle
import numpy as np

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

# sync batch bormalization across multi gpus
from RMI.model.sync_bn import syncbn, parallel
from RMI.model.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from RMI.model.sync_batchnorm.replicate import patch_replication_callback

from RMI import parser_params, full_model
from RMI.model import psp, deeplab
from RMI.dataloaders import factory
from RMI.utils import model_init
from RMI.utils import train_utils
from RMI.utils.metrics import Evaluator
from RMI.losses import loss_factory

# A map from segmentation name to model object.
seg_model_obj_dict = {
	'pspnet': psp.PSPNet,
	'deeplabv3': deeplab.DeepLabv3,
	'deeplabv3+': deeplab.DeepLabv3Plus,
}


class Trainer(object):
	def __init__(self, args):
		"""initialize the Trainer"""
		# about gpus
		self.cuda = args.cuda
		self.gpu_ids = args.gpu_ids
		self.num_gpus = len(self.gpu_ids)
		self.no_val = args.no_val

		# about training schedule
		self.init_global_step = args.init_global_step
		self.start_epoch = args.start_epoch
		self.train_epochs = args.epochs

		# about the learning rate
		self.init_lr = args.init_lr
		self.lr_scheduler = args.lr_scheduler
		self.slow_start_lr = args.slow_start_lr
		self.lr_multiplier = args.lr_multiplier
		self.accumulation_steps = args.accumulation_steps

		# about the model_dir and checkpoint
		self.model_dir = args.model_dir
		self.save_ckpt_steps = args.save_ckpt_steps
		self.max_ckpt_nums = args.max_ckpt_nums
		self.saved_ckpt_filenames = []
		self.checkname = args.checkname

		# define global setp
		self.global_step = 0
		self.main_gpu = args.main_gpu

		# sync bn, both can be used.
		self.norm_layer = syncbn.BatchNorm2d if args.sync_bn else nn.BatchNorm2d
		#self.norm_layer = SynchronizedBatchNorm2d if args.sync_bn else nn.BatchNorm2d

		# define tensorboard summary
		self.train_writer = SummaryWriter(log_dir=self.model_dir)
		self.val_writer = SummaryWriter(log_dir=os.path.join(self.model_dir, 'eval'))

		# define dataloader
		self.train_loader, self.nclass = factory.get_data_loader(args.data_dir,
															batch_size=args.batch_size,
															crop_size=args.crop_size,
															dataset=args.dataset,
															split=args.train_split,
															num_workers=args.workers,
															pin_memory=True)
		self.val_loader, _ = factory.get_data_loader(args.data_dir,
														dataset=args.dataset,
														split="test" if 'camvid' in args.dataset else "val")

		# max iters
		self.steps_per_epochs = len(self.train_loader)
		self.max_iter = self.steps_per_epochs * self.train_epochs

		# define network
		assert args.seg_model in seg_model_obj_dict.keys()
		self.seg_model = args.seg_model
		self.model = seg_model_obj_dict[self.seg_model](num_classes=self.nclass,
											backbone=args.backbone,
											output_stride=args.out_stride,
											norm_layer=self.norm_layer,
											bn_mom=args.bn_mom,
											freeze_bn=args.freeze_bn)

		# define criterion
		self.loss_type = args.loss_type
		self.criterion = loss_factory.criterion_choose(self.nclass,
														weight=None,
														loss_type=args.loss_type,
														ignore_index=255,
														reduction='mean',
														max_iter=self.max_iter,
														args=args)
		self.model_with_loss = full_model.FullModel(seg_model=self.seg_model,
														model=self.model,
														loss_type=self.loss_type,
														criterion=self.criterion)

		# define evaluator
		self.evaluator = Evaluator(self.nclass)

		# using cuda
		# If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.
		# Parameters of a model after .cuda() will be different objects with those before the call.
		# In general, you should make sure that optimized parameters
		# live in consistent locations when optimizers are constructed and used.
		if args.cuda:
			self.model_with_loss = torch.nn.DataParallel(self.model_with_loss, device_ids=self.gpu_ids)
			if self.norm_layer is SynchronizedBatchNorm2d:
				patch_replication_callback(self.model_with_loss)
				print("INFO:PyTorch: The batch norm layer is {}".format(self.norm_layer))
			elif self.norm_layer is syncbn.BatchNorm2d:
				parallel.patch_replication_callback(self.model_with_loss)
				print("INFO:PyTorch: The batch norm layer is Hang Zhang's {}".format(self.norm_layer))
			self.model_with_loss = self.model_with_loss.cuda(self.main_gpu)

		# optimizer parameters, construct optim after module
		self.params_list = []
		self.params_list = model_init.seg_model_get_optim_params(self.params_list,
																	self.model_with_loss.module.model,
																	norm_layer=self.norm_layer,
																	seg_model=args.seg_model,
																	base_lr=args.init_lr,
																	lr_multiplier=self.lr_multiplier,
																	weight_decay=args.weight_decay)
		self.optimizer = torch.optim.SGD(self.params_list, momentum=args.momentum, nesterov=args.nesterov)

		# define learning rate scheduler.
		# Be careful about the learning rate for different params list, check
		# the `params_list` and the `lr_scheduler` to ensure the strategy is right.
		self.scheduler = train_utils.lr_scheduler(init_lr=self.init_lr,
													mode=self.lr_scheduler,
													num_epochs=self.train_epochs,
													max_iter=self.max_iter,
													slow_start_steps=args.slow_start_steps,
													slow_start_lr=args.slow_start_lr,
													multiplier=self.lr_multiplier)
		# resuming checkpoint
		#self.best_pred = 0.0
		if args.resume is not None:
			if os.path.isfile(args.resume):
				#raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
				print("INFO:PyTorch: Restore checkpoint from {}".format(args.resume))
				checkpoint = torch.load(args.resume)
				self.global_step = checkpoint['global_step']
				if args.cuda:
					self.model_with_loss.module.load_state_dict(checkpoint['state_dict'])
				else:
					self.model_with_loss.load_state_dict(checkpoint['state_dict'])
				self.start_epoch = (self.global_step + 1) // self.steps_per_epochs
		#	if not args.ft:
		#		self.optimizer.load_state_dict(checkpoint['optimizer'])
		#	self.best_pred = checkpoint['best_pred']
		#	print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
		# clear start epoch if fine-tuning
		#if args.ft:
		#	args.start_epoch = 0

	def training(self, epoch):
		"""training procedure
		"""
		# set training mode
		self.model_with_loss.train()
		self.evaluator.reset()
		start_time = time.time()
		self.optimizer.zero_grad()
		# training loop
		for i, sample in enumerate(self.train_loader):
			# set grad zero
			self.optimizer.zero_grad()
			# accumulate global steps
			if (i + 1) % self.accumulation_steps == 0:
				self.global_step += 1
			image, target = sample['image'], sample['label']
			if self.cuda:
				image, target = image.cuda(self.main_gpu), target.cuda(self.main_gpu)
			# adjust learning rate, pass input through the model, update
			self.scheduler(self.optimizer, self.global_step, epoch)
			output, loss = self.model_with_loss(inputs=image, target=target, global_step=self.global_step)
			#print(target.size())
			loss = loss.mean()
			loss.backward()
			#if (i + 1) % self.accumulation_steps == 0:
			# update the parameters and set the gradient to 0.
			self.optimizer.step()
			# add batch sample into evaluator
			pred = np.argmax(output.data.cpu().numpy(), axis=1)
			target = target.cpu().numpy()
			self.evaluator.add_batch(target, pred)

			# log info per 20 steps
			#if self.global_step % 20 == 0 and (i + 1) % self.accumulation_steps == 0:
			if self.global_step % 20 == 0:
				# the used time
				used_time = time.time() - start_time
				px_acc = self.evaluator.pixel_accuracy_np()
				miou = self.evaluator.mean_iou_np()
				lr_now = self.optimizer.param_groups[0]['lr']
				print("INFO:PyTorch: epoch={}/{}, steps={}, loss={:.5f}, learning_rate={:.5f}, train_miou={:.5f}, px_accuracy={:.5f}"
						" ({:.3f} sec)".format(epoch + 1, self.train_epochs,
						self.global_step, loss.item(), lr_now, miou, px_acc, used_time))

				# summary per 100 steps
				if self.global_step % 100 == 0:
					self.train_writer.add_scalar('train_miou', miou, global_step=self.global_step)
					self.train_writer.add_scalar('px_accuracy', px_acc, global_step=self.global_step)
					self.train_writer.add_scalar('learning_rate', lr_now, global_step=self.global_step)
					self.train_writer.add_scalar('train_loss', loss.item(), global_step=self.global_step)

				start_time = time.time()

			# save checkpoints
			if (self.global_step % self.save_ckpt_steps == 0) or (i == self.steps_per_epochs - 1):
				filename = os.path.join(self.model_dir, "{0}_ckpt_{1}.pth".format(self.checkname, self.global_step))
				self.saved_ckpt_filenames.append(filename)
				# remove the newest file if saved ckpts if more than max_ckpt_nums
				if len(self.saved_ckpt_filenames) > self.max_ckpt_nums:
					del_filename = self.saved_ckpt_filenames.pop(0)
					os.remove(del_filename)
				# save new ckpt
				state = {
							'global_step': self.global_step,
							'state_dict': self.model_with_loss.module.state_dict(),
							'optimizer': self.optimizer.state_dict(),
						}
				torch.save(state, filename)

	def validation(self, epoch):
		"""validation procedure
		"""
		# set validation mode
		self.model_with_loss.eval()
		self.evaluator.reset()
		test_loss = 0.0
		for i, sample in enumerate(self.val_loader):
			image, target = sample['image'], sample['label']
			# repeat, you can uncomment this the line
			#image, target = image.repeat(self.num_gpus, 1, 1, 1), target.repeat(self.num_gpus, 1, 1)
			if self.cuda:
				image, target = image.cuda(self.main_gpu), target.cuda(self.main_gpu)
			# forward
			with torch.no_grad():
				output = self.model_with_loss(inputs=image, mode='val')

			# Add batch sample into evaluator
			pred = np.argmax(output.data.cpu().numpy(), axis=1)
			target = target.cpu().numpy()
			self.evaluator.add_batch(target, pred)

		# log and summary the validation results
		px_acc = self.evaluator.pixel_accuracy_np()
		val_miou = self.evaluator.mean_iou_np(is_show_per_class=True)
		print("\nINFO:PyTorch: validation results: miou={:5f}, px_acc={:5f}, loss={:5f} \n".
			format(val_miou, px_acc, test_loss))
		self.val_writer.add_scalar('val_loss', test_loss, self.global_step)
		self.val_writer.add_scalar('val_miou', val_miou, self.global_step)
		self.val_writer.add_scalar('val_px_acc', px_acc, self.global_step)


def main():
	# get the parameters
	parser = argparse.ArgumentParser(description="PyTorch Segmentation Model Training")
	args = parser_params.add_parser_params(parser)
	print(args)
	parser_params.save_hp_to_json(args)
	# set the name of the process
	setproctitle.setproctitle(args.proc_name)

	torch.manual_seed(args.seed)
	trainer = Trainer(args)
	print('INFO:PyTorch: Starting Epoch:', trainer.start_epoch)
	print('INFO:PyTorch: Total Epoches:', trainer.train_epochs)

	# train-eval loops
	for epoch in range(trainer.start_epoch, trainer.train_epochs):
		trainer.training(epoch)
		if not trainer.no_val and ((epoch + 1) % args.eval_interval == 0):
			trainer.validation(epoch)

	trainer.train_writer.close()
	trainer.val_writer.close()


if __name__ == "__main__":
	# accelarate the training
	torch.backends.cudnn.benchmark = True
	torch.cuda.seed_all()
	main()
