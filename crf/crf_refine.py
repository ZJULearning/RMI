#coding=utf-8

import os
import time
import timeit
import argparse
import numpy as np

#import cv2
from PIL import Image

import torch
import torch.nn.functional as F
from RMI import parser_params

from RMI.crf import crf
from RMI.model import psp, deeplab
from RMI.dataloaders import factory
from RMI.utils.metrics import Evaluator


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
		self.crf_iter_steps = args.crf_iter_steps
		self.output_dir = args.output_dir
		# define dataloader
		self.val_loader = factory.get_dataset(args.data_dir,
												batch_size=1,
												dataset=args.dataset,
												split=args.train_split)
		self.nclass = self.val_loader.NUM_CLASSES
		# define network
		assert args.seg_model in seg_model_obj_dict.keys()
		self.seg_model = args.seg_model
		self.model = seg_model_obj_dict[self.seg_model](num_classes=self.nclass,
														backbone=args.backbone,
														output_stride=args.out_stride,
														norm_layer=torch.nn.BatchNorm2d,
														bn_mom=args.bn_mom,
														freeze_bn=True)

		# define criterion
		self.criterion = torch.nn.CrossEntropyLoss(weight=None, ignore_index=255, reduction='mean')

		# define evaluator
		self.evaluator = Evaluator(self.nclass)

		# using cuda
		if args.cuda:
			self.model = torch.nn.DataParallel(self.model, device_ids=self.gpu_ids)
			#patch_replication_callback(self.model)
			self.model = self.model.cuda()
			self.criterion = self.criterion.cuda()

		# resuming checkpoint
		if args.resume is not None:
			if not os.path.isfile(args.resume):
				raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
			print('Restore parameters from the {}'.format(args.resume))
			checkpoint = torch.load(args.resume)
			self.global_step = checkpoint['global_step']

			if args.cuda:
				self.model.module.load_state_dict(checkpoint['state_dict'])
			else:
				self.model.load_state_dict(checkpoint['state_dict'])

	def validation(self):
		"""validation procedure
		"""
		# set validation mode
		self.model.eval()
		self.evaluator.reset()
		test_loss = 0.0
		crf_100_steps = 0.0
		start = timeit.default_timer()
		for i in range(len(self.val_loader)):
			#for i, sample in enumerate(self.val_loader):
			sample = self.val_loader[i]
			image, target = sample['image'], sample['label']
			image, target = image.repeat(self.num_gpus, 1, 1, 1), target.repeat(self.num_gpus, 1, 1)
			#print("{}-th sample, Image shape {}, label shape {}".format(i + 1, image.size(), target.size()))
			if self.cuda:
				image, target = image.cuda(), target.cuda()
			# forward
			with torch.no_grad():
				output = self.model(image)
			# the output of the pspnet is a tuple
			if self.seg_model == 'pspnet':
				output = output[0]
			loss = self.criterion(output, target.long())
			test_loss += loss.item()

			# get probs, shape [N, C, H, W] --> [N, H, W, C]
			probs = F.softmax(output, dim=1).permute(0, 2, 3, 1).squeeze_()
			probs_np = probs.data.cpu().numpy()
			#pred = output.data.cpu().numpy()
			target = target.squeeze_().cpu().numpy()

			# CRF post-processing
			image_name = self.val_loader.image_lists[i]
			#real_image = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB)

			real_image = Image.open(image_name).convert('RGB')
			real_image = np.array(real_image).astype(np.uint8)

			crf_start = timeit.default_timer()
			pred = crf.dense_crf(real_image=real_image, probs=probs_np, iter_steps=self.crf_iter_steps)
			crf_end = timeit.default_timer()
			crf_100_steps += (crf_end - crf_start)
			# save output
			path_to_output = os.path.join(self.output_dir, self.val_loader.image_ids[i] + '.png')
			result = Image.fromarray(pred.astype(np.uint8))
			result.save(path_to_output)
			#cv2.imwrite(path_to_output, pred)
			# report time of CRF
			if not i % 100:
				stop = timeit.default_timer()
				print("current step = {} ({:.3f} sec), crf time {:.3f} sec".
									format(i, stop - start, crf_100_steps))
				crf_100_steps = 0.0
				start = timeit.default_timer()
			#pred = np.argmax(pred, axis=1)
			#pred = np.argmax(pred, axis=2)
			# Add batch sample into evaluator
			self.evaluator.add_batch(target, pred)

		# log and summary the validation results
		# log and summary the validation results
		px_acc = self.evaluator.pixel_accuracy_np()
		val_miou = self.evaluator.mean_iou_np(is_show_per_class=True)
		print("\nINFO:PyTorch: validation results: miou={:5f}, px_acc={:5f}, loss={:5f} \n".
			format(val_miou, px_acc, test_loss))
		print("The iteration steps of CRF is {}.".format(self.crf_iter_steps))


def main():
	# get the parameters
	parser = argparse.ArgumentParser(description="PyTorch Segmentation Model Training")
	args = parser_params.add_parser_params(parser)
	print(args)

	torch.manual_seed(args.seed)
	trainer = Trainer(args)
	start_time = time.time()
	trainer.validation()
	total_time = time.time() - start_time
	print("The validation time is {:.5f} sec".format(total_time))


if __name__ == "__main__":
	main()
