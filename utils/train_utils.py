#coding=utf-8
"""
some training utils.
reference:
	https://github.com/zhanghang1989/PyTorch-Encoding
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import os
import math
import torch
from torchvision.utils import make_grid
#from tensorboardX import SummaryWriter
from RMI.dataloaders.utils import decode_seg_map_sequence


class lr_scheduler(object):
	"""learning rate scheduler
	step mode: 		``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``
	cosine mode: 	``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``
	poly mode: 		``lr = baselr * (1 - iter/maxiter) ^ 0.9``

	Args:
		init_lr:			initial learnig rate;
		mode:				['cos', 'poly', 'step'];
		num_epochs:			traing steps;
		max_iter:			max iterations of training;
		lr_step:			hope you do not use this argument;
		slow_start_steps:	slow start steps of training;
		slow_start_lr:		slow start learning rate for slow_start_steps;
		end_lr:				minimum learning rate.
	"""
	def __init__(self, init_lr,
						mode='poly',
						num_epochs=30,
						max_iter=30000,
						lr_step=1,
						slow_start_steps=0,
						slow_start_lr=1e-4,
						end_lr=1e-6,
						multiplier=1.0):
		self.init_lr = init_lr
		self.now_lr = self.init_lr
		self.mode = mode
		self.num_epochs = num_epochs
		self.max_iter = max_iter
		self.slow_start_steps = slow_start_steps
		self.slow_start_lr = slow_start_lr
		self.slow_max_iter = self.max_iter - self.slow_start_steps
		self.end_lr = end_lr
		self.multiplier = multiplier
		# step mode
		if self.mode == 'step':
			assert lr_step
		self.lr_step = lr_step
		# log info
		print('INFO:PyTorch: Using {} learning rate scheduler!'.format(self.mode))

	def __call__(self, optimizer, global_step, epoch=1.0):
		"""call method"""
		step_now = 1.0 * global_step

		if global_step <= self.slow_start_steps:
			# slow start strategy -- warm up
			# see 	https://arxiv.org/pdf/1812.01187.pdf
			# 	Bag of Tricks for Image Classification with Convolutional Neural Networks
			# for details.
			lr = (step_now / self.slow_start_steps) * (self.init_lr - self.slow_start_lr)
			lr = lr + self.slow_start_lr
			lr = min(lr, self.init_lr)
		else:
			step_now = step_now - self.slow_start_steps
			# calculate the learning rate
			if self.mode == 'cos':
				lr = 0.5 * self.init_lr * (1.0 + math.cos(step_now / self.slow_max_iter * math.pi))
			elif self.mode == 'poly':
				lr = self.init_lr * pow(1.0 - step_now / self.slow_max_iter, 0.9)
			#elif self.mode == 'step':
			#	lr = self.init_lr * (0.1 ** (epoch // self.lr_step))
			else:
				raise NotImplementedError
			lr = max(lr, self.end_lr)

		self.now_lr = lr
		# adjust learning rate
		self._adjust_learning_rate(optimizer, lr)

	def _adjust_learning_rate(self, optimizer, lr):
		"""adjust the leaning rate"""
		if len(optimizer.param_groups) == 1:
			optimizer.param_groups[0]['lr'] = lr
		else:
			# BE CAREFUL HERE!!!
			# 0 -- the backbone conv weights with weight decay
			# 1 -- the bn params and bias of backbone without weight decay
			# 2 -- the weights of other layers with weight decay
			# 3 -- the bn params and bias of other layers without weigth decay
			optimizer.param_groups[0]['lr'] = lr
			optimizer.param_groups[1]['lr'] = lr
			for i in range(2, len(optimizer.param_groups)):
				optimizer.param_groups[i]['lr'] = lr * self.multiplier


def visualize_image(writer, dataset, image, target, output, global_step):
	"""summary image during training.
	"""
	grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
	writer.add_image('Image', grid_image, global_step)
	grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),
														dataset=dataset), 3, normalize=False, range=(0, 255))
	writer.add_image('Predicted label', grid_image, global_step)
	grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(),
														dataset=dataset), 3, normalize=False, range=(0, 255))
	writer.add_image('Groundtruth label', grid_image, global_step)
