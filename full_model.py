# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import torch
import torch.nn as nn
import torch.nn.functional as F

_PSP_AUX_WEIGHT = 0.4		# the weight of the auxiliary loss in PSPNet


class FullModel(nn.Module):
	"""The full model wrapper."""
	def __init__(self, seg_model='deeplabv3',
						model=None,
						loss_type=None,
						criterion=None):
		super(FullModel, self).__init__()
		assert seg_model in ['pspnet', 'deeplabv3', 'deeplabv3+']
		self.seg_model = seg_model
		self.model = model
		self.loss_type = loss_type
		self.criterion = criterion

	def forward(self, inputs=None, target=None, global_step=0, mode='train'):
		"""forward step"""
		# output of the model
		output = self.model(inputs)

		# do not calclate the loss during validation or testing
		if 'val' in mode or 'test' in mode:
			if self.seg_model == 'pspnet':
				output = output[0]
			return output

		# PSPNet have auxilary branch
		if self.loss_type == 2:
			if self.seg_model == 'pspnet':
				#loss = self.criterion(output[0], target) + _PSP_AUX_WEIGHT * self.criterion(output[1], target)
				#loss = loss / (1.0 + _PSP_AUX_WEIGHT)
				loss = self.criterion(output[0], target) + _PSP_AUX_WEIGHT * F.cross_entropy(input=output[1],
																								target=target.long(),
																								ignore_index=255,
																								reduction='mean')
				output = output[0]
			else:
				loss = self.criterion(output, target)
		elif self.loss_type == 3:
			if self.seg_model == 'pspnet':
				loss = (self.criterion(output[0], target, global_step=global_step) +
						_PSP_AUX_WEIGHT * self.criterion(output[1], target, global_step=global_step))
				output = output[0]
			else:
				loss = self.criterion(output, target, global_step=global_step)
		elif self.loss_type == 5:
			if self.seg_model == 'pspnet':
				loss = self.criterion(output[0], target) + _PSP_AUX_WEIGHT * self.criterion(output[1], target)
				output = output[0]
			else:
				loss = self.criterion(output, target)
		else:
			if self.seg_model == 'pspnet':
				loss = (self.criterion(output[0], target.long()) + _PSP_AUX_WEIGHT * self.criterion(output[1], target.long()))
				output = output[0]
			else:
				loss = self.criterion(output, target.long())
		#loss = loss.unsqueeze(dim=0)
		return output, loss
