#coding=utf-8
"""
some training utils.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['init_weights', 'group_weight', 'seg_model_get_optim_params']


def init_weights(modules, norm_layer=nn.BatchNorm2d, bn_momentum=0.1):
	"""
	as for he_init normal with std = sqrt(2 / (Cin * k * k))
	"""
	if not isinstance(modules, (list, tuple)):
		modules = (modules,)
	for module in modules:
		__init_weights(module, norm_layer, bn_momentum)


def __init_weights(module, norm_layer=nn.BatchNorm2d, bn_momentum=0.1):
	"""
	The defaut init for conv weight and bias is uniform with stdv = 1 / sqrt(Cin * k * k).
	As for he_init  normal with std = sqrt(2 / (Cin * k * k)).
	"""
	for m in module.modules():
		if isinstance(m, (nn.Linear, nn.Conv2d)):
			nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
			if m.bias is not None:
				m.bias.data.zero_()
		elif isinstance(m, norm_layer) or isinstance(m, nn.BatchNorm2d):
			m.momentum = bn_momentum
			m.weight.data.fill_(1)
			m.bias.data.zero_()


def __group_weight(group_decay, group_no_decay, module, norm_layer):
	for m in module.modules():
		if isinstance(m, (nn.Linear, nn.Conv2d)):
			group_decay.append(m.weight)
			if m.bias is not None:
				group_no_decay.append(m.bias)
		elif isinstance(m, (norm_layer, nn.GroupNorm, nn.BatchNorm2d)):
			group_no_decay.append(m.weight)
			group_no_decay.append(m.bias)
	return group_decay, group_no_decay


def group_weight(params_list, modules, norm_layer, lr, weight_decay):
	"""group the weights.
	no weight decay for the biases, and alpha and gamma of the bn layers.

	ref:
		Bag of Tricks for Image Classification with Convolutional Neural Networks, 2018.
	"""
	group_decay = []
	group_no_decay = []
	params_length = 0

	if not isinstance(modules, (list, tuple)):
		modules = (modules, )
	for module in modules:
		params_length += len(list(module.parameters()))
		group_decay, group_no_decay = __group_weight(group_decay, group_no_decay, module, norm_layer)

	assert params_length == len(group_decay) + len(group_no_decay)
	params_list.append(dict(params=group_decay, weight_decay=weight_decay, lr=lr))
	params_list.append(dict(params=group_no_decay, lr=lr))
	return params_list


def seg_model_get_optim_params(params_list, model,
								norm_layer=nn.BatchNorm2d,
								seg_model='pspnet',
								base_lr=0.007,
								lr_multiplier=1.0,
								weight_decay=4e-5):
		"""
		get the params of the segmentation models.
		"""
		# group weight and config optimizer
		modules_list1 = (model.backbone, )
		if seg_model == 'deeplabv3':
			modules_list2 = (model.aspp, model.last_conv)
		elif seg_model == 'deeplabv3+':
			modules_list2 = (model.aspp, model.decoder)
		elif seg_model == 'pspnet':
			modules_list2 = (model.psp_module, model.main_branch, model.aux_branch)
		else:
			raise 
		# get the param list
		params_list = group_weight(params_list, modules_list1,
									norm_layer, base_lr, weight_decay=weight_decay)
		params_list = group_weight(params_list, modules_list2,
									norm_layer, base_lr * lr_multiplier, weight_decay=weight_decay)
		return params_list
