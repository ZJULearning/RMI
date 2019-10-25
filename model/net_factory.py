#coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from RMI.model.backbone import resnet_v1
#from RMI.model.backbone import resnet_v1_beta


__all__ = ['get_backbone_net']


def get_backbone_net(backbone='resnet101',
						output_stride=16,
						pretrained=True,
						norm_layer=nn.BatchNorm2d,
						bn_mom=0.01,
						root_beta=True):
	"""get the backnbone net of the segmentation model"""
	# A map from network name to network object.
	networks_obj_dict = {
		#'mobilenet_v2': _mobilenet_v2,
		'resnet50': resnet_v1.resnet50,
		'resnet101': resnet_v1.resnet101,
		'resnet152': resnet_v1.resnet152,
		#'resnet50_beta': resnet_v1_beta.resnet50_beta,
		#'resnet101_beta': resnet_v1_beta.resnet101_beta,
		#'resnet152_beta': resnet_v1_beta.resnet152_beta,
		#'xception_41': xception.xception_41,
		#'xception_65': xception.xception_65,
	}
	assert backbone in networks_obj_dict.keys()
	if 'resnet' in backbone:
		backbone_net = networks_obj_dict[backbone](output_stride=output_stride,
													pretrained=pretrained,
													norm_layer=norm_layer,
													bn_mom=bn_mom,
													root_beta=root_beta)
	return backbone_net
