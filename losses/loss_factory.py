# coding=utf-8

# python 2.X, 3.X compatibility
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

#import torch
import torch.nn as nn

from RMI.losses import normal_loss
from RMI.losses import pyramid_loss
from RMI.losses.rmi import rmi
from RMI.losses.affinity import aaf

def criterion_choose(num_classes=21,
						loss_type=0,
						weight=None,
						ignore_index=255,
						reduction='mean',
						max_iter=30000,
						args=None):
	"""choose the criterion to use"""
	info_dict = {
			0: "Normal Softmax Cross Entropy Loss",
			1: "Normal Sigmoid Cross Entropy Loss",
			2: "Region Mutual Information Loss",
			3: "Affinity field Loss",
			5: "Pyramid Loss"
	}
	print("INFO:PyTorch: Using {}.".format(info_dict[loss_type]))
	if loss_type == 0:
		return nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)
	elif loss_type == 1:
		return normal_loss.BCECrossEntropyLoss(num_classes=num_classes, ignore_index=ignore_index)
	elif loss_type == 2:
		return rmi.RMILoss(num_classes=num_classes,
							rmi_radius=args.rmi_radius,
							rmi_pool_way=args.rmi_pool_way,
							rmi_pool_size=args.rmi_pool_size,
							rmi_pool_stride=args.rmi_pool_stride,
							loss_weight_lambda=args.loss_weight_lambda)
	elif loss_type == 3:
		return aaf.AffinityLoss(num_classes=num_classes,
								init_step=args.init_global_step,
								max_iter=max_iter)
	elif loss_type == 5:
		return pyramid_loss.PyramidLoss(num_classes=num_classes, ignore_index=ignore_index)

	else:
		raise NotImplementedError("The loss type {} is not implemented.".format(loss_type))
