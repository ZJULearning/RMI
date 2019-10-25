#coding=utf-8

"""
The pytorch implementation of the paper:
@inproceedings{aaf2018,
	author = {Ke, Tsung-Wei and Hwang, Jyh-Jing and Liu, Ziwei and Yu, Stella X.},
	title = {Adaptive Affinity Fields for Semantic Segmentation},
	booktitle = {European Conference on Computer Vision (ECCV)},
	month = {September},
	year = {2018}
}
"""

# python 2.X, 3.X compatibility
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from RMI.losses.affinity import utils as aaf_utils


__all__ = ['AffinityLoss']


_BOT_EPSILON = 1e-4
_TOP_EPSILON = 1.0


class AffinityLoss(nn.Module):
	"""
	The affinity field loss.
	"""
	def __init__(self,
					num_classes=21,
					ignore_index=255,
					kld_lambda_1=1.0,
					kld_lambda_2=1.0,
					kld_margin=3.0,
					init_step=0,
					max_iter=30000):
		super(AffinityLoss, self).__init__()
		self.num_classes = num_classes
		# factor of aaf
		self.kld_lambda_1 = kld_lambda_1
		self.kld_lambda_2 = kld_lambda_2
		self.kld_margin = kld_margin
		self.ignore_index = ignore_index
		self.reduction = 'mean'
		self.down_stride = 8
		self.init_step = init_step
		self.max_iter = max_iter

	def forward(self, logits_4D, labels_3D, global_step=0):
		"""
		Args:
			logits_4D 	:	[N, C, H, W], dtype=float32
			labels_4D 	:	[N, H, W], dtype=long
		"""
		# PART I -- get the normal cross entropy loss
		normal_loss = F.cross_entropy(input=logits_4D,
										target=labels_3D.long(),
										ignore_index=self.ignore_index,
										reduction=self.reduction)

		# PART II -- get the affinity field loss
		# downsample the logits and labels to save memory
		shape = logits_4D.size()
		new_h, new_w = shape[2] // (self.down_stride), shape[3] // (self.down_stride)
		labels_3D = F.interpolate(labels_3D.unsqueeze(dim=1), size=(new_h, new_w), mode='nearest')
		labels_3D = labels_3D.squeeze(dim=1)
		logits_4D = F.interpolate(logits_4D, size=(new_h, new_w), mode='bilinear', align_corners=True)

		# get the valid label and logits
		# valid label, [N, C, H, W]
		label_mask_3D = labels_3D < self.num_classes
		valid_onehot_labels_4D = F.one_hot(labels_3D.long() * label_mask_3D.long(), num_classes=self.num_classes).float()
		label_mask_3D = label_mask_3D.float()
		valid_onehot_labels_4D = valid_onehot_labels_4D * label_mask_3D.unsqueeze(dim=3)
		valid_onehot_labels_4D = valid_onehot_labels_4D.permute(0, 3, 1, 2).requires_grad_(False)
		# valid probs
		probs_4D = F.softmax(logits_4D, dim=1) * label_mask_3D.unsqueeze(dim=1)
		probs_4D = probs_4D.clamp(min=_BOT_EPSILON, max=_TOP_EPSILON)

		# decay as https://github.com/twke18/Adaptive_Affinity_Fields
		aff_decay = math.pow(20.0, 0.0 - 1.0 * (global_step - self.init_step) / float(self.max_iter))
		aaf_loss = aff_decay * self.affinity_loss(probs_4D, labels_4D=valid_onehot_labels_4D)
		# the final loss
		final_loss = normal_loss + aaf_loss
		return final_loss

	def affinity_loss(self, probs_4D, labels_4D=None, size=1):
		"""
		Args:
			logits_4D 	:	[N, C, H, W], dtype=float32
			labels_4D 	:	[N, C, H, W], dtype=float32
			size		:	default 1.
		"""
		# edge, shape [N, C, 8, H, W]
		edge = aaf_utils.edges_from_label(labels_4D, size=size)
		edge = edge.view(-1)

		# neighbour points, [N, C, 8, H, W]
		probs_paired = aaf_utils.eightcorner_activation(probs_4D, size=size)
		probs_paired = torch.clamp(probs_paired, min=_BOT_EPSILON, max=_TOP_EPSILON)
		probs_4D = probs_4D.unsqueeze(dim=2)
		neg_probs_4D = 1.0 - probs_4D + _BOT_EPSILON
		neg_probs_paired = 1.0 - probs_paired + _BOT_EPSILON

		# compute KL-Divergence
		KL_div = (probs_paired * (probs_paired.log() - probs_4D.log()) +
					neg_probs_paired * (neg_probs_paired.log() - neg_probs_4D.log()))
		KL_div = KL_div.view(-1)
		edge_loss = torch.max(torch.zeros(1).type_as(KL_div), self.kld_margin - KL_div)

		# average
		edge_loss = torch.mean(edge_loss[edge])
		not_edge_loss = torch.mean(KL_div[~edge])
		aaf_loss = edge_loss * self.kld_lambda_1 + not_edge_loss * self.kld_lambda_2
		return aaf_loss
