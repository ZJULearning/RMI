#coding=utf-8

"""
Implementation of some commonly used losses.
"""

# python 2.X, 3.X compatibility
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

#import os
#import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCECrossEntropyLoss(nn.Module):
	"""
	sigmoid with binary cross entropy loss.
	consider the multiclass task as multi binary classification problem.
	one-vs-rest way.
	SUM over the channel.
	"""
	def __init__(self,
					num_classes=21,
					ignore_index=255):
		super(BCECrossEntropyLoss, self).__init__()
		self.num_classes = num_classes
		self.ignore_index = ignore_index

	def forward(self, logits_4D, labels_4D):
		"""
		Args:
			logits_4D 	:	[N, C, H, W], dtype=float32
			labels_4D 	:	[N, H, W], dtype=long
		"""
		label_flat = labels_4D.view(-1).requires_grad_(False)
		label_mask_flat = label_flat < self.num_classes
		onehot_label_flat = F.one_hot(label_flat * label_mask_flat.long(), num_classes=self.num_classes).float()
		onehot_label_flat = onehot_label_flat.requires_grad_(False)
		logits_flat = logits_4D.permute(0, 2, 3, 1).contiguous().view([-1, self.num_classes])

		# binary loss, multiplied by the not_ignore_mask
		label_mask_flat = label_mask_flat.float()
		valid_pixels = torch.sum(label_mask_flat)
		binary_loss = F.binary_cross_entropy_with_logits(logits_flat,
															target=onehot_label_flat,
															weight=label_mask_flat.unsqueeze(dim=1),
															reduction='sum')
		bce_loss = torch.div(binary_loss, valid_pixels + 1.0)
		return bce_loss
