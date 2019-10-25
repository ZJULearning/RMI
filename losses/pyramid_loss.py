#coding=utf-8

# python 2.X, 3.X compatibility
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

#import os
#import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class PyramidLoss(nn.Module):
	"""
	Pyramid Loss.
	"""
	def __init__(self,
					num_classes=21,
					ignore_index=255,
					scales=(0.25, 0.5, 0.75, 1.0)):
		super(PyramidLoss, self).__init__()
		self.num_classes = num_classes
		# ignore class
		self.ignore_index = ignore_index
		self.scales = scales

	def forward(self, logits_4D, labels_3D):
		"""
		Using both softmax and sigmoid operations.
		Args:
			logits_4D 	:	[N, C, H, W], dtype=float32
			labels_4D 	:	[N, H, W], dtype=long
		"""
		h, w = labels_3D.shape[-2], labels_3D.shape[-1]
		total_loss = F.cross_entropy(input=logits_4D,
										target=labels_3D.long(),
										ignore_index=self.ignore_index,
										reduction='mean')
		labels_4D = labels_3D.unsqueeze(dim=1)
		for scale in self.scales:
			if scale == 1.0:
				continue
			assert scale <= 1.0
			now_h, now_w = int(scale * h), int(scale * w)
			now_logits = F.interpolate(logits_4D, size=(now_h, now_w), mode='bilinear')
			now_labels = F.interpolate(labels_4D, size=(now_h, now_w), mode='nearest')
			now_loss = F.cross_entropy(input=now_logits,
										target=now_labels.squeeze(dim=1).long(),
										ignore_index=self.ignore_index,
										reduction='mean')
			total_loss += now_loss
		final_loss = total_loss / len(self.scales)
		return final_loss


if __name__ == '__main__':
	pass
