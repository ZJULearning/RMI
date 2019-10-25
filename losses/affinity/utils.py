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

import torch
import torch.nn.functional as F


__all__ = ['edges_from_label', 'eightcorner_activation']


def edges_from_label(labels, size=1, ignore_class=255):
	"""Retrieves edge positions from the ground-truth labels.
	This function computes the edge map by considering if the pixel values
	are equal between the center and the neighboring pixels on the eight
	corners from a (2 * size + 1) * (2 * size + 1) patch.
	Ignore edges where the any of the paired pixels with label value >= num_classes.

	Args:
		labels: 		A tensor of size [N, C, H, W],
						indicating semantic segmentation ground-truth labels.
		size: 			A number indicating the half size of a patch.
		ignore_class: 	A number indicating the label value to ignore.
	Return:
		A tensor of size [N, C, 8, H, W]
	"""
	# Get the number of channels in the input.
	shape_lab = labels.size()
	assert len(shape_lab) == 4
	n, c, h, w = shape_lab

	# Pad at the margin.
	labels_pad = F.pad(labels, (size, size, size, size), mode='constant', value=0)

	# Get the edge by comparing label value of the center and it paired pixels.
	edge_groups = []
	for st_y in range(0, 2 * size + 1, size):
		for st_x in range(0, 2 * size + 1, size):
			if st_y == size and st_x == size:
				continue
			edge_groups.append(labels_pad[:, :, st_y:st_y+h, st_x:st_x+w] != labels)
	# shape [N, C, 8, H, W]
	return torch.stack(edge_groups, dim=2)


def eightcorner_activation(x, size=1):
	"""Retrieves neighboring pixels one the eight corners from a
		(2 * size + 1) x (2 * size + 1) patch.
	Args:
		x: 		A tensor with shape [N, C, H, W]
		size: 	A number indicating the half size of a patch.
	Returns:
		A tensor with shape [N, C, 8, H, W]
	"""
	# Get the number of channels in the input.
	shape_lab = x.size()
	assert len(shape_lab) == 4
	n, c, h, w = shape_lab

	# Pad at the margin.
	x_pad = F.pad(x, (size, size, size, size), mode='constant', value=0)

	# Get eight corner pixels/features in the patch.
	x_groups = []
	for st_y in range(0, 2 * size + 1, size):
		for st_x in range(0, 2 * size + 1, size):
			if st_y == size and st_x == size:
				# Ignore the center pixel/feature.
				continue
			x_groups.append(x_pad[:, :, st_y:st_y+h, st_x:st_x+w])

	# shape [N, C, 8, H, W]
	return torch.stack(x_groups, dim=2)
