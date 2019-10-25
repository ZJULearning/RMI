#coding=utf-8

"""
dataloader for CamVid dataset
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
from RMI.dataloaders import custom_transforms as tr


__all__ = ['CamVidSegmentation']


# CamVid dataset statistics
_CamVid_R_MEAN = 100
_CamVid_G_MEAN = 103
_CamVid_B_MEAN = 106

_CamVid_R_STD = 75.61
_CamVid_G_STD = 77.81
_CamVid_B_STD = 76.70


# CamVid
camvid_label_colours = [(128, 128, 128),  	#0=Sky
							# 1=Building,   2=Pole, 3=Road, 4=Pavement, 5=Tree
							(128, 0, 0), (192, 192, 128), (128, 64, 128), (60, 40, 222), (128, 128, 0),
							# 6=SignSymbol, 7=Fence,  8=Car,  9=Pedestrian, 10=Bicyclist
							(192, 128, 128), (64, 64, 128), (64, 0, 128), (64, 64, 0), (0, 128, 192),
							# 11=Unlabelled
							(0, 0, 0)]


class CamVidSegmentation(data.Dataset):
	NUM_CLASSES = 12

	def __init__(self,
					data_dir,
					crop_size=479,
					split="train",
					min_scale=0.75,
					max_scale=1.25,
					step_size=0.0):
		"""
		Only support the gtFine part.
		Args:
			data_dir:   	path to CamVidscapes dataset directory.
			crop_size:		the crop size.
			split:      	["train", val", "test"].
		"""
		super().__init__()
		# dataset dir
		self.data_dir = data_dir
		self.split = split

		assert self.split in ['train', 'val', 'test', 'trainval']
		self.data_list_file = os.path.join(self.data_dir, '{}.txt'.format(self.split))
		self.iamge_dir = os.path.join(self.data_dir, self.split)
		self.label_dir = os.path.join(self.data_dir, '{}annot'.format(self.split))

		# crop size and scales
		self.crop_size = crop_size
		self.min_scale = min_scale
		self.max_scale = max_scale
		self.step_size = step_size

		# dataset info
		self.mean = (_CamVid_R_MEAN, _CamVid_G_MEAN, _CamVid_B_MEAN)
		self.std = (_CamVid_R_STD, _CamVid_G_STD, _CamVid_B_STD)
		self.ignore_label = 255

		# read file list
		with open(self.data_list_file, "r") as f:
			lines = f.read().splitlines()
			lines = [line.strip().split(' ')[0] for line in lines]

		# extract the id_now
		self.image_ids = [id_now.split('/')[-1] for id_now in lines]
		# the file list, all are *.png files
		self.image_lists = [os.path.join(self.iamge_dir, filename) for filename in self.image_ids]
		self.label_lists = [os.path.join(self.label_dir, filename) for filename in self.image_ids]

		assert (len(self.image_lists) == len(self.label_lists))

		# print the dataset info
		print('Number of image_lists in {}: {:d}'.format(split, len(self.image_lists)))

	def __len__(self):
		"""len() method"""
		return len(self.image_lists)

	def __getitem__(self, index):
		"""how to get the data"""
		_image, _label = self._make_img_gt_point_pair(index)
		sample = {'image': _image, 'label': _label}

		if 'train' in self.split:
			return self.transform_train(sample)
		elif 'val' in self.split or 'test' in self.split:
			return self.transform_val(sample)
		else:
			raise NotImplementedError

	def _make_img_gt_point_pair(self, index):
		"""open the image and the gorund truth"""
		_image = Image.open(self.image_lists[index]).convert('RGB')
		_label = Image.open(self.label_lists[index])
		return _image, _label

	def transform_train(self, sample):
		composed_transforms = transforms.Compose([
			tr.RandomRescale(self.min_scale, self.max_scale, self.step_size),
			tr.RandomPadOrCrop(crop_height=self.crop_size, crop_width=self.crop_size,
								ignore_label=self.ignore_label, mean=self.mean),
			tr.RandomHorizontalFlip(),
			tr.Normalize(mean=self.mean, std=self.std),
			tr.ToTensor()])

		return composed_transforms(sample)

	def transform_val(self, sample):
		"""transform for validation"""
		composed_transforms = transforms.Compose([
			tr.Normalize(mean=self.mean, std=self.std),
			tr.ToTensor()])

		return composed_transforms(sample)

	def __str__(self):
		return 'CamVid(split=' + str(self.split) + ')'


if __name__ == '__main__':
	# data dir
	data_dir = os.path.join("/home/zhaoshuai/dataset/CamVid")
	print(data_dir)
	dataset = CamVidSegmentation(data_dir=data_dir, split='trainval')
	#print(dataset.image_lists)
	image_mean = np.array([0.0, 0.0, 0.0])
	cov_sum = np.array([0.0, 0.0, 0.0])
	pixel_nums = 0.0
	# mean
	for filename in dataset.image_lists:
		image = Image.open(filename).convert('RGB')
		image = np.array(image).astype(np.float32)
		pixel_nums += image.shape[0] * image.shape[1]
		image_mean += np.sum(image, axis=(0, 1))
	image_mean = image_mean / pixel_nums
	print(image_mean)
	# covariance
	for filename in dataset.image_lists:
		image = Image.open(filename).convert('RGB')
		image = np.array(image).astype(np.float32)
		cov_sum += np.sum(np.square(image - image_mean), axis=(0, 1))
	image_cov = np.sqrt(cov_sum / (pixel_nums - 1))
	print(image_cov)
