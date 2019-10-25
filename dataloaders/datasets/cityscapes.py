#coding=utf-8

"""
dataloader for Cityscapes dataset
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

__all__ = ['CityscapesSegmentation']


# Cityscapes dataset statistics
_City_R_MEAN = 73
_City_G_MEAN = 83
_City_B_MEAN = 72

_City_R_STD = 47.67
_City_G_STD = 48.49
_City_B_STD = 47.74


class CityscapesSegmentation(data.Dataset):
	NUM_CLASSES = 19

	def __init__(self,
					data_dir,
					crop_size=769,
					split="train",
					min_scale=0.75,
					max_scale=1.25,
					step_size=0.0):
		"""
		Only support the gtFine part.
		Args:
			data_dir:   	path to Cityscapes dataset directory.
			crop_size:		the crop size.
			split:      	["train", val", "test"].
		"""
		super().__init__()
		# dataset dir
		self.data_dir = data_dir
		self.iamge_dir = os.path.join(self.data_dir, 'leftImg8bit')
		self.label_dir = os.path.join(self.data_dir, 'gtFine')
		self.split = split

		assert self.split in ['train', 'val', 'test']
		if self.split == 'train':
			self.data_list_file = os.path.join(self.iamge_dir, 'train_images.txt')
		elif self.split == 'val':
			self.data_list_file = os.path.join(self.iamge_dir, 'val_images.txt')
		elif self.split == 'test':
			self.data_list_file = os.path.join(self.iamge_dir, 'test_images.txt')

		# crop size and scales
		self.crop_size = crop_size
		self.min_scale = min_scale
		self.max_scale = max_scale
		self.step_size = step_size

		# dataset info
		self.mean = (_City_R_MEAN, _City_G_MEAN, _City_B_MEAN)
		self.std = (_City_R_STD, _City_G_STD, _City_B_STD)
		self.ignore_label = 255

		# We assume that the label is already converted.
		#self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
		#self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
		#self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence',
		#					'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain',
		#					'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
		#					'motorcycle', 'bicycle']

		with open(self.data_list_file, "r") as f:
			lines = f.read().splitlines()
			lines = [line.strip().split(' ')[0] for line in lines]

		# extract the id_now
		#image_ids = [id_now.strip() for id_now in lines]
		self.image_ids = [id_now.split('/')[-1] for id_now in lines]
		self.image_ids = [id_now.replace('_leftImg8bit.png', '') for id_now in self.image_ids]

		# the file list
		image_base_dir = os.path.join(self.iamge_dir, self.split)
		label_base_dir = os.path.join(self.label_dir, self.split)
		self.image_lists = [os.path.join(image_base_dir, filename.split('_')[0], filename + '_leftImg8bit.png')
								for filename in self.image_ids]
		self.label_lists = [os.path.join(label_base_dir, filename.split('_')[0], filename + '_gtFine_trainIds.png')
								for filename in self.image_ids]

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
		return 'Cityscapes(split=' + str(self.split) + ')'


if __name__ == '__main__':
	# data dir
	data_dir = os.path.join("/home/zhaoshuai/dataset/Cityscapes")
	print(data_dir)
	dataset = CityscapesSegmentation(data_dir=data_dir)
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
