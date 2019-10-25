# coding=utf-8

"""
dataloader for PASCAL VOC 2012 dataset
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

from RMI.dataloaders import custom_transforms as tr


# PASCAL VOC 2012 dataset statistics
_PASCAL_R_MEAN = 116
_PASCAL_G_MEAN = 113
_PASCAL_B_MEAN = 104

_PASCAL_R_STD = 69.58
_PASCAL_G_STD = 68.68
_PASCAL_B_STD = 72.67


class VOCSegmentation(Dataset):
	"""PASCAL VOC 2012 dataset
	"""
	NUM_CLASSES = 21

	def __init__(self,
					data_dir,
					crop_size=513,
					split='train',
					min_scale=0.5,
					max_scale=2.0,
					step_size=0.25):
		"""
		Args:
			data_dir:   	path to VOC dataset directory.
			crop_size:		the crop size.
			split:      	["trainaug", "train", "trainval", "val", "test"].
		"""
		super().__init__()
		# dataset dir
		self.data_dir = data_dir
		self.iamge_dir = os.path.join(self.data_dir, 'JPEGImages')
		self.label_dir = os.path.join(self.data_dir, 'SegmentationClassAug')

		assert split in ["trainaug", "train", "trainval", "val", "test"]
		self.split = split
		# txt lists of images
		list_file_dir = os.path.join(self.data_dir, 'ImageSets/Segmentation')

		# crop size and scales
		self.crop_size = crop_size
		self.min_scale = min_scale
		self.max_scale = max_scale
		self.step_size = step_size

		# dataset info
		self.mean = (_PASCAL_R_MEAN, _PASCAL_G_MEAN, _PASCAL_B_MEAN)
		self.std = (_PASCAL_R_STD, _PASCAL_G_STD, _PASCAL_B_STD)
		self.ignore_label = 255
		self.image_ids = []
		self.image_lists = []
		self.label_lists = []

		# read the dataset file
		with open(os.path.join(os.path.join(list_file_dir, self.split + '.txt')), "r") as f:
			lines = f.read().splitlines()

		for line in lines:
			image_filename = os.path.join(self.iamge_dir, line + ".jpg")
			label_filename = os.path.join(self.label_dir, line + ".png")
			assert os.path.isfile(image_filename)
			if 'test' not in self.split:
				assert os.path.isfile(label_filename)
			self.image_ids.append(line)
			self.image_lists.append(image_filename)
			self.label_lists.append(label_filename)

		assert (len(self.image_lists) == len(self.label_lists))

		# print the dataset info
		print('Number of image_lists in {}: {:d}'.format(split, len(self.image_lists)))

	def __len__(self):
		"""len() method"""
		return len(self.image_lists)

	def __getitem__(self, index):
		"""index method"""
		_image, _label = self._make_img_gt_point_pair(index)

		# different transforms for different splits
		if 'train' in self.split:
			sample = {'image': _image, 'label': _label}
			return self.transform_train(sample)
		elif 'val' in self.split:
			sample = {'image': _image, 'label': _label}
			return self.transform_val(sample)
		elif 'test' in self.split:
			sample = {'image': _image}
			return self.transform_test(sample)
		else:
			raise NotImplementedError

	def _make_img_gt_point_pair(self, index):
		"""open the image and the gorund truth"""
		_image = Image.open(self.image_lists[index]).convert('RGB')
		if 'test' not in self.split:
			_label = Image.open(self.label_lists[index])
		else:
			_label = None
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

	def transform_test(self, sample):
		"""transform for validation"""
		composed_transforms = transforms.Compose([
			tr.Normalize_Image(mean=self.mean, std=self.std),
			tr.ToTensor_Image()])

		return composed_transforms(sample)

	def __str__(self):
		return 'VOC2012(split=' + str(self.split) + ')'


if __name__ == '__main__':
	# data dir
	data_dir = os.path.join("/home/zhaoshuai/dataset/VOCdevkit/VOC2012")
	print(data_dir)
	dataset = VOCSegmentation(data_dir)
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
