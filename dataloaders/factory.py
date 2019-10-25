# coding=utf-8

import torch
from torch.utils.data import DataLoader
from RMI.dataloaders.datasets import cityscapes, pascal, camvid

__all__ = ['get_data_loader', 'get_dataset']

def get_data_loader(data_dir,
						batch_size=16,
						crop_size=513,
						dataset='pascal',
						split="train",
						num_workers=4,
						pin_memory=True,
						distributed=False):
	"""get the dataset loader"""
	kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory}
	if dataset == 'pascal':
		"""PASCAL VOC dataset"""
		assert split in ['trainaug', 'trainval', 'train', 'val', 'test']
		if 'train' in split:
			print("INFO:PyTorch: Using PASCAL VOC dataset, the training batch size {} and crop size is {}.".
						format(batch_size, crop_size))
			train_set = pascal.VOCSegmentation(data_dir, crop_size, split,
													min_scale=0.5,
													max_scale=2.0,
													step_size=0.25)
			num_class = train_set.NUM_CLASSES
			# distributed training
			if distributed:
				train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
			else:
				train_sampler = None
			train_loader = DataLoader(train_set,
										batch_size=batch_size,
										shuffle=(train_sampler is None),
										sampler=train_sampler,
										drop_last=True,
										**kwargs)
			return train_loader, num_class
		else:
			val_set = pascal.VOCSegmentation(data_dir, crop_size, split)
			num_class = val_set.NUM_CLASSES
			val_loader = DataLoader(val_set, batch_size=1, shuffle=False, **kwargs)
			return val_loader, num_class
	elif dataset == 'cityscapes':
		"""Cityscapes dataset"""
		assert split in ['train', 'val', 'test']
		if 'train' in split:
			print("INFO:PyTorch: Using cityscapes dataset, the training batch size {} and crop size is {}.".
						format(batch_size, crop_size))
			train_set = cityscapes.CityscapesSegmentation(data_dir,
															crop_size=crop_size,
															split=split,
															min_scale=0.75,
															max_scale=1.25,
															step_size=0.0)
			num_class = train_set.NUM_CLASSES
			train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
			return train_loader, num_class
		else:
			val_set = cityscapes.CityscapesSegmentation(data_dir, crop_size, split)
			num_class = val_set.NUM_CLASSES
			val_loader = DataLoader(val_set, batch_size=1, shuffle=False, **kwargs)
			return val_loader, num_class
	elif dataset == 'camvid':
		"""CamVid dataset"""
		assert split in ['train', 'trainval', 'val', 'test']
		if 'train' in split:
			print("INFO:PyTorch: Using camvid dataset, the training batch size {} and crop size is {}.".
						format(batch_size, crop_size))
			train_set = camvid.CamVidSegmentation(data_dir,
													crop_size=crop_size,
													split=split,
													min_scale=0.75,
													max_scale=1.25,
													step_size=0.0)
			num_class = train_set.NUM_CLASSES
			train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
			return train_loader, num_class
		else:
			val_set = camvid.CamVidSegmentation(data_dir, crop_size, split)
			num_class = val_set.NUM_CLASSES
			val_loader = DataLoader(val_set, batch_size=1, shuffle=False, **kwargs)
			return val_loader, num_class
	else:
		"""raise error"""
		raise NotImplementedError("The DataLoader for {} is not not implemented.".format(dataset))
	#
	return


def get_dataset(data_dir,
					batch_size=16,
					crop_size=513,
					dataset='pascal',
					split="train"):
	"""get the dataset"""
	if dataset == 'pascal':
		"""PASCAL VOC dataset"""
		assert split in ['trainaug', 'trainval', 'train', 'val', 'test']
		if 'train' in split:
			print("INFO:PyTorch: Using PASCAL VOC dataset, the training batch size {} and crop size is {}.".
						format(batch_size, crop_size))
			train_set = pascal.VOCSegmentation(data_dir, crop_size, split,
													min_scale=0.5,
													max_scale=2.0,
													step_size=0.25)
			return train_set
		else:
			val_set = pascal.VOCSegmentation(data_dir, crop_size, split=split)
			return val_set
	elif dataset == 'cityscapes':
		"""Cityscapes dataset"""
		assert split in ['train', 'val', 'test']
		if 'train' in split:
			print("INFO:PyTorch: Using cityscapes dataset, the training batch size {} and crop size is {}.".
						format(batch_size, crop_size))
			train_set = cityscapes.CityscapesSegmentation(data_dir,
															crop_size=crop_size,
															split=split,
															min_scale=0.75,
															max_scale=1.25,
															step_size=0.0)
			return train_set
		else:
			val_set = cityscapes.CityscapesSegmentation(data_dir, crop_size, split)
			return val_set
	elif dataset == 'camvid':
		"""CamVid dataset"""
		assert split in ['train', 'trainval', 'val', 'test']
		if 'train' in split:
			print("INFO:PyTorch: Using camvid dataset, the training batch size {} and crop size is {}.".
						format(batch_size, crop_size))
			train_set = camvid.CamVidSegmentation(data_dir,
														crop_size=crop_size,
														split=split,
														min_scale=0.75,
														max_scale=1.25,
														step_size=0.0)
			return train_set
		else:
			val_set = camvid.CamVidSegmentation(data_dir, crop_size, split)
			return val_set
	else:
		"""raise error"""
		raise NotImplementedError("The DataLoader for {} is not not implemented.".format(dataset))
	return None
