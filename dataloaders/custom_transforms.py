# coding=utf-8

"""
some custom transforms
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
#from torchvision import transforms

import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter


class RandomRescale(object):
    """rescale an image and label with in target scale
    PIL image version"""
    def __init__(self, min_scale=0.5, max_scale=2.0, step_size=0.25):
        """initialize
        Args:
            min_scale: Min target scale.
            max_scale: Max target scale.
        """
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.step_size = step_size
        # discrete scales
        if (max_scale - min_scale) > step_size and step_size > 0.05:
            self.num_steps = int((max_scale - min_scale) / step_size + 1)
            self.scale_steps = np.linspace(self.min_scale, self.max_scale, self.num_steps)
        elif (max_scale - min_scale) > step_size and step_size < 0.05:
            self.num_steps = 0
            self.scale_steps = np.array([min_scale])
        else:
            self.num_steps = 1
            self.scale_steps = np.array([min_scale])

    def __call__(self, sample):
        """call method"""
        image, label = sample['image'], sample['label']
        width, height = image.size
        # random scale
        if self.num_steps > 0:
            index = random.randint(0, self.num_steps - 1)
            scale_now = self.scale_steps[index]
        else:
            scale_now = random.uniform(self.min_scale, self.max_scale)
        new_width, new_height = int(scale_now * width), int(scale_now * height)
        # resize
        #image = image.resize(self.size, Image.BILINEAR)
        image = image.resize((new_width, new_height), Image.BICUBIC)
        label = label.resize((new_width, new_height), Image.NEAREST)

        return {'image': image,
                'label': label}


class RandomPadOrCrop(object):
    """Crops and/or pads an image to a target width and height
    PIL image version
    """
    def __init__(self, crop_height, crop_width, ignore_label=255, mean=(125, 125, 125)):
        """
        Args:
            crop_height: The new height.
            crop_width: The new width.
            ignore_label: Label class to be ignored.
        """
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.ignore_label = ignore_label
        self.mean = mean

    def __call__(self, sample):
        """call method"""
        image, label = sample['image'], sample['label']
        width, height = image.size
        pad_width, pad_height = max(width, self.crop_width), max(height, self.crop_height)
        pad_width = self.crop_width - width if width < self.crop_width else 0
        pad_height = self.crop_height - height if height < self.crop_height else 0
        # pad the image with constant
        image = ImageOps.expand(image, border=(0, 0, pad_width, pad_height), fill=self.mean)
        label = ImageOps.expand(label, border=(0, 0, pad_width, pad_height), fill=self.ignore_label)
        # random crop image to crop_size
        new_w, new_h = image.size
        x1 = random.randint(0, new_w - self.crop_width)
        y1 = random.randint(0, new_h - self.crop_height)
        image = image.crop((x1, y1, x1 + self.crop_width, y1 + self.crop_height))
        label = label.crop((x1, y1, x1 + self.crop_width, y1 + self.crop_height))

        return {'image': image,
                'label': label}


class RandomHorizontalFlip(object):
    """Randomly flip an image and label horizontally (left to right).
    PIL image version"""
    def __call__(self, sample):
        """call method"""
        image, label = sample['image'], sample['label']
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': image,
                'label': label}


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
        PIL image version.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """call method"""
        image, label = sample['image'], sample['label']
        image = np.array(image).astype(np.float32)
        label = np.array(label).astype(np.float32)
        #image /= 255.0
        image -= self.mean
        image /= self.std

        return {'image': image,
                'label': label}


class Normalize_Image(object):
    """Normalize a tensor image with mean and standard deviation.
        PIL image version.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """call method"""
        image = sample['image']
        image = np.array(image).astype(np.float32)
        #image /= 255.0
        image -= self.mean
        image /= self.std

        return {'image': image}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        # swap color axis because
        # PIL image :   W x H x C
        # numpy image:  H x W x C
        # torch image:  C X H X W
        image, label = sample['image'], sample['label']
        # W x H x C -> H x W x C
        image = np.array(image).astype(np.float32).transpose((2, 0, 1))
        label = np.array(label).astype(np.float32)
        # convet to torch tensor
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        return {'image': image,
                'label': label}


class ToTensor_Image(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        # swap color axis because
        # PIL image :   W x H x C
        # numpy image:  H x W x C
        # torch image:  C X H X W
        image = sample['image']
        # W x H x C -> H x W x C
        image = np.array(image).astype(np.float32).transpose((2, 0, 1))
        # convet to torch tensor
        image = torch.from_numpy(image).float()

        return {'image': image}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        image = image.rotate(rotate_degree, Image.BILINEAR)
        label = label.rotate(rotate_degree, Image.NEAREST)

        return {'image': image,
                'label': label}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        if random.random() < 0.5:
            image = image.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': image,
                'label': label}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        w, h = image.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        image = image.resize((ow, oh), Image.BILINEAR)
        label = label.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = image.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        image = image.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        label = label.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': image,
                'label': label}


class FixedResize(object):
    """resize the image and label to fixed size"""
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        assert image.size == label.size

        image = image.resize(self.size, Image.BILINEAR)
        label = label.resize(self.size, Image.NEAREST)

        return {'image': image,
                'label': label}
