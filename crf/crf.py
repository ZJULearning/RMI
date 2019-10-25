# coding=utf-8
"""
Function which returns the labelled image after applying CRF.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import cv2
import numpy as np
from PIL import Image
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax


def dense_crf(real_image, probs, iter_steps=10):
	"""
	Args:
		real_image  		:   the real world RGB image, numpy array, shape [H, W, 3]
		probs       		:   the predicted probability in (0, 1), shape [H, W, C]
		iter_steps			:	the iterative steps
	Returns:
		return the refined segmentation map in [0,1,2,...,N_label]
	ref:
		https://github.com/milesial/Pytorch-UNet/blob/master/utils/crf.py
		https://github.com/lucasb-eyer/pydensecrf/blob/master/examples/Non%20RGB%20Example.ipynb
	"""
	# converting real -world image to RGB if it is gray
	if(len(real_image.shape) < 3):
		#real_image = cv2.cvtColor(real_image, cv2.COLOR_GRAY2RGB)
		raise ValueError("The input image should be RGB image.")
	# shape, and transpose to [C, H, W]
	H, W, N_classes = probs.shape[0], probs.shape[1], probs.shape[2]
	probs = probs.transpose((2, 0, 1))
	# get unary potentials from the probability distribution
	unary = unary_from_softmax(probs)
	#unary = np.ascontiguousarray(unary)
	# CRF
	d = dcrf.DenseCRF2D(W, H, N_classes)
	d.setUnaryEnergy(unary)
	# add pairwise potentials
	#real_image = np.ascontiguousarray(real_image)
	d.addPairwiseGaussian(sxy=3, compat=3)
	d.addPairwiseBilateral(sxy=30, srgb=13, rgbim=real_image, compat=10)
	#d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
	#d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=real_image,
	#						compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
	# inference
	Q = d.inference(iter_steps)
	Q = np.argmax(np.array(Q), axis=0).reshape((H, W))
	return Q


if __name__ == '__main__':
	import cv2

	def decode_labels(mask, num_images=1, num_classes=21, color_list=None):
		"""Decode batch of segmentation masks.
		Args:
			mask: result of inference after taking argmax.
			num_images: number of images to decode from the batch.
			num_classes: number of classes to predict (including background).
		Returns:
			A batch with num_images RGB images of the same size as the input.
		"""
		n, h, w, c = mask.shape
		assert (n >= num_images)
		outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
		for i in range(num_images):
			img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
			pixels = img.load()
			for j_, j in enumerate(mask[i, :, :, 0]):
				for k_, k in enumerate(j):
					if k < num_classes:
						pixels[k_, j_] = color_dict[k]
			outputs[i] = np.array(img)
		return outputs

	img = cv2.cvtColor(cv2.imread('img.png'), cv2.COLOR_BGR2RGB)
	prob_01 = cv2.cvtColor(cv2.imread('01_bg.png'), cv2.COLOR_BGR2GRAY)
	prob_02 = cv2.cvtColor(cv2.imread('02_dog.png'), cv2.COLOR_BGR2GRAY)
	prob_03 = cv2.cvtColor(cv2.imread('03_sofa.png'), cv2.COLOR_BGR2GRAY)
	prob = np.stack([prob_01, prob_02, prob_03], axis=-1) / 255.0
	H, W = prob.shape[0], prob.shape[1]
	img = cv2.resize(img, (W, H))
	pred = dense_crf(img.astype(np.uint8), prob.astype(np.float32), iter_steps=1)
	print(prob.shape, np.min(prob), np.max(prob))
	print(img.shape)
	print(pred.shape)
	# background, dog, sofa
	color_dict = [(0, 0, 0), (64, 0, 128), (0, 192, 0)]
	pred = np.expand_dims(pred, axis=0)
	pred = np.expand_dims(pred, axis=-1)
	out = decode_labels(pred, num_images=1, num_classes=3, color_list=color_dict)
	out = np.squeeze(out, axis=0)
	cv2.imwrite('./crf.png', out)
