# coding=utf-8

"""
evaluation during training
"""

import numpy as np


class Evaluator(object):
	def __init__(self, num_class):
		"""initialize the evalutor"""
		self.num_class = num_class
		self.confusion_matrix = np.zeros((self.num_class, self.num_class))

	def pixel_accuracy_np(self):
		"""calculate the pixel accuracy with numpy"""
		denominator = self.confusion_matrix.sum().astype(float)
		cm_diag_sum = np.diagonal(self.confusion_matrix).sum().astype(float)

		# If the number of valid entries is 0 (no classes) we return 0.
		accuracy = np.where(denominator > 0, cm_diag_sum / denominator, 0)
		accuracy = float(accuracy)
		#print('Pixel Accuracy: {:.4f}'.format(float(accuracy)))
		return accuracy

	def Pixel_Accuracy_Class(self):
		Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
		Acc = np.nanmean(Acc)
		return Acc

	def mean_iou_np(self, is_show_per_class=False):
		"""compute mean iou with numpy"""
		sum_over_row = np.sum(self.confusion_matrix, axis=0).astype(float)
		sum_over_col = np.sum(self.confusion_matrix, axis=1).astype(float)
		cm_diag = np.diagonal(self.confusion_matrix).astype(float)
		denominator = sum_over_row + sum_over_col - cm_diag

		# The mean is only computed over classes that appear in the
		# label or prediction tensor. If the denominator is 0, we need to
		# ignore the class.
		num_valid_entries = np.sum((denominator != 0).astype(float))

		# If the value of the denominator is 0, set it to 1 to avoid
		# zero division.
		denominator = np.where(denominator > 0,
									denominator,
									np.ones_like(denominator))
		ious = cm_diag / denominator

		if is_show_per_class:
			print('\nIntersection over Union for each class:')
			for i, iou in enumerate(ious):
				print('    class {}: {:.4f}'.format(i, iou))

		# If the number of valid entries is 0 (no classes) we return 0.
		m_iou = np.where(num_valid_entries > 0,
							np.sum(ious) / num_valid_entries,
							0)
		m_iou = float(m_iou)
		if is_show_per_class:
			print('mean Intersection over Union: {:.4f}'.format(float(m_iou)))
		return m_iou

	def Frequency_Weighted_Intersection_over_Union(self):
		"""frequencey weighted miou"""
		freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
		iu = np.diag(self.confusion_matrix) / (np.sum(self.confusion_matrix, axis=1)
					+ np.sum(self.confusion_matrix, axis=0) - np.diag(self.confusion_matrix))

		FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
		return FWIoU

	def _generate_matrix(self, gt_image, pre_image):
		"""calculate confusion matrix"""
		mask = (gt_image >= 0) & (gt_image < self.num_class)
		label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
		count = np.bincount(label, minlength=self.num_class**2)
		confusion_matrix = count.reshape(self.num_class, self.num_class)
		return confusion_matrix

	def add_batch(self, gt_image, pre_image):
		"""add the evluation result to the confusion maxtrix"""
		assert gt_image.shape == pre_image.shape
		self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

	def reset(self):
		"""set the confusion matrix to 0"""
		self.confusion_matrix = np.zeros((self.num_class, self.num_class))


if __name__ == '__main__':
	eval = Evaluator(num_class=5)
	gt = np.array([0, 1, 2, 3, 4, 6])
	pre = np.array([0, 1, 2, 3, 4, 1])
	eval.add_batch(gt, pre)
	print(eval.confusion_matrix)
