#coding=utf-8
"""
some training utils.
reference:
	https://github.com/zhanghang1989/PyTorch-Encoding
	https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from RMI.model import net_factory
from RMI.utils import model_init


__all__ = ['PSPModule', 'PSPNet']

# the feature map used to calculated the auxiliary loss
pspnet_aux_end_point_dict = {
	'resnet50': 'layer3',
	'resnet101': 'layer3',
}

# https://discuss.pytorch.org/t/whats-the-difference-between-nn-relu-and-nn-relu-inplace-true/948
# inplace ReLU save more memory.
_IS_ReLU_INPLACE = True


class PSPModule(nn.Module):
	"""The pyramid pooling module of the PSPNet."""
	def __init__(self,
					in_channels,
					depth=512,
					pool_sizes=[1, 2, 3, 6],
					norm_layer=nn.BatchNorm2d,
					bn_mom=0.05):
		super(PSPModule, self).__init__()
		self.in_channels = in_channels
		self.depth = depth
		self.norm_layer = norm_layer
		self.pool_sizes = pool_sizes
		self.bn_mom = bn_mom
		self.pools = nn.ModuleList([self._pooling(size) for size in self.pool_sizes])

		# fused conv layers, # 2048 + 4 * depth
		self.fuse_conv = nn.Sequential(
			nn.Conv2d(self.in_channels + 4 * depth, out_channels=depth, kernel_size=3, padding=1, bias=False),
			norm_layer(depth, momentum=self.bn_mom),
			nn.ReLU(inplace=_IS_ReLU_INPLACE),
			nn.Dropout2d(0.1, inplace=False)
		)

	def _pooling(self, size):
		return nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(size, size)),
								nn.Conv2d(self.in_channels, self.depth, kernel_size=1, bias=False),
								self.norm_layer(self.depth, momentum=self.bn_mom),
								nn.ReLU(inplace=_IS_ReLU_INPLACE)
							)

	def forward(self, inputs):
		h, w = inputs.shape[2:]
		output_slices = [inputs]
		# pyramid pooling
		for i, size in enumerate(self.pool_sizes):
			pool = self.pools[i](inputs)
			out = F.interpolate(pool, size=(h, w), mode='bilinear', align_corners=True)
			output_slices.append(out)
		# concat and fuse
		outputs = torch.cat(output_slices, dim=1)
		outputs = self.fuse_conv(outputs)

		return outputs


class PSPNet(nn.Module):
	def __init__(self, num_classes=21,
					output_stride=16,
					backbone='resnet50',
					norm_layer=nn.BatchNorm2d,
					bn_mom=0.01,
					depth_aux_branch=256,
					pretrained=True,
					freeze_bn=False):
		super(PSPNet, self).__init__()
		self.num_classes = num_classes
		self.bn_mom = bn_mom
		self.aux_key = pspnet_aux_end_point_dict[backbone]
		# backbone
		self.backbone = net_factory.get_backbone_net(output_stride=output_stride,
														pretrained=pretrained,
														norm_layer=norm_layer,
														bn_mom=bn_mom,
														root_beta=True)
		# pyramid pooling module
		self.psp_module = PSPModule(in_channels=2048,
										depth=512,
										pool_sizes=[1, 2, 3, 6],
										norm_layer=norm_layer,
										bn_mom=bn_mom)
		# main branch
		self.main_branch = nn.Conv2d(in_channels=512, out_channels=self.num_classes, kernel_size=1)

		# auxiliary branch
		self.aux_branch = nn.Sequential(
			nn.Conv2d(in_channels=1024, out_channels=depth_aux_branch, kernel_size=3, padding=1, bias=False),
			norm_layer(depth_aux_branch, momentum=self.bn_mom),
			nn.ReLU(inplace=_IS_ReLU_INPLACE),
			nn.Dropout2d(0.1, inplace=False),
			nn.Conv2d(in_channels=depth_aux_branch, out_channels=self.num_classes, kernel_size=1)
		)

		# initialize weights
		model_init.init_weights([self.psp_module, self.main_branch, self.aux_branch],
								norm_layer=norm_layer,
								bn_momentum=bn_mom)

		if freeze_bn:
			self.freeze_bn()

	def forward(self, inputs):
		h, w = inputs.shape[2:]
		x, end_points = self.backbone(inputs)

		# main branch
		x = self.psp_module(x)
		x_small = self.main_branch(x)
		x = F.interpolate(x_small, size=(h, w), mode='bilinear', align_corners=True)

		# auxiliary out for training
		x_aux_small = self.aux_branch(end_points[self.aux_key])
		x_aux = F.interpolate(x_aux_small, size=(h, w), mode='bilinear', align_corners=True)
		return x, x_aux, x_small, x_aux_small

	def freeze_bn(self):
		"""freeze bn"""
		for m in self.modules():
			if isinstance(m, (self.norm_layer, nn.BatchNorm2d)):
				m.eval()
