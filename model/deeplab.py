# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch
import torch.nn as nn
import torch.nn.functional as F

from RMI.model import net_factory

__all__ = ['DeepLabv3Plus', 'DeepLabv3', 'Decoder', 'ASPP']

# A dictionary from network name to a map of end point features.
decoder_end_points_dict = {
	'resnet50': 'layer1',
	'resnet101': 'layer1',
	'resnet152': 'layer1',
	'resnet50_beta': 'layer1',
	'resnet101_beta': 'layer1',
	'resnet152_beta': 'layer1',
	#'xception_41': 'entry_flow/block2/unit_1/xception_module/separable_conv2_pointwise',
	#'xception_65': 'entry_flow/block2/unit_1/xception_module/separable_conv2_pointwise',
	#'xception_71': 'entry_flow/block3/unit_1/xception_module/separable_conv2_pointwise',
}

# https://discuss.pytorch.org/t/whats-the-difference-between-nn-relu-and-nn-relu-inplace-true/948
# inplace ReLU save more memory.
_IS_ReLU_INPLACE = True

class DeepLabv3Plus(nn.Module):
	"""DeepLabv3+ Segmentation Model"""
	def __init__(self, backbone='resnet101',
						output_stride=16,
						num_classes=21,
						aspp_depth=256,
						norm_layer=nn.BatchNorm2d,
						bn_mom=0.05,
						freeze_bn=False,
						pretrained=True):
		super(DeepLabv3Plus, self).__init__()
		self.norm_layer = norm_layer
		self.decoder_key = decoder_end_points_dict[backbone]
		self.aspp_depth = aspp_depth
		# backbone
		self.backbone = net_factory.get_backbone_net(output_stride=output_stride,
														pretrained=pretrained,
														norm_layer=norm_layer,
														bn_mom=bn_mom,
														root_beta=True)
		self.aspp = ASPP(backbone,
							output_stride,
							norm_layer,
							depth=self.aspp_depth,
							bn_mom=bn_mom)
		self.decoder = Decoder(num_classes,
								backbone=backbone,
								norm_layer=norm_layer,
								bn_mom=bn_mom,
								decoder_depth=256)

		self._init_weight()
		# freeze bn
		if freeze_bn:
			self.freeze_bn()

	def forward(self, input):
		x, end_points = self.backbone(input)
		low_level_feat = end_points[self.decoder_key]
		x = self.aspp(x)
		x = self.decoder(x, low_level_feat)
		x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

		return x

	def freeze_bn(self):
		for m in self.modules():
			if (isinstance(m, self.norm_layer) or isinstance(m, nn.BatchNorm2d)):
				m.eval()

	def _init_weight(self):
		"""initializer"""
		modules = [self.decoder, self.aspp]
		for module in modules:
			for m in module.modules():
				if isinstance(m, nn.Conv2d):
					nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
					if m.bias is not None:
						m.bias.data.zero_()
				elif isinstance(m, (self.norm_layer, nn.BatchNorm2d)):
					m.weight.data.fill_(1)
					m.bias.data.zero_()


class DeepLabv3(nn.Module):
	"""DeepLabv3 Segmentation Model"""
	def __init__(self, backbone='resnet101',
						output_stride=16,
						num_classes=21,
						norm_layer=nn.BatchNorm2d,
						freeze_bn=False,
						bn_mom=0.05,
						aspp_depth=256,
						pretrained=True):
		super(DeepLabv3, self).__init__()

		self.aspp_depth = aspp_depth
		self.output_stride = output_stride
		# choose batchnorm layer
		self.norm_layer = norm_layer
		#norm_layer = SynchronizedBatchNorm2d if sync_bn else nn.BatchNorm2d

		# backbone
		self.backbone = net_factory.get_backbone_net(output_stride=output_stride,
														pretrained=pretrained,
														norm_layer=norm_layer,
														bn_mom=bn_mom,
														root_beta=True)
		self.aspp = ASPP(backbone,
							output_stride,
							norm_layer,
							depth=self.aspp_depth,
							bn_mom=bn_mom)
		self.last_conv = nn.Conv2d(self.aspp_depth, num_classes, kernel_size=1, stride=1)

		# freeze batch normalization or not
		if freeze_bn:
			self.freeze_bn()

		self._init_weight()

	def forward(self, input):
		"""forward process"""
		x, end_points = self.backbone(input)
		x = self.aspp(x)
		x = self.last_conv(x)
		x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

		return x

	def freeze_bn(self):
		"""freeze bn"""
		for m in self.modules():
			if isinstance(m, (self.norm_layer, nn.BatchNorm2d)):
				m.eval()

	def _init_weight(self):
		"""initializer"""
		modules = [self.last_conv, self.aspp]
		for module in modules:
			for m in module.modules():
				if isinstance(m, nn.Conv2d):
					nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
					if m.bias is not None:
						m.bias.data.zero_()
				elif isinstance(m, (self.norm_layer, nn.BatchNorm2d)):
					m.weight.data.fill_(1)
					m.bias.data.zero_()

	#def get_1x_lr_params(self):
	#	modules = [self.backbone]
	#	for i in range(len(modules)):
	#		for m in modules[i].named_modules():
	#			if isinstance(m[1], nn.Conv2d) or isinstance(m[1], self.norm_layer) \
	#					or isinstance(m[1], nn.BatchNorm2d):
	#				for p in m[1].parameters():
	#					if p.requires_grad:
	#						yield p

	#def get_10x_lr_params(self):
	#	modules = [self.aspp, self.last_conv]
	#	for i in range(len(modules)):
	#		for m in modules[i].named_modules():
	#			if (isinstance(m[1], nn.Conv2d) or isinstance(m[1], self.norm_layer) or
	#					isinstance(m[1], nn.BatchNorm2d)):
	#				for p in m[1].parameters():
	#					if p.requires_grad:
	#						yield p


class _ASPPModule(nn.Module):
	"""Atrous Spatial Pyramid Pooling (ASPP)."""
	def __init__(self, inplanes, planes, kernel_size, dilation, norm_layer=nn.BatchNorm2d, bn_mom=0.05):
		super(_ASPPModule, self).__init__()
		self.padding = 0 if (kernel_size == 1 and dilation == 1) else dilation
		self.atrous_conv = nn.Conv2d(inplanes, planes,
										kernel_size=kernel_size,
										stride=1,
										padding=self.padding,
										dilation=dilation,
										bias=False)
		self.bn = norm_layer(planes, momentum=bn_mom)
		self.relu = nn.ReLU(inplace=_IS_ReLU_INPLACE)

	def forward(self, x):
		x = self.atrous_conv(x)
		x = self.bn(x)

		return self.relu(x)


class ASPP(nn.Module):
	def __init__(self, backbone, output_stride, norm_layer, depth=256, bn_mom=0.05):
		super(ASPP, self).__init__()
		# the inplanes of the backbone
		if 'resnet' in backbone or 'xception' in backbone:
			inplanes = 2048
		elif 'mobilenet' in backbone:
			inplanes = 320
		elif 'drn' in backbone:
			inplanes = 512
		else:
			raise NotImplementedError

		# output stride
		assert output_stride in [8, 16]
		dilations = [6, 12, 18] if output_stride == 16 else [12, 24, 36]
		# aspp modules
		self.depth = depth
		self.aspp1 = _ASPPModule(inplanes, depth, 1, dilation=1, norm_layer=norm_layer, bn_mom=bn_mom)
		self.aspp2 = _ASPPModule(inplanes, depth, 3, dilation=dilations[0], norm_layer=norm_layer, bn_mom=bn_mom)
		self.aspp3 = _ASPPModule(inplanes, depth, 3, dilation=dilations[1], norm_layer=norm_layer, bn_mom=bn_mom)
		self.aspp4 = _ASPPModule(inplanes, depth, 3, dilation=dilations[2], norm_layer=norm_layer, bn_mom=bn_mom)
		self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
												nn.Conv2d(inplanes, depth, 1, stride=1, bias=False),
												norm_layer(depth, momentum=bn_mom),
												nn.ReLU(inplace=_IS_ReLU_INPLACE))
		self.conv1 = nn.Conv2d(depth * 5, depth, 1, bias=False)
		self.bn1 = norm_layer(depth, momentum=bn_mom)
		self.relu = nn.ReLU(inplace=_IS_ReLU_INPLACE)
		# the droped probability -- 0.1
		self.dropout = nn.Dropout(0.1, inplace=False)

	def forward(self, x):
		x1 = self.aspp1(x)
		x2 = self.aspp2(x)
		x3 = self.aspp3(x)
		x4 = self.aspp4(x)
		x5 = self.global_avg_pool(x)
		x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
		x = torch.cat((x1, x2, x3, x4, x5), dim=1)

		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.dropout(x)

		return x


class Decoder(nn.Module):
	def __init__(self, num_classes,
						backbone='resnet101',
						norm_layer=nn.BatchNorm2d,
						bn_mom=0.05,
						decoder_depth=256):
		super(Decoder, self).__init__()
		# input channels
		if 'resnet' in backbone or 'drn' in backbone:
			low_level_inplanes = 256
		elif 'xception' in backbone:
			low_level_inplanes = 128
		elif 'mobilenet' in backbone:
			low_level_inplanes = 24
		else:
			raise NotImplementedError

		self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
		self.bn1 = norm_layer(48, momentum=bn_mom)
		self.relu = nn.ReLU(inplace=_IS_ReLU_INPLACE)

		self.last_conv = nn.Sequential(nn.Conv2d(low_level_inplanes + 48, 256, kernel_size=3, stride=1, padding=1, bias=False),
										norm_layer(256, momentum=bn_mom),
										nn.ReLU(inplace=_IS_ReLU_INPLACE),
										#nn.Dropout(0.5),
										nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
										norm_layer(256, momentum=bn_mom),
										nn.ReLU(inplace=_IS_ReLU_INPLACE),
										nn.Dropout(0.1, inplace=False),
										nn.Conv2d(256, num_classes, kernel_size=1, stride=1))

	def forward(self, x, low_level_feat):
		low_level_feat = self.conv1(low_level_feat)
		low_level_feat = self.bn1(low_level_feat)
		low_level_feat = self.relu(low_level_feat)

		x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
		x = torch.cat((x, low_level_feat), dim=1)
		x = self.last_conv(x)

		return x


if __name__ == "__main__":
	pass
