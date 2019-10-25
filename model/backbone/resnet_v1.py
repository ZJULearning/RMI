# coding=utf-8

"""
Reference:
	https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from RMI.utils import model_store

__all__ = ['ResNet', 'resnet50', 'resnet101', 'resnet152']


zhanghang_dir = '~/.encoding/models'

model_urls = {
	'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
	'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
	'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
	'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
	'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


model_dirs = {
	'resnet50': '~/.torch/models/resnet50-19c8e357.pth',
	'resnet101': '~/.torch/models/resnet101-5d3b4d8f.pth',
	#'resnet101': '/home/zhaoshuai/pretrained/resnet_v1_101_20160828/resnet.pth',
}

# https://discuss.pytorch.org/t/whats-the-difference-between-nn-relu-and-nn-relu-inplace-true/948
# inplace ReLU save more memory.
_IS_ReLU_INPLACE = True

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
						padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
	"""1x1 convolution"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
	"""resnet v1 bottleneck block"""
	expansion = 4

	def __init__(self, inplanes,
					planes,
					stride=1,
					downsample=None,
					groups=1,
					base_width=64,
					dilation=1,
					norm_layer=None,
					bn_mom=0.05):
		super(Bottleneck, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		width = int(planes * (base_width / 64.)) * groups
		# Both self.conv2 and self.downsample layers downsample the input when stride != 1
		self.conv1 = conv1x1(inplanes, width)
		self.bn1 = norm_layer(width, momentum=bn_mom)

		self.conv2 = conv3x3(width, width, stride, groups, dilation)
		self.bn2 = norm_layer(width, momentum=bn_mom)

		self.conv3 = conv1x1(width, planes * self.expansion)
		self.bn3 = norm_layer(planes * self.expansion, momentum=bn_mom)

		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out


class ResNet(nn.Module):

	def __init__(self, block,
					layers,
					output_stride=16,
					zero_init_residual=True,
					groups=1,
					width_per_group=64,
					norm_layer=nn.BatchNorm2d,
					bn_mom=0.05,
					root_beta=True):
		super(ResNet, self).__init__()
		self._norm_layer = norm_layer
		self.inplanes = 128 if root_beta else 64
		self.dilation = 1
		self.bn_mom = bn_mom

		# stride and dilations
		assert output_stride in [8, 16]
		self.strides = [1, 2, 2 if output_stride == 16 else 1, 1]
		# slightly different with the official implementation
		self.dilations = [1, 1, 1, 1]

		self.groups = groups
		self.base_width = width_per_group

		# the network modules, use 3 conv3x3 layers to replace the one conv7x7
		if root_beta:
			self.conv1 = nn.Sequential(
							nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
							norm_layer(64, momentum=bn_mom),
							nn.ReLU(inplace=True),
							nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
							norm_layer(64, momentum=bn_mom),
							nn.ReLU(inplace=True),
							nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
			)
		else:
			self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)

		self.bn1 = norm_layer(self.inplanes, momentum=self.bn_mom)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		# four stacked blocks
		self.layer1 = self._make_layer(block, 64, layers[0], stride=self.strides[0], dilation=self.dilations[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=self.strides[1], dilation=self.dilations[1])
		self.layer3 = self._make_layer(block, 256, layers[2], stride=self.strides[2], dilation=self.dilations[2])
		self.layer4 = self._make_layer(block, 512, layers[3], stride=self.strides[3], dilation=self.dilations[3])

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, norm_layer)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		# Zero-initialize the last BN in each residual branch,
		# so that the residual branch starts with zeros, and each residual block behaves like an identity.
		# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
		if zero_init_residual:
			for m in self.modules():
				if isinstance(m, Bottleneck):
					nn.init.constant_(m.bn3.weight, 0)

	def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
		"""construct layers"""
		norm_layer = self._norm_layer
		downsample = None

		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				conv1x1(self.inplanes, planes * block.expansion, stride),
				norm_layer(planes * block.expansion, momentum=self.bn_mom),
			)

		layers = []
		# the dialtion of the first layer
		dilation_first = 1 if dilation in [1, 2] else 2
		layers.append(block(self.inplanes, planes, stride,
								downsample,
								self.groups,
								self.base_width,
								dilation_first,
								norm_layer,
								bn_mom=self.bn_mom))
		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes,
									groups=self.groups,
									base_width=self.base_width,
									dilation=dilation,
									norm_layer=norm_layer,
									bn_mom=self.bn_mom))

		return nn.Sequential(*layers)

	def forward(self, x):
		end_points = {}
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		end_points['layer1'] = x

		x = self.layer2(x)
		end_points['layer2'] = x

		x = self.layer3(x)
		end_points['layer3'] = x

		x = self.layer4(x)

		return x, end_points


def _resnet(arch, block, layers, output_stride=16, pretrained=True, norm_layer=None, bn_mom=0.05, root_beta=True):
	model = ResNet(block, layers, output_stride=output_stride, norm_layer=norm_layer, bn_mom=bn_mom, root_beta=root_beta)
	if pretrained:
		if root_beta:
			old_dict = torch.load(model_store.get_model_file(arch, root=zhanghang_dir))
		else:
			old_dict = model_zoo.load_url(model_urls[arch])
			#old_dict = torch.load(model_dirs[arch])['state_dict']
		#print(old_dict)
		model_dict = model.state_dict()
		old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
		model_dict.update(old_dict)
		model.load_state_dict(model_dict)
	return model


def resnet50(output_stride=16, pretrained=True, norm_layer=None, bn_mom=0.05, root_beta=True):
	"""Constructs a ResNet-50 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], output_stride, pretrained, norm_layer, bn_mom)


def resnet101(output_stride=16, pretrained=True, norm_layer=None, bn_mom=0.05, root_beta=True):
	"""Constructs a ResNet-101 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], output_stride, pretrained, norm_layer, bn_mom)


def resnet152(output_stride=16, pretrained=True, norm_layer=None, bn_mom=0.05, root_beta=True):
	"""Constructs a ResNet-152 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], output_stride, pretrained, norm_layer, bn_mom)
