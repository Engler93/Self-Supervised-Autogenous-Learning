import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models.inception import InceptionC, InceptionD, InceptionE


class InceptionAux(nn.Module):
	def __init__(self, in_channels, pool=0, num_coarse_classes=1000):
		super(InceptionAux, self).__init__()
		self.pool = pool

		if pool >= 2:
			self.cbr1 = BasicConv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3),
									stride=(2, 2) if pool >= 2 else (1, 1))
		else:
			self.cbr1 = 0
		self.inception1 = InceptionC_smaller(in_channels, chf=0.5)
		if pool >= 1:
			self.inception2 = nn.Sequential(InceptionD_smaller(0.5 * 512, chf=0.5),
											InceptionE_smaller(int(0.5 * 512), chf=1), InceptionD_smaller(748, chf=0.5),
											InceptionE_smaller(int(0.5 * 512), chf=2))
		else:
			self.inception2 = nn.Sequential(InceptionD_smaller(0.5 * 512, chf=0.5),
											InceptionE_smaller(int(0.5 * 512), chf=2))
		self.gap = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = nn.Linear(2 * 748, num_coarse_classes)

	def forward(self, x):
		if self.pool >= 2:
			x = self.cbr1(x)
		x = self.inception1(x)
		x = self.inception2(x)
		x = self.gap(x)
		x = torch.flatten(x, 1)
		x = self.fc(x)
		return x


class InceptionC_smaller(nn.Module):
	# out_channels 512
	def __init__(self, in_channels, channels_7x7=160, chf=1.0):
		super(InceptionC_smaller, self).__init__()
		self.branch1x1 = BasicConv2d(int(in_channels), int(chf * 128), kernel_size=1)

		c7 = channels_7x7
		self.branch7x7_1 = BasicConv2d(int(in_channels), int(chf * c7), kernel_size=1)
		self.branch7x7_2 = BasicConv2d(int(chf * c7), int(chf * c7), kernel_size=(1, 7), padding=(0, 3))
		self.branch7x7_3 = BasicConv2d(int(chf * c7), int(chf * 128), kernel_size=(7, 1), padding=(3, 0))

		self.branch7x7dbl_1 = BasicConv2d(int(in_channels), int(chf * c7), kernel_size=1)
		self.branch7x7dbl_2 = BasicConv2d(int(chf * c7), int(chf * c7), kernel_size=(7, 1), padding=(3, 0))
		self.branch7x7dbl_3 = BasicConv2d(int(chf * c7), int(chf * c7), kernel_size=(1, 7), padding=(0, 3))
		self.branch7x7dbl_4 = BasicConv2d(int(chf * c7), int(chf * c7), kernel_size=(7, 1), padding=(3, 0))
		self.branch7x7dbl_5 = BasicConv2d(int(chf * c7), int(chf * 128), kernel_size=(1, 7), padding=(0, 3))

		self.branch_pool = BasicConv2d(int(in_channels), int(chf * 128), kernel_size=1)

	def forward(self, x):
		branch1x1 = self.branch1x1(x)

		branch7x7 = self.branch7x7_1(x)
		branch7x7 = self.branch7x7_2(branch7x7)
		branch7x7 = self.branch7x7_3(branch7x7)

		branch7x7dbl = self.branch7x7dbl_1(x)
		branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
		branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
		branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
		branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

		branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
		branch_pool = self.branch_pool(branch_pool)

		outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
		return torch.cat(outputs, 1)


class InceptionD_smaller(nn.Module):

	def __init__(self, in_channels, chf=1.0):
		super(InceptionD_smaller, self).__init__()
		self.branch3x3_1 = BasicConv2d(int(in_channels), int(chf * 192), kernel_size=1)
		self.branch3x3_2 = BasicConv2d(int(chf * 192), int(chf * 216), kernel_size=3, stride=2)

		self.branch7x7x3_1 = BasicConv2d(int(in_channels), int(chf * 192), kernel_size=1)
		self.branch7x7x3_2 = BasicConv2d(int(chf * 192), int(chf * 192), kernel_size=(1, 7), padding=(0, 3))
		self.branch7x7x3_3 = BasicConv2d(int(chf * 192), int(chf * 192), kernel_size=(7, 1), padding=(3, 0))
		self.branch7x7x3_4 = BasicConv2d(int(chf * 192), int(chf * 168), kernel_size=3, stride=2)

		self.branch_pool = BasicConv2d(int(in_channels), int(chf * 128), kernel_size=1)

	def forward(self, x):
		branch3x3 = self.branch3x3_1(x)
		branch3x3 = self.branch3x3_2(branch3x3)

		branch7x7x3 = self.branch7x7x3_1(x)
		branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
		branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
		branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

		branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
		branch_pool = self.branch_pool(branch_pool)

		outputs = [branch3x3, branch7x7x3, branch_pool]
		return torch.cat(outputs, 1)


class InceptionE_smaller(nn.Module):

	def __init__(self, in_channels, chf=1):
		super(InceptionE_smaller, self).__init__()
		self.branch1x1 = BasicConv2d(in_channels, chf * 128, kernel_size=1)

		self.branch3x3_1 = BasicConv2d(in_channels, chf * 128, kernel_size=1)
		self.branch3x3_2a = BasicConv2d(chf * 128, chf * 128, kernel_size=(1, 3), padding=(0, 1))
		self.branch3x3_2b = BasicConv2d(chf * 128, chf * 128, kernel_size=(3, 1), padding=(1, 0))

		self.branch3x3dbl_1 = BasicConv2d(in_channels, chf * 160, kernel_size=1)
		self.branch3x3dbl_2 = BasicConv2d(chf * 160, chf * 128, kernel_size=3, padding=1)
		self.branch3x3dbl_3a = BasicConv2d(chf * 128, chf * 128, kernel_size=(1, 3), padding=(0, 1))
		self.branch3x3dbl_3b = BasicConv2d(chf * 128, chf * 128, kernel_size=(3, 1), padding=(1, 0))

		self.branch_pool = BasicConv2d(in_channels, chf * 108, kernel_size=1)

	def forward(self, x):
		branch1x1 = self.branch1x1(x)

		branch3x3 = self.branch3x3_1(x)
		branch3x3 = [
			self.branch3x3_2a(branch3x3),
			self.branch3x3_2b(branch3x3),
		]
		branch3x3 = torch.cat(branch3x3, 1)

		branch3x3dbl = self.branch3x3dbl_1(x)
		branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
		branch3x3dbl = [
			self.branch3x3dbl_3a(branch3x3dbl),
			self.branch3x3dbl_3b(branch3x3dbl),
		]
		branch3x3dbl = torch.cat(branch3x3dbl, 1)

		branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
		branch_pool = self.branch_pool(branch_pool)

		outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
		return torch.cat(outputs, 1)


class BasicConv2d(nn.Module):

	def __init__(self, in_channels, out_channels, **kwargs):
		super(BasicConv2d, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
		self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		return F.relu(x, inplace=True)
