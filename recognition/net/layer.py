import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.append('../net/')
from torch.autograd import Function
from tool import get_padding_shape, simplify_padding



#conv layer
class block(nn.Module):
	def __init__(self, inchannels, outchannels, kernelsize, stride, padding='SAME', activation='relu', bn=True, res=False):
		super(block, self).__init__()
		self.res = res
		self.padding = padding
		if padding == 'SAME':
			padding_shape = get_padding_shape(kernelsize, stride)
			simplify_pad, pad_size = simplify_padding(padding_shape)
			self.simplify_pad = simplify_pad
		elif padding == 'VALID':
			padding_shape = 0
		else:
			raise ValueError('padding should be in [VALID|SAME]')
	 	
		if padding == 'SAME':
			if not simplify_pad:
				self.pad = nn.ConstantPad3d(padding_shape, 0)
				self.conv3d = nn.Conv3d(inchannels, outchannels, kernelsize, stride=stride)
			else:
				self.conv3d = nn.Conv3d(inchannels, outchannels, kernelsize, stride=stride, padding=pad_size)
		elif padding=='VALID':
			self.conv3d = nn.Conv3d(inchannels, outchannels, kernelsize, padding=padding_shape, stride=stride)
			
		if bn:
			self.bn = nn.BatchNorm3d(outchannels)
		if activation == 'relu':
			self.activation = nn.ReLU(inplace=True)
		if res:
			self.downsample = nn.Sequential(nn.Conv3d(inchannels, outchannels, kernel_size=1, stride=stride), self.bn)
			
	def forward(self, x):
		residual = x
		if self.padding == 'SAME' and self.simplify_pad is False:
			x = self.pad(x)
		x = self.conv3d(x)
		if self.bn:
			x = self.bn(x)
		if self.res:
			residual = self.downsample(residual)
			x = x + residual
		if self.activation :
			x = self.activation(x)
		return x

#pool layer
class maxpool3d(nn.Module):
	def __init__(self, kernelsize, stride=None, padding='SAME'):
		super(maxpool3d, self).__init__()
		if padding== 'SAME':
			padding_shape = get_padding_shape(kernelsize, stride)
			self.padding_shape = padding_shape
			self.pad = nn.ConstantPad3d(padding_shape, 0)
		self.pool = nn.MaxPool3d(kernelsize, stride, ceil_mode=True)
	
	def forward(self, x):
		if self.pad is not None:
			x = self.pad(x)
			x = self.pool(x)
		return x
	
#create a encoder
class encoder(torch.nn.Module):
	def __init__(self, in_channels, out_channels, mark='human', res=False):
		super(encoder, self).__init__()
		self.block1 = block(in_channels, 64, (3,7,7), stride=(3,2,2), padding='SAME', res=res)
		self.block2 = block(64, 64, (2,3,3), stride=(1,1,1), padding='SAME', res=res)
		if mark == 'human':		
			self.block3 = block(64, out_channels, (4,3,3), stride=(1,1,1), padding='SAME', res=res)

	def forward(self,x):
		# time/3 , size/2
		x = self.block1(x)
		x = self.block2(x)
		if mark == 'human':		
			x = self.block3(x)
		return x


def get_score(x):
    assert x.size(1) == x.size(2)
    batch_size, h, w = x.size(0), x.size(1), x.size(2)
    score = torch.Tensor(batch_size, h)
    x = x.float()
    for b in range(batch_size):
         for i in range(h):
             score[b,i] = x[b,i,i]
    return score

