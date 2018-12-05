import math

import torch.nn as nn
import torch
import sys
from torch.autograd import Variable
sys.path.append('./net/')

def get_padding_shape(filter_shape, stride):
    def _pad_top_bottom(filter_dim, stride_val):
        pad_along = max(filter_dim - stride_val, 0)
        pad_top = pad_along // 2
        pad_bottom = pad_along - pad_top
        return pad_top, pad_bottom
    padding_shape = []
    for filter_dim, stride_val in zip(filter_shape, stride):
        pad_top, pad_bottom = _pad_top_bottom(filter_dim, stride_val)
        padding_shape.append(pad_top)
        padding_shape.append(pad_bottom)
    depth_top = padding_shape.pop(0)
    depth_bottom = padding_shape.pop(0)
    padding_shape.append(depth_top)
    padding_shape.append(depth_bottom)
    """padding_shape from(depth_pad_shape,height_pad_shape,width_pad_shape) 
       transform to(height_pad_shape,width_pad_shape,depth_pad_shape)
    """
    return tuple(padding_shape)


def simplify_padding(padding_shapes):
    all_same = True
    padding_init = padding_shapes[0]
    for pad in padding_shapes[1:]:
        if pad !=padding_init:
            all_same = False
    return all_same, padding_init


def _get_padding(padding_name, conv_shape):
    padding_name = padding_name.decode("utf-8")
    if padding_name == 'VALID':
        return [0, 0]
    elif padding_name == 'SAME':
        return [math.floor(int(conv_shape[0]) / 2),
                math.floor(int(conv_shape[1]) / 2),
                math.floor(int(conv_shape[2]) / 2)]
    else:
        raise ValueError('Invalid padding name' + padding_name)


def get_new_input(input, num_part):
	new_input = []
	for p in range(num_part):
		part = []
		for i in range(input.size(1)):
			if i % num_part == p:
				part.append(input[:, i, :, :, :])
		assert len(part) == (input.size(1)/num_part)
		part = torch.stack(part, 1)
		new_input.append(part)
	return new_input

def get_list_mean(list):
	if list[0].type() == 'torch.cuda.FloatTensor':
		sum = Variable(torch.zeros(list[0].shape).float().cuda())
	else:
		sum = Variable(torch.zeros(list[0].shape).half().cuda())
	for score in list:
		sum += score
	sum = sum/len(list)
	return sum


def caculate_weight(part):
        one = [part[i] for i in range(len(part)-1)]
        another = [ part[i] for i in range(1, len(part))]
        assert len(one) == len(another)
        sum = 0
        for i in range(len(one)):
                cur = one[i]
                next = another[i]
                sum += math.sqrt((cur[0]-next[0])**2 + (cur[1] - next[1])**2)
        return sum


def weight_joint_trace(joint):
	batch_size, num_part, frame = joint.shape[0], joint.shape[1], joint.shape[2]	
	weight_matrix = torch.zeros((batch_size, num_part))
	for b in range(batch_size):
		for n in range(num_part):
			single = []
			for f in range(frame):
				location = []
				location.append(int(joint[b,n,f,0]))
				location.append(int(joint[b,n,f,1]))
				single.append(location)
			weight = caculate_weight(single)
			weight_matrix[b, n] = weight
	#print(weight_matrix[0])
	for b in range(batch_size):
		sum = 0
		for n in range(num_part):
			sum += weight_matrix[b,n]
		weight_matrix[b,:] /= sum	
	#print(weight_matrix[0])
	return weight_matrix.cuda()

class Unit3Dpy(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1, 1),
                 stride=(1, 1, 1), activation='relu', padding='SAME',
                 use_bias=False, use_bn=True):
        super(Unit3Dpy, self).__init__()
        self.padding = padding
        self.activation = activation
        self.use_bn = use_bn

        if padding == 'SAME':
            padding_shape = get_padding_shape(kernel_size, stride)
            simplify_pad, pad_size = simplify_padding(padding_shape)
            self.simplify_pad = simplify_pad
        elif padding == 'VALID':
            padding_shape = 0
        else:
            raise ValueError('padding should be in [VALID|SAME] but'
                             'got {}'.format(padding))

        if padding == 'SAME':
            if not simplify_pad:   # which mean pads in different dims are not same
                self.pad = torch.nn.ConstantPad3d(padding_shape, 0)
                self.conv3d = torch.nn.Conv3d(
                   in_channels, out_channels, kernel_size, stride=stride, bias=use_bias)
            # nn.Conv3d()中的pad要不是int要不是3维tuple，所以要提前添加pad3d函数
            else:
                self.conv3d = torch.nn.Conv3d(in_channels,
                    out_channels, kernel_size, stride = stride, padding=pad_size, bias=use_bias)
        elif padding == 'VALID':
            self.conv3d = torch.nn.Conv3d(in_channels,
            out_channels, kernel_size, padding=padding_shape, stride=stride, bias=use_bias)
        else:
            raise ValueError('padding should be in [VALID|SAME] but'
                             'got {}'.format(padding))

        if self.use_bn:
            self.batch3d = torch.nn.BatchNorm3d(out_channels)

        if activation == 'relu':
            self.activation = torch.nn.functional.relu

    def forward(self, inp):
        if self.padding == 'SAME' and self.simplify_pad is False:
            inp = self.pad(inp)
        out = self.conv3d(inp)
        if self.use_bn:
            out = self.batch3d(out)
        if self.activation is not None:
            out = torch.nn.functional.relu(out)
        return out


class MaxPool3DTFPadding(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding='SAME'):
        super(MaxPool3DTFPadding, self).__init__()
        if padding == 'SAME':
            padding_shape = get_padding_shape(kernel_size, stride)
            self.padding_shape = padding_shape
            self.pad = torch.nn.ConstantPad3d(padding_shape, 0)
        self.pool = torch.nn.MaxPool3d(kernel_size, stride, ceil_mode=True)

    def forward(self, inp):
        inp = self.pad(inp)
        out = self.pool(inp)
        return out


class Mixed(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Mixed, self).__init__()
        # Branch 0
        self.branch_0 = Unit3Dpy(in_channels, out_channels[0], kernel_size=(1, 1, 1))

        # Branch 1
        branch_1_conv1 = Unit3Dpy(in_channels, out_channels[1], kernel_size=(1, 1, 1))
        branch_1_conv2 = Unit3Dpy(out_channels[1], out_channels[2], kernel_size=(3, 3, 3))
        self.branch_1 = torch.nn.Sequential(branch_1_conv1, branch_1_conv2)

        # Branch 2
        branch_2_conv1 = Unit3Dpy(in_channels, out_channels[3], kernel_size=(1, 1, 1))
        branch_2_conv2 = Unit3Dpy(out_channels[3], out_channels[4], kernel_size=(3, 3, 3))
        self.branch_2 = torch.nn.Sequential(branch_2_conv1, branch_2_conv2)

        # Branch 3
        branch_3_pool = MaxPool3DTFPadding(kernel_size=(3, 3, 3), stride=(1, 1, 1), padding='SAME')
        branch_3_conv2 = Unit3Dpy(in_channels, out_channels[5], kernel_size=(1, 1, 1))
        self.branch_3 = torch.nn.Sequential(branch_3_pool, branch_3_conv2)

    def forward(self, inp):
        out_0 = self.branch_0(inp)
        out_1 = self.branch_1(inp)
        out_2 = self.branch_2(inp)
        out_3 = self.branch_3(inp)
        out = torch.cat((out_0, out_1, out_2, out_3), 1)
        return out 

