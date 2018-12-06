import math
import os
import torch
import sys
sys.path.append('./recognition/net')
import numpy as np
import torch.nn as nn
from torch.nn import ReplicationPad3d
from tool import simplify_padding, _get_padding, get_padding_shape


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
            out = self.activation(out)
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


class I3D(torch.nn.Module):
    def __init__(self, num_classes=60, input_size=224, dropout_prob=0.5):
        super(I3D, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        tk = 1
        sk = 1
        conv3d_1a_7x7_1 = Unit3Dpy(out_channels=64, in_channels=3,
                                 kernel_size=(7, 7, 7), stride=(2, 2, 2), padding='SAME')
        tk = tk * 2
        sk = sk * 2
        if input_size == 672:
            self.conv3d = Unit3Dpy(out_channels=64, in_channels=64, kernel_size=(7,7,7), stride=(1,2,2), padding='SAME')
            sk = sk * 2  
        # 1st conv-pool
        self.conv3d_1a_7x7_1 = conv3d_1a_7x7_1
        self.maxPool3d_2a_3x3 = MaxPool3DTFPadding(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='SAME')
        sk = sk * 2
        # 2rd conv-conv-pool
        conv3d_2b_1x1 = Unit3Dpy(out_channels=64, in_channels=64, kernel_size=(1, 1, 1),
                                    stride=(1, 1, 1), padding='SAME')
        self.conv3d_2b_1x1 = conv3d_2b_1x1
        conv3d_2c_3x3 = Unit3Dpy(out_channels=192, in_channels=64, kernel_size=(3, 3, 3),
                                      stride=(1, 1, 1), padding='SAME')
        self.conv3d_2c_3x3 = conv3d_2c_3x3
        self.maxPool3d_3a_3x3 = MaxPool3DTFPadding(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='SAME')
        sk = sk * 2
        # 3th Mixed-Mixed-pool
        self.mixed_3b = Mixed(192, [64, 96, 128, 16, 32, 32])
        self.mixed_3c = Mixed(256, [128, 128, 192, 32, 96, 64])
        self.maxPool3d_4a_3x3 = MaxPool3DTFPadding(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding='SAME')
        tk = tk * 2
        sk = sk * 2
        # 4th Mixed-Mixed-Mixed-Mixed-Mixed-pool
        self.mixed_4b = Mixed(480, [192, 96, 208, 16, 48, 64])
        self.mixed_4c = Mixed(512, [160, 112, 224, 24, 64, 64])
        self.mixed_4d = Mixed(512, [128, 128, 256, 24, 64, 64])
        self.mixed_4e = Mixed(512, [112, 144, 288, 32, 64, 64])
        self.mixed_4f = Mixed(528, [256, 160, 320, 32, 128, 128])
        self.maxPool3d_5a_2x2 = MaxPool3DTFPadding(
            kernel_size=(2, 2, 2), stride=(2, 2, 2), padding='SAME')
        tk = tk * 2
        sk = sk * 2
        # 5th Mixed
        self.mixed_5b = Mixed(832, [256, 160, 320, 32, 128, 128])
        self.mixed_5c = Mixed(832, [384, 192, 384, 48, 128, 128])

        self.avg_pool = torch.nn.AvgPool3d((int(32/tk), int(input_size/sk), int(input_size/sk)), (1, 1, 1))
        self.dropout = torch.nn.Dropout(dropout_prob)
        conv3d_0c_1x1 = Unit3Dpy(in_channels=1024, out_channels=self.num_classes, kernel_size=(1, 1, 1),
                                activation=None, use_bias=True, use_bn=False)
        self.conv3d_0c_1x1 = conv3d_0c_1x1
        self.softmax = torch.nn.Softmax(1)

    def forward(self, inp):
        inp = torch.transpose(inp, 1, 2)
        out = self.conv3d_1a_7x7_1(inp)
        out = self.maxPool3d_2a_3x3(out)
        out = self.conv3d_2b_1x1(out)
        out = self.conv3d_2c_3x3(out)
        out = self.maxPool3d_3a_3x3(out)
        out = self.mixed_3b(out)
        out = self.mixed_3c(out)
        out = self.maxPool3d_4a_3x3(out)
        out = self.mixed_4b(out)
        out = self.mixed_4c(out)
        out = self.mixed_4d(out)
        out = self.mixed_4e(out)
        out = self.mixed_4f(out)
        out = self.maxPool3d_5a_2x2(out)
        out = self.mixed_5b(out)
        out = self.mixed_5c(out)
        out = self.avg_pool(out)
        out = self.dropout(out)
        out = self.conv3d_0c_1x1(out)
        out = out.squeeze(3)    # out is of shape(N,C,D,H,W),now H and W convert to 1
        out = out.squeeze(3)    # shape:(N,C,D)
        out = out.mean(2)       # shape:(batch_size,num_classes)
        out_logits = out
        out = self.softmax(out_logits)
        return out, out_logits
