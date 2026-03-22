# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .conv import DWConv

__all__ = (
    "RepHDW",
    "Pzconv",
    "FCM_3",
    "FCM_2",
    "Combined_Hybrid_Pz_Enhanced",
    "MSGI_FCM",
    "CombinedModule_to",
    "CombinedModule",
    "CombinedModule_1to1",
    "CombinedModule_1to3"

   # "SPDConv",


)


class Conv(nn.Module):
    '''Normal Conv with SiLU activation'''

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, groups=1, bias=False):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class AVG(nn.Module):
    def __init__(self, down_n=2):
        super().__init__()
        self.avg_pool = nn.functional.adaptive_avg_pool2d
        self.down_n = down_n
        # self.output_size = np.array([H, W])

    def forward(self, x):
        B, C, H, W = x.shape
        H = int(H / self.down_n)
        W = int(W / self.down_n)
        output_size = np.array([H, W])
        x = self.avg_pool(x, output_size)
        return x


class RepHDW(nn.Module):
    def __init__(self, in_channels, out_channels, depth=1, shortcut=True, expansion=0.5, kersize=5, depth_expansion=1,
                 small_kersize=3, use_depthwise=True):
        super(RepHDW, self).__init__()
        c1 = int(out_channels * expansion) * 2
        c_ = int(out_channels * expansion)
        self.c_ = c_
        self.conv1 = Conv(in_channels, c1, 1, 1)
        self.m = nn.ModuleList(
            DepthBottleneckUni(self.c_, self.c_, shortcut, kersize, depth_expansion, small_kersize, use_depthwise) for _
            in range(depth))
        self.conv2 = Conv(c_ * (depth + 2), out_channels, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x_out = list(x.split((self.c_, self.c_), 1))
        for conv in self.m:
            y = conv(x_out[-1])
            x_out.append(y)
        y_out = torch.cat(x_out, axis=1)
        y_out = self.conv2(y_out)
        return y_out


class DepthBottleneckUni(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 shortcut=True,
                 kersize=5,
                 expansion_depth=1,
                 small_kersize=3,
                 use_depthwise=True):
        super(DepthBottleneckUni, self).__init__()

        mid_channel = int(in_channels * expansion_depth)
        self.conv1 = Conv(in_channels, mid_channel, 1)
        self.shortcut = shortcut
        if use_depthwise:

            self.conv2 = UniRepLKNetBlock(mid_channel, kernel_size=kersize)
            self.act = nn.SiLU()
            self.one_conv = Conv(mid_channel, out_channels, kernel_size=1)

        else:
            self.conv2 = Conv(out_channels, out_channels, 3, 1)

    def forward(self, x):
        y = self.conv1(x)

        y = self.act(self.conv2(y))

        y = self.one_conv(y)
        return y


class UniRepLKNetBlock(nn.Module):

    def __init__(self,
                 dim,
                 kernel_size,
                 deploy=False,
                 attempt_use_lk_impl=True):
        super().__init__()
        if deploy:
            print('------------------------------- Note: deploy mode')
        if kernel_size == 0:
            self.dwconv = nn.Identity()
        elif kernel_size >= 3:
            self.dwconv = DilatedReparamBlock(dim, kernel_size, deploy=deploy,
                                              attempt_use_lk_impl=attempt_use_lk_impl)
        else:
            assert kernel_size in [3]
            self.dwconv = get_conv2d_uni(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                                         dilation=1, groups=dim, bias=deploy,
                                         attempt_use_lk_impl=attempt_use_lk_impl)

        if deploy or kernel_size == 0:
            self.norm = nn.Identity()
        else:
            self.norm = get_bn(dim)

    def forward(self, inputs):

        out = self.norm(self.dwconv(inputs))
        return out

    def reparameterize(self):
        if hasattr(self.dwconv, 'merge_dilated_branches'):
            self.dwconv.merge_dilated_branches()
        if hasattr(self.norm, 'running_var'):
            std = (self.norm.running_var + self.norm.eps).sqrt()
            if hasattr(self.dwconv, 'lk_origin'):
                self.dwconv.lk_origin.weight.data *= (self.norm.weight / std).view(-1, 1, 1, 1)
                self.dwconv.lk_origin.bias.data = self.norm.bias + (
                        self.dwconv.lk_origin.bias - self.norm.running_mean) * self.norm.weight / std
            else:
                conv = nn.Conv2d(self.dwconv.in_channels, self.dwconv.out_channels, self.dwconv.kernel_size,
                                 self.dwconv.padding, self.dwconv.groups, bias=True)
                conv.weight.data = self.dwconv.weight * (self.norm.weight / std).view(-1, 1, 1, 1)
                conv.bias.data = self.norm.bias - self.norm.running_mean * self.norm.weight / std
                self.dwconv = conv
            self.norm = nn.Identity()


class DilatedReparamBlock(nn.Module):
    """
    Dilated Reparam Block proposed in UniRepLKNet (https://github.com/AILab-CVC/UniRepLKNet)
    We assume the inputs to this block are (N, C, H, W)
    """

    def __init__(self, channels, kernel_size, deploy, use_sync_bn=False, attempt_use_lk_impl=True):
        super().__init__()
        self.lk_origin = get_conv2d_uni(channels, channels, kernel_size, stride=1,
                                        padding=kernel_size // 2, dilation=1, groups=channels, bias=deploy,
                                        )
        self.attempt_use_lk_impl = attempt_use_lk_impl

        if kernel_size == 17:
            self.kernel_sizes = [5, 9, 3, 3, 3]
            self.dilates = [1, 2, 4, 5, 7]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 5, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [7, 5, 3]
            self.dilates = [1, 1, 1]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3]
            self.dilates = [1, 1]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 1]
            self.dilates = [1, 1]
        elif kernel_size == 3:
            self.kernel_sizes = [3, 1]
            self.dilates = [1, 1]


        else:
            raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

        if not deploy:
            self.origin_bn = get_bn(channels)
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__('dil_conv_k{}_{}'.format(k, r),
                                 nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=k, stride=1,
                                           padding=(r * (k - 1) + 1) // 2, dilation=r, groups=channels,
                                           bias=False))
                self.__setattr__('dil_bn_k{}_{}'.format(k, r), get_bn(channels))

    def forward(self, x):
        if not hasattr(self, 'origin_bn'):  # deploy mode
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
            bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
            out = out + bn(conv(x))
        return out

    def merge_dilated_branches(self):
        if hasattr(self, 'origin_bn'):
            origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
                bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
                branch_k, branch_b = fuse_bn(conv, bn)
                origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
                origin_b += branch_b
            merged_conv = get_conv2d_uni(origin_k.size(0), origin_k.size(0), origin_k.size(2), stride=1,
                                         padding=origin_k.size(2) // 2, dilation=1, groups=origin_k.size(0), bias=True,
                                         attempt_use_lk_impl=self.attempt_use_lk_impl)
            merged_conv.weight.data = origin_k
            merged_conv.bias.data = origin_b
            self.lk_origin = merged_conv
            self.__delattr__('origin_bn')
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__delattr__('dil_conv_k{}_{}'.format(k, r))
                self.__delattr__('dil_bn_k{}_{}'.format(k, r))


from itertools import repeat
import collections.abc


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def fuse_bn(conv, bn):
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std


def get_conv2d_uni(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                   attempt_use_lk_impl=True):
    kernel_size = to_2tuple(kernel_size)
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        padding = to_2tuple(padding)
    need_large_impl = kernel_size[0] == kernel_size[1] and kernel_size[0] > 5 and padding == (
    kernel_size[0] // 2, kernel_size[1] // 2)

    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)


def convert_dilated_to_nondilated(kernel, dilate_rate):
    identity_kernel = torch.ones((1, 1, 1, 1), dtype=kernel.dtype, device=kernel.device)
    if kernel.size(1) == 1:
        #   This is a DW kernel
        dilated = F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate)
        return dilated
    else:
        #   This is a dense or group-wise (but not DW) kernel
        slices = []
        for i in range(kernel.size(1)):
            dilated = F.conv_transpose2d(kernel[:, i:i + 1, :, :], identity_kernel, stride=dilate_rate)
            slices.append(dilated)
        return torch.cat(slices, dim=1)


def merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
    large_k = large_kernel.size(2)
    dilated_k = dilated_kernel.size(2)
    equivalent_kernel_size = dilated_r * (dilated_k - 1) + 1
    equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r)
    rows_to_pad = large_k // 2 - equivalent_kernel_size // 2
    merged_kernel = large_kernel + F.pad(equivalent_kernel, [rows_to_pad] * 4)
    return merged_kernel


def get_bn(channels):
    return nn.BatchNorm2d(channels)


class DepthBottleneckUniv2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 shortcut=True,
                 kersize=5,
                 expansion_depth=1,
                 small_kersize=3,
                 use_depthwise=True):
        super(DepthBottleneckUniv2, self).__init__()

        mid_channel = int(in_channels * expansion_depth)
        mid_channel2 = mid_channel
        self.conv1 = Conv(in_channels, mid_channel, 1)
        self.shortcut = shortcut
        if use_depthwise:
            self.conv2 = UniRepLKNetBlock(mid_channel, kernel_size=kersize)
            self.act = nn.SiLU()
            self.one_conv = Conv(mid_channel, mid_channel2, kernel_size=1)

            self.conv3 = UniRepLKNetBlock(mid_channel2, kernel_size=kersize)
            self.act1 = nn.SiLU()
            self.one_conv2 = Conv(mid_channel2, out_channels, kernel_size=1)
        else:
            self.conv2 = Conv(out_channels, out_channels, 3, 1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.act(self.conv2(y))
        y = self.one_conv(y)
        y = self.act1(self.conv3(y))
        y = self.one_conv2(y)
        return y


class RepHMS(nn.Module):
    def __init__(self, in_channels, out_channels, width=3, depth=1, depth_expansion=2, kersize=5, shortcut=True,
                 expansion=0.5,
                 small_kersize=3, use_depthwise=True):
        super(RepHMS, self).__init__()
        self.width = width
        self.depth = depth
        c1 = int(out_channels * expansion) * width
        c_ = int(out_channels * expansion)
        self.c_ = c_
        self.conv1 = Conv(in_channels, c1, 1, 1)
        self.RepElanMSBlock = nn.ModuleList()
        for _ in range(width - 1):
            DepthBlock = nn.ModuleList([
                DepthBottleneckUniv2(self.c_, self.c_, shortcut, kersize, depth_expansion, small_kersize, use_depthwise)
                for _ in range(depth)
            ])
            self.RepElanMSBlock.append(DepthBlock)

        self.conv2 = Conv(c_ * 1 + c_ * (width - 1) * depth, out_channels, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x_out = [x[:, i * self.c_:(i + 1) * self.c_] for i in range(self.width)]
        x_out[1] = x_out[1] + x_out[0]
        cascade = []
        elan = [x_out[0]]
        for i in range(self.width - 1):
            for j in range(self.depth):
                if i > 0:
                    x_out[i + 1] = x_out[i + 1] + cascade[j]
                    if j == self.depth - 1:
                        # cascade = [cascade[-1]]
                        if self.depth > 1:
                            cascade = [cascade[-1]]
                        else:
                            cascade = []
                x_out[i + 1] = self.RepElanMSBlock[i][j](x_out[i + 1])
                elan.append(x_out[i + 1])
                if i < self.width - 2:
                    cascade.append(x_out[i + 1])

        y_out = torch.cat(elan, 1)
        y_out = self.conv2(y_out)
        return y_out


class DepthBottleneckv2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 shortcut=True,
                 kersize=5,
                 expansion_depth=1,
                 small_kersize=3,
                 use_depthwise=True):
        super(DepthBottleneckv2, self).__init__()

        mid_channel = int(in_channels * expansion_depth)
        mid_channel2 = mid_channel
        self.conv1 = Conv(in_channels, mid_channel, 1)
        self.shortcut = shortcut
        if use_depthwise:
            self.conv2 = DWConv(mid_channel, mid_channel, kersize)
            # self.act = nn.SiLU()
            self.one_conv = Conv(mid_channel, mid_channel2, kernel_size=1)

            self.conv3 = DWConv(mid_channel2, mid_channel2, kersize)
            # self.act1 = nn.SiLU()
            self.one_conv2 = Conv(mid_channel2, out_channels, kernel_size=1)
        else:
            self.conv2 = Conv(out_channels, out_channels, 3, 1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.one_conv(y)
        y = self.conv3(y)
        y = self.one_conv2(y)
        return y


class ConvMS(nn.Module):
    def __init__(self, in_channels, out_channels, width=3, depth=1, depth_expansion=2, kersize=5, shortcut=True,
                 expansion=0.5,
                 small_kersize=3, use_depthwise=True):
        super(ConvMS, self).__init__()
        self.width = width
        self.depth = depth
        c1 = int(out_channels * expansion) * width
        c_ = int(out_channels * expansion)
        self.c_ = c_
        self.conv1 = Conv(in_channels, c1, 1, 1)
        self.RepElanMSBlock = nn.ModuleList()
        for _ in range(width - 1):
            DepthBlock = nn.ModuleList([
                DepthBottleneckv2(self.c_, self.c_, shortcut, kersize, depth_expansion, small_kersize, use_depthwise)
                for _ in range(depth)
            ])
            self.RepElanMSBlock.append(DepthBlock)

        self.conv2 = Conv(c_ * 1 + c_ * (width - 1) * depth, out_channels, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x_out = [x[:, i * self.c_:(i + 1) * self.c_] for i in range(self.width)]
        x_out[1] = x_out[1] + x_out[0]
        cascade = []
        elan = [x_out[0]]
        for i in range(self.width - 1):
            for j in range(self.depth):
                if i > 0:
                    x_out[i + 1] = x_out[i + 1] + cascade[j]
                    if j == self.depth - 1:
                        # cascade = [cascade[-1]]
                        if self.depth > 1:
                            cascade = [cascade[-1]]
                        else:
                            cascade = []
                x_out[i + 1] = self.RepElanMSBlock[i][j](x_out[i + 1])
                elan.append(x_out[i + 1])
                if i < self.width - 2:
                    cascade.append(x_out[i + 1])

        y_out = torch.cat(elan, 1)
        y_out = self.conv2(y_out)
        return y_out




class Pzconv(nn.Module): #这是那个0.496的
    def __init__(self, c1, c2, *, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        dim = c2

        # 分支 1: 3x3
        self.dw_conv3x3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.bn1 = nn.BatchNorm2d(dim)
        self.pw_conv3x3 = nn.Conv2d(dim, dim, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(dim)

        # 分支 2: 5x5
        self.dw_conv5x5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.bn3 = nn.BatchNorm2d(dim)
        self.pw_conv5x5 = nn.Conv2d(dim, dim, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(dim)

        # 分支 3: 7x1 + 1x7
        self.dw_conv_7x1 = nn.Conv2d(dim, dim, kernel_size=(7, 1), padding=(3, 0), groups=dim)
        self.bn5 = nn.BatchNorm2d(dim)
        self.dw_conv_1x7 = nn.Conv2d(dim, dim, kernel_size=(1, 7), padding=(0, 3), groups=dim)
        self.bn6 = nn.BatchNorm2d(dim)
        self.pw_conv_7x1_1x7 = nn.Conv2d(dim, dim, kernel_size=1)
        self.bn7 = nn.BatchNorm2d(dim)

        # 分支 4: 9x1 + 1x9
        self.dw_conv_9x1 = nn.Conv2d(dim, dim, kernel_size=(9, 1), padding=(4, 0), groups=dim)
        self.bn8 = nn.BatchNorm2d(dim)
        self.dw_conv_1x9 = nn.Conv2d(dim, dim, kernel_size=(1, 9), padding=(0, 4), groups=dim)
        self.bn9 = nn.BatchNorm2d(dim)
        self.pw_conv_9x1_1x9 = nn.Conv2d(dim, dim, kernel_size=1)
        self.bn10 = nn.BatchNorm2d(dim)

        # 最终融合
        self.final_pw_conv = nn.Conv2d(dim * 5, dim, kernel_size=1)
        self.bn_final = nn.BatchNorm2d(dim)

        # 激活函数
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y1 = self.dw_conv3x3(x)
        y1 = self.bn1(y1)
        y1 = self.act(y1)
        y1 = self.pw_conv3x3(y1)
        y1 = self.bn2(y1)
        y1 = self.act(y1)

        y2 = self.dw_conv5x5(x)
        y2 = self.bn3(y2)
        y2 = self.act(y2)
        y2 = self.pw_conv5x5(y2)
        y2 = self.bn4(y2)
        y2 = self.act(y2)

        y3 = self.dw_conv_7x1(x)
        y3 = self.bn5(y3)
        y3 = self.act(y3)
        y3 = self.dw_conv_1x7(y3)
        y3 = self.bn6(y3)
        y3 = self.act(y3)
        y3 = self.pw_conv_7x1_1x7(y3)
        y3 = self.bn7(y3)
        y3 = self.act(y3)

        y4 = self.dw_conv_9x1(x)
        y4 = self.bn8(y4)
        y4 = self.act(y4)
        y4 = self.dw_conv_1x9(y4)
        y4 = self.bn9(y4)
        y4 = self.act(y4)
        y4 = self.pw_conv_9x1_1x9(y4)
        y4 = self.bn10(y4)
        y4 = self.act(y4)

        y5 = x  # identity

        z = torch.cat([y1, y2, y3, y4, y5], dim=1)
        out = self.final_pw_conv(z)
        out = self.bn_final(out)
        out = self.act(out)

        return out + x




class FCM_3(nn.Module):
    def __init__(self, dim,dim_out):
        super().__init__()
        self.one = dim - dim // 4
        self.two = dim // 4
        self.conv1 = Conv(dim - dim // 4, dim - dim // 4, 3, 1, 1)
        self.conv12 = Conv(dim - dim // 4, dim - dim // 4, 3, 1, 1)
        self.conv123 = Conv(dim - dim // 4, dim, 1, 1)
        self.conv2 = Conv(dim // 4, dim, 1, 1)
        self.spatial = Spatial(dim)
        self.channel = Channel(dim)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)
        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)
        x4 = self.conv2(x2)
        x33 = self.spatial(x4) * x3
        x44 = self.channel(x3) * x4
        x5 = x33 + x44
        return x5


class FCM_2(nn.Module):
    def __init__(self, dim,dim_out):
        super().__init__()
        self.one = dim - dim // 4
        self.two = dim // 4
        self.conv1 = Conv(dim - dim // 4, dim - dim // 4, 3, 1, 1)
        self.conv12 = Conv(dim - dim // 4, dim - dim // 4, 3, 1, 1)
        self.conv123 = Conv(dim - dim // 4, dim, 1, 1)

        self.conv2 = Conv(dim // 4, dim, 1, 1)
        self.spatial = Spatial(dim)
        self.channel = Channel(dim)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)
        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)
        x4 = self.conv2(x2)
        x33 = self.spatial(x4) * x3
        x44 = self.channel(x3) * x4
        x5 = x33 + x44

        return x5







class Combined_Hybrid_Pz_Enhanced(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        if dim_out is None:
            dim_out = dim
        self.dim = dim
        self.dim_out = dim_out

        # ==== Enhanced Hybrid_FCM Part ====
        self.split_ratio = 3  # 3/4 vs. 1/4

        # 分支1: 3x3 + 1x1
        self.hybrid_branch1 = nn.Sequential(
            nn.Conv2d(dim * 3 // 4, dim * 3 // 4, 3, 1, 1),
            nn.BatchNorm2d(dim * 3 // 4),
            nn.ReLU(),
            nn.Conv2d(dim * 3 // 4, dim, 1)
        )
        # 分支2: 1x1
        self.hybrid_branch2 = nn.Conv2d(dim // 4, dim, 1)

        # 轻量级注意力机制（通道 + 空间）
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 8, dim, 1),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(dim, 1, 7, padding=3),
            nn.Sigmoid()
        )

        # 最终融合（增强版：注意力加权分支交互）
        self.hybrid_final_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim_out, 1),  # 注意这里改为接收拼接后的2*dim输入
            nn.BatchNorm2d(dim_out)
        )

        # ==== Pzconv Part ====
        # 分支1: 3x3
        self.pz_dw_conv3x3 = nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=1, groups=dim_out)
        self.pz_bn1 = nn.BatchNorm2d(dim_out)
        self.pz_pw_conv3x3 = nn.Conv2d(dim_out, dim_out, kernel_size=1)
        self.pz_bn2 = nn.BatchNorm2d(dim_out)

        # 分支2: 5x5
        self.pz_dw_conv5x5 = nn.Conv2d(dim_out, dim_out, kernel_size=5, padding=2, groups=dim_out)
        self.pz_bn3 = nn.BatchNorm2d(dim_out)
        self.pz_pw_conv5x5 = nn.Conv2d(dim_out, dim_out, kernel_size=1)
        self.pz_bn4 = nn.BatchNorm2d(dim_out)

        # 分支3: 7x1 + 1x7
        self.pz_dw_conv_7x1 = nn.Conv2d(dim_out, dim_out, kernel_size=(7, 1), padding=(3, 0), groups=dim_out)
        self.pz_bn5 = nn.BatchNorm2d(dim_out)
        self.pz_dw_conv_1x7 = nn.Conv2d(dim_out, dim_out, kernel_size=(1, 7), padding=(0, 3), groups=dim_out)
        self.pz_bn6 = nn.BatchNorm2d(dim_out)
        self.pz_pw_conv_7x1_1x7 = nn.Conv2d(dim_out, dim_out, kernel_size=1)
        self.pz_bn7 = nn.BatchNorm2d(dim_out)

        # 分支4: 9x1 + 1x9
        self.pz_dw_conv_9x1 = nn.Conv2d(dim_out, dim_out, kernel_size=(9, 1), padding=(4, 0), groups=dim_out)
        self.pz_bn8 = nn.BatchNorm2d(dim_out)
        self.pz_dw_conv_1x9 = nn.Conv2d(dim_out, dim_out, kernel_size=(1, 9), padding=(0, 4), groups=dim_out)
        self.pz_bn9 = nn.BatchNorm2d(dim_out)
        self.pz_pw_conv_9x1_1x9 = nn.Conv2d(dim_out, dim_out, kernel_size=1)
        self.pz_bn10 = nn.BatchNorm2d(dim_out)

        # identity
        self.identity = nn.Identity()

        # 最终融合
        self.pz_final_pw_conv = nn.Conv2d(dim_out * 5, dim_out, kernel_size=1)
        self.pz_bn_final = nn.BatchNorm2d(dim_out)

        # 激活函数
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        B, C, H, W = x.shape

        # ====== Enhanced Hybrid_FCM Forward ======
        x1, x2 = torch.split(x, [C * 3 // 4, C // 4], dim=1)

        # 分支处理
        x1 = self.hybrid_branch1(x1)
        x2 = self.hybrid_branch2(x2)

        # 增强版注意力加权融合
        # 1. 通道注意力加权
        channel_weights = self.channel_att(x1 + x2)
        x1_weighted = x1 * channel_weights
        x2_weighted = x2 * (1 - channel_weights)

        # 2. 空间注意力加权
        spatial_weights = self.spatial_att(x1_weighted + x2_weighted)
        x1_spatial = x1_weighted * spatial_weights
        x2_spatial = x2_weighted * spatial_weights

        # 3. 拼接两个分支
        x_fused = torch.cat([x1_spatial, x2_spatial], dim=1)

        # 4. 最终输出
        x_hybrid = self.hybrid_final_conv(x_fused)
        if self.dim_out == C:
            x_hybrid += x
        x_hybrid = self.act(x_hybrid)

        # ====== Pzconv Forward ======
        y1 = self.pz_dw_conv3x3(x_hybrid)
        y1 = self.pz_bn1(y1)
        y1 = self.act(y1)
        y1 = self.pz_pw_conv3x3(y1)
        y1 = self.pz_bn2(y1)
        y1 = self.act(y1)

        y2 = self.pz_dw_conv5x5(x_hybrid)
        y2 = self.pz_bn3(y2)
        y2 = self.act(y2)
        y2 = self.pz_pw_conv5x5(y2)
        y2 = self.pz_bn4(y2)
        y2 = self.act(y2)

        y3 = self.pz_dw_conv_7x1(x_hybrid)
        y3 = self.pz_bn5(y3)
        y3 = self.act(y3)
        y3 = self.pz_dw_conv_1x7(y3)
        y3 = self.pz_bn6(y3)
        y3 = self.act(y3)
        y3 = self.pz_pw_conv_7x1_1x7(y3)
        y3 = self.pz_bn7(y3)
        y3 = self.act(y3)

        y4 = self.pz_dw_conv_9x1(x_hybrid)
        y4 = self.pz_bn8(y4)
        y4 = self.act(y4)
        y4 = self.pz_dw_conv_1x9(y4)
        y4 = self.pz_bn9(y4)
        y4 = self.act(y4)
        y4 = self.pz_pw_conv_9x1_1x9(y4)
        y4 = self.pz_bn10(y4)
        y4 = self.act(y4)

        y5 = self.identity(x_hybrid)

        z = torch.cat([y1, y2, y3, y4, y5], dim=1)
        pz_out = self.pz_final_pw_conv(z)
        pz_out = self.pz_bn_final(pz_out)
        pz_out = self.act(pz_out)

        return pz_out + x_hybrid





class MSGI_FCM(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out if dim_out is not None else dim  # 处理输出维度

        # Step 1: 多尺度特征提取（添加BN和激活函数）
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True)
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(inplace=True)
        )
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(inplace=True)
        )

        # Step 2: 特征融合（添加BN和激活函数）
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        # Step 3: 通道注意力（保持原有结构）
        self.channel_gap = nn.AdaptiveAvgPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Linear(dim, dim // 8, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 8, dim, bias=False),
            nn.Sigmoid()
        )

        # Step 4: 空间注意力（保持原有结构）
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=7, stride=1, padding=3),
            nn.Sigmoid()
        )

        # Step 5: 最终融合（添加BN和激活函数）
        self.final_conv = nn.Sequential(
            nn.Conv2d(dim, self.dim_out, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.dim_out)
            # 激活函数放在残差连接之后（见forward）
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # Step 1: 多尺度特征提取
        x1 = self.conv1x1(x)  # [B, C//2, H, W]
        x3 = self.conv3x3(x)  # [B, C//4, H, W]
        x5 = self.conv5x5(x)  # [B, C//4, H, W]

        # 拼接多尺度特征
        x_multi = torch.cat([x1, x3, x5], dim=1)  # [B, C, H, W]

        # Step 2: 特征整合
        x_fused = self.fusion_conv(x_multi)  # [B, C, H, W]

        # Step 3: 通道注意力
        x_gap = self.channel_gap(x_fused)  # [B, C, 1, 1]
        x_gap = x_gap.view(B, C)  # [B, C]
        channel_weights = self.channel_fc(x_gap).view(B, C, 1, 1)  # [B, C, 1, 1]
        x_channel_enhanced = x_fused * channel_weights  # 通道加权

        # Step 4: 空间注意力
        spatial_weights = self.spatial_conv(x_fused)  # [B, 1, H, W]
        x_spatial_enhanced = x_fused * spatial_weights  # 空间加权

        # Step 5: 融合通道与空间特征
        x_fused_enhanced = x_channel_enhanced + x_spatial_enhanced  # 相加融合
        x_fused_enhanced = self.final_conv(x_fused_enhanced)  # [B, dim_out, H, W]

        # Step 6: 残差连接（处理维度变化）
        if self.dim_out == C:
            x_fused_enhanced += x
        x_fused_enhanced = nn.ReLU(inplace=True)(x_fused_enhanced)  # 最终激活

        return x_fused_enhanced



class ImprovedFCM(nn.Module):
    def __init__(self, dim,):
        super().__init__()
        self.one = dim - dim // 4
        self.two = dim // 4
        self.conv1 = Conv(dim - dim // 4, dim - dim // 4, 3, 1, 1)
        self.conv12 = Conv(dim - dim // 4, dim - dim // 4, 3, 1, 1)
        self.conv123 = Conv(dim - dim // 4, dim, 1, 1)
        self.conv2 = Conv(dim // 4, dim, 1, 1)
        self.spatial = Spatial(dim)
        self.channel = Channel(dim)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)
        x3 = self.conv1(x1)
        x3 = self.conv12(x3)
        x3 = self.conv123(x3)
        x4 = self.conv2(x2)
        # x33 = self.spatial(x4) * x3
        # x44 = self.channel(x3) * x4
        return x3,x4

# ========================
#   特征重标定模块（替代原乘法交互）
# ========================
class FeatureCalibration(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.spatial_att = Spatial(dim)  # 保持原Spatial模块
        self.channel_att = Channel(dim)  # 保持原Channel模块

    def forward(self, x3, x4):
        # 特征重标定（避免直接相乘）
        x3_calib = x3 * self.spatial_att(x4)
        x4_calib = x4 * self.channel_att(x3)
        return x3_calib + x4_calib  # 改为相加融合

    # def forward(self, x):  # 接受单一输入
    #     # 示例：自注意力机制（可以用其他方式）
    #     x_calib = x * self.spatial_att(x)  # 空间注意力
    #     x_calib += x * self.channel_att(x)  # 通道注意力
    #     return x_calib


# ========================
#   完整CombinedModule2
# ========================
class CombinedModule_to(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        if dim_out is None:
            dim_out = dim
        self.dim = dim_out

        # === 改进版FCM ===
        self.fcm = ImprovedFCM(dim)
        self.calib = FeatureCalibration(dim)

        # === PzConv部分（保持原结构）===
        # 分支1: 3x3
        self.dw_conv3x3 = nn.Conv2d(dim_out, dim_out, 3, 1, padding=1, groups=dim_out)
        self.bn1 = nn.BatchNorm2d(dim_out)
        self.pw_conv3x3 = nn.Conv2d(dim_out, dim_out, 1)
        self.bn2 = nn.BatchNorm2d(dim_out)

        # 分支2: 5x5
        self.dw_conv5x5 = nn.Conv2d(dim_out, dim_out, 5, 1, padding=2, groups=dim_out)
        self.bn3 = nn.BatchNorm2d(dim_out)
        self.pw_conv5x5 = nn.Conv2d(dim_out, dim_out, 1)
        self.bn4 = nn.BatchNorm2d(dim_out)

        # 分支3: 7x1 + 1x7
        self.dw_conv_7x1 = nn.Conv2d(dim_out, dim_out, (7,1), padding=(3,0), groups=dim_out)
        self.bn5 = nn.BatchNorm2d(dim_out)
        self.dw_conv_1x7 = nn.Conv2d(dim_out, dim_out, (1,7), padding=(0,3), groups=dim_out)
        self.bn6 = nn.BatchNorm2d(dim_out)
        self.pw_conv_7x1_1x7 = nn.Conv2d(dim_out, dim_out, 1)
        self.bn7 = nn.BatchNorm2d(dim_out)

        # 分支4: 9x1 + 1x9
        self.dw_conv_9x1 = nn.Conv2d(dim_out, dim_out, (9,1), padding=(4,0), groups=dim_out)
        self.bn8 = nn.BatchNorm2d(dim_out)
        self.dw_conv_1x9 = nn.Conv2d(dim_out, dim_out, (1,9), padding=(0,4), groups=dim_out)
        self.bn9 = nn.BatchNorm2d(dim_out)
        self.pw_conv_9x1_1x9 = nn.Conv2d(dim_out, dim_out, 1)
        self.bn10 = nn.BatchNorm2d(dim_out)

        # identity
        self.identity = nn.Identity()

        # 最终融合
        self.final_pw_conv = nn.Conv2d(dim_out * 5, dim_out, 1)
        self.bn_final = nn.BatchNorm2d(dim_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # === FCM Forward ===
        x3,x4 = self.fcm(x)

        # === 特征重标定 ===
        # 假设原分支x3和x4的逻辑需要调整（此处需根据原逻辑适配）
        # x3, x4 = torch.split(x, [self.one, self.two], dim=1)  # 原逻辑可能需要修改
        # 替换为改进后的交互方式（此处需结合原模块的具体分支定义）
        # 以下为示例逻辑，需根据实际分支定义调整
        # x3 = self.calib.branch1(fcm_out)  # 分支1输出
        # x4 = self.calib.branch2(fcm_out)  # 分支2输出
        # calib_out = self.calib(x3, x4)
        calib_out = self.calib(x3,x4)

        # === PzConv Forward ===
        y1 = self.dw_conv3x3(calib_out)
        y1 = self.bn1(y1)
        y1 = self.act(y1)
        y1 = self.pw_conv3x3(y1)
        y1 = self.bn2(y1)
        y1 = self.act(y1)

        y2 = self.dw_conv5x5(calib_out)
        y2 = self.bn3(y2)
        y2 = self.act(y2)
        y2 = self.pw_conv5x5(y2)
        y2 = self.bn4(y2)
        y2 = self.act(y2)

        y3 = self.dw_conv_7x1(calib_out)
        y3 = self.bn5(y3)
        y3 = self.act(y3)
        y3 = self.dw_conv_1x7(y3)
        y3 = self.bn6(y3)
        y3 = self.act(y3)
        y3 = self.pw_conv_7x1_1x7(y3)
        y3 = self.bn7(y3)
        y3 = self.act(y3)

        y4 = self.dw_conv_9x1(calib_out)
        y4 = self.bn8(y4)
        y4 = self.act(y4)
        y4 = self.dw_conv_1x9(y4)
        y4 = self.bn9(y4)
        y4 = self.act(y4)
        y4 = self.pw_conv_9x1_1x9(y4)
        y4 = self.bn10(y4)
        y4 = self.act(y4)

        y5 = self.identity(calib_out)

        z = torch.cat([y1, y2, y3, y4, y5], dim=1)
        pzconv_out = self.act(self.bn_final(self.final_pw_conv(z)))

        return pzconv_out + calib_out  # 残差连接






class Channel(nn.Module):
    """优化版通道注意力 - 用GAP+FC替代3x3 DW卷积，计算量减少约90%"""
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.gap(x)
        y = self.fc(y)
        return y



class Spatial(nn.Module):
    """优化版空间注意力 - 去掉BN，直接用sigmoid，减少计算量"""
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, 1, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv1(x)
        x6 = self.sigmoid(x1)
        return x6
# 目前效果最好的CombinedModule
class CombinedModule(nn.Module):  # 优化版：FCM + pzconv
    def __init__(self , dim , dim_out=None):
        super().__init__()
        if dim_out is None:
            dim_out = dim

        # === 优化版 FCM Part ===
        self.one = dim - dim // 4
        self.two = dim // 4
        # 优化点3: 减少卷积层数 - 只用1个3x3替代原来的2个3x3，计算量减少约50%
        self.conv1 = Conv(self.one , dim , 3 , 1)  # 直接输出到dim维度
        self.conv2 = Conv(self.two , dim , 1 , 1)
        self.spatial = Spatial(dim)
        self.channel = Channel(dim)

        # === Pzconv Part ===
        self.dim = dim_out

        # 分支 1: 3x3
        self.dw_conv3x3 = nn.Conv2d(dim_out , dim_out , kernel_size=3 , padding=1 , groups=dim_out)
        self.bn1 = nn.BatchNorm2d(dim_out)
        self.pw_conv3x3 = nn.Conv2d(dim_out , dim_out , kernel_size=1)
        self.bn2 = nn.BatchNorm2d(dim_out)

        # 分支 2: 分离卷积 (3x1 + 1x3) 替代 5x5，计算量减少约44%
        self.conv_h = nn.Conv2d(dim_out , dim_out , kernel_size=(3 , 1) , padding=(1 , 0) , groups=dim_out)
        self.conv_v = nn.Conv2d(dim_out , dim_out , kernel_size=(1 , 3) , padding=(0 , 1) , groups=dim_out)
        self.bn3 = nn.BatchNorm2d(dim_out)
        self.pw_conv_sep = nn.Conv2d(dim_out , dim_out , kernel_size=1)
        self.bn4 = nn.BatchNorm2d(dim_out)

        # self.dw_conv5x5 = nn.Conv2d(dim_out, dim_out, kernel_size=5, padding=2, groups=dim_out)
        # self.bn3 = nn.BatchNorm2d(dim_out)
        # self.pw_conv_sep = nn.Conv2d(dim_out, dim_out, kernel_size=1)
        # self.bn4 = nn.BatchNorm2d(dim_out)


        # identity
        self.identity = nn.Identity()

        # 最终融合
        self.final_pw_conv = nn.Conv2d(dim_out * 3 , dim_out , kernel_size=1)
        self.bn_final = nn.BatchNorm2d(dim_out)

        # 激活函数
        self.act = nn.ReLU(inplace=True)

    def forward(self , x):
        # === 优化版 FCM Forward ===
        x1 , x2 = torch.split(x , [self.one , self.two] , dim=1)
        # 优化点3: 减少卷积层数 - 只用1个3x3
        x3 = self.conv1(x1)  # 直接输出到dim维度
        x4 = self.conv2(x2)

        # 优化点1&2: 使用优化后的注意力机制
        x33 = self.spatial(x4) * x3
        x44 = self.channel(x3) * x4
        fcm_out = x33 + x44

        # === Pzconv Forward（每个分支都严格还原原始流程）===
        y1 = self.dw_conv3x3(fcm_out)
        y1 = self.bn1(y1)
        y1 = self.act(y1)
        y1 = self.pw_conv3x3(y1)
        y1 = self.bn2(y1)
        y1 = self.act(y1)

        y2 = self.conv_h(fcm_out)  # 3x1 卷积
        y2 = self.conv_v(y2)  # 1x3 卷积
        #y2 = self.dw_conv5x5(fcm_out)  # 直接 5x5 depthwise

        y2 = self.bn3(y2)
        y2 = self.act(y2)
        y2 = self.pw_conv_sep(y2)
        y2 = self.bn4(y2)
        y2 = self.act(y2)

        y5 = self.identity(fcm_out)

        z = torch.cat([y1 , y2 , y5] , dim=1)
        pzconv_out = self.act(self.bn_final(self.final_pw_conv(z)))

        return pzconv_out + fcm_out





# ===================== 3. 1:1切分模块（核心修正） =====================
class CombinedModule_1to1(nn.Module):
    # 关键：添加*args, **kwargs吸收多余参数
    def __init__(self, ch, dim_out=None, *args, **kwargs):
        super().__init__()
        dim = ch  # 输入通道数 = dim
        if dim_out is None:
            dim_out = dim

        # 1:1 对半切分
        self.one = dim // 2
        self.two = dim - self.one
        self.conv1 = Conv(self.one , dim , 3 , 1)
        self.conv2 = Conv(self.two , dim , 1 , 1)
        self.spatial = Spatial(dim)
        self.channel = Channel(dim)

        # Pzconv Part
        self.dim = dim_out
        self.dw_conv3x3 = nn.Conv2d(dim_out , dim_out , kernel_size=3 , padding=1 , groups=dim_out)
        self.bn1 = nn.BatchNorm2d(dim_out)
        self.pw_conv3x3 = nn.Conv2d(dim_out , dim_out , kernel_size=1)
        self.bn2 = nn.BatchNorm2d(dim_out)
        self.conv_h = nn.Conv2d(dim_out , dim_out , kernel_size=(3 , 1) , padding=(1 , 0) , groups=dim_out)
        self.conv_v = nn.Conv2d(dim_out , dim_out , kernel_size=(1 , 3) , padding=(0 , 1) , groups=dim_out)
        self.bn3 = nn.BatchNorm2d(dim_out)
        self.pw_conv_sep = nn.Conv2d(dim_out , dim_out , kernel_size=1)
        self.bn4 = nn.BatchNorm2d(dim_out)
        self.identity = nn.Identity()
        self.final_pw_conv = nn.Conv2d(dim_out * 3 , dim_out , kernel_size=1)
        self.bn_final = nn.BatchNorm2d(dim_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self , x):
        # FCM Forward
        x1 , x2 = torch.split(x , [self.one , self.two] , dim=1)
        x3 = self.conv1(x1)
        x4 = self.conv2(x2)
        x33 = self.spatial(x4) * x3
        x44 = self.channel(x3) * x4
        fcm_out = x33 + x44

        # Pzconv Forward
        y1 = self.dw_conv3x3(fcm_out)
        y1 = self.bn1(y1)
        y1 = self.act(y1)
        y1 = self.pw_conv3x3(y1)
        y1 = self.bn2(y1)
        y1 = self.act(y1)

        y2 = self.conv_h(fcm_out)
        y2 = self.conv_v(y2)
        y2 = self.bn3(y2)
        y2 = self.act(y2)
        y2 = self.pw_conv_sep(y2)
        y2 = self.bn4(y2)
        y2 = self.act(y2)

        y5 = self.identity(fcm_out)
        z = torch.cat([y1 , y2 , y5] , dim=1)
        pzconv_out = self.act(self.bn_final(self.final_pw_conv(z)))

        return pzconv_out + fcm_out




# ===================== 4. 1:3切分模块（核心修正） =====================
class CombinedModule_1to3(nn.Module):
    # 关键：添加*args, **kwargs吸收多余参数
    def __init__(self, ch, dim_out=None, *args, **kwargs):
        super().__init__()
        dim = ch  # 输入通道数 = dim
        if dim_out is None:
            dim_out = dim

        # 1:3 切分
        self.one = dim // 4
        self.two = dim - self.one
        self.conv1 = Conv(self.one , dim , 3 , 1)
        self.conv2 = Conv(self.two , dim , 1 , 1)
        self.spatial = Spatial(dim)
        self.channel = Channel(dim)

        # Pzconv Part
        self.dim = dim_out
        self.dw_conv3x3 = nn.Conv2d(dim_out , dim_out , kernel_size=3 , padding=1 , groups=dim_out)
        self.bn1 = nn.BatchNorm2d(dim_out)
        self.pw_conv3x3 = nn.Conv2d(dim_out , dim_out , kernel_size=1)
        self.bn2 = nn.BatchNorm2d(dim_out)
        self.conv_h = nn.Conv2d(dim_out , dim_out , kernel_size=(3 , 1) , padding=(1 , 0) , groups=dim_out)
        self.conv_v = nn.Conv2d(dim_out , dim_out , kernel_size=(1 , 3) , padding=(0 , 1) , groups=dim_out)
        self.bn3 = nn.BatchNorm2d(dim_out)
        self.pw_conv_sep = nn.Conv2d(dim_out , dim_out , kernel_size=1)
        self.bn4 = nn.BatchNorm2d(dim_out)
        self.identity = nn.Identity()
        self.final_pw_conv = nn.Conv2d(dim_out * 3 , dim_out , kernel_size=1)
        self.bn_final = nn.BatchNorm2d(dim_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self , x):
        # FCM Forward
        x1 , x2 = torch.split(x , [self.one , self.two] , dim=1)
        x3 = self.conv1(x1)
        x4 = self.conv2(x2)
        x33 = self.spatial(x4) * x3
        x44 = self.channel(x3) * x4
        fcm_out = x33 + x44

        # Pzconv Forward
        y1 = self.dw_conv3x3(fcm_out)
        y1 = self.bn1(y1)
        y1 = self.act(y1)
        y1 = self.pw_conv3x3(y1)
        y1 = self.bn2(y1)
        y1 = self.act(y1)

        y2 = self.conv_h(fcm_out)
        y2 = self.conv_v(y2)
        y2 = self.bn3(y2)
        y2 = self.act(y2)
        y2 = self.pw_conv_sep(y2)
        y2 = self.bn4(y2)
        y2 = self.act(y2)

        y5 = self.identity(fcm_out)
        z = torch.cat([y1 , y2 , y5] , dim=1)
        pzconv_out = self.act(self.bn_final(self.final_pw_conv(z)))

        return pzconv_out + fcm_out