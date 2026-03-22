# import torch
# import torch.nn as nn
# import torch_dct as DCT
# import torch.nn.functional as F
# from mmcv.cnn import ConvModule
# from mmengine.model import BaseModule
#
# # https://arxiv.org/pdf/2412.10116
#
#
# #------------------------------------------------------------------#
# # Spatial Path of HFP
# # Only p1&p2 use dct to extract high_frequency response
# #------------------------------------------------------------------#
# class DctSpatialInteraction(BaseModule):
#     def __init__(self,
#                 in_channels,
#                 ratio,
#                 isdct = True,
#                 init_cfg=dict(
#                     type='Xavier', layer='Conv2d', distribution='uniform')):
#         super(DctSpatialInteraction, self).__init__(init_cfg)
#         self.ratio = ratio
#         self.isdct = isdct # true when in p1&p2 # false when in p3&p4
#         if not self.isdct:
#             self.spatial1x1 = nn.Sequential(
#             *[ConvModule(in_channels, 1, kernel_size=1, bias=False)]
#         )
#
#     def forward(self, x):
#         _, _, h0, w0 = x.size()
#         if not self.isdct:
#             return x * torch.sigmoid(self.spatial1x1(x))
#         idct = DCT.dct_2d(x, norm='ortho')
#         weight = self._compute_weight(h0, w0, self.ratio).to(x.device)
#         weight = weight.view(1, h0, w0).expand_as(idct)
#         dct = idct * weight # filter out low-frequency features
#         dct_ = DCT.idct_2d(dct, norm='ortho') # generate spatial mask
#         return x * dct_
#
#     def _compute_weight(self, h, w, ratio):
#         h0 = int(h * ratio[0])
#         w0 = int(w * ratio[1])
#         weight = torch.ones((h, w), requires_grad=False)
#         weight[:h0, :w0] = 0
#         return weight
#
# # ------------------------------------------------------------------#
# # Channel Path of HFP
# # Only p1&p2 use dct to extract high_frequency response
# # ------------------------------------------------------------------#
# class DctChannelInteraction(BaseModule):
#     def __init__(self,
#                  in_channels,
#                  patch,
#                  ratio,
#                  isdct=True,
#                  init_cfg=dict(
#                      type='Xavier', layer='Conv2d', distribution='uniform')
#                  ):
#         super(DctChannelInteraction, self).__init__(init_cfg)
#         self.in_channels = in_channels
#         self.h = patch[0]
#         self.w = patch[1]
#         self.ratio = ratio
#         self.isdct = isdct
#         self.channel1x1 = nn.Sequential(
#             *[ConvModule(in_channels, in_channels, kernel_size=1, groups=8, bias=False)],
#         )
#         self.channel2x1 = nn.Sequential(
#             *[ConvModule(in_channels, in_channels, kernel_size=1, groups=8, bias=False)],
#         )
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         n, c, h, w = x.size()
#         if not self.isdct: # true when in p1&p2 # false when in p3&p4
#             amaxp = F.adaptive_max_pool2d(x,  output_size=(1, 1))
#             aavgp = F.adaptive_avg_pool2d(x,  output_size=(1, 1))
#             channel = self.channel1x1(self.relu(amaxp)) + self.channel1x1(self.relu(aavgp)) # 2025 03 15 szc
#             return x * torch.sigmoid(self.channel2x1(channel))
#
#         idct = DCT.dct_2d(x, norm='ortho')
#         weight = self._compute_weight(h, w, self.ratio).to(x.device)
#         weight = weight.view(1, h, w).expand_as(idct)
#         dct = idct * weight # filter out low-frequency features
#         dct_ = DCT.idct_2d(dct, norm='ortho')
#
#         amaxp = F.adaptive_max_pool2d(dct_,  output_size=(self.h, self.w))
#         aavgp = F.adaptive_avg_pool2d(dct_,  output_size=(self.h, self.w))
#         amaxp = torch.sum(self.relu(amaxp), dim=[2 ,3]).view(n, c, 1, 1)
#         aavgp = torch.sum(self.relu(aavgp), dim=[2 ,3]).view(n, c, 1, 1)
#
#         # channel = torch.cat([self.channel1x1(aavgp), self.channel1x1(amaxp)], dim = 1) # TODO: The values of aavgp and amaxp appear to be on different scales. Add is a better choice instead of concate.
#         channel = self.channel1x1(amaxp) + self.channel1x1(aavgp) # 2025 03 15 szc
#         return x * torch.sigmoid(self.channel2x1(channel))
#
#     def _compute_weight(self, h, w, ratio):
#         h0 = int(h * ratio[0])
#         w0 = int(w * ratio[1])
#         weight = torch.ones((h, w), requires_grad=False)
#         weight[:h0, :w0] = 0
#         return weight
#
#
#     # ------------------------------------------------------------------#
# # High Frequency Perception Module HFP
# # ------------------------------------------------------------------#
# class HFP(BaseModule):
#     def __init__(self,
#                  in_channels,
#                  ratio = (0.2, 0.2),
#                  patch = (8 ,8),
#                  isdct = True,
#                  init_cfg=dict(
#                      type='Xavier', layer='Conv2d', distribution='uniform')):
#         super(HFP, self).__init__(init_cfg)
#         self.spatial = DctSpatialInteraction(in_channels, ratio=ratio, isdct = isdct)
#         self.channel = DctChannelInteraction(in_channels, patch=patch, ratio=ratio, isdct = isdct)
#         self.out =  nn.Sequential(
#             *[ConvModule(in_channels, in_channels, kernel_size=3, padding=1),
#               nn.GroupNorm(8, in_channels)]
#         )
#     def forward(self, x):
#         spatial = self.spatial(x) # output of spatial path
#         channel = self.channel(x) # output of channel path
#         return self.out(spatial + channel)
#
#
#
# def autopad(k, p=None, d=1):  # kernel, padding, dilation
#     """Pad to 'same' shape outputs."""
#     if d > 1:
#         k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
#     if p is None:
#         p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
#     return p
#
#
# class Conv(nn.Module):
#     """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
#
#     default_act = nn.SiLU()  # default activation
#
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
#         """Initialize Conv layer with given arguments including activation."""
#         super().__init__()
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
#
#     def forward(self, x):
#         """Apply convolution, batch normalization and activation to input tensor."""
#         return self.act(self.bn(self.conv(x)))
#
#     def forward_fuse(self, x):
#         """Perform transposed convolution of 2D data."""
#         return self.act(self.conv(x))
#
# class Bottleneck(nn.Module):
#     """Standard bottleneck."""
#
#     def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
#         """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, k[0], 1)
#         self.cv2 = Conv(c_, c2, k[1], 1, g=g)
#         self.add = shortcut and c1 == c2
#
#     def forward(self, x):
#         """Applies the YOLO FPN to input data."""
#         return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
#
# class C2f(nn.Module):
#     """Faster Implementation of CSP Bottleneck with 2 convolutions."""
#
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
#         """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
#         super().__init__()
#         self.c = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, 2 * self.c, 1, 1)
#         self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
#         self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
#
#     def forward(self, x):
#         """Forward pass through C2f layer."""
#         y = list(self.cv1(x).chunk(2, 1))
#         y.extend(m(y[-1]) for m in self.m)
#         return self.cv2(torch.cat(y, 1))
#
#     def forward_split(self, x):
#         """Forward pass using split() instead of chunk()."""
#         y = self.cv1(x).split((self.c, self.c), 1)
#         y = [y[0], y[1]]
#         y.extend(m(y[-1]) for m in self.m)
#         return self.cv2(torch.cat(y, 1))
#
# class C3(nn.Module):
#     """CSP Bottleneck with 3 convolutions."""
#
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c1, c_, 1, 1)
#         self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
#         self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))
#
#     def forward(self, x):
#         """Forward pass through the CSP bottleneck with 2 convolutions."""
#         return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
#
# class Bottleneck_HFP(nn.Module):
#     """Standard bottleneck."""
#
#     def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
#         """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, k[0], 1)
#         self.cv2 = HFP(c_)
#         self.add = shortcut and c1 == c2
#
#     def forward(self, x):
#         """Applies the YOLO FPN to input data."""
#         return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
#
# class C3k(C3):
#     """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""
#
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
#         """Initializes the C3k module with specified channels, number of layers, and configurations."""
#         super().__init__(c1, c2, n, shortcut, g, e)
#         c_ = int(c2 * e)  # hidden channels
#         # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
#         self.m = nn.Sequential(*(Bottleneck_HFP(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
#
# # 在c3k=True时，使用Bottleneck_CGLU特征融合，为false的时候我们使用普通的Bottleneck提取特征
# class C3k2_HFP(C2f):
#     """Faster Implementation of CSP Bottleneck with 2 convolutions."""
#
#     def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
#         """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
#         super().__init__(c1, c2, n, shortcut, g, e)
#         self.m = nn.ModuleList(
#             C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
#         )
#
#
#
#
# if __name__ == '__main__':
#     # 测试配置
#     in_channels = 64  # 输入特征图通道数
#     ratio = (0.2, 0.2)  # 高频滤波比例（控制低频区域大小）
#     patch = (4, 4)  # 通道注意力的池化补丁大小
#     isdct = True  # 是否启用DCT变换（模拟p1/p2场景）
#     input_size = (1, in_channels, 27, 31)  # 输入尺寸：(batch_size, channels, height, width)
#
#     # 初始化模型
#     hfp_module = HFP(
#         in_channels=in_channels,
#         # ratio=ratio,
#         # patch=patch,
#         # isdct=isdct
#     )
#
#     # 生成随机输入
#     x = torch.randn(input_size)  # 随机张量模拟特征图
#     print(f"输入特征图形状: {x.shape}")
#
#     # 前向传播
#     output = hfp_module(x)
#     print(f"输出特征图形状: {output.shape}")
#
#     # 验证输出形状是否与输入一致（确保特征图尺寸不变）
#     assert output.shape == x.shape, "输出形状与输入不一致，模块可能存在尺寸变化问题"
#     print("测试成功：HFP模块前向传播正常，输入输出形状一致")
#
#     # 测试非DCT模式（模拟p3/p4场景）
#     hfp_module_no_dct = HFP(
#         in_channels=in_channels,
#         # ratio=ratio,
#         # patch=patch,
#         # isdct=False
#     )
#     output_no_dct = hfp_module_no_dct(x)
#     print(f"非DCT模式输出形状: {output_no_dct.shape}")
#     assert output_no_dct.shape == x.shape, "非DCT模式下输出形状异常"
#     print("非DCT模式测试成功")