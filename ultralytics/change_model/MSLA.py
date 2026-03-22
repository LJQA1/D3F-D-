import torch
import torch.nn as nn
import torch.nn.functional as F


# https://arxiv.org/pdf/2505.18823


class LinearAttention(nn.Module):

    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.view(b, c, h * w).permute(0, 2, 1)  # (b, h*w, c)

        qkv = self.qkv(x).reshape(b, h * w, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        key = F.softmax(k, dim=-1)
        query = F.softmax(q, dim=-2)
        context = key.transpose(-2, -1) @ v
        x = (query @ context).reshape(b, h * w, c)

        x = self.proj(x)

        x = x.permute(0, 2, 1).view(b, c, h, w)

        return x


class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(DepthwiseConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels,
                                   padding=kernel_size // 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.depthwise(x)
        x = x + residual
        x = self.relu(x)
        return x


class MSLA(nn.Module):

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.dw_conv_3x3 = DepthwiseConv(dim // 4, kernel_size=3)
        self.dw_conv_5x5 = DepthwiseConv(dim // 4, kernel_size=5)
        self.dw_conv_7x7 = DepthwiseConv(dim // 4, kernel_size=7)
        self.dw_conv_9x9 = DepthwiseConv(dim // 4, kernel_size=9)

        self.linear_attention = LinearAttention(dim=dim // 4, num_heads=num_heads)

        self.final_conv = nn.Conv2d(dim, dim, 1)

        self.scale_weights = nn.Parameter(torch.ones(4), requires_grad=True)

    def forward(self, x):
        # 输入形状: (batch_size, channels, height, width)
        b, c, h, w = x.shape

        # 分割通道
        split_size = c // 4
        x_3x3 = x[:, :split_size, :, :]
        x_5x5 = x[:, split_size:2 * split_size, :, :]
        x_7x7 = x[:, 2 * split_size:3 * split_size:, :, :]
        x_9x9 = x[:, 3 * split_size:, :, :]

        # 深度卷积处理
        x_3x3 = self.dw_conv_3x3(x_3x3)
        x_5x5 = self.dw_conv_5x5(x_5x5)
        x_7x7 = self.dw_conv_7x7(x_7x7)
        x_9x9 = self.dw_conv_9x9(x_9x9)

        # 线性注意力处理
        att_3x3 = self.linear_attention(x_3x3)
        att_5x5 = self.linear_attention(x_5x5)
        att_7x7 = self.linear_attention(x_7x7)
        att_9x9 = self.linear_attention(x_9x9)

        # 合并并应用缩放权重
        processed_input = torch.cat([
            att_3x3 * self.scale_weights[0],
            att_5x5 * self.scale_weights[1],
            att_7x7 * self.scale_weights[2],
            att_9x9 * self.scale_weights[3]
        ], dim=1)

        # 最终卷积
        final_output = self.final_conv(processed_input)

        # 输出形状: (batch_size, channels, height, width)
        return final_output



def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization."""
        return self.act(self.conv(x))


class ABlock_MSLA(nn.Module):
    """
    Area-attention block module for efficient feature extraction in YOLO models.

    This module implements an area-attention mechanism combined with a feed-forward network for processing feature maps.
    It uses a novel area-based attention approach that is more efficient than traditional self-attention while
    maintaining effectiveness.

    Attributes:
        attn (AAttn): Area-attention module for processing spatial features.
        mlp (nn.Sequential): Multi-layer perceptron for feature transformation.

    Methods:
        _init_weights: Initializes module weights using truncated normal distribution.
        forward: Applies area-attention and feed-forward processing to input tensor.

    Examples:
        >>> block = ABlock(dim=256, num_heads=8, mlp_ratio=1.2, area=1)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        """
        Initializes an Area-attention block module for efficient feature extraction in YOLO models.

        This module implements an area-attention mechanism combined with a feed-forward network for processing feature
        maps. It uses a novel area-based attention approach that is more efficient than traditional self-attention
        while maintaining effectiveness.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            area (int): Number of areas the feature map is divided.
        """
        super().__init__()

        self.attn = MSLA(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights using a truncated normal distribution."""
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass through ABlock, applying area-attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x)
        return x + self.mlp(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

class A2C2f_MSLA(nn.Module):
    """
    Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.

    This module extends the C2f architecture by incorporating area-attention and ABlock layers for improved feature
    processing. It supports both area-attention and standard convolution modes.

    Attributes:
        cv1 (Conv): Initial 1x1 convolution layer that reduces input channels to hidden channels.
        cv2 (Conv): Final 1x1 convolution layer that processes concatenated features.
        gamma (nn.Parameter | None): Learnable parameter for residual scaling when using area attention.
        m (nn.ModuleList): List of either ABlock or C3k modules for feature processing.

    Methods:
        forward: Processes input through area-attention or standard convolution pathway.

    Examples:
        >>> m = A2C2f(512, 512, n=1, a2=True, area=1)
        >>> x = torch.randn(1, 512, 32, 32)
        >>> output = m(x)
        >>> print(output.shape)
        torch.Size([1, 512, 32, 32])
    """

    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        """
        Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of ABlock or C3k modules to stack.
            a2 (bool): Whether to use area attention blocks. If False, uses C3k blocks instead.
            area (int): Number of areas the feature map is divided.
            residual (bool): Whether to use residual connections with learnable gamma parameter.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            e (float): Channel expansion ratio for hidden channels.
            g (int): Number of groups for grouped convolutions.
            shortcut (bool): Whether to use shortcut connections in C3k blocks.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock_MSLA(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x):
        """Forward pass through R-ELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(-1, len(self.gamma), 1, 1) * y
        return y

def main():
    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)

    # 定义模型参数
    batch_size = 2
    height, width = 21, 17
    dim = 64  # 通道数
    num_heads = 4

    # 创建四维输入张量 (batch, channels, height, width)
    x = torch.randn(batch_size, dim, height, width)
    print(f"输入形状: {x.shape}")

    # 初始化模型
    msla = MSLA(dim=dim, num_heads=num_heads)

    # 前向传播
    with torch.no_grad():
        output = msla(x)

    # 打印输出形状
    print(f"输出形状: {output.shape}")

    # 打印缩放权重
    print(f"缩放权重: {msla.scale_weights}")


if __name__ == "__main__":
    main()