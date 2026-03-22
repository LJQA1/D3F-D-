import torch
import torch.nn as nn

# https://arxiv.org/pdf/2502.07259


class CSA_ConvBlock(nn.Module):
    """
    Channel Self-Attention (CSA) ConvBlock: 应用通道间的自注意力机制来增强特征表示

    该模块通过计算不同通道间的相关性，让网络能够自动关注重要的特征通道，
    从而提升特征提取能力。适用于各种计算机视觉任务中的特征提取阶段。

    属性:
    -----------
    c : int
        模块的输入和输出通道数
    fq, fk, fv : nn.Conv2d
        用于生成查询(Query)、键(Key)和值(Value)的卷积层
    bn : nn.BatchNorm2d
        批归一化层，用于稳定训练
    relu : nn.ReLU
        激活函数，引入非线性

    方法:
    --------
    forward(inputs):
        前向传播函数，实现通道自注意力计算和特征增强
    """

    def __init__(self, c):
        super().__init__()
        self.c = c  # 通道数

        # 定义生成查询、键和值的卷积层
        # 使用3x3卷积保持空间信息，padding=1确保输出尺寸与输入一致
        self.fq = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False)  # 查询卷积层
        self.fk = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False)  # 键卷积层
        self.fv = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False)  # 值卷积层

        self.bn = nn.BatchNorm2d(c)  # 批归一化层
        self.relu = nn.ReLU()  # ReLU激活函数

    def forward(self, inputs):
        """
        前向传播函数

        参数:
        inputs : torch.Tensor
            输入特征图，形状为 [batch_size, channels, height, width]

        返回:
        torch.Tensor
            经过通道自注意力增强后的特征图，形状与输入相同
        """
        # 生成查询、键和值
        fq = self.fq(inputs)  # 查询特征 [B, C, H, W]
        fk = self.fk(inputs)  # 键特征 [B, C, H, W]
        fv = self.fv(inputs)  # 值特征 [B, C, H, W]

        # 获取空间维度
        h, w = inputs.size(2), inputs.size(3)  # 高度和宽度

        # 调整查询和键的维度以进行矩阵乘法
        fq = fq.unsqueeze(2)  # [B, C, 1, H, W]
        # 调整键的维度并转置最后两个维度以便计算相似度
        fk = fk.unsqueeze(1).permute(0, 1, 2, 4, 3)  # [B, 1, C, W, H]

        # 计算通道间的相似度，除以维度的平方根进行归一化
        f_sim_tensor = torch.matmul(fq, fk) / (fq.size(-1) ** 0.5)  # [B, C, C, H, H]

        # 聚合空间信息
        f_sum_tensor = torch.sum(f_sim_tensor, dim=2)  # [B, C, H, H]
        # 计算每个通道的注意力分数
        f_scores = torch.sum(f_sum_tensor, dim=(-2, -1)) / (h ** 2)  # [B, C]

        # 应用softmax获取归一化的注意力权重，并调整维度以便与值相乘
        scores = torch.softmax(f_scores, dim=1).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]

        # 应用注意力权重到值特征上，并与原始输入残差连接
        r = (scores * fv) + inputs  # [B, C, H, W]

        # 批归一化和激活
        r = self.bn(r)
        r = self.relu(r)

        return r





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
        x = x.to(self.conv.weight.device)  # 确保输入张量在卷积层所在设备
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        x = x.to(self.conv.weight.device)  # 确保输入张量在卷积层所在设备
        return self.act(self.conv(x))


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

class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

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

class Bottleneck_CSA_ConvBlock(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = CSA_ConvBlock(c_)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_CSA_ConvBlock(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

# 在c3k=True时，使用Bottleneck_HSMSSD特征融合，为false的时候我们使用普通的Bottleneck提取特征
class C3k2_CSA_ConvBlock(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )

class ABlock_CSA_ConvBlock(nn.Module):
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

        self.attn = CSA_ConvBlock(dim)
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



class A2C2f_CSA_ConvBlock(nn.Module):
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
            nn.Sequential(*(ABlock_CSA_ConvBlock(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
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
    """
    主函数：测试CSA_ConvBlock的功能和输出形状
    """
    # 设置随机种子，确保结果可复现
    torch.manual_seed(42)

    # 创建测试输入：[batch_size=2, channels=32, height=64, width=64]
    # 模拟一个批次的2张图片，32个通道，64x64分辨率
    test_input = torch.randn(2, 32, 27, 31)

    # 初始化CSA卷积块，输入输出通道数均为32
    csa_block = CSA_ConvBlock(c=32)

    # 执行前向传播
    output = csa_block(test_input)

    # 打印输入输出形状，验证模块是否正确工作
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")

    # 验证输出形状是否与输入一致
    assert test_input.shape == output.shape, "输入输出形状不一致！"
    print("测试通过：输入输出形状一致")


if __name__ == "__main__":
    main()
