import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# DiffCLIP: Differential Attention Meets CLIP


class RMSNorm(nn.Module):
    r"""
    RMSNorm normalizes the input tensor by its root-mean-square (RMS) value.

    Given an input x ∈ ℝ^(...×d), it computes:

        RMS(x) = sqrt(mean(x², dim=-1, keepdim=True) + ε)
        output = x / RMS(x)

    Optionally, a learnable weight is applied if elementwise_affine is True.

    Args:
        dim (int): Dimension to normalize.
        eps (float): A value added for numerical stability.
        elementwise_affine (bool): If True, multiply by a learnable weight.
    """
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.weight is not None}'

# ---------------------------------------------------------
# Differential Attention Module
# ---------------------------------------------------------
class DiffAttention(nn.Module):
    r"""
    Differential Attention Module.

    Given an input tensor X ∈ ℝ^(B×N×d_model), we first compute the linear projections:

        Q = X Wᵠ,   K = X Wᵏ,   V = X Wᵛ

    The queries and keys are then reshaped and split into two parts:
        Q → [Q₁; Q₂] ∈ ℝ^(B, N, 2·h_effective, d_head)
        K → [K₁; K₂] ∈ ℝ^(B, N, 2·h_effective, d_head)
    with h_effective = num_heads // 2 and d_head = d_model / num_heads.

    The value projection is reshaped to:
        V ∈ ℝ^(B, N, h_effective, 2·d_head)

    We then compute two attention maps:
        A₁ = softmax((Q₁ K₁ᵀ) / √d_head)
        A₂ = softmax((Q₂ K₂ᵀ) / √d_head)

    A learnable scalar λ is computed via:
        λ = exp(λ_{q1} ⋅ λ_{k1}) − exp(λ_{q2} ⋅ λ_{k2}) + λ_init

    Finally, the differential attention output is:
        DiffAttn(X) = (A₁ − λ · A₂) · V

    The per-head outputs are then normalized headwise with RMSNorm and projected back to d_model.

    Args:
        dim (int): Embedding dimension (d_model).
        num_heads (int): Number of heads in the original transformer (must be even).
        qkv_bias (bool): If True, add a bias term to the Q, K, V projections.
        attn_drop (float): Dropout probability after softmax.
        proj_drop (float): Dropout probability after the output projection.
        lambda_init (float): Initial constant for lambda re-parameterization.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0., lambda_init=0.8):
        super().__init__()
        if num_heads % 2 != 0:
            raise ValueError("num_heads must be even for Differential Attention.")
        self.dim = dim
        self.num_heads = num_heads           # original number of heads
        self.effective_heads = num_heads // 2  # differential attention operates on half as many heads
        self.head_dim = dim // num_heads       # per-head dimension
        self.scaling = self.head_dim ** -0.5

        # Linear projections for Q, K, V: mapping from dim → dim.
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim, bias=True)  # final output projection

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # RMSNorm for headwise normalization on outputs (each head's output has dimension 2·head_dim)
        self.diff_norm = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

        # Learnable lambda parameters (shared across all heads)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_init = lambda_init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, N, d_model).

        Returns:
            Tensor of shape (B, N, d_model) after applying differential attention.
        """
        # 记录原始输入维度
        original_dim = x.dim()

        # 若为四维输入 (B, C, H, W)，转换为三维 (B, N, d_model)，其中 N=H*W, d_model=C
        if original_dim == 4:
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1)  # 转换为 (B, H, W, C)
            x = x.reshape(B, H * W, C)  # 三维形状: (B, N, d_model)
            N = H * W
            d_model = C
        else:
            # 确保三维输入形状正确
            B, N, d_model = x.shape
            assert d_model == self.dim, "输入维度与模型维度不匹配"

        # Compute Q, K, V projections.
        q = self.q_proj(x)  # shape: (B, N, d_model)
        k = self.k_proj(x)  # shape: (B, N, d_model)
        v = self.v_proj(x)  # shape: (B, N, d_model)

        # Reshape Q and K into (B, N, 2 * h_effective, head_dim)
        q = q.view(B, N, 2 * self.effective_heads, self.head_dim)
        k = k.view(B, N, 2 * self.effective_heads, self.head_dim)
        # Reshape V into (B, N, h_effective, 2 * head_dim)
        v = v.view(B, N, self.effective_heads, 2 * self.head_dim)

        # Transpose to bring head dimension forward.
        # q, k: (B, 2 * h_effective, N, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        # v: (B, h_effective, N, 2 * head_dim)
        v = v.transpose(1, 2)

        # Scale Q.
        q = q * self.scaling

        # Compute raw attention scores: (B, 2 * h_effective, N, N)
        attn_scores = torch.matmul(q, k.transpose(-1, -2))

        # Compute attention probabilities.
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_drop(attn_probs)

        # Reshape to separate the two halves: (B, h_effective, 2, N, N)
        attn_probs = attn_probs.view(B, self.effective_heads, 2, N, N)

        # Compute lambda via re-parameterization.
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1))
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2))
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        # Differential attention: subtract the second attention map scaled by lambda_full.
        diff_attn = attn_probs[:, :, 0, :, :] - lambda_full * attn_probs[:, :, 1, :, :]  # shape: (B, h_effective, N, N)

        # Multiply the differential attention weights with V.
        attn_output = torch.matmul(diff_attn, v)  # shape: (B, h_effective, N, 2 * head_dim)

        # Apply RMSNorm (headwise normalization) and scale by (1 - lambda_init)
        attn_output = self.diff_norm(attn_output) * (1 - self.lambda_init)

        # Concatenate heads: reshape from (B, h_effective, N, 2 * head_dim) → (B, N, 2 * h_effective * head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(B, N, 2 * self.effective_heads * self.head_dim)

        # Final linear projection.
        x_out = self.out_proj(attn_output)
        x_out = self.proj_drop(x_out)

        # 计算完成后，若原始输入为四维，将结果转回四维
        if original_dim == 4:
            x_out = x_out.reshape(B, H, W, d_model)  # 三维转四维中间形状
            x_out = x_out.permute(0, 3, 1, 2)  # 转回 (B, C, H, W)
        # return x_out
        return x_out


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


class ABlock_DiffAttention(nn.Module):
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

        self.attn = DiffAttention(dim)
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

class A2C2f_DiffAttention(nn.Module):
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
            nn.Sequential(*(ABlock_DiffAttention(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
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
    """模型功能测试函数"""
    # -------------------------- 配置参数 --------------------------
    batch_size = 2
    embed_dim = 512
    height, width = 17, 21  # 图像高度和宽度

    # -------------------------- 初始化模型 --------------------------
    model = DiffAttention(
        dim=embed_dim,
        num_heads=8,
        qkv_bias=True,
        attn_drop=0.1,
        proj_drop=0.1,
        lambda_init=0.5
    )
    print(f"模型参数总量: {sum(p.numel() for p in model.parameters())}")

    # -------------------------- 测试4D输入 --------------------------
    x_4d = torch.randn(batch_size, embed_dim, height, width)  # 4D输入: (B, C, H, W)
    print(f"\n4D输入张量形状: {x_4d.shape}")

    with torch.no_grad():
        output_4d = model(x_4d)
    print(f"4D输出张量形状: {output_4d.shape}")



if __name__ == "__main__":
    main()