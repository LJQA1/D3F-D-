import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile  # 计算参数量和FLOPs
import time  # 计时


# -------------------------- 原模块定义（未修改）--------------------------
class GRSA(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.qkv_bias = qkv_bias
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
        self.ESRPB_MLP = nn.Sequential(
            nn.Linear(2, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_heads, bias=False)
        )

        self.q1, self.q2 = nn.Linear(dim // 2, dim // 2, bias=True), nn.Linear(dim // 2, dim // 2, bias=True)
        self.k1, self.k2 = nn.Linear(dim // 2, dim // 2, bias=True), nn.Linear(dim // 2, dim // 2, bias=True)
        self.v1, self.v2 = nn.Linear(dim // 2, dim // 2, bias=True), nn.Linear(dim // 2, dim // 2, bias=True)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj1, self.proj2 = nn.Linear(dim // 2, dim // 2, bias=True), nn.Linear(dim // 2, dim // 2, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        b, c, h, w = x.shape
        window_size = (h, w)
        num_tokens = h * w

        x_3d = x.permute(0, 2, 3, 1).contiguous().view(b, num_tokens, c)
        x_3d = x_3d.reshape(b, num_tokens, 2, c // 2).permute(2, 0, 1, 3).contiguous()

        k = torch.stack((x_3d[0] + self.k1(x_3d[0]), x_3d[1] + self.k2(x_3d[1])), dim=0)
        k = k.permute(1, 2, 0, 3).flatten(2)
        k = k.reshape(b, num_tokens, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3).contiguous()

        q = torch.stack((x_3d[0] + self.q1(x_3d[0]), x_3d[1] + self.q2(x_3d[1])), dim=0)
        q = q.permute(1, 2, 0, 3).flatten(2)
        q = q.reshape(b, num_tokens, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3).contiguous()

        v = torch.stack((x_3d[0] + self.v1(x_3d[0]), x_3d[1] + self.v2(x_3d[1])), dim=0)
        v = v.permute(1, 2, 0, 3).flatten(2)
        v = v.reshape(b, num_tokens, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3).contiguous()

        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01, device=self.logit_scale.device))).exp()
        attn = attn * logit_scale

        relative_coords_h = torch.arange(-(window_size[0] - 1), window_size[0], dtype=torch.float32, device=x.device)
        relative_coords_w = torch.arange(-(window_size[1] - 1), window_size[1], dtype=torch.float32, device=x.device)
        relative_position_bias_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w], indexing='ij')).permute(1, 2, 0).contiguous().unsqueeze(0)
        relative_position_bias_table[:, :, :, 0] /= (window_size[0] - 1 + 1e-8)
        relative_position_bias_table[:, :, :, 1] /= (window_size[1] - 1 + 1e-8)
        relative_position_bias_table *= 3.2
        relative_position_bias_table = torch.sign(relative_position_bias_table) * (1 - torch.exp(-torch.abs(relative_position_bias_table)))

        coords_h = torch.arange(window_size[0], device=x.device)
        coords_w = torch.arange(window_size[1], device=x.device)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_position_index = relative_coords[:, :, 0] * (2 * window_size[1] - 1) + relative_coords[:, :, 1]

        relative_position_bias_table = self.ESRPB_MLP(relative_position_bias_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[relative_position_index.view(-1)].view(num_tokens, num_tokens, -1).permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, num_tokens, num_tokens) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, num_tokens, num_tokens)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x_3d = (attn @ v).transpose(1, 2).reshape(b, num_tokens, c)
        x_3d = x_3d.reshape(b, num_tokens, 2, c // 2).permute(2, 0, 1, 3).contiguous()
        x_3d = torch.stack((self.proj1(x_3d[0]), self.proj2(x_3d[1])), dim=0).permute(1, 2, 0, 3).reshape(b, num_tokens, c)
        x_3d = self.proj_drop(x_3d)

        x_4d = x_3d.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return x_4d


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class ABlock_GRSA(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        super().__init__()
        self.attn = GRSA(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=0.1,
            proj_drop=0.1
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x + self.attn(x)
        return x + self.mlp(x)


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class A2C2f_GRSA(nn.Module):
    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        super().__init__()
        c_ = int(c2 * e)
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock_GRSA(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(-1, len(self.gamma), 1, 1) * y
        return y


# -------------------------- 核心测试代码（含分析逻辑）--------------------------
if __name__ == "__main__":
    # 单轮测试参数（输入形状：[1,64,32,32]）
    batch_size = 1     # 批次大小设为1
    num_heads = 4      # 注意力头数（64//4=16，满足整除）
    dim = 64           # 通道维度设为64
    height, width = 32, 32  # 空间维度设为32x32
    warmup_steps = 1   # 仅1步热身（单轮测试）
    test_steps = 1     # 单轮运行测试

    # 1. 初始化设备、数据和模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(batch_size, dim, height, width).to(device)  # 输入形状[1,64,32,32]
    model = GRSA(
        dim=dim,
        num_heads=num_heads,
        qkv_bias=True,
        attn_drop=0.1,
        proj_drop=0.1
    ).to(device)

    # 2. 基础功能测试：输入输出形状一致性
    print("=" * 60)
    print("1. 基础功能测试（输入输出形状）")
    out = model(x)
    print(f"输入形状: {x.shape}")    # 预期：torch.Size([1, 64, 32, 32])
    print(f"输出形状: {out.shape}")  # 预期：torch.Size([1, 64, 32, 32])
    assert x.shape == out.shape, "输入输出形状不一致！"
    print("✅ 基础功能测试通过：输入输出形状一致")
    print("=" * 60)

    # 3. 模块层数统计
    print("\n2. 模块层数统计")
    print(f"GRSA 模块: 1层（单注意力层，含ESRPB位置偏置生成）")
    ablock = ABlock_GRSA(dim=dim, num_heads=num_heads).to(device)
    print(f"ABlock_GRSA 模块: 1层（1个GRSA + 1个MLP）")
    a2c2f = A2C2f_GRSA(c1=dim, c2=dim, n=1).to(device)
    print(f"A2C2f_GRSA 模块: 1个基础块（每个块含2个ABlock_GRSA）")
    print("=" * 60)

    # 4. 参数量与计算量（FLOPs）统计（以GRSA为例）
    print("\n3. 参数量与计算量统计（GRSA模块）")
    flops, params = profile(model, inputs=(x,), verbose=False)
    params_m = params / 1e6  # 转换为百万参数
    flops_g = flops / 1e9    # 转换为十亿次运算
    print(f"可训练参数量: {params_m:.4f} M")
    print(f"单批次计算量: {flops_g:.4f} G")
    print("=" * 60)

    # 5. CPU 运行时间（单轮）
    print("\n4. CPU 运行时间（单轮）")
    model_cpu = GRSA(dim=dim, num_heads=num_heads).cpu()
    x_cpu = torch.randn(batch_size, dim, height, width).cpu()
    # 热身
    for _ in range(warmup_steps):
        _ = model_cpu(x_cpu)
    # 单轮计时
    start_time = time.time()
    _ = model_cpu(x_cpu)
    end_time = time.time()
    cpu_time = (end_time - start_time) * 1000  # 转换为毫秒
    print(f"CPU 运行时间: {cpu_time:.4f} ms")
    print("=" * 60)

    # 6. GPU 运行时间（单轮，仅当GPU可用时）
    print("\n5. GPU 运行时间（单轮）")
    if torch.cuda.is_available():
        model_gpu = model
        x_gpu = x
        # 热身
        for _ in range(warmup_steps):
            _ = model_gpu(x_gpu)
        torch.cuda.synchronize()
        # 单轮计时
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        _ = model_gpu(x_gpu)
        end_event.record()
        torch.cuda.synchronize()
        gpu_time = start_event.elapsed_time(end_event)  # 毫秒
        print(f"GPU 设备: {torch.cuda.get_device_name(0)}")
        print(f"GPU 运行时间: {gpu_time:.4f} ms")
    else:
        print("❌ GPU不可用，跳过GPU时间统计")
    print("=" * 60)

    # 7. 扩展模块参数量统计
    print("\n6. 扩展模块参数量统计")
    # ABlock_GRSA 参数量
    ablock_flops, ablock_params = profile(ablock, inputs=(x,), verbose=False)
    print(f"ABlock_GRSA 可训练参数量: {ablock_params/1e6:.4f} M")
    # A2C2f_GRSA 参数量
    a2c2f_flops, a2c2f_params = profile(a2c2f, inputs=(x,), verbose=False)
    print(f"A2C2f_GRSA (n=1) 可训练参数量: {a2c2f_params/1e6:.4f} M")
    print("=" * 60)
