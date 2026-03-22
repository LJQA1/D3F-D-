import torch
import torch.nn as nn

__all__ = (
    "MFI",

)

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class Squeeze(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Squeeze, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.squeeze(x)
        return out

class MFI(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(MFI, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels

        # 初步卷积：降维到 mid_channels
        self.conv_down = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.bn_down = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)

        # Squeeze 层
        self.squeeze = Squeeze(mid_channels, mid_channels // 4)

        # Bottleneck 层
        group_channels = mid_channels // 4
        self.bottlenecks = nn.ModuleList([
            Bottleneck(group_channels, group_channels),
            Bottleneck(group_channels, group_channels)
        ])

    def forward(self, x):
        # 初步卷积：in_channels -> mid_channels (如 128 -> 64)
        down_out = self.conv_down(x)
        down_out = self.bn_down(down_out)
        down_out = self.relu(down_out)

        b, c, h, w = down_out.size()
        group_size = c // 4
        groups = torch.split(down_out, group_size, dim=1)

        # 分支一：初步卷积输出经过 Squeeze 处理
        branch1 = self.squeeze(down_out)

        # 分支二：Group1 + Group2
        branch2 = groups[0] + groups[1]

        # 分支三：Group3 -> Bottleneck
        branch3 = self.bottlenecks[0](groups[2])

        # 分支四：(Group3 输出 + Group4) -> Bottleneck
        branch4 = self.bottlenecks[1](branch3 + groups[3])

        # 拼接所有分支
        combined = torch.cat([branch1, branch2, branch3, branch4], dim=1)

        # 添加残差连接：将初步卷积的输出加回到拼接结果中
        residual = down_out.repeat(1, 2, 1, 1) if combined.shape[1] == self.mid_channels * 2 else down_out
        final_out = combined + residual

        # 确保输出通道数正确
        assert final_out.shape[1] == self.mid_channels, f"输出通道数错误: {final_out.shape[1]} != {self.mid_channels}"

        return final_out

# 示例用法
if __name__ == "__main__":
    mfi = MFI(in_channels=128, mid_channels=64)
    input_tensor = torch.randn(1, 128, 64, 64)
    output = mfi(input_tensor)
    print("输入尺寸:", input_tensor.shape)
    print("输出尺寸:", output.shape)