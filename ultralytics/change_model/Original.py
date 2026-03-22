import torch
import torch.nn as nn
import torch.nn.functional as F


class Original(nn.Module):
    def __init__(self, c1, c2, k=3):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(2)
        self.reduce = nn.Conv2d(4*c1, c2//2, 1)  # 先降到低通道

        # 分支一：原始信息（轻量）
        self.raw_branch = nn.Identity()

        # 分支二：细节增强
        self.detail_branch = nn.Sequential(
            nn.Conv2d(c2//2, c2//2, k, padding=k//2, groups=c2//2),
            nn.BatchNorm2d(c2//2),
            nn.SiLU(),
            nn.Conv2d(c2//2, c2//2, 1)
        )

        self.fuse = nn.Sequential(
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

    def forward(self, x):
        x = self.unshuffle(x)        # (B, 4*c1, H/2, W/2)
        x = self.reduce(x)           # (B, c2//2, H/2, W/2)

        raw = x
        detail = self.detail_branch(x)
        out = torch.cat([raw, detail], dim=1)
        return self.fuse(out)



class Original2(nn.Module):
    def __init__(self, c1, c2, k=3):
        super().__init__()
        self.pixel_split = nn.PixelUnshuffle(2)
        # 原始分支：保留1/2通道
        self.raw_branch = nn.Conv2d(4*c1, c2//2, kernel_size=1, bias=False)
        # 增强抽象分支：增加1x1卷积和残差连接，强化语义特征
        self.abstract_branch = nn.Sequential(
            nn.Conv2d(4*c1, 4*c1, kernel_size=k, padding=k//2, groups=4*c1, bias=False),
            nn.BatchNorm2d(4*c1),
            nn.SiLU(),
            nn.Conv2d(4*c1, c2//2, kernel_size=1, bias=False),
            nn.BatchNorm2d(c2//2)
        )
        # 残差连接：直接传递部分原始特征到抽象分支
        self.residual = nn.Conv2d(4*c1, c2//2, kernel_size=1, bias=False)
        self.fuse = nn.Sequential(
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

    def forward(self, x):
        x_split = self.pixel_split(x)
        raw = self.raw_branch(x_split)
        # 抽象分支增加残差，强化语义的同时保留原始细节
        abstract = self.abstract_branch(x_split) + self.residual(x_split)
        out = torch.cat([raw, abstract], dim=1)
        return self.fuse(out)