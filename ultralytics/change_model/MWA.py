import torch
import torch.nn as nn
import torch.nn.functional as F


class MWA_Module(nn.Module):
    """
    多尺度加权聚合模块（MWA）：基于论文《Remote Sensing Small Object Detection Based on Multicontextual Information Aggregation》
    输入：特征图 (batch_size, in_channels, height, width)
    输出：特征图 (batch_size, out_channels, height, width)，要求 in_channels == out_channels（论文规定输入输出通道数相等）
    """

    def __init__(self , in_channels , out_channels , dilation_rate=3 , beta_init=0.5):
        super(MWA_Module , self).__init__()
        assert in_channels == out_channels , "MWA模块要求输入输出通道数相等（参考论文{insert\_element\_8\_}、{insert\_element\_9\_}）"
        self.dilation_rate = dilation_rate  # 空洞率，论文固定为3（{insert\_element\_10\_}）

        # 1. 1×1卷积：统一输入通道（若需调整，此处可适配，但论文要求输入输出通道一致）
        self.conv1x1_in = nn.Conv2d(in_channels , out_channels , kernel_size=1 , stride=1 , padding=0)
        self.bn1_in = nn.BatchNorm2d(out_channels)  # 批量归一化，提升训练稳定性

        # 2. 多尺度空洞卷积分支（3条分支，对应论文中绿色、黄色、红色分支{insert\_element\_11\_}）
        # 分支1：3×1空洞卷积 + 空间注意力
        self.conv3x1_dil = nn.Conv2d(out_channels , out_channels , kernel_size=(3 , 1) , stride=1 ,
                                     padding=(dilation_rate , 0) , dilation=dilation_rate)
        self.bn3x1 = nn.BatchNorm2d(out_channels)
        # 分支2：1×3空洞卷积 + 空间注意力
        self.conv1x3_dil = nn.Conv2d(out_channels , out_channels , kernel_size=(1 , 3) , stride=1 ,
                                     padding=(0 , dilation_rate) , dilation=dilation_rate)
        self.bn1x3 = nn.BatchNorm2d(out_channels)
        # 分支3：3×3空洞卷积（无注意力，论文中独立分支{insert\_element\_12\_}）
        self.conv3x3_dil = nn.Conv2d(out_channels , out_channels , kernel_size=3 , stride=1 ,
                                     padding=dilation_rate , dilation=dilation_rate)
        self.bn3x3 = nn.BatchNorm2d(out_channels)

        # 3. 空间注意力模块（论文中“含注意力机制的空洞卷积分支”实现，参考通道注意力+空间注意力常规设计）
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(out_channels * 2 , 1 , kernel_size=3 , stride=1 , padding=1) ,  # 拼接max和avg池化结果
            nn.BatchNorm2d(1) ,
            nn.Sigmoid()  # 注意力权重归一化
        )

        # 4. 拼接后通道调整：3条分支拼接后通道数为3*out_channels，用1×1卷积压缩回out_channels
        self.conv1x1_cat = nn.Conv2d(out_channels * 3 , out_channels , kernel_size=1 , stride=1 , padding=0)
        self.bn_cat = nn.BatchNorm2d(out_channels)

        # 5. 可学习加权参数β（初始值0.5，论文{insert\_element\_13\_}、{insert\_element\_14\_}）
        self.beta = nn.Parameter(torch.tensor(beta_init , dtype=torch.float32) , requires_grad=True)

        # 激活函数（论文未明确，采用ReLU为CNN常用选择）
        self.relu = nn.ReLU(inplace=True)

    def spatial_attention(self , x):
        """空间注意力计算：对输入特征图计算空间权重，增强目标区域、抑制背景（参考论文“注意力机制”描述{insert\_element\_15\_}）"""
        # 全局池化：max池化 + avg池化
        max_pool = F.adaptive_max_pool2d(x , output_size=1)
        avg_pool = F.adaptive_avg_pool2d(x , output_size=1)
        # 拼接后计算注意力权重
        attn = torch.cat([max_pool , avg_pool] , dim=1)  # (batch, 2*out_channels, 1, 1)
        attn = self.spatial_attn(attn)  # (batch, 1, 1, 1)：空间注意力权重
        return x * attn  # 注意力加权

    def forward(self , x):
        # 步骤1：输入通道调整与归一化
        x_in = self.conv1x1_in(x)
        x_in = self.bn1_in(x_in)
        x_in = self.relu(x_in)

        # 步骤2：多尺度分支特征提取
        # 分支1：3×1空洞卷积 + 空间注意力
        branch1 = self.conv3x1_dil(x_in)
        branch1 = self.bn3x1(branch1)
        branch1 = self.relu(branch1)
        branch1 = self.spatial_attention(branch1)

        # 分支2：1×3空洞卷积 + 空间注意力
        branch2 = self.conv1x3_dil(x_in)
        branch2 = self.bn1x3(branch2)
        branch2 = self.relu(branch2)
        branch2 = self.spatial_attention(branch2)

        # 分支3：3×3空洞卷积（无注意力）
        branch3 = self.conv3x3_dil(x_in)
        branch3 = self.bn3x3(branch3)
        branch3 = self.relu(branch3)

        # 步骤3：特征拼接与通道调整
        cat_feat = torch.cat([branch1 , branch2 , branch3] , dim=1)  # (batch, 3*out_channels, H, W)
        cat_feat = self.conv1x1_cat(cat_feat)
        cat_feat = self.bn_cat(cat_feat)
        cat_feat = self.relu(cat_feat)

        # 步骤4：加权融合（类似残差连接，论文{insert\_element\_16\_}）
        out = self.beta * cat_feat + (1 - self.beta) * x_in
        return out


# ------------------- 模块测试代码 -------------------
if __name__ == "__main__":
    # 模拟检测头前的特征图输入（论文中MWA输入尺寸示例：160×160，通道数假设为128{insert\_element\_17\_}）
    batch_size = 2
    in_channels = 128
    height , width = 160 , 160
    x = torch.randn(batch_size , in_channels , height , width)  # 随机生成输入特征

    # 初始化MWA模块（输入输出通道数相等）
    mwa = MWA_Module(in_channels=in_channels , out_channels=in_channels)
    # 前向传播
    output = mwa(x)

    # 验证输出维度（应与输入一致）
    print(f"输入维度: {x.shape}")  # 输出：torch.Size([2, 128, 160, 160])
    print(f"输出维度: {output.shape}")  # 输出：torch.Size([2, 128, 160, 160])
    print(f"可学习参数β的初始值: {mwa.beta.item():.4f}")  # 输出：0.5000（符合论文初始值要求）