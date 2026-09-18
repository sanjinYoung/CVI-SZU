import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from typing import Sequence, Optional, Tuple
from collections.abc import Sequence


class GraphReasoning3D(nn.Module):
    """3D图关系推理模块 - 64 空间锚点 + 可学习邻接矩阵 message passing"""

    def __init__(self, channels: int, graph_nodes: int = 64, reduction_ratio: int = 4):
        super().__init__()
        self.channels = channels
        self.graph_nodes = graph_nodes

        # 关键点提取
        self.keypoint_net = nn.Sequential(
            nn.Conv3d(channels, channels // reduction_ratio, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((4, 4, 4))  # 固定64个节点 (4x4x4)
        )

        # 可学习邻接矩阵（graph_nodes × graph_nodes），用于节点间 message passing
        # 初始化为单位矩阵 + 小噪声：初始以自环为主，训练中逐步学到跨节点结构
        self.adj = nn.Parameter(torch.eye(graph_nodes) + 0.01 * torch.randn(graph_nodes, graph_nodes))

        # 图卷积层
        self.gc1 = nn.Linear(channels // reduction_ratio, channels // reduction_ratio)
        self.gc2 = nn.Linear(channels // reduction_ratio, channels)

        # 特征重建
        self.reconstruct = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape

        # 提取关键点特征
        nodes = self.keypoint_net(x)  # [B, C/r, 4, 4, 4]
        nodes = nodes.view(B, -1, self.graph_nodes).transpose(1, 2)  # [B, 64, C/r]

        # 节点间 message passing（可学习邻接矩阵，softmax 归一化后非负）
        alpha = F.softmax(self.adj, dim=-1)                 # [64, 64]
        nodes = torch.einsum('ij,bjd->bid', alpha, nodes)   # [B, 64, C/r]

        # 图卷积操作
        gc_out = F.relu(self.gc1(nodes))
        gc_out = self.gc2(gc_out)  # [B, 64, C]

        # 重建特征图
        gc_out = gc_out.transpose(1, 2).view(B, C, 4, 4, 4)
        gc_out = F.interpolate(gc_out, size=(D, H, W), mode='trilinear', align_corners=False)

        # 残差连接
        output = x + self.reconstruct(gc_out)
        return output


class EfficientNonLocal3D(nn.Module):
    """高效3D非局部注意力 - 简化版"""

    def __init__(self, channels: int, reduction_ratio: int = 8):
        super().__init__()
        self.channels = channels
        self.reduction_channels = channels // reduction_ratio

        # 投影层
        self.query = nn.Conv3d(channels, self.reduction_channels, 1)
        self.key = nn.Conv3d(channels, self.reduction_channels, 1)
        self.value = nn.Conv3d(channels, self.reduction_channels, 1)

        # 输出层
        self.output = nn.Conv3d(self.reduction_channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape

        # 投影
        q = self.query(x).view(B, -1, D * H * W).permute(0, 2, 1)  # [B, N, C/r]
        k = self.key(x).view(B, -1, D * H * W)  # [B, C/r, N]
        v = self.value(x).view(B, -1, D * H * W).permute(0, 2, 1)  # [B, N, C/r]

        # 注意力计算
        attention = torch.bmm(q, k) / (self.reduction_channels ** 0.5)
        attention = F.softmax(attention, dim=-1)  # [B, N, N]

        # 应用注意力
        out = torch.bmm(attention, v).permute(0, 2, 1)  # [B, C/r, N]
        out = out.view(B, -1, D, H, W)

        # 输出投影
        out = self.output(out)

        # 残差连接
        return x + self.gamma * out


class EnhancedDilatedConvHead(nn.Module):
    """增强版3D空洞卷积头 - 修复版"""

    def __init__(
            self,
            spatial_dims: int,
            in_chns: int,
            out_chns: int,
            act: str = "RELU",
            norm: str = "instance",
            bias: bool = True,
            dropout: float = 0.0,
            dilation_rates: tuple = (1, 2, 4),  # 改为tuple类型
            fusion_type: str = "graph",  # "graph", "nonlocal"
    ):
        super().__init__()

        num_rates = len(dilation_rates)
        # assert out_chns % num_rates == 0
        branch_out_chns = out_chns // num_rates

        self.num_branches = num_rates
        self.fusion_type = fusion_type

        # 3D空洞卷积分支
        self.branches = nn.ModuleList()
        for dilation in dilation_rates:
            conv = Convolution(
                spatial_dims,
                in_chns,
                branch_out_chns,
                kernel_size=3,
                dilation=dilation,
                padding=dilation,
                act=act,
                norm=norm,
                dropout=dropout,
                bias=bias
            )
            self.branches.append(conv)

        # 增强融合模块
        if fusion_type == "graph":
            self.enhanced_fusion = GraphReasoning3D(channels=branch_out_chns)
        else:
            self.enhanced_fusion = EfficientNonLocal3D(channels=branch_out_chns)

        # 最终融合卷积
        self.fusion_conv = Convolution(
            spatial_dims,
            branch_out_chns * num_rates,
            out_chns,
            kernel_size=1,
            act=act,
            norm=norm,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outputs = []

        # 各分支特征提取
        for branch in self.branches:
            branch_outputs.append(branch(x))

        # 应用增强融合
        enhanced_outputs = []
        for feature in branch_outputs:
            enhanced = self.enhanced_fusion(feature)
            enhanced_outputs.append(enhanced)

        # 最终融合
        fused_output = torch.cat(enhanced_outputs, dim=1)
        output = self.fusion_conv(fused_output)

        return output


class TwoConv(nn.Sequential):
    """two convolutions."""

    def __init__(
            self,
            spatial_dims: int,
            in_chns: int,
            out_chns: int,
            act: str = "RELU",
            norm: str = "instance",
            bias: bool = True,
            dropout: float = 0.0,
    ):
        super().__init__()

        conv_0 = Convolution(
            spatial_dims,
            in_chns,
            out_chns,
            kernel_size=3,
            act=act,
            norm=norm,
            bias=bias,
            dropout=dropout,
        )
        conv_1 = Convolution(
            spatial_dims,
            out_chns,
            out_chns,
            kernel_size=3,
            act=act,
            norm=norm,
            bias=bias,
            dropout=dropout,
        )
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)


class Down(nn.Sequential):
    """maxpooling downsampling and two convolutions."""

    def __init__(
            self,
            spatial_dims: int,
            in_chns: int,
            out_chns: int,
            act: str = "RELU",
            norm: str = "instance",
            bias: bool = True,
            dropout: float = 0.0,
    ):
        super().__init__()
        max_pooling = Pool["MAX", spatial_dims](kernel_size=2)
        convs = TwoConv(spatial_dims, in_chns, out_chns, act, norm, bias, dropout)
        self.add_module("max_pooling", max_pooling)
        self.add_module("convs", convs)


class UpCat(nn.Module):
    """upsampling, concatenation with the encoder feature map, two convolutions"""

    def __init__(
            self,
            spatial_dims: int,
            in_chns: int,
            cat_chns: int,
            out_chns: int,
            act: str = "RELU",
            norm: str = "instance",
            bias: bool = True,
            dropout: float = 0.0,
            upsample: str = "deconv",
            halves: bool = True,
    ):
        super().__init__()

        up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(spatial_dims, in_chns, up_chns, 2, mode=upsample)
        self.convs = TwoConv(spatial_dims, cat_chns + up_chns, out_chns, act, norm, bias, dropout)

    def forward(self, x: torch.Tensor, x_e: torch.Tensor) -> torch.Tensor:
        x_0 = self.upsample(x)
        x_cat = torch.cat((x_0, x_e), dim=1)
        return self.convs(x_cat)

class RegressionHead(nn.Module):
    """Regression head for parameter prediction"""

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            hidden_channels: int = 256,
            output_size: int = 2,
    ):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool3d(1) if spatial_dims == 3 else nn.AdaptiveAvgPool2d(1)
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels // 2, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.global_pool(x)
        return self.fc_layers(x)

class EnhancedBasicUNet(nn.Module):
    """增强版BasicUNet"""

    def __init__(
            self,
            spatial_dims: int = 3,
            in_channels: int = 1,
            out_channels: int = 2,
            features: tuple = (32, 32, 64, 128, 256, 32),  # 改为tuple类型
            act: str = "RELU",
            norm: str = "instance",
            bias: bool = True,
            dropout: float = 0.0,
            upsample: str = "deconv",
            use_enhanced_head: bool = True,
            fusion_type: str = "graph",
            dilation_rates: tuple = (1, 2, 4),  # 添加dilation_rates参数
    ):
        super().__init__()
        fea = features

        # 使用增强版卷积头或普通卷积
        if use_enhanced_head:
            self.conv_0 = EnhancedDilatedConvHead(
                spatial_dims=spatial_dims,
                in_chns=in_channels,
                out_chns=fea[0],
                act=act,
                norm=norm,
                bias=bias,
                dropout=dropout,
                dilation_rates=dilation_rates,
                fusion_type=fusion_type
            )
        else:
            self.conv_0 = TwoConv(spatial_dims, in_channels, fea[0], act, norm, bias, dropout)

        # 下采样路径
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)

        # 上采样路径
        self.upcat_4 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)
        self.upcat_3 = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)
        self.upcat_2 = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)
        self.upcat_1 = UpCat(spatial_dims, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample, halves=False)

        # 最终卷积
        self.final_conv = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)

        self.regression_head = RegressionHead(
                spatial_dims=spatial_dims,
                in_channels=fea[4],
                hidden_channels=256,
                output_size=2
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.conv_0(x)
        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)

        regression_params = None
        if self.regression_head:
            regression_params = self.regression_head(x4)

        u4 = self.upcat_4(x4, x3)
        u3 = self.upcat_3(u4, x2)
        u2 = self.upcat_2(u3, x1)
        u1 = self.upcat_1(u2, x0)

        # return self.final_conv(u1)
        return self.final_conv(u1), regression_params


# 测试代码
if __name__ == "__main__":
    # 测试输入 [batch, channel, depth, height, width]
    input_3d = torch.randn(1, 1, 128, 128, 128)

    # 创建模型
    model = EnhancedBasicUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        features=(32, 32, 64, 128, 256, 32),
        use_enhanced_head=True,
        fusion_type="graph",  # 可以改为 "nonlocal"
        dilation_rates=(1, 2, 4)
    )

    # 前向传播
    output = model(input_3d)
    print(f"输入形状: {input_3d.shape}")
    print(f"输出形状: {output.shape}")

    # 测试参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数: {total_params:,}")