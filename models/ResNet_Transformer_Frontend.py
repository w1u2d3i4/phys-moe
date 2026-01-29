"""
ResNet + Transformer 混合前端
ResNet 将 8500 长度缩减到约 100，再用 4 层 Transformer 处理，最终输出 1024 维特征
输入: (B, 1, 8500)
输出: (B, 1024)
"""

import torch
import torch.nn as nn
from .CrystalFusionNet import ResBlock1D


class ResNet_Transformer_Frontend(nn.Module):
    """
    ResNet 压缩序列 + 4 层 Transformer
    - ResNet 部分：从 (B,1,8500) 到 (B, 256, 132)，长度约 100
    - Transformer：4 层，输入 (B, 132, 256)，输出 (B, 132, embed_dim)
    - 池化 + 线性：(B, embed_dim) -> (B, 1024)
    """
    def __init__(self,
                 resnet_out_len=132,   # ResNet 输出序列长度（约 8500/2^6）
                 resnet_out_ch=256,
                 embed_dim=256,
                 num_heads=8,
                 num_layers=4,
                 mlp_ratio=4.0,
                 dropout=0.1,
                 output_dim=1024):
        super().__init__()
        self.resnet_out_len = resnet_out_len
        self.resnet_out_ch = resnet_out_ch
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        # ResNet 部分：8500 -> ~132，通道到 256
        self.intensity_norm = nn.BatchNorm1d(1)
        self.resnet_backbone = nn.Sequential(
            ResBlock1D(1, 32),
            nn.MaxPool1d(2, 2),
            ResBlock1D(32, 32),
            nn.MaxPool1d(2, 2),
            ResBlock1D(32, 64),
            nn.MaxPool1d(2, 2),
            ResBlock1D(64, 128),
            nn.MaxPool1d(2, 2),
            ResBlock1D(128, 128),
            nn.AvgPool1d(2, 2, 1),
            ResBlock1D(128, 256),
            nn.AvgPool1d(2, 2),
        )
        # resnet_backbone: (B,1,8500) -> (B,256,L), L≈132

        # 若通道与 embed 不一致，加投影
        self.ch_proj = nn.Linear(resnet_out_ch, embed_dim) if resnet_out_ch != embed_dim else nn.Identity()

        # 位置编码（多留一点余量，实际 L 可能因池化略有偏差，如 132/133）
        self._max_seq_len = max(resnet_out_len + 10, 150)
        self.pos_embed = nn.Parameter(torch.randn(1, self._max_seq_len, embed_dim) * 0.02)

        # 4 层 Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 序列 -> 1024 维
        self.fc_out = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        """
        x: (B, 1, 8500) 或 (B, 8500)
        return: (B, 1024)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, 8500)
        x = x.float()
        B = x.size(0)

        # ResNet: (B,1,8500) -> (B,256,L)
        x = self.intensity_norm(x)
        x = self.resnet_backbone(x)  # (B, 256, L)

        # (B, C, L) -> (B, L, C)
        x = x.transpose(1, 2)  # (B, L, 256)
        L = x.size(1)

        x = self.ch_proj(x)  # (B, L, embed_dim)
        x = x + self.pos_embed[:, :L, :].to(x.device)

        # Transformer: (B, L, embed_dim)
        x = self.transformer(x)  # (B, L, embed_dim)

        # 沿序列维度均值池化 -> (B, embed_dim)
        x = x.mean(dim=1)
        out = self.fc_out(x)  # (B, 1024)
        return out
