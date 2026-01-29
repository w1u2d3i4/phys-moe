import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class ResBlock1D(torch.nn.Module):
    def __init__(self,in_channel,out_channel) -> None:
        super(ResBlock1D,self).__init__()
        self.pre = torch.nn.Identity() if in_channel == out_channel else torch.nn.Conv1d(in_channel,out_channel,1,bias=False)
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(out_channel,out_channel,3,1,1,bias=False),
            torch.nn.BatchNorm1d(out_channel),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(out_channel,out_channel,3,1,1,bias=False),
            torch.nn.BatchNorm1d(out_channel),
        )
        self.relu = torch.nn.LeakyReLU()

    def forward(self,x):
        x = self.pre(x)
        out = self.conv(x)
        return self.relu(x+out)


class ResTcn(torch.nn.Module):
    def __init__(self,in_c=2,p_dropout=0.1):
        super(ResTcn,self).__init__()
        self.in_c = in_c

        self.intensity_norm = torch.nn.BatchNorm1d(1)

        self.TCN = torch.nn.Sequential(
            ResBlock1D(in_c,32),
            torch.nn.MaxPool1d(2,2),

            ResBlock1D(32,32),
            torch.nn.MaxPool1d(2,2),

            ResBlock1D(32,64),
            torch.nn.MaxPool1d(2,2),

            ResBlock1D(64,128),
            torch.nn.MaxPool1d(2,2),

            ResBlock1D(128,128),
            torch.nn.AvgPool1d(2,2,1),

            ResBlock1D(128,256),
            torch.nn.AvgPool1d(2,2),

            ResBlock1D(256,256),
            torch.nn.AvgPool1d(2,2,1),

            ResBlock1D(256,512),
            torch.nn.AvgPool1d(2,2,1),

            ResBlock1D(512,512),
            torch.nn.AvgPool1d(2,2),

            ResBlock1D(512,1024),
            torch.nn.AvgPool1d(2,2),
            torch.nn.Dropout(p_dropout),

            ResBlock1D(1024,1024),
            torch.nn.AvgPool1d(2,2),
            torch.nn.Dropout(p_dropout),

            ResBlock1D(1024,1024),
            torch.nn.AvgPool1d(2,2),
            torch.nn.Dropout(p_dropout),

            ResBlock1D(1024,1024),
            torch.nn.AvgPool1d(2,2),

            torch.nn.Flatten(),
            torch.nn.Linear(1024,32),    # conv_feat_len = 32

        )
        self.cls = torch.nn.Linear(32,230)

    def forward(self,intensity,angle):
        intensity = intensity.type(torch.float)
        angle = angle.type(torch.float)
        intensity = intensity.view(intensity.shape[0],1,-1)
        intensity = self.intensity_norm(intensity)
        angle = angle.view(angle.shape[0],1,-1)
        data = torch.concat([intensity,angle.deg2rad().sin()],dim=1)
        feat = self.TCN(data)
        return feat # batch_size * 32

class SpectralFusionUnit(torch.nn.Module):
    """Multi-scale feature fusion with residual connections"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = torch.nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else torch.nn.Identity()
        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv1d(out_channels, out_channels, 3, padding=1, bias=False),
            torch.nn.BatchNorm1d(out_channels),
            torch.nn.GELU(),
            torch.nn.Conv1d(out_channels, out_channels, 3, padding=1, bias=False),
            torch.nn.BatchNorm1d(out_channels)
        )
        self.activation = torch.nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return self.activation(x + self.conv_block(x))

class HybridAttentionEngine(torch.nn.Module):
    """Dual-path attention with positional enhancement"""
    def __init__(self, input_dim: int = 8500, embed_dim: int = 64, num_heads: int = 8,
                 dropout: float = 0.1, feedforward_dim: int = 512):
        super().__init__()
        self.position_embed = torch.nn.Parameter(torch.randn(1, 1062, embed_dim))
        self.intensity_proj = torch.nn.Sequential(
            torch.nn.Linear(1, embed_dim//2),
            torch.nn.GELU()
        )
        self.angle_proj = torch.nn.Sequential(
            torch.nn.Linear(1, embed_dim//2),
            torch.nn.GELU()
        )
        self.transformer_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = TransformerEncoder(self.transformer_layer, num_layers=4)

    def forward(self, intensity: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        batch_size = intensity.size(0)
        self.downsample = torch.nn.AvgPool1d(kernel_size=8, stride=8)
        intensity = self.downsample(intensity.unsqueeze(1)).squeeze(1)
        angle = self.downsample(angle.unsqueeze(1)).squeeze(1)
        # 投影强度和角度
        intensity_feat = self.intensity_proj(intensity.unsqueeze(-1))  # (B, L, D/2)
        angle_feat = self.angle_proj(angle.unsqueeze(-1))  # (B, L, D/2)

        # 可学习的positional embedding强化angle特征
        combined = torch.cat([intensity_feat, angle_feat], dim=-1)  # (B, L, D)
        combined += self.position_embed[:, :intensity.size(1), :]

        # transformer encoder
        encoded = self.encoder(combined)  # (B, L, D)
        return encoded.mean(dim=1)  # (B, D)

class CrystalFusionNet(torch.nn.Module):
    def __init__(self, input_dim: int = 8500, output_dim: int = 230):
        super().__init__()
        self.tcn_module = ResTcn(p_dropout=0.25)
        self.attention_module = HybridAttentionEngine(input_dim)

        # Feature refinement
        self.fusion_blocks = torch.nn.Sequential(
            SpectralFusionUnit(32 + 64, 256),
            SpectralFusionUnit(256, 128),
            SpectralFusionUnit(128, 64)
        )

        # Final classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(64),
            torch.nn.Linear(64, output_dim)
        )

    def forward(self, intensity: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        # Input shapes: (B, 8500) for both intensity and angle
        tcn_features = self.tcn_module(intensity, angle)  # (B, 32)
        attn_features = self.attention_module(intensity, angle)  # (B, 64)

        # tcn的特征和attention特征融合
        combined = torch.cat([tcn_features, attn_features], dim=-1)  # (B, 96)
        combined = combined.unsqueeze(-1)  # (B, 96, 1)

        # 融合后的特征通过组合后的SpectralFusionUnit进行重新融合
        refined = self.fusion_blocks(combined)  # (B, 64, 1)
        refined = refined.squeeze(-1)  # (B, 64)

        return self.classifier(refined)  # (B, 230)

if __name__ == "__main__":
    device = 'cpu'
    model = CrystalFusionNet().to(device)
    criterion = torch.nn.CrossEntropyLoss()

    # Test with minimal batch size
    intensity = torch.randn(2, 8500).to(device)
    angle = torch.randn(2, 8500).to(device)
    labels = torch.randint(0, 230, (2,)).to(device)

    logits = model(intensity, angle)
    print(f"Output shape: {logits.shape}")  # Should be (2, 230)

    loss = criterion(logits, labels)
    print(f"Test loss: {loss.item():.4f}")