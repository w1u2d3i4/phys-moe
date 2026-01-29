"""
Vision Transformer (ViT) Frontend for XRD Data Processing
将XRD数据（8500维）通过ViT处理成1024维特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEmbedding1D(nn.Module):
    """
    1D Patch Embedding for XRD data
    将8500维的XRD数据分成patches并嵌入
    """
    def __init__(self, seq_len=8500, patch_size=50, embed_dim=256, in_channels=1):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.embed_dim = embed_dim
        
        # Patch embedding: 将每个patch投影到embed_dim
        self.patch_embed = nn.Conv1d(
            in_channels, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size, 
            bias=False
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        """
        x: (B, 1, 8500)
        返回: (B, num_patches+1, embed_dim)
        """
        B = x.size(0)
        
        # Patch embedding: (B, 1, 8500) -> (B, embed_dim, num_patches)
        x = self.patch_embed(x)
        
        # Transpose: (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = x.transpose(1, 2)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, embed_dim)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Layer norm
        x = self.norm(x)
        
        return x


class TransformerBlock(nn.Module):
    """
    Transformer Encoder Block
    """
    def __init__(self, embed_dim=256, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        
        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        x: (B, N, embed_dim)
        返回: (B, N, embed_dim)
        """
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)
        
        # Feed-forward with residual
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out
        
        return x


class ViT_Frontend(nn.Module):
    """
    Vision Transformer Frontend for XRD Data
    输入: (B, 1, 8500)
    输出: (B, 1024)
    """
    def __init__(self, 
                 seq_len=8500,
                 patch_size=50,
                 embed_dim=256,
                 depth=6,
                 num_heads=8,
                 mlp_ratio=4,
                 dropout=0.1,
                 output_dim=1024):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbedding1D(
            seq_len=seq_len,
            patch_size=patch_size,
            embed_dim=embed_dim,
            in_channels=1
        )
        
        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depth)
        ])
        
        # Output projection: embed_dim -> output_dim
        self.output_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        x: (B, 1, 8500) 或 (B, 8500)
        返回: (B, 1024)
        """
        # 确保输入是 (B, 1, 8500)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Patch embedding: (B, 1, 8500) -> (B, num_patches+1, embed_dim)
        x = self.patch_embed(x)
        
        # Transformer encoder
        for block in self.blocks:
            x = block(x)
        
        # Extract CLS token (first token)
        cls_token = x[:, 0, :]  # (B, embed_dim)
        
        # Project to output dimension
        output = self.output_proj(cls_token)  # (B, output_dim)
        
        return output


class ViT_Frontend_MeanPool(nn.Module):
    """
    ViT Frontend with Mean Pooling (instead of CLS token)
    使用所有patches的平均值而不是CLS token
    """
    def __init__(self, 
                 seq_len=8500,
                 patch_size=50,
                 embed_dim=256,
                 depth=6,
                 num_heads=8,
                 mlp_ratio=4,
                 dropout=0.1,
                 output_dim=1024):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        
        # Patch embedding (without CLS token)
        self.patch_embed = nn.Conv1d(
            1, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size, 
            bias=False
        )
        self.num_patches = seq_len // patch_size
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        
        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depth)
        ])
        
        # Output projection: embed_dim -> output_dim
        self.output_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        x: (B, 1, 8500) 或 (B, 8500)
        返回: (B, 1024)
        """
        # 确保输入是 (B, 1, 8500)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Patch embedding: (B, 1, 8500) -> (B, embed_dim, num_patches)
        x = self.patch_embed(x)
        
        # Transpose: (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = x.transpose(1, 2)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transformer encoder
        for block in self.blocks:
            x = block(x)
        
        # Mean pooling over patches
        x = x.mean(dim=1)  # (B, embed_dim)
        
        # Project to output dimension
        output = self.output_proj(x)  # (B, output_dim)
        
        return output


if __name__ == "__main__":
    # 测试ViT Frontend
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 测试CLS token版本
    print("测试ViT_Frontend (CLS token版本)...")
    model_cls = ViT_Frontend(
        seq_len=8500,
        patch_size=50,
        embed_dim=256,
        depth=6,
        num_heads=8,
        output_dim=1024
    ).to(device)
    
    x = torch.randn(4, 1, 8500).to(device)
    out = model_cls(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    print(f"参数量: {sum(p.numel() for p in model_cls.parameters()) / 1e6:.2f}M")
    
    # 测试Mean Pooling版本
    print("\n测试ViT_Frontend_MeanPool...")
    model_mean = ViT_Frontend_MeanPool(
        seq_len=8500,
        patch_size=50,
        embed_dim=256,
        depth=6,
        num_heads=8,
        output_dim=1024
    ).to(device)
    
    x = torch.randn(4, 1, 8500).to(device)
    out = model_mean(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    print(f"参数量: {sum(p.numel() for p in model_mean.parameters()) / 1e6:.2f}M")
