"""
Lightweight 3D TransUNet
========================

基于 3D-TransUNet 的核心思想实现：
  - UNet 编码/解码
  - bottleneck 处加入 Transformer block

该实现不依赖原仓库的 nnUNet/mask2former 组件，便于直接集成到现有训练脚本。
"""

from typing import Tuple

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Dropout3d(p=dropout),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class TransformerBlock3D(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)

        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, N, C)
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class BottleneckTransformer3D(nn.Module):
    def __init__(self, channels: int, depth: int = 4, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        # 用 depth-wise conv 作为 3D 位置编码
        self.pos_embed = nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.blocks = nn.ModuleList([
            TransformerBlock3D(channels, num_heads=num_heads, dropout=dropout)
            for _ in range(depth)
        ])

    def forward(self, x):
        # x: (B, C, D, H, W)
        x = x + self.pos_embed(x)
        b, c, d, h, w = x.shape
        x_seq = x.flatten(2).transpose(1, 2)  # (B, N, C)

        for blk in self.blocks:
            x_seq = blk(x_seq)

        x = x_seq.transpose(1, 2).reshape(b, c, d, h, w)
        return x


class Encoder(nn.Module):
    def __init__(self, in_ch: int, features: Tuple[int, ...], dropout: float = 0.0):
        super().__init__()
        self.stages = nn.ModuleList()
        self.pools = nn.ModuleList()

        ch = in_ch
        for i, f in enumerate(features):
            self.stages.append(ConvBlock(ch, f, dropout=dropout))
            if i < len(features) - 1:
                self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            ch = f

    def forward(self, x):
        skips = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.stages) - 1:
                skips.append(x)
                x = self.pools[i](x)
        return x, skips


class Decoder(nn.Module):
    def __init__(self, features: Tuple[int, ...], dropout: float = 0.0):
        super().__init__()
        self.upconvs = nn.ModuleList()
        self.stages = nn.ModuleList()

        for i in range(len(features) - 1):
            in_f = features[i]
            out_f = features[i + 1]
            self.upconvs.append(nn.ConvTranspose3d(in_f, out_f, kernel_size=2, stride=2))
            self.stages.append(ConvBlock(out_f * 2, out_f, dropout=dropout))

    def forward(self, x, skips):
        skips = skips[::-1]
        for i, (upconv, stage) in enumerate(zip(self.upconvs, self.stages)):
            x = upconv(x)
            skip = skips[i]
            if x.shape != skip.shape:
                d = skip.shape[2] - x.shape[2]
                h = skip.shape[3] - x.shape[3]
                w = skip.shape[4] - x.shape[4]
                x = nn.functional.pad(x, (0, w, 0, h, 0, d))
            x = torch.cat([x, skip], dim=1)
            x = stage(x)
        return x


class TransUNet3D(nn.Module):
    """3D UNet + bottleneck Transformer，输出接口与 BasicUNet3D 保持一致。"""

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        features: Tuple[int, ...] = (32, 64, 128, 256),
        dropout: float = 0.1,
        transformer_depth: int = 4,
        num_heads: int = 8,
    ):
        super().__init__()
        self.encoder = Encoder(in_channels, features, dropout=dropout)

        bottleneck_ch = features[-1]
        while num_heads > 1 and bottleneck_ch % num_heads != 0:
            num_heads -= 1

        self.transformer = BottleneckTransformer3D(
            channels=bottleneck_ch,
            depth=transformer_depth,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.decoder = Decoder(features[::-1], dropout=dropout)
        self.head = nn.Conv3d(features[0], out_channels, kernel_size=1)

        self.bottleneck_channels = bottleneck_ch
        self._return_features = False

    def set_return_features(self, flag: bool):
        self._return_features = flag

    def forward(self, x):
        bottleneck, skips = self.encoder(x)
        bottleneck = self.transformer(bottleneck)
        out = self.decoder(bottleneck, skips)
        seg_out = self.head(out)

        if self._return_features:
            return seg_out, bottleneck
        return seg_out
