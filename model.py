"""
Model Module
============
Defines segmentation models for BraTS2021.

包含:
  ┌─────────────────────────────────────────────────────────────┐
  │  自定义模型 (手写实现)                                        │
  │    - basic_unet : 从零实现的 3D UNet                          │
  │                                                              │
  │  MONAI 内置模型 (直接调用)                                     │
  │    - monai_unet      : MONAI 的 UNet                         │
  │    - attention_unet   : UNet with attention gates             │
  │    - unetr            : Vision-Transformer encoder + CNN      │
  │    - swin_unetr       : Swin-Transformer based UNet           │
  └─────────────────────────────────────────────────────────────┘

所有模型统一接口:
  - Input:  (B, 4, D, H, W)   4通道 MRI
  - Output: (B, 3, D, H, W)   3通道 分割 (TC, WT, ET)

添加你自己的模型只需要两步:
  1. 在本文件中定义你的 nn.Module 类
  2. 在 build_model() 工厂函数中注册一个 elif 分支
"""

from typing import Tuple

import torch
import torch.nn as nn
from monai.networks.nets import UNet, AttentionUnet, UNETR, SwinUNETR


# ═════════════════════════════════════════════════════════════
#  自定义模型: 手写 3D UNet
# ═════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    """
    基础卷积块: Conv3d → InstanceNorm → LeakyReLU → Conv3d → InstanceNorm → LeakyReLU

    为什么用 InstanceNorm 而不是 BatchNorm?
      - 医学图像 batch 通常很小 (1~2), InstanceNorm 在小 batch 下更稳定
    为什么用 LeakyReLU?
      - 避免 ReLU 的 "dead neuron" 问题
    """
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


class Encoder(nn.Module):
    """
    UNet 编码器路径 (下采样)

    每一层:  ConvBlock → MaxPool3d(2)
    最后一层 (bottleneck) 不做 pool

    Example with features=(32, 64, 128, 256):
        输入 (B, 4, D, H, W)
         ↓ ConvBlock → skip1 (B, 32, D, H, W)
         ↓ MaxPool
         ↓ ConvBlock → skip2 (B, 64, D/2, H/2, W/2)
         ↓ MaxPool
         ↓ ConvBlock → skip3 (B, 128, D/4, H/4, W/4)
         ↓ MaxPool
         ↓ ConvBlock → bottleneck (B, 256, D/8, H/8, W/8)
    """
    def __init__(self, in_ch: int, features: Tuple[int, ...], dropout: float = 0.0):
        super().__init__()
        self.stages = nn.ModuleList()
        self.pools  = nn.ModuleList()

        ch = in_ch
        for i, f in enumerate(features):
            self.stages.append(ConvBlock(ch, f, dropout=dropout))
            if i < len(features) - 1:  # 最后一层不 pool
                self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            ch = f

    def forward(self, x):
        skips = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.stages) - 1:
                skips.append(x)        # 保存 skip connection
                x = self.pools[i](x)   # 下采样
        return x, skips  # x = bottleneck, skips = [skip1, skip2, ...]


class Decoder(nn.Module):
    """
    UNet 解码器路径 (上采样 + skip connection)

    每一层:
        1. ConvTranspose3d 上采样 2x
        2. Concatenate skip connection
        3. ConvBlock 融合特征

    Example with features=(256, 128, 64, 32):
        bottleneck (B, 256, D/8, H/8, W/8)
         ↑ ConvTranspose → (B, 128, D/4, H/4, W/4)
         ↑ Cat skip3     → (B, 256, D/4, H/4, W/4)
         ↑ ConvBlock     → (B, 128, D/4, H/4, W/4)
         ↑ ...
         ↑ ConvBlock     → (B, 32, D, H, W)
    """
    def __init__(self, features: Tuple[int, ...], dropout: float = 0.0):
        super().__init__()
        self.upconvs = nn.ModuleList()
        self.stages  = nn.ModuleList()

        # features 从 bottleneck 到最浅层, 如 (256, 128, 64, 32)
        # 解码时从 256 → 128 → 64 → 32
        for i in range(len(features) - 1):
            in_f  = features[i]
            out_f = features[i + 1]
            self.upconvs.append(
                nn.ConvTranspose3d(in_f, out_f, kernel_size=2, stride=2)
            )
            # cat 之后通道数 = out_f (上采样) + out_f (skip) = 2 * out_f
            self.stages.append(ConvBlock(out_f * 2, out_f, dropout=dropout))

    def forward(self, x, skips):
        """
        Args:
            x: bottleneck feature map
            skips: list of skip connections [skip1, skip2, ...] (浅→深)
                   我们从最深的 skip 开始用, 所以 reverse
        """
        skips = skips[::-1]  # reverse: 从最深层的 skip 开始

        for i, (upconv, stage) in enumerate(zip(self.upconvs, self.stages)):
            x = upconv(x)

            # 处理尺寸不匹配 (因为 pool 时奇数尺寸会丢 1 pixel)
            skip = skips[i]
            if x.shape != skip.shape:
                x = nn.functional.pad(x, self._calc_pad(x, skip))

            x = torch.cat([x, skip], dim=1)  # 沿 channel 维度拼接
            x = stage(x)

        return x

    @staticmethod
    def _calc_pad(x, target):
        """计算让 x 和 target 空间维度对齐所需的 padding."""
        d_diff = target.shape[2] - x.shape[2]
        h_diff = target.shape[3] - x.shape[3]
        w_diff = target.shape[4] - x.shape[4]
        return (0, w_diff, 0, h_diff, 0, d_diff)


class BasicUNet3D(nn.Module):
    """
    从零手写的 3D UNet

    结构图:
        Input (B, 4, D, H, W)
          │
          ├──► Encoder Level 1  ──────────────────────► skip1 ──┐
          │        ↓ MaxPool                                     │
          ├──► Encoder Level 2  ──────────────────► skip2 ──┐   │
          │        ↓ MaxPool                                 │   │
          ├──► Encoder Level 3  ──────────► skip3 ──┐       │   │
          │        ↓ MaxPool                         │       │   │
          └──► Bottleneck                            │       │   │
                   ↓ ConvTranspose                   │       │   │
               Cat(↑, skip3) → ConvBlock ◄───────────┘       │   │
                   ↓ ConvTranspose                           │   │
               Cat(↑, skip2) → ConvBlock ◄───────────────────┘   │
                   ↓ ConvTranspose                               │
               Cat(↑, skip1) → ConvBlock ◄───────────────────────┘
                   ↓
               1×1×1 Conv → Output (B, 3, D, H, W)

    用法:
        model = BasicUNet3D(in_channels=4, out_channels=3, features=(32, 64, 128, 256))
        output = model(input)  # input: (B, 4, D, H, W) → output: (B, 3, D, H, W)
    """
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        features: Tuple[int, ...] = (32, 64, 128, 256),
        dropout: float = 0.0,
    ):
        super().__init__()

        self.encoder = Encoder(in_channels, features, dropout=dropout)
        self.decoder = Decoder(features[::-1], dropout=dropout)  # reverse for decoding

        # 最终 1x1x1 卷积: 将最浅层特征映射到输出类别数
        self.head = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # 编码
        bottleneck, skips = self.encoder(x)
        # 解码
        out = self.decoder(bottleneck, skips)
        # 输出 head (不加 sigmoid, 交给 loss 函数处理)
        out = self.head(out)
        return out


# ═════════════════════════════════════════════════════════════
#  你可以在这里继续添加自己的模型, 比如:
# ═════════════════════════════════════════════════════════════
#
# class MyResUNet3D(nn.Module):
#     """带残差连接的 UNet — 只需把 ConvBlock 改成带 shortcut 的版本"""
#     def __init__(self, in_channels, out_channels, features, dropout=0.0):
#         super().__init__()
#         # ... 你的实现 ...
#
#     def forward(self, x):
#         # ... 你的实现 ...
#         return out
#
#
# class MyUNetPlusPlus(nn.Module):
#     """UNet++ (嵌套 UNet / Dense skip connections)"""
#     def __init__(self, ...):
#         super().__init__()
#         # ... 你的实现 ...
#
#     def forward(self, x):
#         # ... 你的实现 ...
#         return out


# ═════════════════════════════════════════════════════════════
#  工厂函数 — train.py 的唯一入口
# ═════════════════════════════════════════════════════════════

def build_model(
    model_name: str = "basic_unet",
    in_channels: int = 4,
    out_channels: int = 3,
    roi_size: Tuple[int, int, int] = (128, 128, 128),
    features: Tuple[int, ...] = (32, 64, 128, 256),
    dropout: float = 0.1,
) -> nn.Module:
    """
    工厂函数: 根据名字构建模型.

    ┌──────────────────┬────────────────────────────────────┐
    │  model_name      │  来源                              │
    ├──────────────────┼────────────────────────────────────┤
    │  basic_unet      │  ★ 手写实现 (本文件)               │
    │  monai_unet      │  MONAI UNet                       │
    │  attention_unet  │  MONAI AttentionUnet              │
    │  unetr           │  MONAI UNETR                      │
    │  swin_unetr      │  MONAI SwinUNETR                  │
    └──────────────────┴────────────────────────────────────┘

    添加新模型只需:
      1. 在上面写一个 class MyModel(nn.Module)
      2. 在下面加一个 elif name == "my_model": model = MyModel(...)
      3. 在 train.py 的 --model choices 列表里加上 "my_model"

    Args:
        model_name: 模型名称
        in_channels: 输入通道数 (BraTS = 4)
        out_channels: 输出通道数 (3: TC, WT, ET)
        roi_size: 输入 patch 尺寸 (Transformer 模型需要)
        features: 各层通道数 (UNet 系列)
        dropout: Dropout 概率

    Returns:
        nn.Module
    """
    name = model_name.lower().replace("-", "_")

    # ── 自定义模型 ────────────────────────────────────────────
    if name == "basic_unet":
        model = BasicUNet3D(
            in_channels=in_channels,
            out_channels=out_channels,
            features=features,
            dropout=dropout,
        )

    # ── MONAI 内置模型 ───────────────────────────────────────
    elif name == "monai_unet":
        model = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=features,
            strides=(2,) * (len(features) - 1),
            num_res_units=2,
            dropout=dropout,
        )

    elif name == "attention_unet":
        model = AttentionUnet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=features,
            strides=(2,) * (len(features) - 1),
            dropout=dropout,
        )

    elif name == "unetr":
        model = UNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=roi_size,
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            proj_type="conv",
            norm_name="instance",
            res_block=True,
            dropout_rate=dropout,
        )

    elif name == "swin_unetr":
        model = SwinUNETR(
            img_size=roi_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=48,
            drop_rate=dropout,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=True,
        )

    # ── 你的新模型在这里注册 ──────────────────────────────────
    #
    # elif name == "my_resunet":
    #     model = MyResUNet3D(in_channels, out_channels, features, dropout)
    #
    # elif name == "my_unet_plusplus":
    #     model = MyUNetPlusPlus(in_channels, out_channels, ...)

    else:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: basic_unet, monai_unet, attention_unet, unetr, swin_unetr"
        )

    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {name}")
    print(f"  Total params     : {total_params:,}")
    print(f"  Trainable params : {trainable_params:,}")

    return model


# ═════════════════════════════════════════════════════════════
#  Smoke test — 验证手写模型和 MONAI 模型输出一致
# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for name in ["basic_unet", "monai_unet"]:
        print(f"\n{'='*60}")
        model = build_model(name, features=(16, 32, 64, 128)).to(device)
        x = torch.randn(1, 4, 64, 64, 64, device=device)
        with torch.no_grad():
            y = model(x)
        print(f"  Input  : {x.shape}")
        print(f"  Output : {y.shape}")
        assert y.shape == (1, 3, 64, 64, 64), f"Shape mismatch! Got {y.shape}"
        print("  ✓ Pass")