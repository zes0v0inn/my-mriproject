"""
Model Module
============
Defines segmentation models for BraTS2021 with optional auxiliary classification head.

核心设计:
  ┌─────────────────────────────────────────────────────────────────────┐
  │                                                                     │
  │  任何 backbone (手写 or MONAI)                                       │
  │      │                                                              │
  │      ├─→ seg_output   (B, 3, D, H, W)   分割结果                     │
  │      │                                                              │
  │      └─→ bottleneck features ──→ ClassificationHead ──→ cls_output  │
  │              ↑                        (GAP → FC)         (B, 1)     │
  │              │                                                      │
  │         通过两种方式获取:                                              │
  │           1. 手写模型: 直接从 encoder 拿                               │
  │           2. MONAI 模型: 用 forward hook 截取                         │
  │                                                                     │
  └─────────────────────────────────────────────────────────────────────┘

使用方式:
  # 纯分割 (和之前一样)
  model = build_model("basic_unet")
  seg_out = model(x)                    # (B, 3, D, H, W)

  # 分割 + 分类 (多任务)
  model = build_model("basic_unet", with_classifier=True)
  seg_out, cls_out = model(x)           # seg: (B, 3, D, H, W), cls: (B, 1)
"""

from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn
from monai.networks.nets import UNet, AttentionUnet, UNETR, SwinUNETR


# ═════════════════════════════════════════════════════════════
#  Part 1: 基础构件 — 手写 3D UNet
# ═════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    """Conv3d × 2 + InstanceNorm + LeakyReLU"""
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
    """UNet 编码器: ConvBlock + MaxPool, 最后一层不 pool (即 bottleneck)"""
    def __init__(self, in_ch: int, features: Tuple[int, ...], dropout: float = 0.0):
        super().__init__()
        self.stages = nn.ModuleList()
        self.pools  = nn.ModuleList()

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
        return x, skips  # x = bottleneck


class Decoder(nn.Module):
    """UNet 解码器: ConvTranspose + Concat skip + ConvBlock"""
    def __init__(self, features: Tuple[int, ...], dropout: float = 0.0):
        super().__init__()
        self.upconvs = nn.ModuleList()
        self.stages  = nn.ModuleList()

        for i in range(len(features) - 1):
            in_f  = features[i]
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


class BasicUNet3D(nn.Module):
    """
    手写 3D UNet, 支持返回 bottleneck 特征.

    forward 返回:
      - 默认:     seg_output  (B, 3, D, H, W)
      - 需要特征: (seg_output, bottleneck)   bottleneck: (B, C_deep, d, h, w)
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
        self.decoder = Decoder(features[::-1], dropout=dropout)
        self.head = nn.Conv3d(features[0], out_channels, kernel_size=1)

        # 记录 bottleneck 通道数, 方便外部构建分类头
        self.bottleneck_channels = features[-1]

        # 控制是否返回 bottleneck 特征
        self._return_features = False

    def set_return_features(self, flag: bool):
        """开关: 是否让 forward 同时返回 bottleneck 特征"""
        self._return_features = flag

    def forward(self, x):
        bottleneck, skips = self.encoder(x)
        out = self.decoder(bottleneck, skips)
        seg_out = self.head(out)

        if self._return_features:
            return seg_out, bottleneck
        return seg_out


# ═════════════════════════════════════════════════════════════
#  Part 2: MONAI 模型特征提取 — Forward Hook 机制
# ═════════════════════════════════════════════════════════════
#
#  问题: MONAI 的模型是封装好的, 我们没法像手写模型那样
#        直接拿 encoder 的 bottleneck 输出.
#
#  解决: 用 PyTorch 的 register_forward_hook() 在指定层上
#        "截取" 该层的输出. 这不需要修改 MONAI 的源码.
#
#  原理图:
#
#    Input ──→ [ Layer_1 ] ──→ [ Layer_2 ] ──→ ... ──→ [ Bottleneck ] ──→ ... ──→ Output
#                                                            │
#                                                       hook 截取!
#                                                            │
#                                                            ↓
#                                                     captured_features
#

class FeatureHook:
    """
    一个简单的 forward hook, 用来捕获某一层的输出.

    用法:
        hook = FeatureHook()
        model.some_layer.register_forward_hook(hook)
        output = model(input)
        bottleneck_features = hook.features  # 就是 some_layer 的输出
    """
    def __init__(self):
        self.features: Optional[torch.Tensor] = None

    def __call__(self, module, input, output):
        # output 就是这一层的前向输出
        # 有些层返回 tuple, 我们取第一个 tensor
        if isinstance(output, tuple):
            self.features = output[0]
        else:
            self.features = output

    def clear(self):
        self.features = None


def find_bottleneck_layer(model: nn.Module, model_name: str) -> Tuple[nn.Module, int]:
    """
    根据模型类型, 找到 bottleneck 层并返回 (layer, channels).

    这是整个 hook 机制中最关键的函数 — 你需要知道 MONAI 模型
    内部的层名字. 可以通过 print(model) 查看完整结构.

    ┌──────────────┬─────────────────────────┬────────────────────────┐
    │  模型         │  bottleneck 层路径       │  怎么找到的             │
    ├──────────────┼─────────────────────────┼────────────────────────┤
    │  MONAI UNet  │  model.model[2]         │  print(model) 看结构   │
    │              │  (最深的 down block)     │  encoder 部分的最后一块  │
    │              │                         │                        │
    │  AttnUnet    │  model.model[2]         │  同上, 结构类似          │
    │              │                         │                        │
    │  UNETR       │  model.encoder          │  ViT encoder 的输出     │
    │              │  (ViT backbone)         │  768-dim feature       │
    │              │                         │                        │
    │  SwinUNETR   │  model.swinViT          │  Swin encoder 最后一层  │
    │              │  (SwinTransformer)      │  768-dim feature       │
    └──────────────┴─────────────────────────┴────────────────────────┘

    如果你不确定某个模型的层名, 只需要:
        model = build_backbone("monai_unet", ...)
        print(model)      ← 打印完整结构
        # 或者
        for name, module in model.named_modules():
            print(name, type(module))

    Args:
        model: MONAI 模型实例
        model_name: 模型名字 (用于判断结构)

    Returns:
        (target_layer, bottleneck_channels)
    """
    name = model_name.lower().replace("-", "_")

    if name == "monai_unet":
        # MONAI UNet 结构: model.model = Sequential(down_1, down_2, ..., bottom, up_1, ...)
        # 编码器的最深层是中间那个 block
        # channels 的最后一个值就是 bottleneck 通道数
        n_layers = len(model.model)
        bottleneck_idx = n_layers // 2  # 中间层就是 bottleneck
        layer = model.model[bottleneck_idx]
        channels = model.channels[-1]  # features 的最后一个
        return layer, channels

    elif name == "attention_unet":
        # AttentionUnet 结构类似, bottleneck 也在中间
        n_layers = len(model.model)
        bottleneck_idx = n_layers // 2
        layer = model.model[bottleneck_idx]
        channels = model.channels[-1]
        return layer, channels

    elif name == "unetr":
        # UNETR: ViT encoder 输出 hidden_size (默认 768)
        layer = model.encoder
        channels = model.hidden_size  # 768
        return layer, channels

    elif name == "swin_unetr":
        # SwinUNETR: SwinTransformer backbone
        layer = model.swinViT
        channels = 768  # SwinUNETR 最深层特征维度
        return layer, channels

    else:
        raise ValueError(f"Don't know how to find bottleneck for '{model_name}'")


# ═════════════════════════════════════════════════════════════
#  Part 3: 分类头 + 多任务包装器
# ═════════════════════════════════════════════════════════════

class ClassificationHead(nn.Module):
    """
    轻量分类头: Global Average Pooling → FC layers → 输出

    结构:
        bottleneck (B, C, d, h, w)
            ↓ AdaptiveAvgPool3d(1)
        (B, C, 1, 1, 1)
            ↓ Flatten
        (B, C)
            ↓ FC → ReLU → Dropout → FC
        (B, num_classes)

    为什么用 GAP 而不是 Flatten?
      - bottleneck 的空间尺寸不固定 (取决于输入大小和下采样次数)
      - GAP 把任意空间尺寸压缩成 1×1×1, 输出维度只取决于通道数
      - 参数量也小很多
    """
    def __init__(self, in_channels: int, num_classes: int = 1, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),       # (B, C, d, h, w) → (B, C, 1, 1, 1)
            nn.Flatten(),                   # (B, C)
            nn.Linear(in_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),    # (B, num_classes)
        )

    def forward(self, x):
        return self.head(x)


class SegWithClassifier(nn.Module):
    """
    多任务包装器: 分割 backbone + 分类头

    这个 wrapper 可以包装 **任何** 分割模型 (手写的或 MONAI 的),
    统一提供 (seg_output, cls_output) 的双输出接口.

    工作原理:
    ┌──────────────────────────────────────────────────────────┐
    │                                                          │
    │  对于手写模型 (BasicUNet3D):                               │
    │    直接调用 model.set_return_features(True)                │
    │    forward 时自动返回 (seg_out, bottleneck)                │
    │                                                          │
    │  对于 MONAI 模型:                                         │
    │    在 bottleneck 层注册 forward hook                      │
    │    forward 时正常跑, hook 偷偷截取 bottleneck 特征          │
    │                                                          │
    │  然后:                                                    │
    │    bottleneck → ClassificationHead → cls_output           │
    │                                                          │
    └──────────────────────────────────────────────────────────┘

    用法:
        model = SegWithClassifier(backbone, backbone_name, num_classes=1)
        seg_out, cls_out = model(x)
        # seg_out: (B, 3, D, H, W)
        # cls_out: (B, 1)
    """
    def __init__(
        self,
        backbone: nn.Module,
        backbone_name: str,
        num_classes: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone = backbone
        self.backbone_name = backbone_name.lower().replace("-", "_")

        # ── 根据 backbone 类型选择特征提取策略 ────────────────
        if self.backbone_name == "basic_unet":
            # 手写模型: 直接让它返回 features
            self.backbone.set_return_features(True)
            bottleneck_ch = self.backbone.bottleneck_channels
            self._use_hook = False
        else:
            # MONAI 模型: 用 hook 截取
            layer, bottleneck_ch = find_bottleneck_layer(backbone, backbone_name)
            self._hook = FeatureHook()
            self._hook_handle = layer.register_forward_hook(self._hook)
            self._use_hook = True

        # ── 分类头 ────────────────────────────────────────────
        self.cls_head = ClassificationHead(
            in_channels=bottleneck_ch,
            num_classes=num_classes,
            dropout=dropout,
        )

        print(f"[Model] Added ClassificationHead: {bottleneck_ch}ch → {num_classes} output")

    def forward(self, x):
        if not self._use_hook:
            # BasicUNet3D: forward 直接返回 (seg, features)
            seg_out, bottleneck = self.backbone(x)
        else:
            # MONAI 模型: 正常 forward, hook 自动截取
            seg_out = self.backbone(x)
            bottleneck = self._hook.features
            assert bottleneck is not None, \
                "Hook failed to capture features. Check find_bottleneck_layer()."

        # 分类头
        cls_out = self.cls_head(bottleneck)

        return seg_out, cls_out

    def remove_hooks(self):
        """清理 hook (在 model 不再使用时调用, 避免内存泄漏)"""
        if self._use_hook and hasattr(self, "_hook_handle"):
            self._hook_handle.remove()


# ═════════════════════════════════════════════════════════════
#  Part 4: 工厂函数
# ═════════════════════════════════════════════════════════════

def _build_backbone(
    model_name: str,
    in_channels: int = 4,
    out_channels: int = 3,
    roi_size: Tuple[int, int, int] = (128, 128, 128),
    features: Tuple[int, ...] = (32, 64, 128, 256),
    dropout: float = 0.1,
) -> nn.Module:
    """构建纯分割 backbone (不带分类头)"""
    name = model_name.lower().replace("-", "_")

    if name == "basic_unet":
        return BasicUNet3D(in_channels, out_channels, features, dropout)

    elif name == "monai_unet":
        return UNet(
            spatial_dims=3, in_channels=in_channels, out_channels=out_channels,
            channels=features, strides=(2,) * (len(features) - 1),
            num_res_units=2, dropout=dropout,
        )

    elif name == "attention_unet":
        return AttentionUnet(
            spatial_dims=3, in_channels=in_channels, out_channels=out_channels,
            channels=features, strides=(2,) * (len(features) - 1), dropout=dropout,
        )

    elif name == "unetr":
        return UNETR(
            in_channels=in_channels, out_channels=out_channels, img_size=roi_size,
            feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12,
            proj_type="conv", norm_name="instance", res_block=True, dropout_rate=dropout,
        )

    elif name == "swin_unetr":
        return SwinUNETR(
            img_size=roi_size, in_channels=in_channels, out_channels=out_channels,
            feature_size=48, drop_rate=dropout, attn_drop_rate=0.0,
            dropout_path_rate=0.0, use_checkpoint=True,
        )

    else:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: basic_unet, monai_unet, attention_unet, unetr, swin_unetr"
        )


def build_model(
    model_name: str = "basic_unet",
    in_channels: int = 4,
    out_channels: int = 3,
    roi_size: Tuple[int, int, int] = (128, 128, 128),
    features: Tuple[int, ...] = (32, 64, 128, 256),
    dropout: float = 0.1,
    # ── 分类头相关 (新增) ──────────────────────────────────
    with_classifier: bool = False,
    num_classes: int = 1,
    cls_dropout: float = 0.3,
) -> nn.Module:
    """
    工厂函数 — train.py 的唯一入口.

    ┌──────────────────┬────────────────────────────────────┐
    │  model_name      │  来源                              │
    ├──────────────────┼────────────────────────────────────┤
    │  basic_unet      │  ★ 手写实现                        │
    │  monai_unet      │  MONAI UNet                       │
    │  attention_unet  │  MONAI AttentionUnet              │
    │  unetr           │  MONAI UNETR                      │
    │  swin_unetr      │  MONAI SwinUNETR                  │
    └──────────────────┴────────────────────────────────────┘

    Args:
        model_name: 模型架构名称
        with_classifier: 是否附加分类头 (多任务学习)
        num_classes: 分类头输出维度 (默认 1, 用于回归 score)
        cls_dropout: 分类头的 dropout
        (其余参数同之前)

    Returns:
        with_classifier=False → model(x) returns seg_output
        with_classifier=True  → model(x) returns (seg_output, cls_output)
    """
    backbone = _build_backbone(
        model_name, in_channels, out_channels, roi_size, features, dropout
    )

    if with_classifier:
        model = SegWithClassifier(
            backbone=backbone,
            backbone_name=model_name,
            num_classes=num_classes,
            dropout=cls_dropout,
        )
    else:
        model = backbone

    # 打印参数量
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {model_name}" + (" + ClassificationHead" if with_classifier else ""))
    print(f"  Total params     : {total:,}")
    print(f"  Trainable params : {trainable:,}")

    return model


# ═════════════════════════════════════════════════════════════
#  Smoke test
# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_features = (16, 32, 64, 128)
    x = torch.randn(1, 4, 64, 64, 64, device=device)

    print("=" * 70)
    print("  Test 1: 纯分割模式")
    print("=" * 70)
    for name in ["basic_unet", "monai_unet"]:
        model = build_model(name, features=test_features, with_classifier=False).to(device)
        with torch.no_grad():
            y = model(x)
        print(f"  {name:16s}  Input: {x.shape}  →  Seg: {y.shape}")
        assert y.shape == (1, 3, 64, 64, 64)
        print(f"  {'':16s}  ✓ Pass\n")

    print("=" * 70)
    print("  Test 2: 分割 + 分类 多任务模式")
    print("=" * 70)
    for name in ["basic_unet", "monai_unet"]:
        model = build_model(name, features=test_features, with_classifier=True, num_classes=1).to(device)
        with torch.no_grad():
            seg, cls = model(x)
        print(f"  {name:16s}  Input: {x.shape}  →  Seg: {seg.shape}  Cls: {cls.shape}")
        assert seg.shape == (1, 3, 64, 64, 64)
        assert cls.shape == (1, 1)
        print(f"  {'':16s}  ✓ Pass\n")