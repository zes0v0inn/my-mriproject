"""
mamba_transunet3d.py
====================
MambaTransUNet3D: 完整的多任务脑肿瘤分割+分类模型

架构创新点：
  1. Mamba3D Encoder        - 替换 CNN Encoder，建模长程依赖
  2. Atlas-aware PE         - 解剖先验指导的可学习位置编码
  3. Multi-task heads       - 分割 + MGMT + IDH（部分标注）
  4. Classification Feedback - IDH/MGMT 预测结果反馈调制 Mask2Former queries
  5. T1/T2 ratio            - 作为第5个输入通道

数据流：
  Input (B, 5, D, H, W)       ← T1, T2, T1ce, FLAIR + T1/T2 ratio
        │
  Mamba3D Encoder             ← 替换 CNN，保留 skip connections
        │ bottleneck
        │
  Atlas-aware PE              ← Atlas 体积指导位置编码
        │
  ViT Bottleneck              ← Transformer Encoder
        │
        ├──→ ClassificationHead (MGMT)    ← 所有样本
        ├──→ ClassificationHead (IDH)     ← 部分样本（has_idh mask）
        │         │
        │    分类置信度反馈
        │         ↓
  CNN Decoder + Mask2Former   ← 分类引导的精细分割
        │
  seg_out (B, 3, D, H, W)

依赖：
  pip install mamba-ssm causal-conv1d   （Mamba 官方实现）
  如果没有安装，自动 fallback 到等效的 CNN block
"""

import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Mamba 依赖（可选，没装则 fallback）
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    warnings.warn(
        "[MambaTransUNet3D] mamba-ssm not found. "
        "Falling back to gated CNN block. "
        "Install with: pip install mamba-ssm causal-conv1d"
    )

# 复用 transunet3d.py 里已有的 CNN/Transformer 组件
from transunet3d import (
    StackedConvBlock,
    ViTBottleneck,
    Mask2FormerDecoder3D,
)


# ═══════════════════════════════════════════════════════════════
#  Part 1 ── Mamba3D Encoder
# ═══════════════════════════════════════════════════════════════

class GatedCNNBlock(nn.Module):
    """
    Mamba fallback：门控 CNN 块，近似模拟 SSM 的选择性机制。
    结构：Conv3d → SiLU gate × Conv3d value + residual
    """
    def __init__(self, ch: int, dropout: float = 0.0):
        super().__init__()
        self.norm  = nn.InstanceNorm3d(ch, affine=True)
        self.gate  = nn.Conv3d(ch, ch, 3, padding=1, bias=False)
        self.value = nn.Conv3d(ch, ch, 3, padding=1, bias=False)
        self.proj  = nn.Conv3d(ch, ch, 1, bias=False)
        self.drop  = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return x + self.drop(self.proj(F.silu(self.gate(h)) * self.value(h)))


class MambaBlock3D(nn.Module):
    """
    3D Mamba Block：
      1. LayerNorm
      2. Reshape (B,C,D,H,W) → (B, D*H*W, C)  把空间展平成序列
      3. Mamba SSM（或 fallback GatedCNN）
      4. Reshape 回 (B,C,D,H,W)
      5. Residual

    为什么展平？
      Mamba 是序列模型，需要 (B, L, d) 的输入。
      把 3D 体积展平成长度 L=D×H×W 的序列，
      SSM 的选择性扫描在这个序列上建模全局依赖。
    """
    def __init__(self, ch: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(ch)

        if MAMBA_AVAILABLE:
            self.ssm = Mamba(
                d_model=ch,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            # Fallback：用 GatedCNN 模拟（不需要展平）
            self.ssm = None
            self.fallback = GatedCNNBlock(ch, dropout)

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape

        if self.ssm is not None:
            # Mamba 路径：展平 → SSM → reshape 回来
            tokens = x.flatten(2).transpose(1, 2)          # (B, L, C)
            tokens = tokens + self.drop(self.ssm(self.norm(tokens)))
            return tokens.transpose(1, 2).reshape(B, C, D, H, W)
        else:
            # Fallback 路径：直接在 3D feature map 上操作
            return self.fallback(x)


class MambaStage3D(nn.Module):
    """
    Mamba Encoder 的一个 Stage：
      StackedConvBlock（调整通道数）→ N × MambaBlock3D（建模全局依赖）

    为什么先 Conv 再 Mamba？
      Conv 负责局部特征提取和通道数调整（类似 patch embedding），
      Mamba 负责在这个基础上建模长程依赖。
      这和原始 TransUNet 的 CNN→ViT 混合思路一致。
    """
    def __init__(self, in_ch: int, out_ch: int, num_mamba: int = 1,
                 d_state: int = 16, dropout: float = 0.0):
        super().__init__()
        self.output_channels = out_ch
        self.conv  = StackedConvBlock(in_ch, out_ch, num_convs=2, dropout=dropout)
        self.mamba = nn.Sequential(*[
            MambaBlock3D(out_ch, d_state=d_state, dropout=dropout)
            for _ in range(num_mamba)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mamba(self.conv(x))


class Mamba3DEncoder(nn.Module):
    """
    Mamba3D Encoder：替换原始 CNN Encoder。
    每个 stage 包含：Conv（局部）+ Mamba SSM（全局）+ MaxPool（下采样）

    保留 skip connections，格式和原 CNN Encoder 完全一致，
    因此 CNN Decoder 部分无需修改。
    """
    def __init__(self, in_ch: int, features: Tuple[int, ...],
                 num_mamba: int = 1, d_state: int = 16, dropout: float = 0.0):
        super().__init__()
        self.stages = nn.ModuleList()
        self.pools  = nn.ModuleList()

        ch = in_ch
        for i, f in enumerate(features):
            self.stages.append(MambaStage3D(ch, f, num_mamba, d_state, dropout))
            if i < len(features) - 1:
                self.pools.append(nn.MaxPool3d(2, 2))
            ch = f

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Returns (bottleneck, skips)"""
        skips = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.stages) - 1:
                skips.append(x)
                x = self.pools[i](x)
        return x, skips


# ═══════════════════════════════════════════════════════════════
#  Part 2 ── Atlas-aware PE
# ═══════════════════════════════════════════════════════════════

class AtlasAwarePE(nn.Module):
    """
    Atlas-aware 可学习位置编码。

    核心思路：
      标准 PE 只知道"我在第几个位置"。
      Atlas PE 知道"我在哪个脑区，这个脑区的解剖先验是什么"。
      两者通过可学习的融合门控加权组合。

    为什么用 Embedding 而不是直接用 Atlas 值？
      Atlas 是离散的脑区 ID（比如 AAL 的 116 个区），
      每个 ID 对应一个可学习的 pe_dim 维向量，
      模型可以学到"右侧额叶应该有更高的注意力权重"这类知识。

    Args:
      num_regions : 脑区数量（AAL=116, Brainnetome=246）
      pe_dim      : 和 ViT hidden_dim 一致
      max_seq     : 预分配的最大序列长度
    """
    def __init__(self, num_regions: int, pe_dim: int, max_seq: int = 4096):
        super().__init__()

        # 标准位置 PE（保留，作为基础）
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq, pe_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 每个脑区一个可学习 embedding
        # padding_idx=0 → background/unknown 区域贡献为 0
        self.region_embed = nn.Embedding(
            num_regions + 1, pe_dim, padding_idx=0
        )
        nn.init.trunc_normal_(self.region_embed.weight[1:], std=0.02)

        # 标量融合门控：学习 pos_PE 和 atlas_PE 各占多少
        # 初始化为 0 → sigmoid(0)=0.5 → 各占一半
        self.fusion_gate = nn.Parameter(torch.zeros(1))

    def _get_pos_pe(self, n: int) -> torch.Tensor:
        max_n = self.pos_embed.shape[1]
        if n == max_n:
            return self.pos_embed
        return F.interpolate(
            self.pos_embed.transpose(1, 2),
            size=n, mode="linear", align_corners=False
        ).transpose(1, 2)

    def forward(self, x: torch.Tensor, atlas: torch.Tensor) -> torch.Tensor:
        """
        x     : (B, C, d, h, w)   bottleneck feature map
        atlas : (B, 1, D, H, W)   atlas volume（原始分辨率，会被下采样）

        Returns pe : (B, N, pe_dim)   组合后的位置编码
        """
        B, C, d, h, w = x.shape
        N = d * h * w

        # 1. Atlas 下采样到 bottleneck 尺寸（nearest 保留离散 ID）
        atlas_down = F.interpolate(
            atlas.float(), size=(d, h, w), mode="nearest"
        ).long().squeeze(1).clamp(min=0)          # (B, d, h, w)

        # 2. 查找每个 voxel 对应脑区的 embedding
        region_ids = atlas_down.flatten(1)         # (B, N)
        region_pe  = self.region_embed(region_ids) # (B, N, pe_dim)

        # 3. 标准位置 PE
        pos_pe = self._get_pos_pe(N)               # (1, N, pe_dim)

        # 4. 门控融合
        gate = self.fusion_gate.sigmoid()          # ∈ (0, 1)
        pe   = gate * pos_pe + (1.0 - gate) * region_pe

        return pe                                  # (B, N, pe_dim)


class ViTBottleneckWithAtlas(nn.Module):
    """
    ViT Bottleneck，支持 Atlas-aware PE。
    当 atlas=None 时退化为标准 ViT Bottleneck（完全向后兼容）。
    """
    def __init__(self, in_ch: int, num_layers: int = 4, num_heads: int = 8,
                 mlp_ratio: float = 4.0, attn_drop: float = 0.0,
                 ffn_drop: float = 0.1, max_seq: int = 4096,
                 # Atlas PE 参数
                 use_atlas_pe: bool = False,
                 num_regions: int = 116):
        super().__init__()

        # 直接复用 transunet3d.py 的 ViTBottleneck，但重写 forward 以支持 atlas
        from transunet3d import TransformerEncoderLayer
        mlp_dim = int(in_ch * mlp_ratio)
        self.layers    = nn.ModuleList([
            TransformerEncoderLayer(in_ch, num_heads, mlp_dim, attn_drop, ffn_drop)
            for _ in range(num_layers)
        ])
        self.norm      = nn.LayerNorm(in_ch)

        self.use_atlas_pe = use_atlas_pe
        if use_atlas_pe:
            self.pe = AtlasAwarePE(num_regions, in_ch, max_seq)
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, max_seq, in_ch))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _std_pe(self, n: int) -> torch.Tensor:
        max_n = self.pos_embed.shape[1]
        if n == max_n:
            return self.pos_embed
        return F.interpolate(
            self.pos_embed.transpose(1, 2), size=n,
            mode="linear", align_corners=False
        ).transpose(1, 2)

    def forward(self, x: torch.Tensor,
                atlas: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, d, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)     # (B, N, C)

        if self.use_atlas_pe and atlas is not None:
            pe = self.pe(x, atlas)                 # (B, N, C)
        else:
            pe = self._std_pe(d * h * w)           # (1, N, C)

        tokens = tokens + pe
        for layer in self.layers:
            tokens = layer(tokens)
        tokens = self.norm(tokens)
        return tokens.transpose(1, 2).reshape(B, C, d, h, w)


# ═══════════════════════════════════════════════════════════════
#  Part 3 ── Multi-task Classification Heads
# ═══════════════════════════════════════════════════════════════

class ClinicalHead(nn.Module):
    """
    临床分类头：GAP → MLP → 二分类（MGMT/IDH）

    支持部分标注：has_label mask 为 False 的样本不计算 loss，
    但仍然做 forward（输出置信度用于反馈分割）。
    """
    def __init__(self, in_ch: int, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(in_ch, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),    # 二分类 logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, d, h, w)  →  (B, 1) logit"""
        return self.head(x)


# ═══════════════════════════════════════════════════════════════
#  Part 4 ── Classification-guided Mask2Former Decoder
# ═══════════════════════════════════════════════════════════════

class GuidedMask2FormerDecoder3D(Mask2FormerDecoder3D):
    """
    在 Mask2Former Decoder 基础上加入分类结果反馈。

    IDH/MGMT 的预测置信度用来调制 learnable queries：
      IDH 突变型（prob 高）→ 肿瘤边界更清晰 → query 增强
      IDH 野生型（prob 低）→ 肿瘤更弥散    → query 抑制

    实现：
      query = query * (1 + alpha * cls_feedback)
      alpha 是可学习的缩放系数
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 可学习的反馈强度（初始化为 0，训练初期不干扰）
        self.feedback_alpha = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        multi_scale_feats: List[torch.Tensor],
        mask_features: torch.Tensor,
        cls_feedback: Optional[torch.Tensor] = None,  # (B, 1) IDH/MGMT 置信度
    ) -> Dict:
        """
        cls_feedback : (B, 1) sigmoid 后的分类置信度
                       None → 和原版 Mask2Former 完全一样
        """
        assert len(multi_scale_feats) == self.num_layers
        B = multi_scale_feats[0].shape[0]

        # 初始化 queries
        q    = self.query_feat.weight.unsqueeze(0).expand(B, -1, -1)
        q_pe = self.query_pe.weight.unsqueeze(0).expand(B, -1, -1)

        # 分类反馈调制 queries
        # alpha 初始为 0 → 训练稳定后逐渐学到合适的调制强度
        if cls_feedback is not None:
            alpha  = self.feedback_alpha.tanh()              # ∈ (-1, 1)
            modulation = 1.0 + alpha * cls_feedback.unsqueeze(1)  # (B, 1, 1)
            q = q * modulation

        proj_feats  = [self.input_proj[i](f) for i, f in enumerate(multi_scale_feats)]
        pred_masks  = self.mask_head(q, mask_features)
        aux_outputs: List[Dict] = []

        for i in range(self.num_layers):
            feat = proj_feats[i]
            d_i, h_i, w_i = feat.shape[2:]
            kv = feat.flatten(2).transpose(1, 2)

            attn_mask = self._build_attn_mask(
                pred_masks, (d_i, h_i, w_i),
                self.cross_attn_layers[i].num_heads,
            )
            q = self.self_attn_layers[i](q + q_pe)
            q = self.cross_attn_layers[i](q + q_pe, kv, attn_mask)
            q = self.ffn_layers[i](q)

            pred_masks = self.mask_head(q, mask_features)
            aux_outputs.append({
                "pred_masks":  pred_masks,
                "pred_logits": self.class_head(q),
            })

        return {
            "pred_masks":  pred_masks,
            "pred_logits": self.class_head(q),
            "aux_outputs": aux_outputs,
        }


# ═══════════════════════════════════════════════════════════════
#  Part 5 ── MambaTransUNet3D 主模型
# ═══════════════════════════════════════════════════════════════

class MambaTransUNet3D(nn.Module):
    """
    完整的 MambaTransUNet3D 多任务模型。

    创新点整合：
      1. Mamba3D Encoder        建模长程依赖
      2. Atlas-aware PE         解剖先验位置编码
      3. Multi-task heads       分割 + MGMT + IDH（部分标注）
      4. Classification Feedback IDH 置信度调制 Mask2Former queries
      5. T1/T2 ratio            in_channels=5（第5通道）

    Forward 输入：
      x     : (B, 5, D, H, W)   4 modalities + T1/T2 ratio
      atlas : (B, 1, D, H, W)   atlas volume（可选）

    Forward 输出（dict）：
      "seg"       : (B, 3, D, H, W)   分割结果
      "mgmt"      : (B, 1)             MGMT 预测 logit
      "idh"       : (B, 1)             IDH 预测 logit
      "m2f"       : dict               Mask2Former 输出（训练时用）

    model.py 接口兼容：
      .bottleneck_channels
      .set_return_features(flag)
    """

    def __init__(
        self,
        in_channels: int = 5,                    # 4 modalities + T1/T2
        out_channels: int = 3,                   # TC / WT / ET
        features: Tuple[int, ...] = (32, 64, 128, 256, 320, 320),
        dropout: float = 0.0,
        num_conv_per_stage: int = 2,
        # Mamba
        num_mamba_per_stage: int = 1,
        mamba_d_state: int = 16,
        # ViT Bottleneck
        vit_layers: int = 4,
        vit_heads: int = 8,
        vit_mlp_ratio: float = 4.0,
        vit_attn_drop: float = 0.0,
        vit_ffn_drop: float = 0.1,
        # Atlas PE
        use_atlas_pe: bool = True,
        num_regions: int = 116,                  # AAL atlas
        # Mask2Former
        use_mask2former: bool = True,
        m2f_hidden_dim: int = 192,
        m2f_num_layers: Optional[int] = 3,       # C2F stage = 3
        m2f_num_heads: int = 8,
        m2f_mlp_ratio: float = 8.0,
        m2f_dropout: float = 0.1,
        # Multi-task classification
        cls_hidden: int = 128,
        cls_dropout: float = 0.3,
    ):
        super().__init__()
        assert len(features) >= 2

        self.out_channels        = out_channels
        self.bottleneck_channels = features[-1]
        self._return_features    = False
        self._return_m2f_dict    = False
        self.use_mask2former     = use_mask2former

        # ── Mamba3D Encoder ───────────────────────────────────
        self.encoder = Mamba3DEncoder(
            in_ch=in_channels,
            features=features,
            num_mamba=num_mamba_per_stage,
            d_state=mamba_d_state,
            dropout=dropout,
        )

        # ── ViT Bottleneck with Atlas PE ──────────────────────
        bn_ch = features[-1]
        heads = vit_heads
        while heads > 1 and bn_ch % heads != 0:
            heads //= 2

        self.transformer = ViTBottleneckWithAtlas(
            in_ch=bn_ch,
            num_layers=vit_layers,
            num_heads=heads,
            mlp_ratio=vit_mlp_ratio,
            attn_drop=vit_attn_drop,
            ffn_drop=vit_ffn_drop,
            use_atlas_pe=use_atlas_pe,
            num_regions=num_regions,
        )

        # ── Multi-task Classification Heads ───────────────────
        self.mgmt_head = ClinicalHead(bn_ch, cls_hidden, cls_dropout)
        self.idh_head  = ClinicalHead(bn_ch, cls_hidden, cls_dropout)

        # ── CNN Decoder ───────────────────────────────────────
        dec_features    = list(features[::-1])
        self.up_convs   = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in range(len(dec_features) - 1):
            in_f, out_f = dec_features[i], dec_features[i + 1]
            self.up_convs.append(
                nn.ConvTranspose3d(in_f, out_f, 2, stride=2, bias=False))
            self.dec_blocks.append(
                StackedConvBlock(out_f * 2, out_f, num_conv_per_stage, dropout))

        # ── Segmentation Head（Classic）──────────────────────
        self.seg_head = nn.Conv3d(dec_features[-1], out_channels, 1, bias=True)

        # ── Guided Mask2Former Decoder ────────────────────────
        if use_mask2former:
            n_dec     = m2f_num_layers if m2f_num_layers is not None else len(features) - 1
            mask_dim  = dec_features[-1]
            all_dec_ch = [dec_features[min(i, len(dec_features) - 1)]
                          for i in range(n_dec)]

            m2f_h = m2f_num_heads
            while m2f_h > 1 and m2f_hidden_dim % m2f_h != 0:
                m2f_h //= 2

            self.mask2former = GuidedMask2FormerDecoder3D(
                num_classes=out_channels,
                hidden_dim=m2f_hidden_dim,
                num_layers=n_dec,
                num_heads=m2f_h,
                mask_dim=mask_dim,
                feature_dims=all_dec_ch,
                mlp_ratio=m2f_mlp_ratio,
                dropout=m2f_dropout,
            )
            self.mask_feat_proj = (
                nn.Identity() if dec_features[-1] == mask_dim
                else nn.Conv3d(dec_features[-1], mask_dim, 1, bias=False)
            )

        self._init_weights()

    # ── Init ─────────────────────────────────────────────────

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, a=1e-2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── model.py 接口 ─────────────────────────────────────────

    def set_return_features(self, flag: bool):
        self._return_features = flag

    def set_return_m2f_dict(self, flag: bool):
        self._return_m2f_dict = flag

    # ── Forward ──────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,                          # (B, 5, D, H, W)
        atlas: Optional[torch.Tensor] = None,     # (B, 1, D, H, W)
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict:
          "seg"   : (B, 3, D, H, W)
          "mgmt"  : (B, 1)
          "idh"   : (B, 1)
          "m2f"   : dict  （仅 use_mask2former=True 且 _return_m2f_dict=True 时）
        """
        # ── Mamba3D Encoder ───────────────────────────────────
        bottleneck, skips = self.encoder(x)        # skips: list of (B,C,d,h,w)

        # ── 分类头（在 ViT 之前，用纯 CNN 特征，更局部）────────
        # 也可以放 ViT 之后，取决于你想用全局还是局部特征做分类
        mgmt_logit = self.mgmt_head(bottleneck)    # (B, 1)
        idh_logit  = self.idh_head(bottleneck)     # (B, 1)

        # ── ViT Bottleneck + Atlas PE ─────────────────────────
        x = self.transformer(bottleneck, atlas)

        # ── CNN Decoder ───────────────────────────────────────
        ds_feats: List[torch.Tensor] = [x]
        skips_rev = skips[::-1]
        for i, (up, dec) in enumerate(zip(self.up_convs, self.dec_blocks)):
            x = up(x)
            skip = skips_rev[i]
            if x.shape[2:] != skip.shape[2:]:
                pd = skip.shape[2] - x.shape[2]
                ph = skip.shape[3] - x.shape[3]
                pw = skip.shape[4] - x.shape[4]
                x  = F.pad(x, (0, pw, 0, ph, 0, pd))
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
            ds_feats.append(x)

        # ── Mask2Former（分类反馈） ────────────────────────────
        if self.use_mask2former:
            n          = self.mask2former.num_layers
            m2f_feats  = ds_feats[:n]
            mask_feat  = self.mask_feat_proj(ds_feats[-1])

            # IDH 置信度反馈（sigmoid 后作为调制信号）
            cls_feedback = idh_logit.detach().sigmoid()  # detach 避免梯度环路

            m2f_out = self.mask2former(m2f_feats, mask_feat, cls_feedback)
            seg_out = m2f_out["pred_masks"]
        else:
            seg_out  = self.seg_head(x)
            m2f_out  = None

        # ── 返回值组装 ─────────────────────────────────────────
        result: Dict[str, torch.Tensor] = {
            "seg":  seg_out,
            "mgmt": mgmt_logit,
            "idh":  idh_logit,
        }
        if m2f_out is not None and self._return_m2f_dict:
            result["m2f"] = m2f_out

        return result


# ═══════════════════════════════════════════════════════════════
#  Part 6 ── Uncertainty Weighting Loss
# ═══════════════════════════════════════════════════════════════

class UncertaintyWeightedLoss(nn.Module):
    """
    Kendall et al. (2018) 多任务不确定性加权损失。

    每个任务学一个 log_sigma（log 同方差不确定性）：
      L_total = Σ_i [ L_i / (2σ_i²) + log(σ_i) ]

    效果：
      - 任务 loss 大时，σ 自动变大，降低该任务权重
      - 任务 loss 小时，σ 自动变小，提高该任务权重
      - 对部分标注任务（IDH）天然适应：
        没有 IDH label 的 batch 里 loss_idh=0，σ_idh 会自动调大

    Args:
      task_names : list of task name strings, e.g. ["seg", "mgmt", "idh"]
    """
    def __init__(self, task_names: List[str]):
        super().__init__()
        self.task_names = task_names
        # 初始化为 0 → σ = exp(0) = 1 → 各任务初始权重相等
        self.log_sigmas = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1))
            for name in task_names
        })

    def forward(
        self,
        losses: Dict[str, torch.Tensor],  # {"seg": tensor, "mgmt": tensor, ...}
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns:
          total_loss : scalar tensor
          weights    : dict of effective weights for logging
        """
        total = torch.zeros(1, device=next(iter(losses.values())).device)
        weights = {}

        for name in self.task_names:
            if name not in losses:
                continue
            loss     = losses[name]
            sigma    = self.log_sigmas[name].exp()           # σ_i
            weighted = loss / (2.0 * sigma ** 2) + self.log_sigmas[name]
            total    = total + weighted
            weights[name] = (1.0 / (2.0 * sigma ** 2)).item()

        return total.squeeze(), weights


# ═══════════════════════════════════════════════════════════════
#  Smoke Test
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    print(f"Mamba available: {MAMBA_AVAILABLE}\n")

    feats = (16, 32, 64, 128)
    B = 2

    x     = torch.randn(B, 5, 64, 64, 64, device=device)      # 5 channels
    atlas = torch.randint(0, 116, (B, 1, 64, 64, 64), device=device)

    print("=" * 60)
    print("Test 1: Full model forward")
    print("=" * 60)
    model = MambaTransUNet3D(
        in_channels=5,
        out_channels=3,
        features=feats,
        use_atlas_pe=True,
        use_mask2former=True,
        m2f_num_layers=3,
    ).to(device)

    with torch.no_grad():
        out = model(x, atlas)

    assert out["seg"].shape  == (B, 3, 64, 64, 64), out["seg"].shape
    assert out["mgmt"].shape == (B, 1),              out["mgmt"].shape
    assert out["idh"].shape  == (B, 1),              out["idh"].shape
    print(f"  seg  : {out['seg'].shape}   OK")
    print(f"  mgmt : {out['mgmt'].shape}  OK")
    print(f"  idh  : {out['idh'].shape}   OK\n")

    print("=" * 60)
    print("Test 2: Uncertainty Weighted Loss")
    print("=" * 60)
    loss_fn = UncertaintyWeightedLoss(["seg", "mgmt", "idh"]).to(device)

    # 模拟部分标注：batch 里只有 1 个样本有 IDH label
    has_idh = torch.tensor([True, False], device=device)

    seg_loss  = torch.tensor(0.5, device=device)
    mgmt_loss = torch.tensor(0.3, device=device)
    idh_loss  = torch.tensor(0.2, device=device) if has_idh.any() else torch.tensor(0.0, device=device)

    total, weights = loss_fn({"seg": seg_loss, "mgmt": mgmt_loss, "idh": idh_loss})
    print(f"  Total loss : {total.item():.4f}")
    print(f"  Weights    : { {k: f'{v:.3f}' for k,v in weights.items()} }\n")

    total = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total:,}")
