"""
BraTS2021 Dataset Module
========================
Handles data loading, preprocessing, and splitting for BraTS2021 brain tumor segmentation.

支持两种数据来源方式:
  1. 扫描目录: 自动扫描 data_root 下所有 BraTS subject 文件夹
  2. CSV 文件: 根据 CSV 中的 subject ID 筛选数据 (推荐)

CSV 格式 (两列):
  BraTS2021_00000, 0.85
  BraTS2021_00001, 0.72
  ...
  - 第一列: subject ID (对应文件夹名)
  - 第二列: 得分 (float), 会作为 "score" 字段返回到每个 batch 中

BraTS2021 provides 4 MRI modalities per subject:
  - T1, T1ce (contrast-enhanced), T2, FLAIR
Label classes:
  - 0: Background
  - 1: Necrotic / Non-Enhancing Tumor (NCR/NET)
  - 2: Peritumoral Edema (ED)
  - 4: Enhancing Tumor (ET)  (注意: 官方标签中没有 class 3)

We convert labels to 3-channel:
  - Channel 0: TC  (Tumor Core)      = label 1 + label 4
  - Channel 1: WT  (Whole Tumor)     = label 1 + label 2 + label 4
  - Channel 2: ET  (Enhancing Tumor) = label 4
"""

import os
import csv
import glob
from typing import List, Dict, Tuple, Optional

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    NormalizeIntensityd,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    ConvertToMultiChannelBasedOnBratsClassesd,
    EnsureTyped,
    Resized,
)
from monai.data import Dataset, CacheDataset, DataLoader, decollate_batch
from sklearn.model_selection import train_test_split


# ══════════════════════════════════════════════════════════════
#  CSV 解析
# ══════════════════════════════════════════════════════════════

def parse_csv(csv_path: str) -> Dict[str, float]:
    """
    解析 CSV 文件, 返回 {subject_id: score} 的字典.

    支持的 CSV 格式:
      - 有表头:   BraTS2021_ID, score  (自动识别, 跳过表头)
      - 无表头:   BraTS2021_00000, 0.85 (直接解析)
      - 分隔符:   逗号 or 其他 (csv.Sniffer 自动检测)

    Args:
        csv_path: CSV 文件路径

    Returns:
        Dict mapping subject_id -> score (float)
    """
    id_to_score = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        # 自动检测分隔符
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = "excel"  # fallback: comma-separated

        reader = csv.reader(f, dialect)

        # 检测是否有表头: 如果第二列不能转成 float, 就当作表头跳过
        first_row = next(reader)
        try:
            score = float(first_row[1].strip())
            # 能转成 float → 无表头, 这一行就是数据
            subj_id = first_row[0].strip()
            id_to_score[subj_id] = score
        except (ValueError, IndexError):
            # 不能转 → 这是表头, 跳过
            pass

        for row in reader:
            if len(row) < 2:
                continue
            subj_id = row[0].strip()
            print(subj_id)
            try:
                score = float(row[1].strip())
            except ValueError:
                print(f"[Dataset] Warning: skipping invalid row: {row}")
                continue
            id_to_score[subj_id] = score

    print(f"[Dataset] Parsed CSV: {len(id_to_score)} subjects from '{csv_path}'")
    return id_to_score


# ══════════════════════════════════════════════════════════════
#  构建文件列表
# ══════════════════════════════════════════════════════════════

def get_brats_file_list(
    data_root: str,
    csv_path: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> List[Dict]:
    """
    构建数据列表.

    两种模式:
      1. csv_path=None  → 扫描 data_root 下所有 subject, score 默认为 0.0
      2. csv_path=有效路径 → 只使用 CSV 中列出的 subject, 并带上 score

    每个样本返回的 dict:
      {
          "image":      [t1_path, t1ce_path, t2_path, flair_path],
          "label":      seg_path,
          "score":      float,    ← CSV 中的得分 (无 CSV 则默认 0.0)
          "subject_id": str,      ← 方便 debug 追踪
      }

    注意: "score" 和 "subject_id" 是非图像字段, MONAI 的图像 transforms
    不会处理它们, 但它们会原样保留在 batch dict 中, 训练时可以直接取用:
        batch_data["score"]      → tensor of shape (B,)
        batch_data["subject_id"] → list of strings

    Args:
        data_root: BraTS2021 数据根目录
        csv_path: CSV 文件路径 (可选)
        max_samples: 限制样本数 (调试用)

    Returns:
        List of data dicts
    """
    # ── 确定要加载哪些 subjects ───────────────────────────────
    if csv_path is not None:
        id_to_score = parse_csv(csv_path)
        keys = list(id_to_score.keys())
        subject_ids = ["BraTS2021_" + key for key in keys]
        print(f"[Dataset] Loading {len(subject_ids)} subjects from CSV '{csv_path}'")
        print(f"[Test] Sample subjects from CSV: {subject_ids[:5]} ...")
    else:
        # 扫描目录获取所有 subjects
        subject_dirs = sorted(glob.glob(os.path.join(data_root, "BraTS2021_*")))
        if not subject_dirs:
            subject_dirs = sorted(glob.glob(os.path.join(data_root, "BraTS*")))
        if not subject_dirs:
            raise FileNotFoundError(
                f"No BraTS subject directories found in '{data_root}'. "
                f"Please check that your data follows the expected naming convention."
            )
        subject_ids = [os.path.basename(d) for d in subject_dirs]
        id_to_score = {}  # 无 CSV 时 score 默认 0.0

    # ── 根据 subject ID 构建文件路径 ─────────────────────────
    data_list = []
    missing_count = 0

    for subj_id in subject_ids:
        subj_dir = os.path.join(data_root, subj_id)

        if not os.path.isdir(subj_dir):
            missing_count += 1
            continue

        t1    = os.path.join(subj_dir, f"{subj_id}_t1.nii.gz")
        t1ce  = os.path.join(subj_dir, f"{subj_id}_t1ce.nii.gz")
        t2    = os.path.join(subj_dir, f"{subj_id}_t2.nii.gz")
        flair = os.path.join(subj_dir, f"{subj_id}_flair.nii.gz")
        seg   = os.path.join(subj_dir, f"{subj_id}_seg.nii.gz")

        modalities = [t1, t1ce, t2, flair]
        if all(os.path.isfile(f) for f in modalities) and os.path.isfile(seg):
            data_list.append({
                "image": modalities,
                "label": seg,
                "score": id_to_score.get(subj_id, 0.0),
                "subject_id": subj_id,
            })
        else:
            missing_count += 1

    if missing_count > 0:
        print(f"[Dataset] Warning: {missing_count} subjects not found or incomplete on disk")

    if max_samples is not None and max_samples > 0:
        data_list = data_list[:max_samples]

    print(f"[Dataset] Found {len(data_list)} valid subjects"
          + (f" (filtered by CSV)" if csv_path else ""))
    return data_list


def split_dataset(
    data_list: List[Dict],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split data_list into training and validation sets.

    Args:
        data_list: Full list of data dicts.
        val_ratio: Fraction of data to use for validation.
        seed: Random seed for reproducibility.

    Returns:
        (train_list, val_list)
    """
    train_list, val_list = train_test_split(
        data_list, test_size=val_ratio, random_state=seed
    )
    print(f"[Dataset] Split -> Train: {len(train_list)}, Val: {len(val_list)}")
    return train_list, val_list


def get_train_transforms(roi_size: Tuple[int, int, int] = (128, 128, 128)) -> Compose:
    """
    Build MONAI training transforms pipeline.

    Pipeline:
        1. Load NIfTI images
        2. Ensure channel-first format
        3. Convert BraTS labels to multi-channel (TC, WT, ET)
        4. Resample to uniform spacing
        5. Normalize intensity per-channel (z-score)
        6. Crop foreground to remove background
        7. Random crop around positive/negative voxels
        8. Random augmentations (flip, rotate)
        9. Convert to PyTorch tensors
    """
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        EnsureChannelFirstd(keys="label"),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=roi_size,
            pos=1,
            neg=1,
            num_samples=1,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        EnsureTyped(keys=["image", "label"]),
    ])


def get_val_transforms() -> Compose:
    """
    Build MONAI validation transforms pipeline (no augmentation).
    """
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        EnsureChannelFirstd(keys="label"),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        EnsureTyped(keys=["image", "label"]),
    ])


def get_dataloaders(
    data_root: str,
    train_csv: Optional[str] = None,
    val_csv: Optional[str] = None,
    batch_size: int = 2,
    val_ratio: float = 0.2,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
    roi_size: Tuple[int, int, int] = (128, 128, 128),
    cache_rate: float = 0.0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    End-to-end helper: build datasets & dataloaders.

    三种模式:
      1. train_csv + val_csv 都提供  → 各自从 CSV 加载, 不做自动划分
      2. 只提供 train_csv            → train 从 CSV, val 从 train 里按比例划分
      3. 都不提供                     → 扫描目录, 按比例划分

    Args:
        data_root: Path to BraTS2021 data.
        train_csv: Path to training CSV (subject_id, score).
        val_csv: Path to validation CSV (subject_id, score).
        batch_size: Batch size for training loader.
        val_ratio: Fraction for validation (only used when val_csv is None).
        num_workers: DataLoader workers.
        max_samples: Limit total samples per split (for debugging).
        roi_size: Random crop size for training.
        cache_rate: Fraction of data to cache in memory.
        seed: Random seed.

    Returns:
        (train_loader, val_loader)
    """
    if train_csv and val_csv:
        # ── 模式 1: 两个 CSV 分别指定 train / val ────────────
        train_list = get_brats_file_list(data_root, csv_path=train_csv, max_samples=max_samples)
        val_list   = get_brats_file_list(data_root, csv_path=val_csv,   max_samples=max_samples)
        print(f"[Dataset] Train: {len(train_list)}, Val: {len(val_list)} (from separate CSVs)")
    else:
        # ── 模式 2/3: 单个来源, 自动划分 ─────────────────────
        data_list = get_brats_file_list(data_root, csv_path=train_csv, max_samples=max_samples)
        train_list, val_list = split_dataset(data_list, val_ratio=val_ratio, seed=seed)

    train_transforms = get_train_transforms(roi_size=roi_size)
    val_transforms = get_val_transforms()

    if cache_rate > 0:
        train_ds = CacheDataset(data=train_list, transform=train_transforms, cache_rate=cache_rate, num_workers=num_workers)
        val_ds   = CacheDataset(data=val_list,   transform=val_transforms,   cache_rate=cache_rate, num_workers=num_workers)
    else:
        train_ds = Dataset(data=train_list, transform=train_transforms)
        val_ds   = Dataset(data=val_list,   transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1,          shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader