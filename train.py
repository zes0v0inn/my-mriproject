#!/usr/bin/env python3
"""
BraTS2021 Training Script
=========================
Full-featured training loop with:
  - CLI arguments for all hyperparameters
  - Dice Loss + optional CE Loss
  - Dice & IoU (Jaccard) metrics per class (TC / WT / ET)
  - Learning rate scheduler
  - Model checkpointing (best + last)
  - TensorBoard logging

Usage examples:
  # Basic UNet training
  python train.py --data_root /path/to/BraTS2021 --model unet --epochs 100

  # Small-scale debug run on CPU
  python train.py --data_root /path/to/BraTS2021 --model unet --max_samples 10 \
                  --epochs 5 --batch_size 1 --no_cuda

  # Attention UNet on GPU with custom params
  python train.py --data_root /path/to/BraTS2021 --model attention_unet \
                  --features 32 64 128 256 --lr 1e-4 --batch_size 2 --epochs 200

  # UNETR with mixed precision
  python train.py --data_root /path/to/BraTS2021 --model unetr --amp
"""

import os
import sys
import time
import argparse
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from monai.losses import DiceLoss
from monai.metrics import DiceMetric, MeanIoU
from monai.inferers import sliding_window_inference
from monai.transforms import Activations, AsDiscrete
from monai.data import decollate_batch
from monai.utils import set_determinism

# Local imports
from dataset import get_dataloaders
from model import build_model


# ──────────────────────────────────────────────────────────────
# CLI Argument Parser
# ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="BraTS2021 Brain Tumor Segmentation Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ──────────────────────────────────────────────────
    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--data_root", type=str, required=True,
                            help="Path to BraTS2021 dataset root directory")
    data_group.add_argument("--csv_path", type=str, default=None,
                            help="Path to CSV file (subject_id, score). "
                                 "If provided, only subjects listed in CSV are used, "
                                 "and score is included in each batch.")
    data_group.add_argument("--max_samples", type=int, default=None,
                            help="Limit number of samples (for debugging)")
    data_group.add_argument("--val_ratio", type=float, default=0.2,
                            help="Fraction of data for validation")
    data_group.add_argument("--cache_rate", type=float, default=0.0,
                            help="MONAI CacheDataset cache rate (0=no cache, 1=full)")
    data_group.add_argument("--num_workers", type=int, default=4,
                            help="DataLoader worker processes")

    # ── Model ─────────────────────────────────────────────────
    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--model", type=str, default="basic_unet",
                             choices=["basic_unet", "monai_unet", "attention_unet", "unetr", "swin_unetr"],
                             help="Model architecture: basic_unet(手写) / monai_unet(MONAI内置) / attention_unet / unetr / swin_unetr")
    model_group.add_argument("--in_channels", type=int, default=4,
                             help="Number of input channels (4 for BraTS)")
    model_group.add_argument("--out_channels", type=int, default=3,
                             help="Number of output channels (3: TC, WT, ET)")
    model_group.add_argument("--features", type=int, nargs="+", default=[32, 64, 128, 256, 512],
                             help="Channel sizes for UNet encoder levels")
    model_group.add_argument("--dropout", type=float, default=0.1,
                             help="Dropout probability")
    model_group.add_argument("--with_classifier", action="store_true",
                             help="Attach auxiliary classification head (multi-task)")
    model_group.add_argument("--num_classes", type=int, default=1,
                             help="Classification head output dim (1 for score regression)")
    model_group.add_argument("--cls_dropout", type=float, default=0.3,
                             help="Classification head dropout")

    # ── Training ──────────────────────────────────────────────
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--epochs", type=int, default=100,
                             help="Number of training epochs")
    train_group.add_argument("--batch_size", type=int, default=2,
                             help="Training batch size")
    train_group.add_argument("--lr", type=float, default=1e-4,
                             help="Initial learning rate")
    train_group.add_argument("--weight_decay", type=float, default=1e-5,
                             help="AdamW weight decay")
    train_group.add_argument("--scheduler", type=str, default="cosine",
                             choices=["cosine", "step", "none"],
                             help="Learning rate scheduler type")
    train_group.add_argument("--warmup_epochs", type=int, default=5,
                             help="Linear warmup epochs (for cosine scheduler)")

    # ── Loss ──────────────────────────────────────────────────
    loss_group = parser.add_argument_group("Loss")
    loss_group.add_argument("--loss", type=str, default="dice",
                            choices=["dice", "dice_ce"],
                            help="Loss function: 'dice' or 'dice_ce' (Dice + CrossEntropy)")
    loss_group.add_argument("--dice_weight", type=float, default=1.0,
                            help="Weight for Dice loss in dice_ce mode")
    loss_group.add_argument("--ce_weight", type=float, default=1.0,
                            help="Weight for CE loss in dice_ce mode")
    loss_group.add_argument("--cls_loss_weight", type=float, default=0.1,
                            help="Weight for classification loss when --with_classifier is used")

    # ── Inference ─────────────────────────────────────────────
    infer_group = parser.add_argument_group("Inference")
    infer_group.add_argument("--roi_size", type=int, nargs=3, default=[128, 128, 128],
                             help="Patch size for training crop and sliding window inference")
    infer_group.add_argument("--sw_batch_size", type=int, default=4,
                             help="Batch size for sliding window inference during validation")
    infer_group.add_argument("--overlap", type=float, default=0.5,
                             help="Overlap ratio for sliding window inference")

    # ── Hardware ──────────────────────────────────────────────
    hw_group = parser.add_argument_group("Hardware")
    hw_group.add_argument("--no_cuda", action="store_true",
                          help="Disable CUDA (force CPU training)")
    hw_group.add_argument("--gpu_id", type=int, default=0,
                          help="GPU device ID to use")
    hw_group.add_argument("--amp", action="store_true",
                          help="Enable automatic mixed precision (FP16)")

    # ── Logging / Checkpointing ───────────────────────────────
    log_group = parser.add_argument_group("Logging")
    log_group.add_argument("--output_dir", type=str, default="./runs",
                           help="Directory for checkpoints and logs")
    log_group.add_argument("--exp_name", type=str, default=None,
                           help="Experiment name (auto-generated if not set)")
    log_group.add_argument("--save_interval", type=int, default=10,
                           help="Save checkpoint every N epochs")
    log_group.add_argument("--val_interval", type=int, default=1,
                           help="Run validation every N epochs")
    log_group.add_argument("--seed", type=int, default=42,
                           help="Random seed for reproducibility")

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────
# Loss function builder
# ──────────────────────────────────────────────────────────────

class DiceCELoss(nn.Module):
    """Combined Dice + Binary Cross-Entropy loss."""
    def __init__(self, dice_weight=1.0, ce_weight=1.0):
        super().__init__()
        self.dice_loss = DiceLoss(sigmoid=True, smooth_nr=0, smooth_dr=1e-5, squared_pred=True)
        self.ce_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, pred, target):
        return self.dice_weight * self.dice_loss(pred, target) + \
               self.ce_weight * self.ce_loss(pred, target.float())


def build_loss(args):
    if args.loss == "dice":
        return DiceLoss(sigmoid=True, smooth_nr=0, smooth_dr=1e-5, squared_pred=True)
    elif args.loss == "dice_ce":
        return DiceCELoss(dice_weight=args.dice_weight, ce_weight=args.ce_weight)
    else:
        raise ValueError(f"Unknown loss: {args.loss}")


# ──────────────────────────────────────────────────────────────
# Training & Validation loops
# ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler, use_amp,
                    with_classifier=False, cls_loss_weight=0.1):
    """Run one training epoch. Returns dict of losses."""
    model.train()
    epoch_seg_loss = 0.0
    epoch_cls_loss = 0.0
    step = 0

    cls_loss_fn = nn.MSELoss() if with_classifier else None

    for batch_data in loader:
        step += 1
        inputs = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            if with_classifier:
                seg_out, cls_out = model(inputs)
                seg_loss = loss_fn(seg_out, labels)

                # score 从 CSV 来, 用于分类/回归辅助任务
                scores = batch_data["score"].float().to(device).unsqueeze(1)  # (B, 1)
                cls_loss = cls_loss_fn(cls_out, scores)

                loss = seg_loss + cls_loss_weight * cls_loss
                epoch_cls_loss += cls_loss.item()
            else:
                seg_out = model(inputs)
                loss = loss_fn(seg_out, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        epoch_seg_loss += loss.item()

    result = {"total_loss": epoch_seg_loss / step}
    if with_classifier:
        result["cls_loss"] = epoch_cls_loss / step
    return result


@torch.no_grad()
def validate(model, loader, device, roi_size, sw_batch_size, overlap, use_amp,
             with_classifier=False):
    """
    Run validation with sliding window inference.
    Returns per-class Dice and IoU scores, and optionally classification MAE.
    """
    model.eval()

    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
    iou_metric  = MeanIoU(include_background=True, reduction="mean_batch")

    post_sigmoid = Activations(sigmoid=True)
    post_pred    = AsDiscrete(threshold=0.5)

    cls_errors = []

    for batch_data in loader:
        inputs = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            if with_classifier:
                # sliding_window_inference 只给 model 喂 image, 但 SegWithClassifier
                # 返回 tuple, 所以需要一个小 wrapper 只取分割部分做滑窗
                # 分类部分我们单独跑一次完整 forward
                seg_out, cls_out = model(inputs)

                # 分类 MAE
                scores = batch_data["score"].float().to(device).unsqueeze(1)
                cls_errors.append((cls_out - scores).abs().mean().item())

                # 对于分割, 用 sliding window 获得更准确的结果
                def seg_only(x):
                    s, _ = model(x)
                    return s

                outputs = sliding_window_inference(
                    inputs, roi_size=roi_size, sw_batch_size=sw_batch_size,
                    predictor=seg_only, overlap=overlap,
                )
            else:
                outputs = sliding_window_inference(
                    inputs, roi_size=roi_size, sw_batch_size=sw_batch_size,
                    predictor=model, overlap=overlap,
                )

        outputs_list = decollate_batch(outputs)
        labels_list  = decollate_batch(labels)
        outputs_post = [post_pred(post_sigmoid(x)) for x in outputs_list]

        dice_metric(y_pred=outputs_post, y=labels_list)
        iou_metric(y_pred=outputs_post, y=labels_list)

    dice_scores = dice_metric.aggregate().cpu().numpy()
    iou_scores  = iou_metric.aggregate().cpu().numpy()
    dice_metric.reset()
    iou_metric.reset()

    result = {"dice": dice_scores, "iou": iou_scores}
    if with_classifier and cls_errors:
        result["cls_mae"] = sum(cls_errors) / len(cls_errors)

    return result


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Reproducibility ───────────────────────────────────────
    set_determinism(seed=args.seed)

    # ── Device ────────────────────────────────────────────────
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
        print("[Hardware] Using CPU")
    else:
        device = torch.device(f"cuda:{args.gpu_id}")
        print(f"[Hardware] Using GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")

    use_amp = args.amp and device.type == "cuda"
    if use_amp:
        print("[Hardware] Automatic Mixed Precision (AMP) enabled")

    # ── Experiment directory ──────────────────────────────────
    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.exp_name = f"{args.model}_{timestamp}"

    exp_dir = os.path.join(args.output_dir, args.exp_name)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(exp_dir, "tb_logs"))

    # Print full config
    print("\n" + "=" * 60)
    print("  BraTS2021 Segmentation Training")
    print("=" * 60)
    for group_name in ["Data", "Model", "Training", "Loss", "Inference", "Hardware", "Logging"]:
        print(f"\n  [{group_name}]")
        # Ugly but functional: just dump all args
    for k, v in sorted(vars(args).items()):
        print(f"    {k:20s}: {v}")
    print("=" * 60 + "\n")

    # ── Data ──────────────────────────────────────────────────
    roi_size = tuple(args.roi_size)

    train_loader, val_loader = get_dataloaders(
        data_root=args.data_root,
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        roi_size=roi_size,
        cache_rate=args.cache_rate,
        seed=args.seed,
    )

    # ── Model ─────────────────────────────────────────────────
    model = build_model(
        model_name=args.model,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        roi_size=roi_size,
        features=tuple(args.features),
        dropout=args.dropout,
        with_classifier=args.with_classifier,
        num_classes=args.num_classes,
        cls_dropout=args.cls_dropout,
    ).to(device)

    # ── Optimizer & Scheduler ─────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        scheduler = None

    # ── Loss ──────────────────────────────────────────────────
    loss_fn = build_loss(args)

    # ── AMP scaler ────────────────────────────────────────────
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ── Training loop ─────────────────────────────────────────
    best_mean_dice = 0.0
    class_names = ["TC", "WT", "ET"]

    print("Starting training...\n")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_result = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, scaler, use_amp,
            with_classifier=args.with_classifier,
            cls_loss_weight=args.cls_loss_weight,
        )
        train_loss = train_result["total_loss"]

        # Scheduler step
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        # Log
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/lr", current_lr, epoch)
        if "cls_loss" in train_result:
            writer.add_scalar("train/cls_loss", train_result["cls_loss"], epoch)

        elapsed = time.time() - epoch_start
        loss_str = f"Loss: {train_loss:.4f}"
        if "cls_loss" in train_result:
            loss_str += f" (cls: {train_result['cls_loss']:.4f})"
        print(f"Epoch [{epoch:3d}/{args.epochs}]  "
              f"{loss_str}  "
              f"LR: {current_lr:.2e}  "
              f"Time: {elapsed:.1f}s", end="")

        # Validate
        if epoch % args.val_interval == 0:
            val_start = time.time()
            val_result = validate(
                model, val_loader, device,
                roi_size=roi_size,
                sw_batch_size=args.sw_batch_size,
                overlap=args.overlap,
                use_amp=use_amp,
                with_classifier=args.with_classifier,
            )
            val_time = time.time() - val_start

            dice_scores = val_result["dice"]
            iou_scores  = val_result["iou"]
            mean_dice = dice_scores.mean()
            mean_iou  = iou_scores.mean()

            # TensorBoard logging
            writer.add_scalar("val/mean_dice", mean_dice, epoch)
            writer.add_scalar("val/mean_iou",  mean_iou,  epoch)
            for i, cname in enumerate(class_names):
                writer.add_scalar(f"val/dice_{cname}", dice_scores[i], epoch)
                writer.add_scalar(f"val/iou_{cname}",  iou_scores[i],  epoch)

            val_extra = ""
            if "cls_mae" in val_result:
                writer.add_scalar("val/cls_mae", val_result["cls_mae"], epoch)
                val_extra = f"  ClsMAE: {val_result['cls_mae']:.4f}"

            print(f"  |  Val ({val_time:.1f}s)  "
                  f"Dice: {mean_dice:.4f} [{', '.join(f'{cname}={dice_scores[i]:.4f}' for i, cname in enumerate(class_names))}]  "
                  f"IoU: {mean_iou:.4f} [{', '.join(f'{cname}={iou_scores[i]:.4f}' for i, cname in enumerate(class_names))}]"
                  f"{val_extra}", end="")

            # Save best model
            if mean_dice > best_mean_dice:
                best_mean_dice = mean_dice
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_mean_dice": best_mean_dice,
                    "args": vars(args),
                }, os.path.join(ckpt_dir, "best_model.pth"))
                print("  ★ New best!", end="")

        print()  # newline

        # Save periodic checkpoint
        if epoch % args.save_interval == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_mean_dice": best_mean_dice,
                "args": vars(args),
            }, os.path.join(ckpt_dir, f"checkpoint_epoch{epoch:04d}.pth"))

    # Save final model
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_mean_dice": best_mean_dice,
        "args": vars(args),
    }, os.path.join(ckpt_dir, "last_model.pth"))

    writer.close()

    print("\n" + "=" * 60)
    print(f"  Training complete!  Best Mean Dice: {best_mean_dice:.4f}")
    print(f"  Checkpoints saved to: {ckpt_dir}")
    print(f"  TensorBoard logs:     tensorboard --logdir {exp_dir}/tb_logs")
    print("=" * 60)


if __name__ == "__main__":
    main()