%%writefile /kaggle/working/submission/train.py
"""
Training script for Off-Road Segmentation using DINOv2
Usage: python train.py --data_dir <path> --save_dir <path> [--epochs N] [--checkpoint <path>]
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import os
import argparse

from config import *
from model import build_model, DeepSegmentationHead
from dataset import SegmentationDataset
from losses import FocalLoss, DiceLoss


def compute_iou(pred, target, num_classes):
    pred = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)
    iou_per_class = []
    for c in range(num_classes):
        pi, ti = pred == c, target == c
        inter = (pi & ti).sum().float()
        union = (pi | ti).sum().float()
        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append((inter / union).cpu().numpy())
    return np.nanmean(iou_per_class), iou_per_class


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # Data
    train_dataset = SegmentationDataset(args.data_dir, augment=True)
    eval_dataset = SegmentationDataset(args.data_dir, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, pin_memory=True)

    print(f"🎯 Training: {len(train_dataset)} images | {args.epochs} epochs")

    # Model
    backbone, classifier = build_model(device, args.checkpoint)

    # Freeze backbone except last N blocks
    for param in backbone.parameters():
        param.requires_grad = False
    for block in backbone.blocks[-UNFREEZE_LAST_N_BLOCKS:]:
        for param in block.parameters():
            param.requires_grad = True

    backbone_params = [p for p in backbone.parameters() if p.requires_grad]
    n_bb = sum(p.numel() for p in backbone_params)
    n_head = sum(p.numel() for p in classifier.parameters())
    print(f"  Trainable params: backbone={n_bb:,} head={n_head:,}")

    # Optimizer
    optimizer = torch.optim.AdamW([
        {'params': classifier.parameters(), 'lr': LEARNING_RATE},
        {'params': backbone_params, 'lr': BACKBONE_LR}
    ], weight_decay=WEIGHT_DECAY)

    # Loss
    weights = torch.tensor(CLASS_WEIGHTS).to(device)
    focal_loss = FocalLoss(weight=weights, gamma=FOCAL_GAMMA)
    dice_loss = DiceLoss(N_CLASSES, ignore_classes=DICE_IGNORE_CLASSES)
    scaler = GradScaler('cuda')

    # Scheduler: warmup + cosine
    def get_lr_scale(epoch):
        if epoch <= WARMUP_EPOCHS:
            return epoch / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / (args.epochs - WARMUP_EPOCHS)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_scale)

    # Training loop
    best_iou = 0.0
    print(f"\n{'='*70}")
    print(f" Ep  TrLoss  EvLoss  TrIoU  EvIoU   LR")
    print(f"{'='*70}")

    for epoch in range(1, args.epochs + 1):
        backbone.train(); classifier.train()
        train_loss, train_iou_sum, train_count = 0, 0, 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.squeeze(1).long().to(device)
            optimizer.zero_grad()

            with autocast('cuda'):
                tokens = backbone.forward_features(imgs)["x_norm_patchtokens"]
                logits = classifier(tokens)
                out = F.interpolate(logits, size=imgs.shape[2:],
                                    mode="bilinear", align_corners=False)
                loss = focal_loss(out, labels) + 0.5 * dice_loss(out, labels)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(
                list(classifier.parameters()) + backbone_params, GRAD_CLIP)
            scaler.step(optimizer); scaler.update()

            train_loss += loss.item()
            iou, _ = compute_iou(out.detach(), labels, N_CLASSES)
            train_iou_sum += iou; train_count += 1

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        # Evaluate
        backbone.eval(); classifier.eval()
        eval_loss, eval_iou_sum, eval_count = 0, 0, 0

        with torch.no_grad():
            for imgs, labels in eval_loader:
                imgs = imgs.to(device)
                labels = labels.squeeze(1).long().to(device)
                with autocast('cuda'):
                    tokens = backbone.forward_features(imgs)["x_norm_patchtokens"]
                    logits = classifier(tokens)
                    out = F.interpolate(logits, size=imgs.shape[2:],
                                        mode="bilinear", align_corners=False)
                    loss = focal_loss(out, labels) + 0.5 * dice_loss(out, labels)
                eval_loss += loss.item()
                iou, _ = compute_iou(out, labels, N_CLASSES)
                eval_iou_sum += iou; eval_count += 1

        avg_tl = train_loss / train_count
        avg_el = eval_loss / eval_count
        avg_ti = train_iou_sum / train_count
        avg_ei = eval_iou_sum / eval_count

        is_best = avg_ei > best_iou
        if is_best:
            best_iou = avg_ei
            bb_ft = {n: p.data.clone() for n, p in backbone.named_parameters()
                     if p.requires_grad}
            torch.save({
                'model_state_dict': classifier.state_dict(),
                'backbone_finetuned': bb_ft,
                'best_iou': best_iou,
                'epoch': epoch,
            }, os.path.join(args.save_dir, 'checkpoint_best.pth'))

        marker = " ⭐" if is_best else ""
        print(f" {epoch:2d}  {avg_tl:.4f}  {avg_el:.4f}  {avg_ti:.4f}  "
              f"{avg_ei:.4f}  {lr:.2e}{marker}")

    print(f"\n{'='*60}")
    print(f"  Training complete! Best mIoU: {best_iou:.4f}")
    print(f"  Saved to: {args.save_dir}/checkpoint_best.pth")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Off-Road Segmentation Model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Save directory")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()
    train(args)