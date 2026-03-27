"""
Test/Inference script for Off-Road Segmentation
Usage: python test.py --data_dir <path> --checkpoint <path> [--tta]

Outputs: mIoU, mAP50, per-class IoU, per-class AP50, Pixel Accuracy
"""
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from PIL import Image
import numpy as np
import os
import argparse
from tqdm import tqdm

from config import *
from model import build_model


def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw, new in VALUE_MAP.items():
        new_arr[arr == raw] = new
    return Image.fromarray(new_arr)


class TestDataset(Dataset):
    """Returns raw PIL images for TTA support"""
    def __init__(self, data_dir):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.data_ids = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        image = Image.open(os.path.join(self.image_dir, data_id)).convert("RGB")
        mask = Image.open(os.path.join(self.masks_dir, data_id))
        mask = convert_mask(mask)
        return image, mask, data_id


def compute_image_level_ap(all_ious_scores, iou_threshold=0.5):
    """Compute Average Precision at IoU threshold"""
    all_scores, all_tp, total_gt = [], [], 0
    for iou_val, has_gt in all_ious_scores:
        if has_gt:
            total_gt += 1
        if iou_val > 0 or has_gt:
            all_scores.append(iou_val)
            all_tp.append(1 if iou_val >= iou_threshold else 0)
    if total_gt == 0:
        return float('nan')
    if len(all_scores) == 0:
        return 0.0
    sorted_idx = np.argsort(-np.array(all_scores))
    tp_sorted = np.array(all_tp)[sorted_idx]
    tp_cum = np.cumsum(tp_sorted)
    fp_cum = np.cumsum(1 - tp_sorted)
    prec = tp_cum / (tp_cum + fp_cum)
    rec = tp_cum / total_gt
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([1.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def predict(backbone, classifier, img_tensor, target_size, device, use_tta=False):
    """Run inference, optionally with Test-Time Augmentation (horizontal flip)"""
    with autocast('cuda'):
        tokens = backbone.forward_features(img_tensor)["x_norm_patchtokens"]
        logits = classifier(tokens)
        out = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)

        if use_tta:
            img_flip = torch.flip(img_tensor, dims=[3])
            tokens_f = backbone.forward_features(img_flip)["x_norm_patchtokens"]
            logits_f = classifier(tokens_f)
            out_f = F.interpolate(logits_f, size=target_size, mode="bilinear", align_corners=False)
            out_f = torch.flip(out_f, dims=[3])
            out = (out + out_f) / 2.0

    return out


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_tta = args.tta

    print(f"🎯 EVALUATION {'(with TTA)' if use_tta else '(no TTA)'}")

    # Model
    backbone, classifier = build_model(device, args.checkpoint)
    backbone.eval(); classifier.eval()

    # Data
    dataset = TestDataset(args.data_dir)
    print(f"Test images: {len(dataset)}")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()
    resize_img = transforms.Resize((IMG_HEIGHT, IMG_WIDTH))
    resize_mask = transforms.Resize((IMG_HEIGHT, IMG_WIDTH),
                                     interpolation=transforms.InterpolationMode.NEAREST)

    # Metrics
    global_inter = np.zeros(N_CLASSES, dtype=np.float64)
    global_union = np.zeros(N_CLASSES, dtype=np.float64)
    ap_data = {c: [] for c in range(N_CLASSES)}
    total_correct, total_pixels = 0, 0

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Evaluating"):
            image, mask, data_id = dataset[i]
            img_t = normalize(to_tensor(resize_img(image))).unsqueeze(0).to(device)
            mask_t = (to_tensor(resize_mask(mask)) * 255).squeeze(0).long().to(device)

            out = predict(backbone, classifier, img_t, (IMG_HEIGHT, IMG_WIDTH),
                          device, use_tta=use_tta)
            preds = torch.argmax(out, dim=1).squeeze(0)

            total_correct += (preds == mask_t).sum().cpu().numpy()
            total_pixels += mask_t.numel()

            pred_np = preds.cpu().numpy()
            gt_np = mask_t.cpu().numpy()

            for c in range(N_CLASSES):
                pc, gc = (pred_np == c), (gt_np == c)
                inter = (pc & gc).sum()
                union = (pc | gc).sum()
                global_inter[c] += inter
                global_union[c] += union
                ap_data[c].append((inter / union if union > 0 else 0.0, gc.sum() > 0))

    # Compute metrics
    class_iou = [global_inter[c] / global_union[c] if global_union[c] > 0
                 else float('nan') for c in range(N_CLASSES)]
    class_ap = [compute_image_level_ap(ap_data[c], 0.5) for c in range(N_CLASSES)]
    mean_iou = np.nanmean(class_iou)
    valid_ap = [a for a in class_ap if not np.isnan(a)]
    map50 = np.mean(valid_ap) if valid_ap else 0.0
    pix_acc = total_correct / total_pixels

    # Print results
    tta_label = " (TTA)" if use_tta else ""
    print(f"\n{'='*65}")
    print(f"  🏆 RESULTS{tta_label} — {len(dataset)} images")
    print(f"{'='*65}")
    print(f"  {'Class':<20} {'IoU':>8}  {'AP50':>8}")
    print(f"  {'-'*40}")
    for i, name in enumerate(CLASS_NAMES):
        iou_s = f"{class_iou[i]:.4f}" if not np.isnan(class_iou[i]) else "  N/A"
        ap_s = f"{class_ap[i]:.4f}" if not np.isnan(class_ap[i]) else "  N/A"
        print(f"  {name:<20} {iou_s:>8}  {ap_s:>8}")
    print(f"  {'-'*40}")
    print(f"  {'MEAN':<20} {mean_iou:>8.4f}  {map50:>8.4f}")
    print(f"{'='*65}")
    print(f"  📊 mIoU:           {mean_iou:.4f}")
    print(f"  📊 mAP50:          {map50:.4f}")
    print(f"  📊 Pixel Accuracy: {pix_acc:.4f}")
    print(f"{'='*65}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Off-Road Segmentation Model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to test dataset")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation")
    args = parser.parse_args()
    evaluate(args)
