"""
Loss functions: Focal Loss + Dice Loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance"""
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, pred, target):
        ce = F.cross_entropy(pred, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


class DiceLoss(nn.Module):
    """Dice loss with option to ignore specific classes"""
    def __init__(self, n_classes, smooth=1e-6, ignore_classes=None):
        super().__init__()
        self.n_classes = n_classes
        self.smooth = smooth
        self.ignore_classes = ignore_classes or []

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target_oh = F.one_hot(target, self.n_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = (pred * target_oh).sum(dim=dims)
        union = pred.sum(dim=dims) + target_oh.sum(dim=dims)
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        mask = torch.ones(self.n_classes, device=pred.device)
        for c in self.ignore_classes:
            mask[c] = 0
        return 1 - (dice * mask).sum() / mask.sum()
