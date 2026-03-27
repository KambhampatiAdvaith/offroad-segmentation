%%writefile /kaggle/working/submission/model.py
"""
Model architecture: DINOv2 backbone + Deep Segmentation Head
"""
import torch
import torch.nn as nn

class DeepSegmentationHead(nn.Module):
    """
    Multi-scale segmentation head with depthwise separable convolutions
    and residual connections for DINOv2 patch token features.
    """
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1),
            nn.BatchNorm2d(256), nn.GELU())

        self.block1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=7, padding=3, groups=256),
            nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256), nn.GELU())

        self.block2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=5, padding=2, groups=256),
            nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256), nn.GELU())

        self.block3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=256),
            nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128), nn.GELU())

        self.upsample_refine = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.GELU())

        self.classifier = nn.Conv2d(64, out_channels, 1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        residual = x; x = self.block1(x) + residual
        residual = x; x = self.block2(x) + residual
        x = self.block3(x)
        x = self.upsample_refine(x)
        x = self.dropout(x)
        return self.classifier(x)


def build_model(device, checkpoint_path=None):
    """
    Build full model: DINOv2 backbone + segmentation head.
    Optionally load from checkpoint.
    """
    from config import N_CLASSES, IMG_WIDTH, IMG_HEIGHT, BACKBONE_NAME

    backbone = torch.hub.load("facebookresearch/dinov2", BACKBONE_NAME)
    backbone.to(device)

    with torch.no_grad():
        dummy = torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH).to(device)
        n_emb = backbone.forward_features(dummy)["x_norm_patchtokens"].shape[2]

    classifier = DeepSegmentationHead(
        n_emb, N_CLASSES, IMG_WIDTH // 14, IMG_HEIGHT // 14).to(device)

    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        classifier.load_state_dict(ckpt['model_state_dict'])
        if 'backbone_finetuned' in ckpt:
            backbone_sd = backbone.state_dict()
            for name, param in ckpt['backbone_finetuned'].items():
                if name in backbone_sd:
                    backbone_sd[name] = param
            backbone.load_state_dict(backbone_sd)
        print(f"✅ Loaded checkpoint: mIoU={ckpt.get('best_iou', 'N/A')}")

    return backbone, classifier