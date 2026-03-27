%%writefile /kaggle/working/submission/dataset.py
"""
Dataset and data loading utilities
"""
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import torchvision.transforms as transforms
from config import VALUE_MAP, IMG_HEIGHT, IMG_WIDTH, FLIP_PROB, BRIGHTNESS_RANGE, CONTRAST_RANGE


def convert_mask(mask):
    """Convert raw mask values to class indices"""
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw, new in VALUE_MAP.items():
        new_arr[arr == raw] = new
    return Image.fromarray(new_arr)


def get_transforms(augment=False):
    """Get image and mask transforms"""
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH),
                          interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    return transform, mask_transform


class SegmentationDataset(Dataset):
    """Off-road segmentation dataset"""
    def __init__(self, data_dir, augment=False):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.augment = augment
        self.data_ids = sorted(os.listdir(self.image_dir))
        self.transform, self.mask_transform = get_transforms(augment)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        image = Image.open(os.path.join(self.image_dir, data_id)).convert("RGB")
        mask = Image.open(os.path.join(self.masks_dir, data_id))
        mask = convert_mask(mask)

        if self.augment:
            if np.random.random() > FLIP_PROB:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if np.random.random() > 0.7:
                from torchvision.transforms import functional as TF
                image = TF.adjust_brightness(image, np.random.uniform(*BRIGHTNESS_RANGE))
                image = TF.adjust_contrast(image, np.random.uniform(*CONTRAST_RANGE))

        image = self.transform(image)
        mask = self.mask_transform(mask) * 255
        return image, mask