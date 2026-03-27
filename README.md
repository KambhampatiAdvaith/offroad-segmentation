# Off-Road Terrain Segmentation using DINOv2

## Kaggle Notebook

Full training and evaluation notebook with all outputs and logs:

🔗 **[Kaggle Notebook — GRIET](https://www.kaggle.com/code/kadvaith/griet)**

## Overview
Semantic segmentation model for off-road terrain using **DINOv2 ViT-B/14** backbone with a custom **Deep Segmentation Head**. The model classifies each pixel into 10 terrain classes.

---

## Final Results (1002 test images)

| Metric | Without TTA | With TTA (recommended) |
|---|---|---|
| **mIoU** | 0.5341 | **0.5375** |
| **mAP50** | 0.3983 | **0.4058** |
| **Pixel Accuracy** | 0.7537 | **0.7552** |

### Per-Class Performance (with TTA)

| Class | IoU | AP50 |
|---|---|---|
| Background | N/A (absent) | N/A |
| Trees | 0.4921 | 0.2211 |
| Lush Bushes | 0.2280 | 0.0000 |
| Dry Grass | 0.4546 | 0.1267 |
| Dry Bushes | 0.5179 | 0.4541 |
| Ground Clutter | N/A (absent) | N/A |
| Logs | N/A (absent) | N/A |
| Rocks | 0.4425 | 0.0908 |
| Landscape | 0.6433 | 0.9481 |
| Sky | 0.9840 | 1.0000 |

---

## Environment & Dependencies

### Requirements
- Python 3.8+
- CUDA-compatible GPU (tested on NVIDIA T4/P100)
- ~4GB GPU memory minimum

### Install dependencies
```bash
pip install -r requirements.txt
```

### requirements.txt contents
```
torch>=2.0.0
torchvision>=0.15.0
numpy
Pillow
tqdm
```

---

## File Structure

```
├── config.py              # All hyperparameters and settings
├── model.py               # DINOv2 backbone + Deep Segmentation Head
├── dataset.py             # Dataset loading and augmentation
├── losses.py              # Focal Loss + Dice Loss
├── train.py               # Training script
├── test.py                # Inference and evaluation script
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── checkpoint_best.pth    # Trained model weights (Google Drive link below)
```

---

## Model Checkpoint

The trained checkpoint is hosted on Google Drive (too large for GitHub):

**📥 [Download checkpoint_best.pth](https://drive.google.com/file/d/1tl3DShW0rjZJQO_DxDkkl9V5i_GFTUw7/view?usp=sharing)**

Place it in the project root directory after downloading.

---

## Dataset Structure

Your dataset should be organized as:
```
dataset_folder/
├── Color_Images/
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
└── Segmentation/
    ├── image_001.png
    ├── image_002.png
    └── ...
```

---

## Step-by-Step Instructions

### 1. Clone the repository
```bash
git clone https://github.com/KambhampatiAdvaith/offroad-segmentation.git
cd offroad-segmentation
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the checkpoint
Download `checkpoint_best.pth` from the Google Drive link above and place it in the project root.

### 4. Run evaluation (reproduce final results)

**Without TTA:**
```bash
python test.py --data_dir /path/to/test_data --checkpoint ./checkpoint_best.pth
```

**With TTA (recommended — gives best scores):**
```bash
python test.py --data_dir /path/to/test_data --checkpoint ./checkpoint_best.pth --tta
```

### 5. Train from scratch (optional)
```bash
python train.py --data_dir /path/to/dataset --save_dir ./checkpoints --epochs 40
```

### 6. Fine-tune from checkpoint (optional)
```bash
python train.py --data_dir /path/to/dataset --save_dir ./checkpoints --checkpoint ./checkpoint_best.pth --epochs 30
```

---

## How to Reproduce Final Results

1. Install dependencies: `pip install -r requirements.txt`
2. Download `checkpoint_best.pth` from Google Drive link
3. Run:
```bash
python test.py --data_dir /path/to/test_data --checkpoint ./checkpoint_best.pth --tta
```
4. Expected output:
```
🎯 EVALUATION (with TTA)
✅ Loaded checkpoint
Evaluating: 100%|████████████████████| 1002/1002

=================================================================
  🏆 RESULTS (TTA) — 1002 images
=================================================================
  Class                     IoU      AP50
  ----------------------------------------
  Background                N/A       N/A
  Trees                  0.4921    0.2211
  Lush Bushes            0.2280    0.0000
  Dry Grass              0.4546    0.1267
  Dry Bushes             0.5179    0.4541
  Ground Clutter            N/A       N/A
  Logs                      N/A       N/A
  Rocks                  0.4425    0.0908
  Landscape              0.6433    0.9481
  Sky                    0.9840    1.0000
  ----------------------------------------
  MEAN                   0.5375    0.4058
=================================================================
  📊 mIoU:           0.5375
  📊 mAP50:          0.4058
  📊 Pixel Accuracy: 0.7552
=================================================================
```

---

## Interpreting the Output

| Metric | What it means |
|---|---|
| **mIoU** | Mean Intersection-over-Union across all classes. Higher = better overlap between predicted and ground truth masks. |
| **mAP50** | Mean Average Precision at IoU threshold 0.5. Measures detection quality — whether per-image class regions are correctly identified. |
| **Pixel Accuracy** | Percentage of correctly classified pixels across all images. |
| **Per-class IoU** | How well each individual class is segmented. N/A means the class is absent from the dataset. |
| **Per-class AP50** | Detection precision for each class at IoU ≥ 0.5 threshold. |

### Notes:
- **Background, Ground Clutter, Logs** show N/A because they have zero pixels in the test set
- **Lush Bushes** has very low scores due to extreme scarcity (~0.003% of total pixels)
- **Sky** and **Landscape** perform best as they have clear visual features and sufficient data
- TTA (Test-Time Augmentation) averages predictions from original and horizontally-flipped images for a ~0.5-1% free boost

---

## Model Architecture

- **Backbone:** DINOv2 ViT-B/14 with registers (pretrained, last 6 transformer blocks fine-tuned)
- **Head:** Custom Deep Segmentation Head
  - 1x1 stem convolution (768 → 256 channels)
  - 7x7 depthwise separable block with residual connection
  - 5x5 depthwise separable block with residual connection
  - 3x3 depthwise separable block (256 → 128 channels)
  - 2x transposed convolution upsampling + 3x3 refinement (128 → 64)
  - 1x1 classifier (64 → 10 classes)
- **Loss:** Focal Loss (γ=2.0, class-weighted) + 0.5 × Dice Loss
- **Training:** AdamW optimizer, warmup + cosine annealing LR schedule

---

## Classes (10)

| ID | Class | Raw Pixel Value |
|---|---|---|
| 0 | Background | 0 |
| 1 | Trees | 100 |
| 2 | Lush Bushes | 200 |
| 3 | Dry Grass | 300 |
| 4 | Dry Bushes | 500 |
| 5 | Ground Clutter | 550 |
| 6 | Logs | 700 |
| 7 | Rocks | 800 |
| 8 | Landscape | 7100 |
| 9 | Sky | 10000 |
