# Off-Road Terrain Segmentation using DINOv2

## Kaggle Notebook
Full training and evaluation notebook with all outputs and logs:
🔗 **[Kaggle Notebook — GRIET](https://www.kaggle.com/code/kadvaith/griet)**

---

## 📄 **Hackathon Report**
**📋 [Read Full Hackathon Report (BY APEX AUTOMATORS)](./hackathon_report%20apex%20automators.pdf)**

**Report Includes:**
- ✅ Methodology & 4-Stage Training Strategy
- ✅ Results with mIoU 0.5375 & All Visualizations  
- ✅ Challenges & Solutions (Class Imbalance, Occlusion)
- ✅ Optimization Techniques & Future Work

---

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

## 📊 Failure Case Analysis

See [FAILURE_CASE_ANALYSIS.md](./FAILURE_CASE_ANALYSIS.md) for detailed analysis of challenging cases, model limitations, and future improvements.

**Quick Summary:**
- **Lush Bushes (0.228 IoU):** Dense vegetation easily confused with trees
- **Rocks (0.4425 IoU):** Small objects with high scale variation
- **Dry Grass (0.4546 IoU):** Texture similarity with dry bushes

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

### Quick Start

#### Training
```bash
python train.py --data_dir <path_to_dataset> --save_dir ./checkpoints --epochs 40
```

#### Testing
```bash
python test.py --data_dir <path_to_dataset> --checkpoint ./checkpoints/checkpoint_best.pth --tta
```

#### Results
- mIoU: **0.5375** (with TTA)
- mAP50: **0.4058**
- Pixel Accuracy: **0.7552**