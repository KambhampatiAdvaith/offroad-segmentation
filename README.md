# Off-Road Terrain Segmentation using DINOv2

**Semantic segmentation model for off-road terrain classification using DINOv2 ViT-B/14 backbone with a custom Deep Segmentation Head. Achieves 0.5375 mIoU on 1002 test images.**

---

## 📑 Table of Contents

- [Quick Start](#quick-start)
- [Model Overview](#model-overview)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset Structure](#dataset-structure)
- [Architecture Details](#architecture-details)
- [Training Configuration](#training-configuration)
- [Failure Case Analysis](#failure-case-analysis)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## 🚀 Quick Start

### Installation (2 minutes)
```bash
# Clone repository
git clone https://github.com/KambhampatiAdvaith/offroad-segmentation.git
cd offroad-segmentation

# Install dependencies
pip install -r requirements.txt
Training (40 epochs ≈ 2-3 hours on T4 GPU)
Bashpython train.py \
  --data_dir ./data \
  --save_dir ./checkpoints \
  --epochs 40
Testing (Evaluation)
Bashpython test.py \
  --data_dir ./data \
  --checkpoint ./checkpoints/checkpoint_best.pth \
  --tta
Expected Output
text===================================================================
  🏆 RESULTS (with TTA) — 1002 images
===================================================================
  mIoU:           0.5375
  mAP50:          0.4058
  Pixel Accuracy: 0.7552
===================================================================
📊 Model Overview
Architecture
Backbone: DINOv2 ViT-B/14 (Pre-trained Vision Transformer)

Self-supervised pre-trained on diverse visual data
Strong semantic feature extraction
14×14 patch size for fine-grained features
12 transformer blocks total

Segmentation Head: Deep Segmentation Head

Multi-scale upsampling layers
10-class output for terrain classification
Bilinear interpolation for smooth upsampling
Custom batch normalization and ReLU activations

Key Strengths:

✅ Patch-based attention mechanism captures global context
✅ Hierarchical feature extraction from multiple levels
✅ Efficient inference (~100ms per image)
✅ Support for Test-Time Augmentation (TTA)
✅ Combined Focal + Dice loss handles class imbalance

Class Definitions (10 Classes)







































































IDClassTypical ColorChallenge Level0BackgroundVariesN/A (Absent)1TreesGreenMedium2Lush BushesBright GreenHigh ⚠️3Dry GrassBrown/TanHigh ⚠️4Dry BushesBrownMedium5Ground ClutterMixedN/A (Absent)6LogsBrownN/A (Absent)7RocksGray/BrownHigh ⚠️8LandscapeDark/MixedLow ✅9SkyBlueNone ✅
📈 Results
Overall Performance (1002 Test Images)

























MetricWithout TTAWith TTA (Recommended)mIoU0.53410.5375mAP500.39830.4058Pixel Accuracy0.75370.7552
Per-Class Performance (with TTA)







































































ClassIoUAP50StatusSky0.98401.0000✅ ExcellentLandscape0.64330.9481✅ GoodDry Bushes0.51790.4541⚠️ ModerateTrees0.49210.2211⚠️ ModerateDry Grass0.45460.1267❌ Needs WorkRocks0.44250.0908❌ Needs WorkLush Bushes0.22800.0000🔴 CriticalBackgroundN/AN/AAbsentGround ClutterN/AN/AAbsentLogsN/AN/AAbsent
Performance Notes:

TTA Improvement: +0.3% mIoU (horizontal flip augmentation)
Best Performance: Sky (98.4%), Landscape (64.3%)
Worst Performance: Lush Bushes (22.8%) — severely undersampled in training
Pixel Accuracy: 75.5% — high per-pixel classification accuracy

🔧 Installation
Requirements

Python: 3.8 or higher
GPU: CUDA-compatible (tested on NVIDIA T4, P100)
Memory: ~4GB GPU VRAM minimum (8GB recommended)
Disk Space: ~2GB for model checkpoint + dependencies

Step-by-Step Installation
Bash# 1. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import timm; print(f'TIMM version: {timm.__version__}')"
Dependencies (from requirements.txt)
txttorch>=2.0.0              # PyTorch deep learning framework
torchvision>=0.15.0       # Computer vision utilities
timm>=0.9.0              # DINOv2 and transformer models
numpy>=1.24.0            # Numerical computing
Pillow>=9.0.0            # Image processing
tqdm>=4.65.0             # Progress bars
opencv-python>=4.8.0     # Image I/O and processing
scipy>=1.10.0            # Scientific computing
📚 Usage
Training
Basic Training Command
Bashpython train.py --data_dir ./data --save_dir ./checkpoints --epochs 40
Resume Training from Checkpoint
Bashpython train.py \
  --data_dir ./data \
  --save_dir ./checkpoints \
  --epochs 50 \
  --checkpoint ./checkpoints/checkpoint_best.pth
Training Arguments:

--data_dir: Path to dataset directory (required)
--save_dir: Directory to save checkpoints (default: ./checkpoints)
--epochs: Number of training epochs (default: 40)
--checkpoint: Path to checkpoint for resuming (optional)

Output Files:

checkpoint_best.pth: Best model (automatically saved)
Console output: mIoU, loss, learning rate printed each epoch

Training Metrics Printed (Each Epoch):
textEp  TrLoss  EvLoss  TrIoU  EvIoU   LR
  1  0.8234  0.7921  0.3821  0.3956  1.33e-05 ⭐
  2  0.7156  0.7234  0.4123  0.4287  2.67e-05
  ...
 40  0.2134  0.2456  0.5123  0.5375  1.00e-07 ⭐
Training Time:

~2-3 hours on NVIDIA T4 GPU (40 epochs)
~1-2 hours on NVIDIA A100 GPU (40 epochs)
GPU memory: ~3.5 GB

Testing/Evaluation
Standard Evaluation (No TTA)
Bashpython test.py \
  --data_dir ./data \
  --checkpoint ./checkpoints/checkpoint_best.pth
Evaluation with Test-Time Augmentation (Recommended)
Bashpython test.py \
  --data_dir ./data \
  --checkpoint ./checkpoints/checkpoint_best.pth \
  --tta
Testing Arguments:

--data_dir: Path to test dataset (required)
--checkpoint: Path to model checkpoint (required)
--tta: Enable Test-Time Augmentation flag (optional)

Expected Output Format:
text🎯 EVALUATION (with TTA)
Test images: 1002

===================================================================
  🏆 RESULTS (with TTA) — 1002 images
===================================================================
  Class                IoU        AP50
  ------------------------------------------
  Background           N/A         N/A
  Trees             0.4921      0.2211
  Lush Bushes       0.2280      0.0000
  Dry Grass         0.4546      0.1267
  Dry Bushes        0.5179      0.4541
  Ground Clutter       N/A         N/A
  Logs                 N/A         N/A
  Rocks             0.4425      0.0908
  Landscape         0.6433      0.9481
  Sky               0.9840      1.0000
  ------------------------------------------
  MEAN              0.5375      0.4058
===================================================================
  📊 mIoU:           0.5375
  📊 mAP50:          0.4058
  📊 Pixel Accuracy: 0.7552
===================================================================
📁 Dataset Structure
Required Directory Organization
Bashdata/
├── Color_Images/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   ├── image_003.jpg
│   └── ... (all RGB images)
│
└── Segmentation/
    ├── image_001.png
    ├── image_002.png
    ├── image_003.png
    └── ... (corresponding segmentation masks)
Image Specifications
Color Images:

Format: .jpg, .jpeg, .png, .bmp
Recommended: RGB JPEG format
Resolution: ~960×540 pixels (automatically resized to 476×266)
Aspect ratio: All images should match

Segmentation Masks:

Format: .png (8-bit grayscale)
Size: Same dimensions as color image
Values: Integer pixel values mapped to classes

Pixel-to-Class Mapping:
text0      → Background (ID: 0)
100    → Trees (ID: 1)
200    → Lush Bushes (ID: 2)
300    → Dry Grass (ID: 3)
500    → Dry Bushes (ID: 4)
550    → Ground Clutter (ID: 5)
700    → Logs (ID: 6)
800    → Rocks (ID: 7)
7100   → Landscape (ID: 8)
10000  → Sky (ID: 9)
Preparing Your Dataset
Bash# Create directory structure
mkdir -p data/Color_Images data/Segmentation

# Copy images
cp /your/images/*.jpg data/Color_Images/
cp /your/masks/*.png data/Segmentation/

# Verify structure
ls -la data/Color_Images/ | wc -l    # Count images
ls -la data/Segmentation/ | wc -l    # Count masks (should match)

# Check sample mask values
python -c "
import numpy as np
from PIL import Image
mask = Image.open('data/Segmentation/image_001.png')
print('Unique pixel values in mask:', np.unique(mask))
"
🧠 Architecture Details
Model Components
1. DINOv2 Backbone (Frozen + Fine-tuned)
Architecture:
textInput Image (476×266×3)
    ↓
Patch Embedding Layer (14×14 patches = 35×19 patches)
    ↓
Vision Transformer Blocks (12 layers total)
    │
    ├─ Blocks 0-5: FROZEN (general vision features)
    └─ Blocks 6-11: FINE-TUNED (terrain-specific features)
    ↓
Patch Tokens Output (35×19×768 features)
Configuration (config.py):
PythonBACKBONE_NAME = "dinov2_vitb14_reg"
UNFREEZE_LAST_N_BLOCKS = 6  # Out of 12 total blocks
BACKBONE_LR = 5e-6          # Lower learning rate for backbone
Fine-tuning Strategy:

Keep first 6 blocks frozen (general visual features)
Fine-tune last 6 blocks (adapt to terrain-specific patterns)
Use lower learning rate (5e-6 vs 1e-4 for head)

2. Deep Segmentation Head
Architecture:
textInput Features (35×19×768)
    ↓
Decoder Block 1
  ├─ Bilinear Upsample (35×19 → 70×38)
  ├─ Convolution 3×3
  ├─ Batch Norm + ReLU
    ↓
Decoder Block 2
  ├─ Bilinear Upsample (70×38 → 140×76)
  ├─ Convolution 3×3
  ├─ Batch Norm + ReLU
    ↓
Decoder Block 3
  ├─ Bilinear Upsample (140×76 → 476×266)
  ├─ Convolution 3×3
  ├─ Batch Norm + ReLU
    ↓
Final Output (476×266×10)  [10 class logits]
Key Features:

Bilinear interpolation for smooth, boundary-preserving upsampling
Batch normalization after each upsampling layer
ReLU activations between layers
No skip connections from backbone (patch-based approach)

3. Loss Function (Combined)
Loss Computation:
textTotal Loss = 0.5 × Focal Loss + 0.5 × Dice Loss
Focal Loss:

Focuses on hard-to-classify pixels
Weight: γ (gamma) = 2.0
Class weights: [0.0, 2.0, 0.5, 1.0, 1.5, 0.0, 0.0, 1.0, 0.8, 1.0]
Helps with class imbalance and hard negatives

Dice Loss:

Directly optimizes IoU metric
Particularly good for multi-class segmentation
Ignores classes: Background (0), Ground Clutter (5), Logs (6)
Weight: 0.5 of total loss

Why This Combination:

Focal loss handles hard examples and class imbalance
Dice loss directly optimizes the evaluation metric (IoU)
Combined: balances local accuracy with global IoU

⚙️ Training Configuration
Hyperparameters (from config.py)













































































ParameterValueDescriptionImpactBATCH_SIZE8Samples per training iterationLimited by ~4GB GPU memoryEPOCHS40Total training iterations~2-3 hours on T4 GPULEARNING_RATE1e-4Segmentation head learning rateFast adaptation to taskBACKBONE_LR5e-6Backbone fine-tuning learning rateSlow, stable adaptationWEIGHT_DECAY0.01L2 regularization coefficientPrevents overfittingFOCAL_GAMMA2.0Hard example mining strengthFocuses on difficult pixelsWARMUP_EPOCHS3Learning rate warmup phaseStable initializationGRADIENT_CLIP1.0Max gradient magnitudePrevents training instabilityUNFREEZE_LAST_N_BLOCKS6Backbone blocks to fine-tuneBalance between adaptation and stabilityIMG_WIDTH476Image width (divisible by 14)DINOv2 patch-based processingIMG_HEIGHT266Image height (divisible by 14)DINOv2 patch-based processing
Class Weights Explanation (config.py)
PythonCLASS_WEIGHTS = [0.0, 2.0, 0.5, 1.0, 1.5, 0.0, 0.0, 1.0, 0.8, 1.0]
#                [BG, Trees, LushBushes, DryGrass, DryBushes, GndClutter, Logs, Rocks, Landscape, Sky]
Analysis:





















































ClassWeightRationaleActual PerformanceTrees2.0High priority (confuses with bushes)0.4921 IoULush Bushes0.5⚠️ TOO LOW (severely undersampled)0.2280 IoU ← WorstDry Bushes1.5Medium priority0.5179 IoURocks1.0Neutral (moderate undersampling)0.4425 IoULandscape0.8Low (well-represented)0.6433 IoUSky1.0Neutral (already performs well)0.9840 IoU ← BestAbsent Classes0.0Background, Ground Clutter, LogsN/A
⚠️ Recommendation: Increase Lush Bushes weight to 2.0-3.0 for significant improvement.
Data Augmentation (config.py)





























AugmentationTypeRange/ProbabilityPurposeHorizontal FlipSpatial50% probabilityIncrease dataset sizeBrightnessColor0.85 - 1.15Handle lighting variationsContrastColor0.85 - 1.15Handle contrast variations
Current Limitations:

No rotation (could help with orientation invariance)
No zoom/scale (would help with scale variations)
No color jittering (would help vegetation discrimination)

Learning Rate Schedule
textEpoch 1-3:   Linear Warmup
  LR = initial_lr × (epoch / warmup_epochs)
  
Epoch 4-40:  Cosine Annealing
  LR = initial_lr × 0.5 × (1 + cos(π × progress))
Visualization:
textLR │
   │     /‾‾‾‾‾‾‾‾‾\
   │    /             \___
   │   /                   \___
   │  /________________________\
   └───────────────────────────── → Epochs
     0   3              40
     |---|  |-----------|
     warmup  cosine annealing
📊 Failure Case Analysis
See FAILURE_CASE_ANALYSIS.md for comprehensive analysis including:
✅ Detailed Breakdown:

Root causes for each underperforming class
Current solutions and their effectiveness
Specific recommendations with expected improvements
3-phase implementation roadmap (Quick Wins → Medium → Long-term)

✅ Key Findings:

Lush Bushes (0.228 IoU): Severely undersampled — PRIMARY BOTTLENECK
Rocks (0.4425 IoU): Scale variability — needs multi-scale features
Dry Grass (0.4546 IoU): Class confusion — needs better augmentation

✅ Quick Wins (1-2 hours):

Increase Lush Bushes class weight: 0.5 → 3.0
Increase focal gamma: 2.0 → 2.5
Add color jittering augmentation
Expected improvement: +5-10% mIoU

✅ Medium Improvements (4-8 hours):

Multi-scale feature pyramid (FPN)
Progressive training strategy
CRF post-processing
Expected improvement: +5-10% additional

🐛 Troubleshooting
1. CUDA Out of Memory Error
Error Message:
textRuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
Solutions (in order of recommendation):
a) Reduce batch size:
Python# Edit config.py
BATCH_SIZE = 8   # Change to 4 or 2
b) Clear GPU cache:
Bashpython -c "import torch; torch.cuda.empty_cache()"
c) Use CPU (very slow):
Bash# Remove GPU requirement - not recommended for training
d) Use smaller image size:
Python# Edit config.py
IMG_HEIGHT = 200  # Reduce from 266
IMG_WIDTH = 356   # Reduce from 476
2. ModuleNotFoundError: No module named 'timm'
Error Message:
textModuleNotFoundError: No module named 'timm'
Solutions:
a) Install timm specifically:
Bashpip install timm>=0.9.0
b) Reinstall all dependencies:
Bashpip install -r requirements.txt --upgrade --force-reinstall
c) Check installation:
Bashpython -c "import timm; print(timm.__version__)"
3. Dataset Not Found Error
Error Message:
textFileNotFoundError: [Errno 2] No such file or directory: './data/Color_Images'
Solutions:
a) Verify dataset structure:
Bash# Check if directories exist
ls -la ./data/
ls -la ./data/Color_Images/
ls -la ./data/Segmentation/

# Count images
ls ./data/Color_Images/ | wc -l
ls ./data/Segmentation/ | wc -l  # Should match above
b) Use absolute path:
Bashpython train.py --data_dir /absolute/path/to/data
c) Check file permissions:
Bashchmod -R 755 ./data/
4. Checkpoint Not Loading
Error Message:
textRuntimeError: Error(s) in loading state_dict for Module...
Solutions:
a) Start fresh (no checkpoint):
Bashpython train.py --data_dir ./data --save_dir ./checkpoints --epochs 40
# Don't use --checkpoint flag
b) Verify checkpoint integrity:
Bashpython -c "
import torch
checkpoint = torch.load('./checkpoints/checkpoint_best.pth')
print('Keys:', checkpoint.keys())
print('Model state dict keys:', len(checkpoint['model_state_dict']))
"
c) Use compatible checkpoint:

Ensure model architecture hasn't changed
Try downloading fresh checkpoint if available

5. Poor Model Performance (mIoU < 0.50)
Debugging Steps:
a) Verify dataset loading:
Bashpython -c "
from dataset import SegmentationDataset
import numpy as np

ds = SegmentationDataset('./data', augment=False)
print(f'Dataset size: {len(ds)} images')

# Check sample
img, mask = ds[0]
print(f'Image shape: {img.shape}')
print(f'Mask shape: {mask.shape}')
print(f'Mask unique values: {np.unique(mask.numpy())}')
print(f'Expected values: 0-9 (10 classes)')
"
b) Check mask encoding:
Bashpython -c "
import numpy as np
from PIL import Image

mask = Image.open('./data/Segmentation/image_001.png')
vals = np.unique(np.array(mask))
print(f'Unique pixel values in mask: {vals}')

# Should be subset of: [0, 100, 200, 300, 500, 550, 700, 800, 7100, 10000]
"
c) Verify images and masks align:
Bash# Check file counts match
COUNT_IMG=$(ls ./data/Color_Images/ | wc -l)
COUNT_MASK=$(ls ./data/Segmentation/ | wc -l)
echo "Images: $COUNT_IMG, Masks: $COUNT_MASK"
# Should be equal
d) Check training metrics:

If loss doesn't decrease → learning rate too high/low
If mIoU stagnates → insufficient epochs or poor data quality
If training very slow → check GPU utilization

6. Low GPU Utilization
Symptoms: GPU memory usage <50%, training very slow
Solutions:
a) Increase batch size (if GPU memory allows):
Python# Edit config.py
BATCH_SIZE = 16  # Increase from 8
b) Enable mixed precision:
Bash# Add to train.py
torch.set_float32_matmul_precision('high')
c) Pin memory for DataLoader:
Python# In dataset.py
DataLoader(..., pin_memory=True, num_workers=4)
Performance Tuning Tips
To Improve mIoU:

Train longer: Increase epochs: 40 → 50-60
Better augmentation: Add rotation, zoom, color jitter (see FAILURE_CASE_ANALYSIS.md)
Adjust class weights: Increase weight for poorly performing classes (start with Lush Bushes 2.0-3.0 and Rocks 1.5-2.0)
More backbone fine-tuning: Unfreeze more blocks: 6 → 8-10
Post-processing: Add CRF for spatial smoothness + morphological operations

🔗 External Resources

Kaggle Notebook — Full training and evaluation with all outputs: Kaggle Notebook — GRIET
Hackathon Report — Comprehensive methodology and analysis: Full Hackathon Report (APEX AUTOMATORS)

Key References

DINOv2: Meta AI Self-Supervised Vision Transformers — Paper: "Emerging Properties in Self-Supervised Vision Transformers"
Focal Loss: Lin et al. — Paper: "Focal Loss for Dense Object Detection" (ICCV 2017)
Dice Loss: Milletari et al. — Paper: "V-Net: Fully Convolutional Neural Networks" (3DV 2016)

📋 File Structure & Explanation
textoffroad-segmentation/
│
├── README.md                              ← Complete documentation (this file)
├── FAILURE_CASE_ANALYSIS.md              ← Detailed performance analysis
│
├── requirements.txt                       ← Python dependencies
│                                          (torch, timm, numpy, opencv, etc.)
│
├── config.py                             ← Hyperparameters & configuration
│                                          (batch size, learning rate, class weights)
│
├── train.py                              ← Training script
│                                          (run: python train.py --data_dir ./data)
│
├── test.py                               ← Testing/Evaluation script
│                                          (run: python test.py --checkpoint ./checkpoints/...)
│
├── model.py                              ← Model architecture
│                                          (DINOv2 backbone + segmentation head)
│
├── dataset.py                            ← Data loading & augmentation
│                                          (handles .jpg & .png files)
│
├── losses.py                             ← Loss functions
│                                          (Focal Loss + Dice Loss)
│
├── checkpoint_head_only.pth              ← Pre-trained model weights (~348 MB)
│
├── hackathon_report apex automators.pdf  ← Official competition report
│
└── data/                                 ← Your dataset directory
    ├── Color_Images/                     ← RGB images (.jpg)
    │   ├── image_001.jpg
    │   ├── image_002.jpg
    │   └── ...
    └── Segmentation/                     ← Segmentation masks (.png)
        ├── image_001.png
        ├── image_002.png
        └── ...
📈 Performance Summary









































MetricValueBest mIoU0.5375 (with Test-Time Augmentation)mAP500.4058Pixel Accuracy0.7552Model Size~348 MBInference Speed~100 ms per imageTraining Time~2-3 hours (40 epochs on T4 GPU)GPU Memory~3.5 GBTest Images1002
📞 Getting Help

Check Troubleshooting section above for common issues
Review FAILURE_CASE_ANALYSIS.md for model-specific analysis
Check Kaggle Notebook for working example
Review error message — usually indicates what went wrong
