"""
Configuration for Off-Road Segmentation Model
DINOv2 ViT-B/14 + Deep Segmentation Head
"""

# Dataset
VALUE_MAP = {0: 0, 100: 1, 200: 2, 300: 3, 500: 4, 550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9}
N_CLASSES = 10
CLASS_NAMES = ['Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
               'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky']

# Image dimensions (must be divisible by 14 for DINOv2 patch size)
IMG_WIDTH = 476   # int(((960 / 2) // 14) * 14)
IMG_HEIGHT = 266  # int(((540 / 2) // 14) * 14)

# Training hyperparameters
BATCH_SIZE = 8
EPOCHS = 40
LEARNING_RATE = 1e-4
BACKBONE_LR = 5e-6
WEIGHT_DECAY = 0.01
WARMUP_EPOCHS = 3
GRAD_CLIP = 1.0

# Backbone
BACKBONE_NAME = "dinov2_vitb14_reg"
UNFREEZE_LAST_N_BLOCKS = 6

# Loss
FOCAL_GAMMA = 2.0
CLASS_WEIGHTS = [0.0, 2.0, 0.5, 1.0, 1.5, 0.0, 0.0, 1.0, 0.8, 1.0]
DICE_IGNORE_CLASSES = [0, 5, 6]  # Background, Ground Clutter, Logs (absent)

# Augmentation
FLIP_PROB = 0.5
BRIGHTNESS_RANGE = (0.85, 1.15)
CONTRAST_RANGE = (0.85, 1.15)

# TTA
USE_TTA = True
