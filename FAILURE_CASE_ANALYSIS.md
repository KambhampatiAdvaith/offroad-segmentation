# Failure Case Analysis: Off-Road Terrain Segmentation

## 1. Performance Summary

| Class | IoU | AP50 | Status |
|---|---|---|---|
| Sky | 0.9840 | 1.0000 | ✅ Excellent |
| Landscape | 0.6433 | 0.9481 | ✅ Good |
| Dry Bushes | 0.5179 | 0.4541 | ⚠️ Moderate |
| Trees | 0.4921 | 0.2211 | ⚠️ Moderate |
| Dry Grass | 0.4546 | 0.1267 | ❌ Needs Work |
| Rocks | 0.4425 | 0.0908 | ❌ Needs Work |
| **Lush Bushes** | **0.2280** | 0.0000 | 🔴 **CRITICAL** |

## 2. Worst Performing Classes

### Lush Bushes (IoU: 0.2280) - CRITICAL

**Root Causes:**
- Severely underrepresented in training data (PRIMARY cause)
- Visual similarity to trees and landscape
- Data imbalance - insufficient diverse samples

**Current Solutions Applied (from config.py):**
- Class weight: 0.5 (CLASS_WEIGHTS[2]) ← TOO LOW
- Focal Loss: gamma 2.0 (FOCAL_GAMMA)
- Dice Loss: 50% weight in combined loss
- Data augmentation: flip 50%, brightness 0.85-1.15, contrast 0.85-1.15
- Backbone fine-tuning: last 6 blocks (UNFREEZE_LAST_N_BLOCKS: 6)
- TTA: horizontal flips (+0.3% improvement)

**Why Insufficient:**
- Class weight 0.5 is TOO LOW (trees get 2.0)
- Generic augmentations don't address vegetation confusion
- Only 6 blocks unfrozen - limited domain adaptation
- No vegetation-specific techniques

**Recommended Quick Wins (1-2 hours):**
1. Increase class weight: 0.5 → 2.0-3.0 in config.py
2. Increase focal gamma: 2.0 → 2.5
3. Add color jittering augmentation (Hue-Saturation)
4. Expand augmentation ranges

Expected Impact: +5-10% IoU

### Rocks (IoU: 0.4425)

**Root Causes:**
- Scale variability (tiny pebbles to large boulders)
- DINOv2 patch size (14x14) misses small objects
- Texture diversity across rock types

**Recommended:** Increase class weight to 1.5-2.0, add FPN for multi-scale

### Dry Grass (IoU: 0.4546)

**Root Causes:**
- Color confusion with dry bushes
- Boundary ambiguity in transition zones

**Recommended:** Increase weight to 1.5, add contrastive loss

## 3. Implementation Roadmap

**Phase 1 (1-2 hours): Quick Wins**
- Update CLASS_WEIGHTS in config.py
- Increase FOCAL_GAMMA
- Add augmentations
- Expected: +5-10% mIoU

**Phase 2 (4-8 hours): Medium Effort**
- Multi-scale training
- FPN implementation
- More backbone blocks unfrozen
- Expected: +5-10% additional

**Phase 3 (1-2 weeks): Long-term**
- Collect more training data
- Implement advanced architectures

## 4. Success Metrics

| Metric | Current | Goal |
|---|---|---|
| mIoU | 0.5375 | 0.6875+ |
| Lush Bushes | 0.2280 | 0.6000+ |

---
*March 28, 2026 | mIoU (TTA): 0.5375*
