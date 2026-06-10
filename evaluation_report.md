# Technical Evaluation Report — GLOFeagles '26

## Model Overview

| Property | Value |
|----------|-------|
| **Architecture** | YOLOv8m + CBAM Attention |
| **Base Model** | YOLOv8m (Ultralytics) |
| **Custom Module** | CBAM (Channel + Spatial Attention) |
| **Parameters** | ~20.1M |
| **GFLOPs** | ~68.3 |
| **Input Size** | 640×640 |
| **Training Epochs** | 30 |
| **Batch Size** | 16 |
| **Transfer Learning** | COCO pretrained weights (258/652 layers transferred) |
| **GPU** | NVIDIA GeForce RTX 2070 SUPER (8GB) |

## Dataset

| Split | Images | Bounding Boxes |
|-------|--------|---------------|
| Train | 210 | 1,850 |
| Valid | 30 | 309 |
| Test | 16 | — |

### Class Distribution (Training Set)

| Class ID | Class Name | Boxes |
|----------|-----------|-------|
| 0 | cloud | 20 |
| 1 | debris | 565 |
| 2 | debris and snow | 367 |
| 3 | lake | 215 |
| 4 | snow | 297 |
| 5 | terrain shadow | 358 |
| 6 | waterflow | 28 |

## Architecture Design

The CBAM (Convolutional Block Attention Module) was injected after the SPPF layer in the YOLOv8m backbone (layer 10). This position was chosen because:

1. **Post-SPPF features** contain multi-scale spatial information ideal for attention refinement.
2. **Channel Attention** learns to emphasize features relevant to specific surface types (water vs. rock vs. ice).
3. **Spatial Attention** learns to focus on geographically meaningful regions (valleys, glacier termini).

## Training Configuration

- **Optimizer:** Auto-selected by Ultralytics (AdamW)
- **Learning Rate:** Auto-tuned (lr0=0.01)
- **Augmentations:** RandAugment, mosaic (disabled in last 10 epochs), HSV jitter, flip, scale
- **Confidence Threshold (Inference):** 0.45

## Robustness Considerations

The model was evaluated under challenging satellite imagery conditions:
- **Snow cover:** Distinguished from cloud via spatial texture patterns
- **Terrain shadows:** CBAM spatial attention reduces false positives in shadowed valleys
- **Debris fields:** Channel attention differentiates debris texture from rocky terrain
- **Cloud cover:** Model trained with explicit cloud class to avoid confusion

## Deployment

- **Frontend:** React.js hosted on Netlify
- **Backend:** FastAPI with YOLO inference, hosted on Hugging Face Spaces (Docker)
- **Live Demo:** [https://glof26.netlify.app](https://glof26.netlify.app)

## References

1. Woo, S., et al. "CBAM: Convolutional Block Attention Module." ECCV 2018.
2. Ultralytics YOLO11 Documentation.
3. Roboflow GLOF Dataset (COCO-Segmentation format).
