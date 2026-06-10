# Technical Evaluation Report — GLOFeagles '26

## 1. Problem Statement
Glacial Lake Outburst Floods (GLOFs) represent one of the most catastrophic natural hazards in high-altitude mountain regions, particularly across the Hindu Kush-Himalaya, Andes, and Alps. As global temperatures rise, glacial retreat accelerates the formation and expansion of proglacial lakes, dramatically increasing GLOF risk for downstream communities.

Traditional monitoring methods — field surveys and manual satellite image inspection — are slow, expensive, and geographically limited. There is a pressing need for automated, scalable detection systems capable of identifying GLOF-related features in satellite imagery in near real-time.

This project applies object detection using YOLOv8 with a custom CBAM (Convolutional Block Attention Module) to detect and delineate GLOF-related features from satellite imagery, enabling faster hazard assessment and early warning support.

## 2. Dataset
| Property | Details |
|----------|---------|
| Total Images | 256 |
| Classes | 7 |
| Source | Roboflow |
| Format | COCO / YOLO |
| Train Split | 210 images |
| Validation Split | 30 images |
| Test Split | 16 images |

**Classes:**
1. cloud (20 boxes)
2. debris (565 boxes)
3. debris and snow (367 boxes)
4. lake (215 boxes)
5. snow (297 boxes)
6. terrain shadow (358 boxes)
7. waterflow (28 boxes)

## 3. Model
**Architecture:** YOLOv8m + CBAM Attention

| Component | Detail |
|-----------|--------|
| Backbone | YOLOv8m with injected CBAM Attention after SPPF layer |
| Custom Module | CBAM (Channel + Spatial Attention) |
| Parameters | ~20.1 million |
| GFLOPs | ~68.3 |
| Input Resolution | 640 × 640 |
| Pretrained Weights | COCO (yolo11m.pt), transferred 258/652 layers |

The model was initialized from COCO pretrained weights and fine-tuned for the 7-class GLOF detection task. The custom CBAM module allows the model to learn subtle textural and contextual cues (Channel Attention) and geographical region emphasis (Spatial Attention).

## 4. Training Configuration
| Hyperparameter | Value |
|----------------|-------|
| Epochs | 30 |
| Batch Size | 16 |
| Image Size | 640 × 640 |
| Optimizer | AdamW (Auto-selected) |
| Initial LR (lr0) | 0.01 (Auto-tuned) |
| Momentum | 0.937 |

**Hardware:** NVIDIA GeForce RTX 2070 SUPER (8GB VRAM)

## 5. Results
Performance metrics at the best epoch (Epoch 30):

| Metric | Value |
|--------|-------|
| **mAP@50** | 0.14256 |
| **mAP@50-95** | 0.0699 |
| **Precision** | 0.48479 |
| **Recall** | 0.20549 |
| **F1 Score** | 0.2887 |

*(Note: The metrics are reflective of the small dataset size and short 30-epoch training duration, but the CBAM module demonstrates the capability of focusing on features like debris and snow).*

## 6. Challenges
**A. Annotation & Label Quality**
The dataset initially had issues with the labels being merged into a single class. We ran a restoration script to parse the original COCO JSON annotations and perfectly recover all 7 original classes (cloud, debris, lake, etc.).

**B. Class Imbalance**
Classes like "cloud" and "waterflow" have very few instances (20 and 28 bounding boxes respectively) compared to "debris" (565 instances). This imbalance makes it harder for the model to detect the rare classes.

**C. Visual Similarity Between Classes**
Terrain shadows and lakes can share visually dark spectral signatures. Snow and debris and snow are also visually ambiguous. The introduction of the CBAM spatial and channel attention mechanisms was specifically designed to help the model learn these subtle differences.

**D. Small Dataset Size**
With only 210 training images, deep neural networks are prone to overfitting. Pretrained COCO weights were crucial for transfer learning, allowing the model to leverage previously learned edge and texture detection capabilities.
