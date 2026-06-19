# GLOFeagles '26 — GLOF Image Segmentation

**Team:** GLOFeagles
**Challenge:** NCVPRIPG 2026 GLOFeagles Challenge

## Overview

YOLO11l-Seg based segmentation model for automated Glacial Lake Outburst Flood (GLOF) detection from satellite imagery. The model identifies and segments glacial lakes under varying environmental conditions such as snow cover, terrain shadows, debris cover, moraine-dammed lakes, and varying turbidity.

## Repository Structure

├── best.pt                   # Trained PyTorch model
├── best.onnx                 # ONNX export
├── train.py
├── inference.py
├── model_architecture.py
├── utils.py
├── GLOF_Lake_Detection.ipynb
├── segmentation_masks.zip
├── technical_report.pdf
├── evaluation_report.pdf
├── requirements.txt
└── README.md

## Setup

```bash
pip install -r requirements.txt
```

## Run Inference

```bash
python inference.py --model best.pt --input images/ --output masks/
```

## Dataset

The dataset contains satellite imagery with glacial lakes observed under:

* Moraine Dammed
* Snow Cover
* Terrain Shadow
* Debris Cover
* Varying Turbidity

## Model

* Architecture: YOLO11m-Seg
* Image Size: 640 × 640
* Optimizer: AdamW
* Epochs: 30

## Evaluation Results

| Metric    | Score |
| --------- | ----: |
| mAP50     | 78.82 |
| mAP50-95  | 47.78 |
| Precision | 81.40 |
| Recall    | 70.85 |

## Deliverables

* Trained Model Files (.pt, .onnx)
* Source Code
* Python Notebook
* Segmentation Masks
* Technical Report
* Evaluation Report
* Requirements File

## Explanation Video

https://youtu.be/jKZPJ0i8uwg?feature=shared

## Conclusion

The proposed YOLO11l-Seg model successfully performs automatic segmentation of glacial lakes from satellite imagery and can assist in monitoring glacial lake outburst flood risks in mountainous regions.
