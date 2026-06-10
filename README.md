# GLOF Detection — GLOFeagles '26 Challenge

> **Live Demo:** [https://glof26.netlify.app](https://glof26.netlify.app)  
> **YouTube Link:** [https://youtu.be/jKZPJ0i8uwg?feature=shared](https://youtu.be/jKZPJ0i8uwg?feature=shared)

━━━━━━━━━━━━━━━━━━━━━━
TEAM INNOVATORS 
━━━━━━━━━━━━━━━━━━━━━━
Medwin Manuel  S 
Dakshan Karthic S
Vishagan E
Mentor : Uma Maheswari Pandyan

## Glacial Lake Outburst Flood Detection using YOLOv8 + CBAM Attention

This repository contains our solution for the **GLOFeagles '26 Challenge** — a satellite imagery-based detection system for identifying Glacial Lake Outburst Flood (GLOF) hazards using deep learning.

---

## 🏗️ Architecture

- **Base Model:** YOLOv8m (Ultralytics)
- **Custom Enhancement:** CBAM (Convolutional Block Attention Module) injected after the SPPF layer
- **Classes (7):** `cloud`, `debris`, `debris and snow`, `lake`, `snow`, `terrain shadow`, `waterflow`
- **Input Size:** 640×640
- **Parameters:** ~20.1M
- **Transfer Learning:** COCO pretrained weights with fine-tuning

## 📁 Repository Structure

```
glof/
├── inference.py              # Run inference on images
├── train.py                  # Training script
├── model_architecture.py     # CBAM Attention module + architecture details
├── utils.py                  # Utility functions
├── requirements.txt          # Python dependencies
├── best.pt                   # Trained model weights
├── glof_notebook.ipynb       # Full pipeline notebook
├── yolov8-cbam.yaml          # Model architecture YAML config
├── evaluation_report.md      # Technical evaluation report
├── backend/                  # FastAPI backend (Hugging Face Spaces)
│   ├── main.py
│   ├── custom_modules.py
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/                 # React frontend (Netlify)
│   └── src/
├── training/                 # Training data & scripts
│   ├── dataset/
│   ├── custom_modules.py
│   └── train_yolo11_attention.py
└── segmentation_masks/       # Generated segmentation masks
```

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run Inference
```bash
python inference.py --source path/to/satellite/image.jpg --weights best.pt
```

### Train Model
```bash
python train.py --data dataset/glof.yaml --epochs 30 --batch 16 --device 0
```

## 🌐 Live Demo

- **Frontend:** [https://glof26.netlify.app](https://glof26.netlify.app)
- **Backend API:** [https://dk1112-glof-detection-api.hf.space](https://dk1112-glof-detection-api.hf.space)

## 📊 Evaluation Metrics

| Metric | Value |
|--------|-------|
| mAP@50 | See evaluation_report.md |
| Precision | See evaluation_report.md |
| Recall | See evaluation_report.md |
| F1 Score | See evaluation_report.md |



## 📜 License

This project is developed for the GLOFeagles '26 Challenge.
