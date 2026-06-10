"""
train.py
========
Training script for GLOF Detection using YOLOv8m + CBAM Attention.

Usage:
    python train.py --data dataset/glof.yaml --epochs 30 --batch 16 --device 0
"""

import argparse
from ultralytics import YOLO
import ultralytics.nn.modules
import ultralytics.nn.tasks
from model_architecture import CBAM


def main():
    parser = argparse.ArgumentParser(description="GLOF Detection Training")
    parser.add_argument("--data", type=str, default="training/dataset/glof.yaml",
                        help="Path to dataset YAML config")
    parser.add_argument("--model", type=str, default="training/yolo11-cbam.yaml",
                        help="Path to model architecture YAML")
    parser.add_argument("--pretrained", type=str, default="yolo11m.pt",
                        help="Pretrained weights for transfer learning")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size")
    parser.add_argument("--device", type=str, default="0",
                        help="CUDA device (0 for GPU, cpu for CPU)")
    parser.add_argument("--project", type=str, default="runs",
                        help="Project directory for saving results")
    parser.add_argument("--name", type=str, default="glof_yolo11m_cbam",
                        help="Experiment name")
    args = parser.parse_args()

    # Monkey-patch Ultralytics to recognize custom CBAM module
    print("Injecting CBAM Attention module into YOLO architecture...")
    ultralytics.nn.modules.CBAM = CBAM
    ultralytics.nn.tasks.CBAM = CBAM

    # Build model from custom YAML architecture
    print(f"Building model from {args.model}...")
    model = YOLO(args.model)

    # Transfer learning: load pretrained COCO weights for matching layers
    print(f"Loading pretrained weights from {args.pretrained}...")
    model.load(args.pretrained)

    # Start training
    print(f"Starting training for {args.epochs} epochs...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
    )

    print("Training complete!")
    print(f"Results saved to {args.project}/{args.name}")


if __name__ == "__main__":
    main()
