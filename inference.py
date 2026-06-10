"""
inference.py
============
Run GLOF detection inference on satellite images using YOLOv8m + CBAM Attention.

Usage:
    python inference.py --source path/to/image.jpg --weights best.pt
    python inference.py --source path/to/folder/ --weights best.pt --conf 0.45
"""

import argparse
import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import ultralytics.nn.modules
import ultralytics.nn.tasks
from model_architecture import CBAM


def setup_model(weights_path, conf_threshold=0.45):
    """Load the YOLO model with custom CBAM module support."""
    # Monkey-patch Ultralytics to recognize our custom CBAM module
    ultralytics.nn.modules.CBAM = CBAM
    ultralytics.nn.tasks.CBAM = CBAM

    model = YOLO(weights_path)
    model.conf = conf_threshold
    return model


def run_inference(model, source, output_dir="results", save=True):
    """
    Run inference on a single image or directory of images.
    
    Args:
        model: Loaded YOLO model.
        source: Path to image file or directory.
        output_dir: Directory to save results.
        save: Whether to save annotated images.
    
    Returns:
        List of results from the model.
    """
    os.makedirs(output_dir, exist_ok=True)

    results = model(source, save=save, project=output_dir, name="predictions")

    # Print detection summary
    class_names = model.names
    for i, result in enumerate(results):
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            print(f"\n--- Image {i+1}: {result.path} ---")
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = class_names.get(cls_id, f"class_{cls_id}")
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                print(f"  {cls_name}: {conf:.2f} [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
        else:
            print(f"\n--- Image {i+1}: {result.path} --- No detections")

    return results


def main():
    parser = argparse.ArgumentParser(description="GLOF Detection Inference")
    parser.add_argument("--source", type=str, required=True,
                        help="Path to image file or directory")
    parser.add_argument("--weights", type=str, default="best.pt",
                        help="Path to model weights (.pt file)")
    parser.add_argument("--conf", type=float, default=0.45,
                        help="Confidence threshold (default: 0.45)")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size (default: 640)")
    args = parser.parse_args()

    print(f"Loading model from {args.weights}...")
    model = setup_model(args.weights, args.conf)

    print(f"Running inference on {args.source}...")
    results = run_inference(model, args.source, args.output)

    print(f"\nInference complete! Results saved to {args.output}/")


if __name__ == "__main__":
    main()
