"""
utils.py
========
Utility functions for GLOF Detection pipeline.
Includes: image preprocessing, mask generation, metric computation, and visualization.
"""

import os
import cv2
import json
import numpy as np
from pathlib import Path


def load_image(path, size=640):
    """Load and resize an image for inference."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.resize(img, (size, size))
    return img


def generate_segmentation_mask(image, results, num_classes=7):
    """
    Generate a segmentation mask from YOLO detection results.
    Each detected bounding box region is filled with its class ID.
    
    Args:
        image: Original image (H, W, 3).
        results: YOLO results object.
        num_classes: Number of classes.
    
    Returns:
        mask: (H, W) numpy array with class IDs.
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0]) + 1  # 0 = background
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                mask[y1:y2, x1:x2] = cls_id

    return mask


def save_mask(mask, output_path):
    """Save segmentation mask as a PNG image."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(str(output_path), mask)


def colorize_mask(mask, num_classes=7):
    """
    Convert a class-ID mask into a color visualization.
    
    Args:
        mask: (H, W) array with class IDs (0=background).
        num_classes: Number of classes.
    
    Returns:
        color_mask: (H, W, 3) BGR color image.
    """
    # Color palette for each class
    palette = {
        0: (0, 0, 0),         # background - black
        1: (255, 255, 255),   # cloud - white
        2: (0, 128, 128),     # debris - teal
        3: (0, 200, 200),     # debris and snow - cyan-ish
        4: (255, 200, 0),     # lake - bright blue
        5: (200, 200, 220),   # snow - light grey
        6: (80, 80, 80),      # terrain shadow - dark grey
        7: (255, 100, 50),    # waterflow - orange-blue
    }

    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for cls_id, color in palette.items():
        color_mask[mask == cls_id] = color

    return color_mask


def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1, box2: [x1, y1, x2, y2] format.
    
    Returns:
        IoU value (float).
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def compute_precision_recall(tp, fp, fn):
    """Compute precision and recall from counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


def compute_f1(precision, recall):
    """Compute F1 score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def batch_generate_masks(model, image_dir, output_dir="segmentation_masks"):
    """
    Generate segmentation masks for all images in a directory.
    
    Args:
        model: Loaded YOLO model.
        image_dir: Path to directory with images.
        output_dir: Output directory for masks.
    """
    os.makedirs(output_dir, exist_ok=True)
    image_dir = Path(image_dir)

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    images = [f for f in image_dir.iterdir() if f.suffix.lower() in image_extensions]

    print(f"Generating masks for {len(images)} images...")

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        results = model(img, conf=0.45)
        mask = generate_segmentation_mask(img, results)

        mask_path = os.path.join(output_dir, img_path.stem + "_mask.png")
        save_mask(mask, mask_path)

        color_path = os.path.join(output_dir, img_path.stem + "_color.png")
        color_mask = colorize_mask(mask)
        cv2.imwrite(color_path, color_mask)

    print(f"Masks saved to {output_dir}/")


if __name__ == "__main__":
    print("GLOF Detection Utilities")
    print("Available functions: load_image, generate_segmentation_mask, compute_iou, compute_f1")
