import json, os, yaml
from pathlib import Path
from collections import defaultdict
from roboflow import Roboflow

def download_and_convert(api_key):
    # 1. Download
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("medwins-workspace").project("glof-wyr2m")
    dataset = project.version(4).download("coco-segmentation")
    
    data_root = dataset.location
    print(f"\nDownloaded to {data_root}")

    # 2. Convert COCO to YOLO detection format
    splits=["train", "valid", "test"]
    for split in splits:
        ann_file = f"{data_root}/{split}/_annotations.coco.json"
        
        if not os.path.exists(ann_file):
            print(f"Skipping {split}, annotation file not found.")
            continue

        with open(ann_file) as f:
            coco = json.load(f)

        id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
        id_to_dims = {img["id"]: (img["width"], img["height"]) for img in coco["images"]}

        anns_by_image = defaultdict(list)
        for ann in coco["annotations"]:
            bbox = ann.get("bbox", [])
            anns_by_image[ann["image_id"]].append({
                "category_id": ann["category_id"],
                "bbox": bbox
            })

        written = 0
        det_dir = f"{data_root}/{split}/labels_det"
        os.makedirs(det_dir, exist_ok=True)

        for img_id, anns in anns_by_image.items():
            fname = Path(id_to_file[img_id]).stem + ".txt"
            w, h  = id_to_dims[img_id]
            bbox_lines = []

            for ann in anns:
                cls = ann["category_id"] - 1
                if ann["bbox"] and len(ann["bbox"]) == 4:
                    x, y, bw, bh = ann["bbox"]
                    cx = (x + bw/2) / w
                    cy = (y + bh/2) / h
                    bw /= w
                    bh /= h
                    bbox_lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            with open(f"{det_dir}/{fname}", "w") as f:
                f.write("\n".join(bbox_lines))

            written += 1

        print(f"{split}: {written} images processed for detection labels.")

    print(f"\n✅ Data conversion complete. Your dataset is ready at: {data_root}")
    print("Please copy this absolute path, you will need it when running train_yolonas.py!")

if __name__ == "__main__":
    print("--- GLOF Dataset Downloader ---")
    key = input("Enter your Roboflow API key: ").strip()
    download_and_convert(key)
