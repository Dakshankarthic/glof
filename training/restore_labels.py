import json, os
from pathlib import Path
from collections import defaultdict

data_root = r"D:\glof\training\dataset"
splits = ["train", "valid", "test"]

for split in splits:
    ann_file = os.path.join(data_root, split, "_annotations.coco.json")
    
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
    labels_dir = os.path.join(data_root, split, "labels")
    os.makedirs(labels_dir, exist_ok=True)

    for img_id, anns in anns_by_image.items():
        fname = Path(id_to_file[img_id]).stem + ".txt"
        w, h  = id_to_dims[img_id]
        bbox_lines = []

        for ann in anns:
            # category_id in COCO is 1-indexed (0 is usually background or supercategory)
            # but wait, the categories list:
            # [{'id': 0, 'name': 'GLOF'}, {'id': 1, 'name': 'cloud'}, ...]
            # So if ann["category_id"] is 1, it corresponds to 'cloud'.
            # We will map it to 0-indexed for YOLO (e.g. cloud = 0, debris = 1, etc.)
            # Wait, let's keep it strictly mapped:
            cls = ann["category_id"] - 1
            if cls < 0:
                continue # Skip supercategory
            
            if ann["bbox"] and len(ann["bbox"]) == 4:
                x, y, bw, bh = ann["bbox"]
                cx = (x + bw/2) / w
                cy = (y + bh/2) / h
                bw /= w
                bh /= h
                bbox_lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        with open(os.path.join(labels_dir, fname), "w") as f:
            f.write("\n".join(bbox_lines))

        written += 1

    print(f"{split}: {written} images processed and RESTORED.")

print("All original labels have been restored successfully!")
