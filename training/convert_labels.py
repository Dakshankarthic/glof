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
        anns_by_image[ann["image_id"]].append(ann)

    det_dir = os.path.join(data_root, split, "labels_det")
    os.makedirs(det_dir, exist_ok=True)
    written = 0

    for img_id, anns in anns_by_image.items():
        fname = Path(id_to_file[img_id]).stem + ".txt"
        w, h = id_to_dims[img_id]
        lines = []
        for ann in anns:
            cls = ann["category_id"] - 1
            bbox = ann.get("bbox", [])
            if bbox and len(bbox) == 4:
                x, y, bw, bh = bbox
                cx = (x + bw / 2) / w
                cy = (y + bh / 2) / h
                bw /= w
                bh /= h
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        with open(os.path.join(det_dir, fname), "w") as f:
            f.write("\n".join(lines))
        written += 1

    print(f"{split}: {written} images converted")

print("\nDone! Dataset is ready at:", data_root)
