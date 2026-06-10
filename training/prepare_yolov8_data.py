import os
import shutil
from glob import glob

data_root = r"D:\glof\training\dataset"
splits = ["train", "valid", "test"]

for split in splits:
    split_dir = os.path.join(data_root, split)
    images_dir = os.path.join(split_dir, "images")
    labels_dir = os.path.join(split_dir, "labels")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Move images
    for img in glob(os.path.join(split_dir, "*.jpg")) + glob(os.path.join(split_dir, "*.png")):
        shutil.move(img, os.path.join(images_dir, os.path.basename(img)))
        
    # Move labels
    old_labels_dir = os.path.join(split_dir, "labels_det")
    if os.path.exists(old_labels_dir):
        for txt in glob(os.path.join(old_labels_dir, "*.txt")):
            shutil.move(txt, os.path.join(labels_dir, os.path.basename(txt)))
        # Clean up old labels dir
        shutil.rmtree(old_labels_dir)

# Create yaml file
yaml_content = f"""path: {data_root}
train: train/images
val: valid/images
test: test/images

nc: 1
names: ['GLOF']
"""
with open(os.path.join(data_root, "glof.yaml"), "w") as f:
    f.write(yaml_content)

print("Dataset reorganized for YOLOv8 and glof.yaml created!")
