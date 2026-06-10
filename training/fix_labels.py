import os
import glob

data_root = r"D:\glof\training\dataset"
splits = ["train", "valid", "test"]

for split in splits:
    labels_dir = os.path.join(data_root, split, "labels")
    for txt_file in glob.glob(os.path.join(labels_dir, "*.txt")):
        with open(txt_file, "r") as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                # Force the class ID to be 0 (since it's a 1-class problem: GLOF)
                parts[0] = "0"
                new_lines.append(" ".join(parts))
        
        with open(txt_file, "w") as f:
            f.write("\n".join(new_lines) + "\n")

print("All label classes forced to 0!")
