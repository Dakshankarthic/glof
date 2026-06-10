import os, cv2, numpy as np
import sys
sys.path.insert(0, r'D:\glof\backend')

import ultralytics.nn.modules, ultralytics.nn.tasks
from custom_modules import CBAM
ultralytics.nn.modules.CBAM = CBAM
ultralytics.nn.tasks.CBAM = CBAM
from ultralytics import YOLO

model = YOLO(r'D:\glof\best.pt')
os.makedirs(r'D:\glof\segmentation_masks', exist_ok=True)

palette = {0:(0,0,0), 1:(255,255,255), 2:(0,128,128), 3:(0,200,200),
           4:(255,200,0), 5:(200,200,220), 6:(80,80,80), 7:(255,100,50)}

count = 0
for split in ['train', 'valid', 'test']:
    img_dir = os.path.join(r'D:\glof\training\dataset', split, 'images')
    if not os.path.exists(img_dir):
        continue
    for fname in os.listdir(img_dir):
        if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        img = cv2.imread(os.path.join(img_dir, fname))
        if img is None:
            continue
        h, w = img.shape[:2]
        results = model(img, conf=0.45, verbose=False)
        mask = np.zeros((h, w), dtype=np.uint8)
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0]) + 1
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = cls_id
        stem = os.path.splitext(fname)[0]
        cv2.imwrite(os.path.join(r'D:\glof\segmentation_masks', stem + '_mask.png'), mask)
        color = np.zeros((h, w, 3), dtype=np.uint8)
        for cid, c in palette.items():
            color[mask == cid] = c
        cv2.imwrite(os.path.join(r'D:\glof\segmentation_masks', stem + '_color.png'), color)
        count += 1

print(f'Done! Generated masks for {count} images')
