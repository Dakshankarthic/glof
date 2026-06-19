from ultralytics import YOLO

model = YOLO("yolo11l-seg.pt")

model.train(
    data="lake.yaml",
    epochs=30,
    imgsz=640,
    batch=4,
    optimizer="AdamW",
    lr0=0.001,
    patience=15,
    device=0,
    project="LakeRuns",
    name="Quick_Test"
)