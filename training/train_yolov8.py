from ultralytics import YOLO

def main():
    print("Loading YOLOv8m model...")
    # Load the user's provided model
    model = YOLO(r"C:\Users\DK11\Downloads\yolov8m.pt")
    
    print("Starting training...")
    # Train the model
    results = model.train(
        data=r"D:\glof\training\dataset\glof.yaml",
        epochs=100,
        batch=16,
        imgsz=640,
        device=0,  # RTX 2070 Super
        project="runs",
        name="glof_yolov8m"
    )
    print("Training complete!")

if __name__ == '__main__':
    main()
