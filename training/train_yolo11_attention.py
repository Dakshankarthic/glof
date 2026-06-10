from ultralytics import YOLO
import ultralytics.nn.modules
import ultralytics.nn.tasks
from custom_modules import CBAM

def main():
    print("Monkey-patching YOLO11 to include CBAM Attention module...")
    
    # Inject our custom CBAM class into Ultralytics internal modules
    # This allows the YAML parser to understand the 'CBAM' block in our config
    ultralytics.nn.modules.CBAM = CBAM
    ultralytics.nn.tasks.CBAM = CBAM

    print("Building custom YOLO11m+CBAM model from YAML...")
    
    # We load the architecture from our new YAML file
    # We CANNOT load the pre-trained weights for the entire model because the architecture is different,
    # but YOLO will automatically attempt to transfer weights for the layers that match!
    model = YOLO("yolo11-cbam.yaml")
    
    # Load COCO pretrained weights for all matching layers! This is CRITICAL for small datasets.
    model.load("yolo11m.pt") 

    print("Starting training with CBAM Attention...")
    
    # Train the custom model
    results = model.train(
        data=r"D:\glof\training\dataset\glof.yaml",
        epochs=30,
        batch=16,
        imgsz=640,
        device=0,  # RTX 2070 Super
        project="runs",
        name="glof_yolo11m_cbam"
    )
    print("Training complete!")

if __name__ == '__main__':
    main()
