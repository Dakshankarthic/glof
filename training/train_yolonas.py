import torch
import torch.nn as nn
from super_gradients.training import models
from super_gradients import Trainer
from super_gradients.training import dataloaders
import os

# --- 1. CBAM Attention Definition ---
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, channels):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

def main():
    print("Welcome to YOLO-NAS + CBAM Training on Windows!")
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ Warning: CUDA not detected! Training will be EXTREMELY slow on CPU.")

    data_dir = input("\nEnter the absolute path to your downloaded dataset (from the download script output): ").strip()
    
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} does not exist. Please run download_data.py first.")
        return

    # --- 2. Setup Dataset Loaders ---
    dataset_params = {
        'data_dir': data_dir,
        'train_images_dir': 'train',
        'train_labels_dir': 'train/labels_det',
        'val_images_dir': 'valid',
        'val_labels_dir': 'valid/labels_det',
        'classes': ['GLOF', 'cloud', 'debris', 'debris and snow', 'lake', 'snow', 'terrain shadow', 'waterflow']
    }

    num_classes = len(dataset_params['classes'])

    print("\nSetting up Data Loaders...")
    train_data = dataloaders.get(
        "coco_detection_yolo_format_train", 
        dataloader_params={
            "dataset_params": dataset_params,
            "batch_size": 16,  # Increased to 16 for faster training
            "num_workers": 2
        }
    )

    valid_data = dataloaders.get(
        "coco_detection_yolo_format_val", 
        dataloader_params={
            "dataset_params": dataset_params,
            "batch_size": 16,
            "num_workers": 2
        }
    )

    # --- 3. Model Setup & Injection ---
    print("\nLoading YOLO-NAS Model...")
    model = models.get(
        "yolo_nas_s",
        num_classes=num_classes,
        pretrained_weights="coco"
    )

    print("Injecting CBAM Attention...")
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            channels = module.out_channels
            attention = CBAM(channels)
            module.add_module("cbam_attention", attention)

    print("✅ CBAM Added Successfully!")

    # --- 4. Training Setup ---
    trainer = Trainer(
        experiment_name="YOLO_NAS_CBAM_GLOF",
        ckpt_root_dir=os.path.join(os.path.dirname(__file__), "checkpoints")
    )

    train_params = {
        "max_epochs": 100,
        "lr_mode": "CosineLRScheduler",
        "initial_lr": 1e-3,
        "optimizer": "AdamW",
        "optimizer_params": {"weight_decay": 5e-4},
        "ema": True,
        "mixed_precision": True,
        "warmup_mode": "linear_epoch_step",
        "warmup_initial_lr": 1e-6,
        "loss": "PPYoloELoss",
        "criterion_params": {
            "use_focal_loss": True,
            "focal_loss_gamma": 2.0
        },
        "average_best_models": True,
        "save_ckpt_epoch_list": [50, 100],
        "metric_to_watch": "mAP@0.50",
        "greater_metric_to_watch_is_better": True
    }

    # --- 5. Start Training ---
    print("\n🚀 Starting Training...")
    trainer.train(
        model=model,
        training_params=train_params,
        train_loader=train_data,
        valid_loader=valid_data
    )

if __name__ == "__main__":
    main()
