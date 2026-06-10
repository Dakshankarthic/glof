"""
model_architecture.py
=====================
Custom YOLOv8m + CBAM (Convolutional Block Attention Module) Architecture
for Glacial Lake Outburst Flood (GLOF) Detection.

Architecture: YOLOv8m backbone with injected CBAM Attention after SPPF layer.
Classes: 7 (cloud, debris, debris and snow, lake, snow, terrain shadow, waterflow)
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (CAM).
    
    Learns inter-channel relationships by aggregating spatial information
    using both average-pooling and max-pooling operations, followed by a 
    shared MLP network.
    
    Args:
        channels (int): Number of input channels.
        reduction (int): Reduction ratio for the bottleneck. Default: 16.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced_channels = max(1, channels // reduction)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        mx  = self.fc(self.max_pool(x))
        return self.sigmoid(avg + mx)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (SAM).
    
    Learns inter-spatial relationships by aggregating channel information
    using average-pooling and max-pooling along the channel axis, then
    applying a convolution to produce a spatial attention map.
    
    Args:
        kernel_size (int): Size of the convolution kernel. Default: 7.
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(
            2, 1, kernel_size,
            padding=kernel_size // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg, mx], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    
    Sequentially applies Channel Attention followed by Spatial Attention
    to refine feature maps. Injected after the SPPF layer in the YOLOv8m
    backbone to enhance the model's ability to focus on discriminative 
    regions in satellite imagery (e.g., glacial lakes vs. terrain shadows).
    
    Reference: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.
    
    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels (unused, kept for YOLO parser compatibility).
    """
    def __init__(self, c1, c2=None):
        super().__init__()
        channels = c1
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x  # Channel attention
        x = self.sa(x) * x  # Spatial attention
        return x


# YOLO11m + CBAM Architecture Configuration (YAML equivalent)
ARCHITECTURE_CONFIG = {
    "nc": 7,
    "classes": ["cloud", "debris", "debris and snow", "lake", "snow", "terrain shadow", "waterflow"],
    "backbone": "YOLO11m with CBAM injected after SPPF (layer 10)",
    "scales": {"m": [0.50, 1.00, 512]},
    "total_params": "~20.1M",
    "GFLOPs": "~68.3",
    "input_size": 640,
    "attention_position": "After SPPF, before C2PSA (backbone layer 10)",
}

if __name__ == "__main__":
    # Quick test
    model = CBAM(512)
    x = torch.randn(1, 512, 20, 20)
    out = model(x)
    print(f"CBAM Input:  {x.shape}")
    print(f"CBAM Output: {out.shape}")
    print(f"Architecture: {ARCHITECTURE_CONFIG}")
