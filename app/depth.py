# depth.py

import torch
import torch.nn as nn
import cv2
import numpy as np

class SimpleDepthCNN(nn.Module):
    def __init__(self):
        super(SimpleDepthCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),  # Input RGB
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),

            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),

            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # output depth between 0 and 1
        )

    def forward(self, x):
        return self.encoder(x)

def load_depth_model():
    model = SimpleDepthCNN()
    model.eval()
    return model

def estimate_depth(model, image):
    img_resized = cv2.resize(image, (128, 128))  # smaller size = faster inference
    img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0  # Normalize to [0,1]
    with torch.no_grad():
        depth_pred = model(img_tensor)
        depth_np = depth_pred.squeeze().cpu().numpy()
        depth_resized = cv2.resize(depth_np, (image.shape[1], image.shape[0]))  # Resize to original image size
    return depth_resized * 255  # scale to 0–255 for compatibility
