# app/utils/image_processing.py
import torch
from PIL import Image
from torchvision import transforms
from typing import Tuple

def preprocess_image(image: Image.Image, target_size: Tuple[int, int]) -> torch.Tensor:
    """Preprocess image for model inference"""
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Apply transformations and add batch dimension
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def allowed_file(filename: str, allowed_extensions: set) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions
