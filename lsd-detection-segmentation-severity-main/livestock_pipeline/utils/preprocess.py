import cv2
import torch
from torchvision import transforms
from PIL import Image

def load_image(img_path):
    """Loads an image from disk and converts to RGB."""
    img = Image.open(img_path).convert("RGB")
    return img

def preprocess_for_unet(img, size=(256, 256)):
    """Resize and convert image to tensor for UNet."""
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0)  # add batch dimension

def preprocess_for_yolo(img_path):
    """YOLO takes file paths directly, but this ensures the file is valid."""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image at {img_path}")
    return img_path  # YOLOv8 model() accepts file path
