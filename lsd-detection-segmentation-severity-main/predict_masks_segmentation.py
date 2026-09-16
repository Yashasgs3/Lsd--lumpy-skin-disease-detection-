# red overlay, not used
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import os
from model_unet import UNet  # make sure your UNet class is in this file

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best_model.pth"   # path to your trained segmentation model
IMAGE_PATH = r"C:\Users\samik\OneDrive\Pictures\Screenshots\Screenshot 2025-10-31 071619.png"  # path to your test image
SAVE_DIR = "runs/segmentation/final_predictions"
THRESHOLD = 0.53  # adjust if needed

# ---------------- CREATE SAVE FOLDER ----------------
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------- LOAD MODEL ----------------
model = UNet(in_channels=3, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ---------------- PREPROCESS IMAGE ----------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

image = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(DEVICE)

# ---------------- PREDICT ----------------
with torch.no_grad():
    pred = model(input_tensor).cpu().squeeze().numpy()

# Threshold prediction to binary mask
binary_mask = (pred > THRESHOLD).astype(np.uint8)

# ---------------- CREATE OVERLAY ----------------
image_np = np.array(image.resize((256, 256)))
red_mask = np.zeros_like(image_np)
red_mask[..., 0] = binary_mask * 255  # Red channel

overlay = (0.6 * image_np + 0.4 * red_mask).astype(np.uint8)

# ---------------- SAVE RESULTS ----------------
base_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
save_path = os.path.join(SAVE_DIR, f"{base_name}_overlay.jpg")
Image.fromarray(overlay).save(save_path)
print(f"✅ Saved overlay to: {save_path}")

# ---------------- SHOW RESULTS ----------------
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(image_np)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(binary_mask, cmap="gray")
plt.title("Predicted Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(overlay)
plt.title("Overlay (Mask + Image)")
plt.axis("off")

plt.tight_layout()
plt.show()
