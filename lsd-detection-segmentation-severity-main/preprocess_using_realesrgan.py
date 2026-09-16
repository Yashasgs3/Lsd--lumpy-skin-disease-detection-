import sys
sys.path.append(r"C:\Users\samik\OneDrive\Desktop\Samiksha\project\Real-ESRGAN")
from realesrgan import RealESRGAN

import torch
from PIL import Image
import os

# ------------------------
# 1️⃣ Set device (GPU if available)
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------
# 2️⃣ Load RealESRGAN model
# ------------------------
# You can use "RealESRGAN_x4plus" for general 4x upscaling
model = RealESRGAN(device, scale=4)

# Make sure the weights file exists. You can download from:
# https://github.com/xinntao/Real-ESRGAN/releases
weights_path = "weights/RealESRGAN_x4plus.pth"
if not os.path.isfile(weights_path):
    raise FileNotFoundError(f"Weights not found: {weights_path}. Download from the official repo!")

model.load_weights(weights_path)
model.eval()

# ------------------------
# 3️⃣ Load your image
# ------------------------
input_path = r"C:\Users\samik\OneDrive\Desktop\Samiksha\project\dataset\livestock_dataset\test\images\Lumpy_Skin_134_png.rf.a07b03036ee50af1b738cbe73defc469.jpg"
img = Image.open(input_path).convert("RGB")

# ------------------------
# 4️⃣ Apply super-resolution
# ------------------------
with torch.no_grad():  # avoids memory issues
    sr_image = model.predict(img)

# ------------------------
# 5️⃣ Save the enhanced image
# ------------------------
output_path = r"C:\Users\samik\OneDrive\Desktop\Samiksha\project\dataset\livestock_dataset_enhanced\test\images\image1_sr.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
sr_image.save(output_path)

print("Super-resolved image saved at:", output_path)
