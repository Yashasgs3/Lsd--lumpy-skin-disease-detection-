# evaluate_segmentation.py
import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from model_unet import UNet
from custom_dataset import CowMaskDataset

# ---------------- PATHS ----------------
BASE = "dataset/livestock_dataset_enhanced"
TEST_IMG_DIR = os.path.join(BASE, "valid_balanced", "images")  # same as you used in visualization
TEST_MASK_DIR = os.path.join(BASE, "valid_balanced", "masks")
MODEL_PATH = "checkpoints/best_model_segmentation.pth"

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# ---------------- LOAD MODEL ----------------
model = UNet(in_channels=3, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ---------------- METRICS ----------------
def dice_coeff(pred, target):
    smooth = 1e-6
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def iou_score(pred, target):
    smooth = 1e-6
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

# ---------------- EVALUATION ----------------
ious, dices = [], []
img_files = sorted([f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith((".jpg", ".png"))])

print(f"\n📊 Evaluating {len(img_files)} images using {MODEL_PATH}\n")

for img_name in img_files:
    img_path = os.path.join(TEST_IMG_DIR, img_name)
    mask_path = os.path.join(TEST_MASK_DIR, os.path.splitext(img_name)[0] + "_mask.png")
    if not os.path.exists(mask_path):
        print(f"⚠️ Missing GT mask for {img_name}")
        continue

    # Read and preprocess image
    img = cv2.imread(img_path)
    h0, w0 = img.shape[:2]
    inp = transforms.ToPILImage()(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    inp = transforms.Resize((256, 256))(inp)
    inp = transforms.ToTensor()(inp).unsqueeze(0).to(DEVICE)

    # Predict
    with torch.no_grad():
        pred = model(inp).squeeze().cpu().numpy()

    # Resize to original
    pred_resized = cv2.resize(pred, (w0, h0))
    pred_bin = (pred_resized > 0.5).astype(np.uint8)
    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    gt_mask = (gt_mask > 127).astype(np.uint8)

    # Metrics
    dice = dice_coeff(pred_bin, gt_mask)
    iou = iou_score(pred_bin, gt_mask)
    dices.append(dice)
    ious.append(iou)

    print(f"{img_name:<20} → IoU: {iou:.4f}, Dice: {dice:.4f}")

# ---------------- RESULTS ----------------
print("\n📈 MEAN IoU:", np.mean(ious))
print("📈 MEAN Dice:", np.mean(dices))
