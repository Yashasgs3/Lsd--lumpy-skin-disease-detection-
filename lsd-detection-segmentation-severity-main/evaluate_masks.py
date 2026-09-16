import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from model_unet import UNet

# ---------------- PATHS ----------------
BASE = "dataset/livestock_dataset_enhanced"
TEST_IMG_DIR = os.path.join(BASE, "train", "images")
TEST_MASK_DIR = os.path.join(BASE, "train", "masks")
CHECKPOINT = "checkpoints/unet_best_masks_retry.pth"

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# ---------------- LOAD MODEL ----------------
model = UNet(in_channels=3, out_channels=1).to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()

# ---------------- METRICS ----------------
def iou_and_dice(gt, pred):
    gt = (gt > 127).astype(np.uint8)
    pred = (pred > 127).astype(np.uint8)
    inter = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    iou = inter / union if union > 0 else 1.0 if inter == 0 else 0.0
    dice = 2 * inter / (gt.sum() + pred.sum()) if (gt.sum() + pred.sum()) > 0 else 1.0
    return iou, dice

# ---------------- EVALUATION ----------------
ious, dices = [], []
img_files = sorted([f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

for img_name in img_files:
    img_path = os.path.join(TEST_IMG_DIR, img_name)
    base_name = os.path.splitext(img_name)[0]

    # Try both mask naming styles
    possible_masks = [
        os.path.join(TEST_MASK_DIR, base_name + "_mask.png"),
        os.path.join(TEST_MASK_DIR, base_name + ".png"),
        os.path.join(TEST_MASK_DIR, base_name + ".jpg")
    ]

    gt_mask_path = next((p for p in possible_masks if os.path.exists(p)), None)
    if gt_mask_path is None:
        print(f"⚠️ Missing GT mask for {img_name}")
        continue

    # Load image
    img = cv2.imread(img_path)
    h0, w0 = img.shape[:2]
    inp = transforms.ToPILImage()(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    inp = transforms.Resize((256, 256))(inp)
    inp = transforms.ToTensor()(inp).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        pred = model(inp).squeeze().cpu().numpy()  # [256,256] in [0,1]

    pred_resized = cv2.resize((pred * 255).astype(np.uint8), (w0, h0))
    _, pred_bin = cv2.threshold(pred_resized, 127, 255, cv2.THRESH_BINARY)
    gt = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)

    # Compute metrics
    iou, dice = iou_and_dice(gt, pred_bin)
    ious.append(iou)
    dices.append(dice)

    print(f"{img_name} → IoU: {iou:.4f}, Dice: {dice:.4f}")

# ---------------- SUMMARY ----------------
if len(ious) > 0:
    print(f"\n✅ MEAN IoU: {np.mean(ious):.4f}")
    print(f"✅ MEAN Dice: {np.mean(dices):.4f}")
else:
    print("\n⚠️ No ground truth masks were found — please check file names or extensions.")
