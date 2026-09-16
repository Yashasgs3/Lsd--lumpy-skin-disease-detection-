# infer_generate_masks.py
import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from custom_dataset import CowMaskDataset
from model_unet import UNet

BASE = "dataset/livestock_dataset_enhanced"
SPLITS = ["train", "valid_balanced"]
IMG_SUB = "images"
OUT_SUB = "masks"   # folder to save generated masks
CHECKPOINT = "checkpoints/unet_best_masks_retry.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model
model = UNet(in_channels=3, out_channels=1).to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
])

for split in SPLITS:
    img_dir = os.path.join(BASE, split, IMG_SUB)
    out_dir = os.path.join(BASE, split, OUT_SUB)
    os.makedirs(out_dir, exist_ok=True)
    ds = CowMaskDataset(img_dir, mask_dir=None, transform=transform)
    for img_tensor, img_name in ds:
        # img_tensor: [3,256,256]
        inp = img_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(inp)                # [1,1,256,256], sigmoid already applied
        pred = pred.squeeze().cpu().numpy()   # H,W in [0,1]
        # resize back to original size (optional: preserve original resolution)
        # load original size:
        orig = cv2.imread(os.path.join(img_dir, img_name))
        h0, w0 = orig.shape[:2]
        pred_resized = cv2.resize((pred*255).astype(np.uint8), (w0, h0))
        # threshold
        _, bin_mask = cv2.threshold(pred_resized, 127, 255, cv2.THRESH_BINARY)
        out_name = os.path.splitext(img_name)[0] + ".png"
        cv2.imwrite(os.path.join(out_dir, out_name), bin_mask)
        print(f"Saved {split}/masks/{out_name}")

print("Done generating pseudo masks for train & valid.")
