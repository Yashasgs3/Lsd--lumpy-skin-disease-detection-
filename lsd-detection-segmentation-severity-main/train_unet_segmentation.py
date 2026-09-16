import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from custom_dataset import CowMaskDataset
from model_unet import UNet
import gc

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE = "dataset/livestock_dataset_enhanced"

TRAIN_IMG_DIR = os.path.join(BASE, "train", "images")
TRAIN_MASK_DIR = os.path.join(BASE, "train", "masks")
VAL_IMG_DIR = os.path.join(BASE, "valid_balanced", "images")
VAL_MASK_DIR = os.path.join(BASE, "valid_balanced", "masks")

BATCH_SIZE = 4
LR = 1e-4
EPOCHS = 50
PATIENCE = 5
MODEL_PATH = "checkpoints/unet_model_segmentation_retry.pth"

# ---------------- TRANSFORMS ----------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# ---------------- DATA ----------------
train_dataset = CowMaskDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, transform=transform)
val_dataset = CowMaskDataset(VAL_IMG_DIR, VAL_MASK_DIR, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

# ---------------- MODEL ----------------
model = UNet(in_channels=3, out_channels=1).to(DEVICE)
criterion = nn.BCELoss()  # Since UNet already has sigmoid
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

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

# ---------------- TRAINING ----------------
best_val_loss = float('inf')
no_improve = 0

for epoch in range(1, EPOCHS + 1):
    print(f"\n🟩 Epoch {epoch}/{EPOCHS}")
    model.train()
    train_loss, train_dice, train_iou = 0, 0, 0

    for imgs, masks in train_loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()

        preds = model(imgs)
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()

        # Metrics
        preds_bin = (preds > 0.5).float()
        train_dice += dice_coeff(preds_bin, masks).item()
        train_iou += iou_score(preds_bin, masks).item()
        train_loss += loss.item()

        # Free up memory
        del imgs, masks, preds, loss
        torch.cuda.empty_cache()
        gc.collect()

    train_loss /= len(train_loader)
    train_dice /= len(train_loader)
    train_iou /= len(train_loader)

    # ---------------- VALIDATION ----------------
    model.eval()
    val_loss, val_dice, val_iou = 0, 0, 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = model(imgs)
            loss = criterion(preds, masks)
            preds_bin = (preds > 0.5).float()

            val_loss += loss.item()
            val_dice += dice_coeff(preds_bin, masks).item()
            val_iou += iou_score(preds_bin, masks).item()

            del imgs, masks, preds, loss
            torch.cuda.empty_cache()
            gc.collect()

    val_loss /= len(val_loader)
    val_dice /= len(val_loader)
    val_iou /= len(val_loader)

    print(f"📊 Train Loss: {train_loss:.4f} | Dice: {train_dice:.4f} | IoU: {train_iou:.4f}")
    print(f"📈 Val   Loss: {val_loss:.4f} | Dice: {val_dice:.4f} | IoU: {val_iou:.4f}")

    # ---------------- EARLY STOPPING ----------------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve = 0
        torch.save(model.state_dict(), MODEL_PATH)
        print("✅ Model improved — saved!")
    else:
        no_improve += 1
        print(f"⏸ No improvement for {no_improve}/{PATIENCE} epoch(s).")
        if no_improve >= PATIENCE:
            print("🛑 Early stopping triggered.")
            break

print("\n🎉 Training complete! Best model saved to:", MODEL_PATH)
