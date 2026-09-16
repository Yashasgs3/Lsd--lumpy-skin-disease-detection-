# train_unet.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from custom_dataset import CowMaskDataset
from model_unet import UNet
import gc

# ---------------- PATHS ----------------
BASE = "dataset/livestock_dataset_enhanced"
TRAIN_IMG_DIR = os.path.join(BASE, "test_manual", "images")   # manually labeled data
TRAIN_MASK_DIR = os.path.join(BASE, "test_manual", "masks_retry")
SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------- CONFIG ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 4
epochs = 50
lr = 1e-4

# ---------------- TRANSFORMS ----------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# ---------------- MEMORY HELPER ----------------
def free_memory():
    torch.cuda.empty_cache()
    gc.collect()

# ---------------- METRIC FUNCTIONS ----------------
def dice_coefficient(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum()
    return (2.0 * intersection + eps) / (pred.sum() + target.sum() + eps)

def iou_score(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + eps) / (union + eps)

# ---------------- TRAIN FUNCTION ----------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        preds = model(imgs)
        preds = preds.squeeze(1)
        masks = masks.squeeze(1)

        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()

        # Metrics
        dice = dice_coefficient(preds, masks).item()
        iou = iou_score(preds, masks).item()

        total_loss += loss.item() * imgs.size(0)
        total_dice += dice * imgs.size(0)
        total_iou += iou * imgs.size(0)

        # memory cleanup per batch
        del imgs, masks, preds, loss
        free_memory()

    n = len(loader.dataset)
    return total_loss / n, total_dice / n, total_iou / n

# ---------------- MAIN TRAIN LOOP ----------------
if __name__ == "__main__":
    train_ds = CowMaskDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        print(f"\n🟩 Epoch {epoch}/{epochs} ----------------------------")
        epoch_loss, epoch_dice, epoch_iou = train_one_epoch(model, train_loader, optimizer, criterion, device)

        print(f"📊 Loss: {epoch_loss:.4f} | Dice: {epoch_dice:.4f} | IoU: {epoch_iou:.4f}")

        # Save best model by lowest loss
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "unet_best_masks_retry.pth"))
            print("  ✅ Saved best model")

        free_memory()

    print(f"\n🏁 Training complete. Best loss: {best_loss:.4f}")
