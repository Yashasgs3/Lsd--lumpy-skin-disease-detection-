# metrics_segmentation.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from custom_dataset import CowMaskDataset
from model_unet import UNet
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE = "dataset/livestock_dataset_enhanced"
VAL_IMG_DIR = f"{BASE}/valid_balanced/images"
VAL_MASK_DIR = f"{BASE}/valid_balanced/masks"
MODEL_PATH = "checkpoints/unet_model_segmentation_retry.pth"

# IoU and Dice
def iou_score(pred, target, eps=1e-6):
    pred = (pred > 0.55).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + eps) / (union + eps)

def dice_score(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2 * intersection + eps) / (pred.sum() + target.sum() + eps)

def main():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    val_dataset = CowMaskDataset(VAL_IMG_DIR, VAL_MASK_DIR, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    ious, dices, accs = [], [], []

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = model(imgs)

            preds = preds.squeeze(1)
            masks = masks.squeeze(1)

            iou = iou_score(preds, masks)
            dice = dice_score(preds, masks)
            acc = ((preds > 0.5).float() == masks).float().mean()

            ious.append(iou.item())
            dices.append(dice.item())
            accs.append(acc.item())

    print("\n✅ Validation Complete!")
    print(f"Mean IoU:   {np.mean(ious):.4f}")
    print(f"Mean Dice:  {np.mean(dices):.4f}")
    print(f"Pixel Acc:  {np.mean(accs):.4f}")

    return {"IoU": np.mean(ious), "Dice": np.mean(dices), "Pixel_Accuracy": np.mean(accs)}

if __name__ == "__main__":
    metrics = main()
    print(metrics)
