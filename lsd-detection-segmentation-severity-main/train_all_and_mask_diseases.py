import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from model_unet import UNet

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CLASS_NAMES = ("healthy", "lumpy", "foot-and-mouth")
IMAGE_SIZE = (256, 256)


class FlatSegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE, interpolation=Image.Resampling.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        mask = Image.open(self.mask_paths[index]).convert("L")
        return self.image_transform(image), (self.mask_transform(mask) > 0).float()


def image_files(folder):
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint)


def predict_mask(model, image_path, device, threshold):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])
    with torch.no_grad():
        prediction = model(transform(image).unsqueeze(0).to(device))
    mask = (prediction.squeeze().cpu().numpy() >= threshold).astype(np.uint8) * 255
    return Image.fromarray(mask).resize(image.size, Image.Resampling.NEAREST)


def create_bootstrap_masks(model, class_images, working_masks, device, threshold):
    for class_name, paths in class_images.items():
        for image_path in paths:
            output_path = working_masks / f"{class_name}__{image_path.name}.png"
            if output_path.exists():
                continue
            if class_name == "healthy":
                mask = Image.new("L", Image.open(image_path).size, 0)
            else:
                mask = predict_mask(model, image_path, device, threshold)
            mask.save(output_path)


def train_model(model, dataset, device, epochs, batch_size, learning_rate):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        average_loss = total_loss / len(dataset)
        print(f"Epoch {epoch}/{epochs} - loss: {average_loss:.4f}")


def save_disease_masks(model, class_images, output_dir, device, threshold):
    output_dir.mkdir(parents=True, exist_ok=True)
    for class_name in ("lumpy", "foot-and-mouth"):
        for image_path in class_images[class_name]:
            mask = predict_mask(model, image_path, device, threshold)
            output_path = output_dir / f"{class_name}__{image_path.stem}.png"
            mask.save(output_path)
            print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train on healthy and diseased cow images, then mask diseased images only."
    )
    parser.add_argument("--images", default="dataset/livestock_dataset_enhanced/train/images")
    parser.add_argument("--masks", default="dataset/livestock_dataset_enhanced/train/masks")
    parser.add_argument("--checkpoint", default="checkpoints/unet_best_masks_retry.pth")
    parser.add_argument("--output-checkpoint", default="checkpoints/unet_all_classes.pth")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.55)
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")

    image_root = Path(args.images)
    output_masks = Path(args.masks)
    checkpoint_path = Path(args.checkpoint)
    working_root = image_root.parent / "_segmentation_training"
    working_images = working_root / "images"
    working_masks = working_root / "masks"

    class_images = {}
    for class_name in CLASS_NAMES:
        class_dir = image_root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        class_images[class_name] = image_files(class_dir)
        if not class_images[class_name]:
            raise FileNotFoundError(f"No images found in {class_dir}")
        print(f"{class_name}: {len(class_images[class_name])} image(s)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = UNet(in_channels=3, out_channels=1, backbone_weights=None).to(device)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing bootstrap checkpoint: {checkpoint_path}")
    load_checkpoint(model, checkpoint_path, device)

    working_images.mkdir(parents=True, exist_ok=True)
    working_masks.mkdir(parents=True, exist_ok=True)
    flat_images = []
    flat_masks = []
    for class_name, paths in class_images.items():
        for image_path in paths:
            flat_name = f"{class_name}__{image_path.name}"
            copied_image = working_images / flat_name
            shutil.copy2(image_path, copied_image)
            flat_images.append(copied_image)
            flat_masks.append(working_masks / f"{flat_name}.png")

    print("Creating bootstrap masks. Healthy masks are blank; diseased masks use the existing checkpoint.")
    create_bootstrap_masks(model, class_images, working_masks, device, args.threshold)
    print("Training on all three classes...")
    train_model(
        model,
        FlatSegmentationDataset(flat_images, flat_masks),
        device,
        args.epochs,
        args.batch_size,
        1e-4,
    )
    Path(args.output_checkpoint).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_checkpoint)
    save_disease_masks(model, class_images, output_masks, device, args.threshold)
    print(f"Done. Disease masks are in {output_masks}")


if __name__ == "__main__":
    main()
