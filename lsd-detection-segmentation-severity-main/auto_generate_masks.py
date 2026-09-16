import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model_unet import UNet

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IMAGE_SIZE = (256, 256)


def load_model(checkpoint_path, device):
    model = UNet(in_channels=3, out_channels=1).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def make_overlay(image, mask):
    image_array = np.array(image).copy()
    red = np.zeros_like(image_array)
    red[..., 0] = 255
    mask_pixels = mask > 0
    image_array[mask_pixels] = (
        image_array[mask_pixels] * 0.55 + red[mask_pixels] * 0.45
    ).astype(np.uint8)
    return Image.fromarray(image_array)


def generate_masks(model, image_dir, mask_dir, overlay_dir, threshold, device):
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])
    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            original_size = image.size
            input_tensor = transform(image).unsqueeze(0).to(device)
            prediction = model(input_tensor).squeeze().cpu().numpy()
            binary_mask = (prediction >= threshold).astype(np.uint8) * 255
            mask_image = Image.fromarray(binary_mask).resize(
                original_size, Image.Resampling.NEAREST
            )
            mask_image.save(mask_dir / f"{image_path.stem}.png")
            overlay = make_overlay(image, np.array(mask_image))
            overlay.save(overlay_dir / f"{image_path.stem}_overlay.jpg", quality=95)
            print(f"Created {mask_dir / image_path.stem}.png")

    print(f"Finished {len(image_paths)} image(s).")
    print(f"Review overlays in: {overlay_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate disease segmentation masks with a trained UNet."
    )
    parser.add_argument(
        "--images",
        default="dataset/livestock_dataset_enhanced/train/images",
        help="Folder containing cow images.",
    )
    parser.add_argument(
        "--masks",
        default="dataset/livestock_dataset_enhanced/auto_masks/masks",
        help="Folder where PNG masks will be saved.",
    )
    parser.add_argument(
        "--overlays",
        default="dataset/livestock_dataset_enhanced/auto_masks/overlays",
        help="Folder where red overlay previews will be saved.",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/unet_best_masks_retry.pth",
        help="Trained UNet checkpoint.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="Probability threshold from 0 to 1 used to create binary masks.",
    )
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")

    image_dir = Path(args.images)
    checkpoint_path = Path(args.checkpoint)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image folder does not exist: {image_dir}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}. Train the UNet first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = load_model(checkpoint_path, device)
    generate_masks(
        model,
        image_dir,
        Path(args.masks),
        Path(args.overlays),
        args.threshold,
        device,
    )


if __name__ == "__main__":
    main()
