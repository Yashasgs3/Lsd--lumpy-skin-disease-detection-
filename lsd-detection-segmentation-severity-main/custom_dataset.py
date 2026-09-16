# custom_dataset.py
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CowMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, transform=None, mask_transform=None):
        """
        If mask_dir is None → dataset for inference (no masks).
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform if mask_transform is not None else transform
        self.images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png"))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.mask_dir is None:
            if self.transform:
                image = self.transform(image)
            return image, img_name  # inference mode

        mask_path = os.path.join(self.mask_dir, os.path.splitext(img_name)[0] + "_mask.png")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.mask_dir, os.path.splitext(img_name)[0] + ".png")
        mask = Image.open(mask_path).convert("L")

        # consistent resizing for both
        resize_transform = transforms.Resize((256, 256), interpolation=Image.NEAREST)
        image = resize_transform(image)
        mask = resize_transform(mask)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = (mask > 0).float()
        return image, mask
