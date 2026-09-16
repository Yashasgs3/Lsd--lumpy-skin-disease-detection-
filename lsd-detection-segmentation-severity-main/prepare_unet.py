import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ----------------------------
# Paths
# ----------------------------Q
BASE_DIR = "dataset/livestock_dataset_enhanced"
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train/images")
TRAIN_MASK_DIR = os.path.join(BASE_DIR, "train/masks")
VALID_IMG_DIR = os.path.join(BASE_DIR, "valid_balanced/images")
VALID_MASK_DIR = os.path.join(BASE_DIR, "valid_balanced/masks")

# ----------------------------
# Dataset
# ----------------------------
class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace(".jpg", ".png"))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Ensure mask is 1-channel float tensor
        mask = (mask > 0).float()
        return image, mask

# ----------------------------
# Transforms & DataLoader
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

train_dataset = SegmentationDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, transform)
valid_dataset = SegmentationDataset(VALID_IMG_DIR, VALID_MASK_DIR, transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False)
