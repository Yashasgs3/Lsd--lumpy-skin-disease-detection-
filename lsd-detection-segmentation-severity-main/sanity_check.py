from custom_dataset import CowMaskDataset
from torchvision import transforms

dataset = CowMaskDataset("dataset/livestock_dataset_enhanced/valid_balanced/images", "dataset/livestock_dataset_enhanced/valid_balanced/masks_sam", transform=transforms.ToTensor())

img, mask = dataset[0]
print("Image shape:", img.shape)
print("Mask shape:", mask.shape)

