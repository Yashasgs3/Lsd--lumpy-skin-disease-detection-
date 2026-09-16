import os
import shutil
from PIL import Image
import random

# Paths
train_images_dir = "dataset/livestock_dataset_enhanced/train/images"
train_labels_dir = "dataset/livestock_dataset_enhanced/train/labels"

# Identify minority class (infected = class 1)
minority_class = "1"  # label in YOLO txt files
all_labels = os.listdir(train_labels_dir)

# Collect files of minority class
minority_files = []
for lbl_file in all_labels:
    with open(os.path.join(train_labels_dir, lbl_file), "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith(minority_class):
                minority_files.append(lbl_file.replace(".txt", ".jpg"))  # assuming .jpg images
                break

print(f"Found {len(minority_files)} minority class images")

# Number of majority class images
majority_count = 2784  # from your earlier stats
minority_count = len(minority_files)
needed = majority_count - minority_count

print(f"Need to create {needed} more minority images to balance")

# Duplicate and augment images
for i in range(needed):
    original_file = random.choice(minority_files)
    img_path = os.path.join(train_images_dir, original_file)
    lbl_path = os.path.join(train_labels_dir, original_file.replace(".jpg", ".txt"))

    # Load image
    img = Image.open(img_path)

    # Apply simple augmentation: flip or rotate randomly
    aug_type = random.choice(["flip", "rotate", "none"])
    if aug_type == "flip":
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif aug_type == "rotate":
        img = img.rotate(random.choice([90, 180, 270]))

    # Save new image and label
    new_name = f"{original_file.split('.')[0]}_aug{i}.jpg"
    img.save(os.path.join(train_images_dir, new_name))
    shutil.copy(lbl_path, os.path.join(train_labels_dir, new_name.replace(".jpg", ".txt")))

print("Minority class balancing done!")
