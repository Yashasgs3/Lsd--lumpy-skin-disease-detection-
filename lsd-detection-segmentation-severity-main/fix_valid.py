import os
import random
import shutil

# Base dataset path
base_path = "dataset/livestock_dataset_enhanced"
train_img_dir = os.path.join(base_path, "train", "images")
train_lbl_dir = os.path.join(base_path, "train", "labels")

valid_img_dir = os.path.join(base_path, "valid_balanced", "images")
valid_lbl_dir = os.path.join(base_path, "valid_balanced", "labels")

os.makedirs(valid_img_dir, exist_ok=True)
os.makedirs(valid_lbl_dir, exist_ok=True)

# Read all label files to group images by class
class_to_files = {0: [], 1: []}

for label_file in os.listdir(train_lbl_dir):
    if label_file.endswith(".txt"):
        path = os.path.join(train_lbl_dir, label_file)
        with open(path, "r") as f:
            lines = f.readlines()
            if not lines:
                continue
            # extract first class id from label file
            class_id = int(lines[0].split()[0])
            class_to_files[class_id].append(label_file)

# Set how many images per class to include
num_per_class = min(len(class_to_files[0]), len(class_to_files[1]), 100)

for cls_id, files in class_to_files.items():
    selected = random.sample(files, min(num_per_class, len(files)))
    for label_file in selected:
        img_file = os.path.splitext(label_file)[0] + ".jpg"
        img_path = os.path.join(train_img_dir, img_file)
        lbl_path = os.path.join(train_lbl_dir, label_file)

        if os.path.exists(img_path):
            shutil.copy(img_path, valid_img_dir)
            shutil.copy(lbl_path, valid_lbl_dir)

print("✅ Balanced validation set created successfully!")
