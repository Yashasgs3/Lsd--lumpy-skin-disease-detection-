import os
from collections import Counter

# Change this path to the folder where your dataset is stored
base_dir = r"dataset\livestock_dataset_enhanced"

splits = ["train", "valid_balanced", "test"]

for split in splits:
    label_dir = os.path.join(base_dir, split, "labels")
    counts = Counter()
    total_files = 0

    for f in os.listdir(label_dir):
        if not f.endswith(".txt"):
            continue
        total_files += 1
        with open(os.path.join(label_dir, f)) as file:
            lines = [line.strip() for line in file.readlines() if line.strip()]
        for line in lines:
            cls = int(line.split()[0])  # first number = class ID
            counts[cls] += 1

    print(f"\n📂 {split.upper()} SET:")
    print(f"  Total label files: {total_files}")
    print(f"  Object count per class: {dict(counts)}")
