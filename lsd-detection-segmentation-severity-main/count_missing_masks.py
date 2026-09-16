import os

base_dir = "dataset/livestock_dataset_enhanced"
folders = ["train", "valid_balanced", "test"]

for folder in folders:
    img_dir = os.path.join(base_dir, folder, "images")
    mask_dir = os.path.join(base_dir, folder, "masks_sam")

    images = set([os.path.splitext(f)[0] for f in os.listdir(img_dir)])
    masks = set([os.path.splitext(f)[0] for f in os.listdir(mask_dir)])

    missing_masks = images - masks
    extra_masks = masks - images

    print(f"\nFolder: {folder}")
    print(f"Total images: {len(images)}, Total masks: {len(masks)}")
    print(f"Images missing masks: {len(missing_masks)}")
    if missing_masks:
        print("Missing masks:", list(missing_masks)[:10])
    print(f"Masks without images: {len(extra_masks)}")
    if extra_masks:
        print("Extra masks:", list(extra_masks)[:10])
