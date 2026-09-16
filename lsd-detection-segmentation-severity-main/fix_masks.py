# not used
import os
from PIL import Image
import numpy as np

base_dir = "dataset/livestock_dataset_enhanced"
folders = ["train", "valid_balanced", "test"]

for folder in folders:
    img_dir = os.path.join(base_dir, folder, "images")
    mask_dir = os.path.join(base_dir, folder, "masks_sam")

    images = sorted(os.listdir(img_dir))
    masks = sorted(os.listdir(mask_dir))

    # Map image base_name -> full image filename
    image_map = {os.path.splitext(f)[0]: f for f in images}

    for base_name, img_file in image_map.items():
        # Find all masks that contain this image base_name
        matched_masks = [f for f in masks if base_name in f and os.path.exists(os.path.join(mask_dir, f))]

        if not matched_masks:
            print(f"⚠️ No mask found for {img_file}")
            continue

        # Combine multiple masks safely
        combined_mask = None
        for i, mask_file in enumerate(matched_masks):
            mask_path = os.path.join(mask_dir, mask_file)

            # Extra safety check
            if not os.path.exists(mask_path):
                print(f"⚠️ Skipping missing mask: {mask_file}")
                continue

            mask_img = np.array(Image.open(mask_path).convert("L"))

            if combined_mask is None:
                combined_mask = mask_img
            else:
                combined_mask = np.maximum(combined_mask, mask_img)

            # Remove old mask after processing
            if len(matched_masks) > 1:
                if i > 0:
                    os.remove(mask_path)

        # Save final mask with exactly the same name as the image
        new_mask_name = os.path.splitext(img_file)[0] + ".png"
        new_mask_path = os.path.join(mask_dir, new_mask_name)
        Image.fromarray(combined_mask).save(new_mask_path)

    print(f"✅ Masks fixed to match images in {folder} folder")
