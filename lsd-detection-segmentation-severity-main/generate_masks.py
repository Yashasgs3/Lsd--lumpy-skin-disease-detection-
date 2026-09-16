import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

# ---------------- PATHS ----------------
DATASET_DIR = "dataset/livestock_dataset_enhanced"
ANNOTATION_DIR = os.path.join(DATASET_DIR, "test_manual", "annotations")
IMAGES_DIR = os.path.join(DATASET_DIR, "test_manual", "images")
MASKS_DIR = os.path.join(DATASET_DIR, "test_manual", "masks_retry")

os.makedirs(MASKS_DIR, exist_ok=True)

# ---------------- PARSE & CREATE MASKS ----------------
for xml_file in os.listdir(ANNOTATION_DIR):
    if not xml_file.endswith(".xml"):
        continue

    xml_path = os.path.join(ANNOTATION_DIR, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for image_tag in root.findall("image"):
        image_name = image_tag.attrib["name"]
        image_path = os.path.join(IMAGES_DIR, image_name)

        if not os.path.exists(image_path):
            print(f"⚠️ Skipping {image_name}: not found in test_manual/images/")
            continue

        # Load image to get size
        img = cv2.imread(image_path)
        if img is None:
            print(f"⚠️ Could not read {image_name}. Skipping.")
            continue

        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Fill polygons or points as white (255) on black mask
        for shape_tag in image_tag.findall("polygon") + image_tag.findall("points"):
            pts = np.array(
                [[float(x), float(y)] for x, y in 
                (point.split(",") for point in shape_tag.attrib["points"].split(";"))],
                np.int32
            ).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)


        # Save mask
        mask_name = os.path.splitext(image_name)[0] + ".png"
        mask_path = os.path.join(MASKS_DIR, mask_name)
        cv2.imwrite(mask_path, mask)

        if np.any(mask > 0):
            print(f"✅ Diseased mask created for {image_name}")
        else:
            print(f"⚪ Healthy (blank) mask created for {image_name}")

print("\n🎉 All masks saved inside dataset/livestock_dataset_enhanced/test_manual/masks_retry/")
