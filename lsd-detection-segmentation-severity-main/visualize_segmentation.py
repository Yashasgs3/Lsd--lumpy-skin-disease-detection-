import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model_unet import UNet  # your model file

# ---------------- PATHS ----------------
TEST_IMG_DIR = "dataset/livestock_dataset_enhanced/test_2"   # folder with unseen test images
SAVE_DIR = "runs/test2_bboxes"
MODEL_PATH = "checkpoints/unet_model_segmentation_retry.pth"

os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------- DEVICE ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- MODEL ----------------
model = UNet(in_channels=3, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),   # same as training
    transforms.ToTensor(),
])

# ---------------- CONFIG ----------------
THRESHOLD = 0.6         # Lower constant threshold → catch faint lesions
AREA_THRESHOLD = 250     # Ignore very tiny regions
KERNEL = np.ones((3, 3), np.uint8)

# ---------------- FUNCTION ----------------
def predict_and_draw_polygons(img_path):
    # Load and preprocess image
    image = Image.open(img_path).convert("RGB")
    orig_w, orig_h = image.size
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Predict mask
    with torch.no_grad():
        pred = model(input_tensor)
        sig = torch.sigmoid(pred).squeeze().cpu().numpy()

    # Resize mask to original size
    sig_resized = cv2.resize(sig, (orig_w, orig_h))

    # Step 1: Threshold
    binary_mask = (sig_resized > THRESHOLD).astype(np.uint8) * 255

    # Step 2: Gentle smoothing
    binary_mask = cv2.GaussianBlur(binary_mask, (3, 3), 0)

    # Step 3: Morphological cleanup and dilation
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, KERNEL)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, KERNEL)
    binary_mask = cv2.dilate(binary_mask, KERNEL, iterations=2)

    # Step 4: Find detailed polygon contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Convert original image to OpenCV BGR format
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    bbox_count = 0
    polygon_points_total = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < AREA_THRESHOLD:
            continue

        # Count polygon points
        polygon_points_total += len(contour)

        # Draw polygon contour and bounding box
        cv2.drawContours(img_bgr, [contour], -1, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 0, 255), 2)
        bbox_count += 1

    # Step 5: Label image with counts
    label_text = f"Boxes: {bbox_count} | Points: {polygon_points_total}"
    cv2.putText(img_bgr, label_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Save output
    save_path = os.path.join(SAVE_DIR, os.path.basename(img_path))
    cv2.imwrite(save_path, img_bgr)

    # Optional: also save the binary mask for inspection
    cv2.imwrite(os.path.join(SAVE_DIR, f"mask_{os.path.basename(img_path)}"), binary_mask)

    # Log
    if bbox_count > 0:
        print(f"✅ {os.path.basename(img_path)} → {bbox_count} boxes ({polygon_points_total} pts) saved")
    else:
        print(f"⚪ {os.path.basename(img_path)} → Likely healthy (no boxes)")

# ---------------- RUN ----------------
if __name__ == "__main__":
    image_files = [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"\n📦 Processing {len(image_files)} unseen test images...\n")
    for img_name in image_files:
        img_path = os.path.join(TEST_IMG_DIR, img_name)
        predict_and_draw_polygons(img_path)
