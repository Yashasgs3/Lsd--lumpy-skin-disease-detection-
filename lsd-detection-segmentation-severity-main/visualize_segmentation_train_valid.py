import os
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
from torchvision import transforms
from custom_dataset import CowMaskDataset
from model_unet import UNet

# ---------------- PATHS ----------------
BASE = "dataset/livestock_dataset_enhanced"
TEST_IMG_DIR = os.path.join(BASE, "valid_balanced", "images")
TEST_MASK_DIR = os.path.join(BASE, "valid_balanced", "masks")  # ground truth masks
SAVE_DIR = "runs/visual_comparison_results_with_severity"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------- DEVICE ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- TRANSFORMS ----------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# ---------------- DATASET ----------------
test_dataset = CowMaskDataset(TEST_IMG_DIR, mask_dir=None, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ---------------- MODEL ----------------
model = UNet(in_channels=3, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load("checkpoints/unet_model_segmentation_retry.pth", map_location=DEVICE))
model.eval()

# ---------------- MAIN ----------------
def visualize_results(threshold=0.55):
    print(f"\n📦 Processing {len(test_dataset)} test images (threshold = {threshold})...\n")

    with torch.no_grad():
        for idx, (img, img_name) in enumerate(test_loader):
            name = img_name[0]
            img_path = os.path.join(TEST_IMG_DIR, name)
            mask_path = os.path.join(TEST_MASK_DIR, os.path.splitext(name)[0] + ".png")

            # Load original image
            orig = cv2.imread(img_path)
            orig_resized = cv2.resize(orig, (256, 256))

            # Load actual mask (if available)
            gt_mask = np.zeros((256, 256), np.uint8)
            if os.path.exists(mask_path):
                gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                gt_mask = cv2.resize(gt_mask, (256, 256))

            # Predict mask
            img = img.to(DEVICE)
            pred = model(img)
            sig = torch.sigmoid(pred).squeeze().cpu().numpy()
            pred_mask = (sig > threshold).astype(np.uint8) * 255

            # Smooth and refine predicted mask
            pred_mask = cv2.GaussianBlur(pred_mask, (3, 3), 0)
            pred_mask = cv2.medianBlur(pred_mask, 3)
            kernel = np.ones((3, 3), np.uint8)
            pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
            pred_mask = cv2.dilate(pred_mask, kernel, iterations=2)

            # ---------------- SEVERITY SCORE ----------------
            infected_pixels = np.count_nonzero(pred_mask > 127)
            total_pixels = pred_mask.shape[0] * pred_mask.shape[1]
            severity = (infected_pixels / total_pixels) * 100

            # Classification levels
            if severity < 1:
                level = "Healthy"
            elif severity < 5:
                level = "Mild"
            elif severity < 20:
                level = "Moderate"
            else:
                level = "Severe"


            # ---------------- CONTOURS & BOXES ----------------
            contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            bbox_count = 0
            total_points = 0
            vis_bbox = orig_resized.copy()

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 700:
                    continue

                total_points += len(contour)
                x, y, w, h = cv2.boundingRect(contour)
                cv2.drawContours(vis_bbox, [contour], -1, (0, 255, 0), 2)
                cv2.rectangle(vis_bbox, (x, y), (x + w, y + h), (0, 0, 255), 2)
                bbox_count += 1

            # ---------------- LABEL TEXT ----------------
           # ---------------- LABEL TEXT ----------------
            label_text1 = f"Boxes: {bbox_count} | Points: {total_points}"
            label_text2 = f"Severity: {severity:.2f}% ({level})"

            # First line (boxes & points)
            cv2.putText(vis_bbox, label_text1, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            # Second line (severity info, below)
            cv2.putText(vis_bbox, label_text2, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


            # Convert single-channel masks to color
            gt_mask_colored = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR)
            pred_mask_colored = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)

            # Combine all into one image horizontally
            combined = np.hstack((orig_resized, gt_mask_colored, pred_mask_colored, vis_bbox))
            cv2.putText(combined, "Original | Ground Truth | Predicted | Boxes + Severity", (10, 245),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Save visualization
            save_path = os.path.join(SAVE_DIR, name)
            cv2.imwrite(save_path, combined)

            print(f"✅ {name}: {bbox_count} boxes ({total_points} pts) | Severity: {severity:.2f}% ({level}) → saved to {save_path}")

# ---------------- RUN ----------------
if __name__ == "__main__":
    visualize_results(threshold=0.55)
