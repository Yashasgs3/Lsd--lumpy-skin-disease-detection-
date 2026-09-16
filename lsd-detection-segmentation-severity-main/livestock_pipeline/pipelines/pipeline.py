import os
import cv2
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO
from livestock_pipeline.utils.model_unet import UNet

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DETECTION_MODEL_PATH = "livestock_pipeline/models/detection_model.pt"
SEGMENTATION_MODEL_PATH = "livestock_pipeline/models/segmentation_model.pth"
RESULTS_DIR = "livestock_pipeline/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

LEVELS = {
    "Healthy": (0, 1),
    "Mild": (1, 5),
    "Moderate": (5, 20),
    "Severe": (20, 100)
}

# ---------------- LOAD MODELS ----------------
def load_models():
    print("📦 Loading models...")
    det_model = YOLO(DETECTION_MODEL_PATH)
    seg_model = UNet(in_channels=3, out_channels=1)
    seg_model.load_state_dict(torch.load(SEGMENTATION_MODEL_PATH, map_location=DEVICE))
    seg_model.to(DEVICE).eval()
    print("✅ Models loaded successfully.")
    return det_model, seg_model

# ---------------- DETECTION ----------------
def detect_animal(det_model, image_path, conf_threshold=0.3):
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Could not read image:", image_path)
        return None, None, None

    # Run detection directly on image (no internal resizing issues)
    results = det_model(img, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy()

    h, w = img.shape[:2]
    print(f"🖼️ Image size: {w}x{h}")
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        print(f"Box {i}: ({x1:.1f}, {y1:.1f}) → ({x2:.1f}, {y2:.1f})  "
            f"Width={x2-x1:.1f}, Height={y2-y1:.1f}")


    confidences = results.boxes.conf.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)

    if len(boxes) == 0:
        print("⚠️ No detections found.")
        return None, None, img

    det_vis = img.copy()
    h, w, _ = img.shape

    # Draw all detected boxes
    for (x1, y1, x2, y2), conf, cls in zip(boxes, confidences, classes):
        if conf < conf_threshold:
            continue
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(det_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{results.names[cls]}: {conf:.2f}"
        cv2.putText(det_vis, label, (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)

    # Select the highest-confidence detection
    top_idx = int(np.argmax(confidences))
    x1, y1, x2, y2 = map(int, boxes[top_idx])

    # ✅ Ensure coordinates are within image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    # ✅ Crop using correct bounding box
    crop = img[y1:y2, x1:x2].copy()

    print(f"🟩 Detected box: ({x1}, {y1}) → ({x2}, {y2}) | Confidence: {confidences[top_idx]:.2f}")

    return crop, (x1, y1, x2, y2), det_vis



# ---------------- SEGMENTATION ----------------
def segment_animal(seg_model, crop):
    img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (256, 256)) / 255.0
    tensor = torch.tensor(img_resized.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        pred = seg_model(tensor)
        mask = torch.sigmoid(pred).squeeze().cpu().numpy()

    mask_bin = (mask > 0.55).astype(np.uint8) * 255
    mask_bin = cv2.resize(mask_bin, (crop.shape[1], crop.shape[0]))
    return mask_bin

# ---------------- SEVERITY ----------------
def compute_severity(mask):
    infected_pixels = np.count_nonzero(mask > 127)
    total_pixels = mask.size
    severity = (infected_pixels / total_pixels) * 100
    level = next(k for k, (low, high) in LEVELS.items() if low <= severity < high)
    return round(severity, 2), level

# ---------------- OVERLAY ----------------
def create_overlay(original, mask, box, severity, level):
    x1, y1, x2, y2 = box
    overlay = original.copy()
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Resize mask to box size and overlay it
    resized_mask = cv2.resize(mask_colored, (x2 - x1, y2 - y1))
    alpha = 0.5
    overlay[y1:y2, x1:x2] = cv2.addWeighted(overlay[y1:y2, x1:x2], 1 - alpha, resized_mask, alpha, 0)

    # Draw contours on overlay
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_points, bbox_count = 0, 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 700:
            continue
        total_points += len(contour)
        bx, by, bw, bh = cv2.boundingRect(contour)
        cv2.rectangle(overlay, (x1 + bx, y1 + by), (x1 + bx + bw, y1 + by + bh), (0, 255, 0), 2)
        cv2.drawContours(overlay, [contour + np.array([[x1, y1]])], -1, (255, 0, 0), 1)
        bbox_count += 1

    # Add severity text
    cv2.putText(overlay, f"Boxes: {bbox_count} | Points: {total_points}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(overlay, f"Severity: {severity:.2f}% ({level})", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return overlay

# ---------------- MAIN PIPELINE ----------------
def process_image(image_path):
    det_model, seg_model = load_models()
    crop, box, det_vis = detect_animal(det_model, image_path)

    if crop is None:
        return

    mask = segment_animal(seg_model, crop)
    severity, level = compute_severity(mask)

    img = cv2.imread(image_path)
    overlay = create_overlay(img, mask, box, severity, level)

    # Combine all views horizontally
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_resized = cv2.resize(mask_colored, (256, 256))
    det_resized = cv2.resize(det_vis, (256, 256))
    overlay_resized = cv2.resize(overlay, (256, 256))
    orig_resized = cv2.resize(img, (256, 256))

    combined = np.hstack((orig_resized, det_resized, mask_resized, overlay_resized))
    cv2.putText(combined, "Original | Detection | Mask | Overlay + Severity", (10, 245),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    save_path = os.path.join(RESULTS_DIR, f"combined_{os.path.basename(image_path)}")
    cv2.imwrite(save_path, combined)
    print(f"✅ Saved combined visualization → {save_path}")

    # Save CSV record
    csv_path = os.path.join(RESULTS_DIR, "severity_results.csv")
    df = pd.DataFrame([[os.path.basename(image_path), severity, level]],
                      columns=["Image", "Severity (%)", "Level"])
    df.to_csv(csv_path, mode='a', index=False, header=not os.path.exists(csv_path))

# ---------------- RUN ----------------
if __name__ == "__main__":
    test_img = "dataset/livestock_dataset_enhanced/test_2/healthy-4.jpg"  # replace with your image
    process_image(test_img)
