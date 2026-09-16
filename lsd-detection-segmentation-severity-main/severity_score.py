# severity_score_from_masks.py
import os
import cv2
import numpy as np
import pandas as pd

# ---------------- PATHS ----------------
BASE = "dataset/livestock_dataset_enhanced"
IMG_DIR = os.path.join(BASE, "valid_balanced", "images")   # or "train" if needed
PRED_MASK_DIR = os.path.join(BASE, "valid_balanced", "masks")  # folder with your predicted masks
SAVE_CSV = "runs/severity_scores.csv"

# ---------------- THRESHOLDS ----------------
LEVELS = {
    "Healthy": (0, 1),
    "Mild": (1, 5),
    "Moderate": (5, 30),
    "Severe": (30, 100)
}

# ---------------- MAIN ----------------
results = []

# Preload all image file paths by base name (ignore extension)
image_files = {}
for f in os.listdir(IMG_DIR):
    base, _ = os.path.splitext(f)
    image_files[base] = os.path.join(IMG_DIR, f)

for fname in sorted(os.listdir(PRED_MASK_DIR)):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    base, _ = os.path.splitext(fname)
    mask_path = os.path.join(PRED_MASK_DIR, fname)

    # Find image with same base name (any extension)
    img_path = image_files.get(base)
    if img_path is None:
        print(f"⚠️ Missing image for {fname}")
        continue

    # Read files
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(img_path)

    if mask is None or img is None:
        print(f"⚠️ Could not read {fname}")
        continue

    # Resize mask to match image
    h, w = img.shape[:2]
    mask = cv2.resize(mask, (w, h))

    # Count infected area (non-zero pixels)
    infected_pixels = np.count_nonzero(mask > 127)
    total_pixels = h * w
    severity = (infected_pixels / total_pixels) * 100

    # Assign level
    level = next(k for k, (low, high) in LEVELS.items() if low <= severity < high)

    results.append({
        "Image": fname,
        "Severity (%)": round(severity, 2),
        "Level": level
    })

    print(f"{fname}: Severity = {severity:.2f}% ({level})")

# Save results
df = pd.DataFrame(results)
os.makedirs(os.path.dirname(SAVE_CSV), exist_ok=True)
df.to_csv(SAVE_CSV, index=False)

print(f"\n✅ Saved severity results to: {SAVE_CSV}")
