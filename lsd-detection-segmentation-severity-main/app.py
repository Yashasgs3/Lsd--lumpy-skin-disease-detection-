import csv
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO

from livestock_pipeline.utils.model_unet import UNet

ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "livestock_pipeline" / "results" / "uploads"
RESULTS_DIR = ROOT / "livestock_pipeline" / "results"
DETECTION_MODEL_PATH = ROOT / "livestock_pipeline" / "models" / "detection_model.pt"
SEGMENTATION_MODEL_PATH = ROOT / "livestock_pipeline" / "models" / "segmentation_model.pth"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
IMAGE_SIZE = (256, 256)
SEGMENTATION_THRESHOLD = 0.55
LEVELS = (
    ("Healthy", 0, 1),
    ("Mild", 1, 5),
    ("Moderate", 5, 20),
    ("Severe", 20, 100.01),
)

app = Flask(__name__, template_folder="frontend")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_detection_model = None
_segmentation_model = None


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_models():
    global _detection_model, _segmentation_model
    if _detection_model is None:
        _detection_model = YOLO(str(DETECTION_MODEL_PATH))
    if _segmentation_model is None:
        _segmentation_model = UNet(
            in_channels=3,
            out_channels=1,
            backbone_weights=None,
        ).to(DEVICE)
        _segmentation_model.load_state_dict(
            torch.load(SEGMENTATION_MODEL_PATH, map_location=DEVICE)
        )
        _segmentation_model.eval()
    return _detection_model, _segmentation_model


def normalize_label(label):
    value = label.lower().replace("_", "-").strip()
    if "foot" in value or "mouth" in value or value in {"fmd", "fm disease"}:
        return "Foot-and-mouth"
    if "lumpy" in value or "skin" in value:
        return "Lumpy"
    if "healthy" in value or "normal" in value:
        return "Healthy"
    return label.title()


def severity_level(severity):
    for label, low, high in LEVELS:
        if low <= severity < high:
            return label
    return "Severe"


def segment_crop(model, crop):
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMAGE_SIZE).astype(np.float32) / 255.0
    tensor = torch.from_numpy(resized.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        prediction = model(tensor).squeeze().cpu().numpy()
    mask = (prediction >= SEGMENTATION_THRESHOLD).astype(np.uint8) * 255
    return cv2.resize(mask, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)


def make_overlay(image, mask, boxes, severity, level):
    overlay = image.copy()
    if mask.any():
        color_mask = np.zeros_like(overlay)
        color_mask[:, :, 2] = mask
        overlay = cv2.addWeighted(overlay, 0.55, color_mask, 0.45, 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) >= 40:
                cv2.drawContours(overlay, [contour], -1, (255, 90, 70), 2)

    for box in boxes:
        x1, y1, x2, y2, label, confidence = box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (70, 230, 150), 3)
        cv2.putText(overlay, f"{label} {confidence:.0%}", (x1, max(30, y1 - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        overlay,
        f"Severity: {severity:.2f}% ({level})",
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay


def create_combined(original, detection_view, mask, overlay):
    target_size = (320, 240)
    original_view = cv2.resize(original, target_size)
    detection_view = cv2.resize(detection_view, target_size)
    mask_view = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), target_size)
    overlay_view = cv2.resize(overlay, target_size)
    combined = np.hstack((original_view, detection_view, mask_view, overlay_view))
    labels = ("Original", "Detection", "Disease mask", "Overlay")
    for index, label in enumerate(labels):
        cv2.putText(
            combined,
            label,
            (index * target_size[0] + 12, 226),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return combined


def append_result(row):
    csv_path = RESULTS_DIR / "severity_results.csv"
    columns = ["timestamp", "image", "diagnosis", "confidence", "severity_percent", "severity_level", "result_file"]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def analyze_image(image_path):
    image_path = Path(image_path)
    detector, segmenter = load_models()
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("The uploaded file is not a readable image.")

    results = detector(image, conf=0.15, iou=0.5, max_det=30, verbose=False)[0]
    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        raise ValueError("No cow was detected. Upload a clear image showing one cow.")

    confidences = boxes.conf.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy().astype(int)
    height, width = image.shape[:2]
    detection_view = image.copy()
    full_mask = np.zeros((height, width), dtype=np.uint8)
    detection_boxes = []
    detections = []
    for box, confidence, class_id in zip(boxes.xyxy.detach().cpu().numpy(), confidences, classes):
        bx1, by1, bx2, by2 = box.astype(int)
        bx1, by1 = max(0, bx1), max(0, by1)
        bx2, by2 = min(width, bx2), min(height, by2)
        if bx2 <= bx1 or by2 <= by1:
            continue
        name = normalize_label(results.names[int(class_id)])
        cv2.rectangle(detection_view, (bx1, by1), (bx2, by2), (70, 110, 255), 3)
        cv2.putText(detection_view, f"{name} {confidence:.0%}", (bx1, max(28, by1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        crop = image[by1:by2, bx1:bx2]
        crop_mask = np.zeros(crop.shape[:2], dtype=np.uint8) if name == "Healthy" else segment_crop(segmenter, crop)
        full_mask[by1:by2, bx1:bx2] = np.maximum(full_mask[by1:by2, bx1:bx2], crop_mask)
        detection_boxes.append((bx1, by1, bx2, by2, name, float(confidence)))
        detections.append({"diagnosis": name, "confidence": round(float(confidence) * 100, 1)})

    if not detection_boxes:
        raise ValueError("The detected cow regions are invalid. Try another image.")
    primary = max(detection_boxes, key=lambda item: item[5])
    diagnosis = primary[4]
    confidence = primary[5]
    cow_area = sum((box[2] - box[0]) * (box[3] - box[1]) for box in detection_boxes)
    infected_pixels = int(np.count_nonzero(full_mask > 127))
    severity = round((infected_pixels / max(1, cow_area)) * 100, 2)
    level = severity_level(severity)
    if not any(item[4] != "Healthy" for item in detection_boxes):
        severity = 0.0
        level = "Healthy"
    overlay = make_overlay(image, full_mask, detection_boxes, severity, level)
    combined = create_combined(image, detection_view, full_mask, overlay)

    token = uuid.uuid4().hex[:10]
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", secure_filename(image_path.name))
    stem = Path(safe_name).stem
    result_name = f"result_{stem}_{token}.jpg"
    mask_name = f"mask_{stem}_{token}.png"
    cv2.imwrite(str(RESULTS_DIR / result_name), combined, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(RESULTS_DIR / mask_name), full_mask)
    append_result({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image": safe_name,
        "diagnosis": diagnosis,
        "confidence": f"{confidence:.4f}",
        "severity_percent": f"{severity:.2f}",
        "severity_level": level,
        "result_file": result_name,
    })
    return {
        "diagnosis": diagnosis,
        "confidence": round(confidence * 100, 1),
        "severity": severity,
        "severity_level": level,
        "detections": detections,
        "detection_count": len(detections),
        "result_url": f"/results/{result_name}",
        "mask_url": f"/results/{mask_name}",
        "device": str(DEVICE),
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok", "device": str(DEVICE)})


@app.post("/api/analyze")
def analyze():
    uploaded = request.files.get("image")
    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "Choose an image first."}), 400
    if not allowed_file(uploaded.filename):
        return jsonify({"error": "Use a JPG, PNG, JPEG, or WEBP image."}), 400

    safe_name = secure_filename(uploaded.filename)
    upload_path = UPLOAD_DIR / f"{uuid.uuid4().hex[:10]}_{safe_name}"
    uploaded.save(upload_path)
    try:
        return jsonify(analyze_image(upload_path))
    except Exception as error:
        upload_path.unlink(missing_ok=True)
        return jsonify({"error": str(error)}), 422


@app.get("/results/<path:filename>")
def results_file(filename):
    return send_from_directory(RESULTS_DIR, filename)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
