from ultralytics import YOLO

# Load your trained model (replace path with your model's .pt file)
model = YOLO("runs/detect/train/weights/best.pt")

# Run prediction on an image, folder, or video
results = model.predict(
    source="dataset/livestock_dataset_enhanced/test/images",  # 👈 path to image or folder
    conf=0.5,   # confidence threshold (0.25–0.7 works well)
    save=True,  # saves output images with boxes to 'runs/predict'
)

# Print detailed prediction info (optional)
for r in results:
    print(r.boxes)       # bounding box coordinates
    print(r.names)       # class names
