from ultralytics import YOLO
import pandas as pd

def main():
    # Load the trained model
    model = YOLO("runs/detect/train100_balanced2/weights/best.pt")

    # Evaluate on validation set
    metrics = model.val()
    print(metrics)

if __name__ == '__main__':
    main()