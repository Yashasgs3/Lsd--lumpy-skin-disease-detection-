from ultralytics import YOLO

def main():
    # Load YOLOv8 model (medium recommended for your GPU)
    model = YOLO("runs/detect/train100_balanced2/weights/last.pt")  # pretrained weights

    # Train the model
    results = model.train(
        data="dataset/livestock_dataset_enhanced/data.yaml",  # path to your YAML
        epochs=100,                # max epochs
        imgsz=640,                 # image size
        batch=4,                   # adjust based on GPU memory
        optimizer="Adam",          # optimizer
        device=0,                  # GPU index
        augment=True,              # data augmentation for train set
        # cache=True,                # speeds up training
        name="train100_balanced",  # output folder
        patience=15,               # EARLY STOPPING: stop if no improvement in 15 epochs
        save=True,                 # save checkpoints
        pretrained=True,           # use pretrained weights
        plots=True,                 # generate plots during validation
        resume=True
    )

    # Print final results
    print("Training complete!")
    print("Best fitness:", results.fitness)
    print("Validation metrics:", results.metrics)
    print("Saved in:", results.save_dir)

if __name__ == "__main__":
    main()
