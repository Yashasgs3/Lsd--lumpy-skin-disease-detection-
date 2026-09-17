import json
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ROOT = Path(__file__).resolve().parent
IMAGE_ROOT = ROOT / "dataset" / "livestock_dataset_enhanced" / "train" / "images"
OUTPUT_DIR = ROOT / "dataset" / "livestock_dataset_enhanced" / "annotations"
TASK_FILE = OUTPUT_DIR / "label_studio_tasks.json"


def main():
    image_paths = sorted(
        path for path in IMAGE_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {IMAGE_ROOT}")

    tasks = [
        {
            "data": {
                "image": image_path.as_uri(),
                "source_class": image_path.parent.name,
            }
        }
        for image_path in image_paths
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TASK_FILE.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    print(f"Created {len(tasks)} Label Studio tasks")
    print(f"Task file: {TASK_FILE}")


if __name__ == "__main__":
    main()
