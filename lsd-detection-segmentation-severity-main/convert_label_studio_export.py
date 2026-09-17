import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
CLASS_IDS = {"healthy": 0, "lumpy": 1, "foot-and-mouth": 2}


def image_path_from_task(task):
    image_uri = task.get("data", {}).get("image", "")
    if image_uri.startswith("file:///"):
        return Path(image_uri[8:])
    return Path(image_uri)


def label_name(result):
    values = result.get("value", {})
    labels = values.get("rectanglelabels") or values.get("polygonlabels") or []
    return labels[0].strip().lower() if labels else None


def polygon_pixels(points, width, height):
    return [
        (round(point["x"] * width / 100), round(point["y"] * height / 100))
        for point in points
    ]


def convert(export_path, output_dir):
    tasks = json.loads(export_path.read_text(encoding="utf-8"))
    if isinstance(tasks, dict):
        tasks = [tasks]

    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    masks_dir = output_dir / "masks"
    for directory in (images_dir, labels_dir, masks_dir):
        directory.mkdir(parents=True, exist_ok=True)

    converted = 0
    for task in tasks:
        source_image = image_path_from_task(task)
        if not source_image.is_file() or source_image.suffix.lower() not in IMAGE_EXTENSIONS:
            print(f"Skipping missing image: {source_image}")
            continue

        image = Image.open(source_image).convert("RGB")
        width, height = image.size
        stem = source_image.stem
        image.save(images_dir / source_image.name)
        mask = Image.new("L", (width, height), 0)
        mask_draw = ImageDraw.Draw(mask)
        yolo_lines = []

        annotations = task.get("annotations", [])
        results = annotations[-1].get("result", []) if annotations else []
        for result in results:
            values = result.get("value", {})
            label = label_name(result)
            if label not in CLASS_IDS:
                continue

            if "x" in values and "width" in values:
                x_center = (values["x"] + values["width"] / 2) / 100
                y_center = (values["y"] + values["height"] / 2) / 100
                box_width = values["width"] / 100
                box_height = values["height"] / 100
                yolo_lines.append(
                    f"{CLASS_IDS[label]} {x_center:.6f} {y_center:.6f} "
                    f"{box_width:.6f} {box_height:.6f}"
                )

            if "polygon" in values and label != "healthy":
                points = polygon_pixels(values["polygon"], width, height)
                if len(points) >= 3:
                    mask_draw.polygon(points, fill=255)

        (labels_dir / f"{stem}.txt").write_text("\n".join(yolo_lines), encoding="utf-8")
        mask.save(masks_dir / f"{stem}.png")
        converted += 1

    print(f"Converted {converted} annotated image(s) into {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert Label Studio export to YOLO labels and PNG masks.")
    parser.add_argument("export", type=Path, help="Label Studio JSON export file")
    parser.add_argument("--output", type=Path, default=Path("dataset/labeled"))
    args = parser.parse_args()
    convert(args.export, args.output)


if __name__ == "__main__":
    main()
