# Disease annotation workflow

The current model cannot be retrained reliably from its generated pseudo-masks. Use the annotation project to create verified labels.

## 1. Install and start Label Studio

From the project directory:

```powershell
.venv\Scripts\python.exe -m pip install label-studio
.venv\Scripts\label-studio.exe start --allow-serving-local-files
```

Open `http://localhost:8080` and create a project.

## 2. Configure the project

Use the contents of `annotation_config.xml` as the project's labeling interface. Import:

```text
dataset/livestock_dataset_enhanced/annotations/label_studio_tasks.json
```

The task images are local `file:///` paths. Keep Label Studio running with `--allow-serving-local-files`.

## 3. Label carefully

- Draw one rectangle around every visible cow, including partially visible cows.
- Use `Healthy`, `Lumpy`, or `Foot-and-mouth` for each cow rectangle.
- Draw polygons around every visible diseased skin region, not only the largest patch.
- Use `Lumpy` or `Foot-and-mouth` for disease polygons.
- Do not draw disease polygons on healthy cows.
- Zoom into skin lesions and include the full visible lesion boundary.
- Do not use the current predicted masks as ground truth without correcting them.

For quality, manually review at least 200 images per class before training, and keep a separate unseen test set.

## 4. Export

Export the completed project as Label Studio JSON. The next training step must convert:

- Rectangle labels to YOLO `.txt` files.
- Disease polygons to binary PNG masks.

Do not train the final model until those exports are reviewed.
