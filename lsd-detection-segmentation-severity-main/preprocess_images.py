import glob

label_files = glob.glob("dataset/livestock_dataset_enhanced/valid/labels/*.txt")

classes = set()
for file in label_files:
    with open(file, "r") as f:
        for line in f:
            if line.strip():
                cls = int(line.split()[0])
                classes.add(cls)

print("Classes found in labels:", classes)
