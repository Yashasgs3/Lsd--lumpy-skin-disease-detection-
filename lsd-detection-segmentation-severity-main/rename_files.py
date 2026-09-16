import os

# Path to your images folder
folder = r"dataset\livestock_dataset_enhanced\test\images"

# Loop through all files in the folder
for filename in os.listdir(folder):
    # Check if "_out" is in the filename
    if "_out" in filename:
        # Create the new filename (remove "_out")
        new_name = filename.replace("_out", "")
        # Get the full paths
        old_path = os.path.join(folder, filename)
        new_path = os.path.join(folder, new_name)
        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} → {new_name}")

print("✅ All files renamed successfully!")

# Path to your images folder
folder = r"dataset\livestock_dataset_enhanced\train\images"

# Loop through all files in the folder
for filename in os.listdir(folder):
    # Check if "_out" is in the filename
    if "_out" in filename:
        # Create the new filename (remove "_out")
        new_name = filename.replace("_out", "")
        # Get the full paths
        old_path = os.path.join(folder, filename)
        new_path = os.path.join(folder, new_name)
        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} → {new_name}")

print("✅ All files renamed successfully!")

# Path to your images folder
folder = r"dataset\livestock_dataset_enhanced\valid_balanced\images"

# Loop through all files in the folder
for filename in os.listdir(folder):
    # Check if "_out" is in the filename
    if "_out" in filename:
        # Create the new filename (remove "_out")
        new_name = filename.replace("_out", "")
        # Get the full paths
        old_path = os.path.join(folder, filename)
        new_path = os.path.join(folder, new_name)
        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} → {new_name}")

print("✅ All files renamed successfully!")