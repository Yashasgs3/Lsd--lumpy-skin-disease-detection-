import zipfile
import os

# Path to your zip file
zip_path = r"C:\Users\samik\OneDrive\Desktop\Samiksha\project\dataset\lsd_new.v1i.yolov8.zip"

# Folder to extract to (create livestock_dataset inside dataset)
extract_folder = os.path.join(os.path.dirname(zip_path), "livestock_dataset")
os.makedirs(extract_folder, exist_ok=True)  # Create folder if it doesn't exist

# Extract the zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

print(f"Unzipped successfully to: {extract_folder}")
