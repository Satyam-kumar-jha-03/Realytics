import kagglehub
import os
import shutil
import random

# Download the dataset
print("Downloading CIFAKE dataset...")
path = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")
print(f"Dataset downloaded to: {path}")

# Create local directory to save the dataset
LOCAL_DATA_DIR = "dataset"
os.makedirs(LOCAL_DATA_DIR, exist_ok=True)

# Copy the dataset to local directory
print("Copying dataset to local directory...")
for folder in ['train', 'test']:
    source_folder = os.path.join(path, folder)
    dest_folder = os.path.join(LOCAL_DATA_DIR, folder)
    
    if os.path.exists(source_folder):
        shutil.copytree(source_folder, dest_folder, dirs_exist_ok=True)
        print(f"✓ Copied {folder}")

print(f"✅ Dataset saved to: {LOCAL_DATA_DIR}")

# Explore the dataset structure
print("\n📁 Dataset Structure:")
for folder in ['train', 'test']:
    folder_path = os.path.join(LOCAL_DATA_DIR, folder)
    if os.path.exists(folder_path):
        print(f"\n{folder.upper()}:")
        for cls in ['REAL', 'FAKE']:
            cls_path = os.path.join(folder_path, cls)
            if os.path.exists(cls_path):
                num_files = len(os.listdir(cls_path))
                print(f"  - {cls}: {num_files} images")
                
                # Show a sample
                sample_files = os.listdir(cls_path)[:3]
                for f in sample_files:
                    print(f"    • {f}")