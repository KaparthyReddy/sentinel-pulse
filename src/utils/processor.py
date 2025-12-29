import os
import cv2
import numpy as np
from glob import glob

def process_sentinel_data(sat_dir, flood_dir, output_dir):
    """
    Stitches satellite images and flood maps into 512x256 pairs.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all files in satellite directory
    sat_files = sorted(os.listdir(sat_dir))
    count = 0

    print(f"🧵 Starting the stitching process...")

    for file_name in sat_files:
        # Construct paths
        name_only = os.path.splitext(file_name)[0]
        sat_path = os.path.join(sat_dir, file_name)
        
        # Try to find a matching flood map (checking .png and .jpg)
        flood_path_png = os.path.join(flood_dir, name_only + ".png")
        flood_path_jpg = os.path.join(flood_dir, name_only + ".jpg")
        
        target_path = None
        if os.path.exists(flood_path_png):
            target_path = flood_path_png
        elif os.path.exists(flood_path_jpg):
            target_path = flood_path_jpg

        if target_path:
            # Read images
            img_sat = cv2.imread(sat_path)
            img_flood = cv2.imread(target_path)

            if img_sat is not None and img_flood is not None:
                # 1. Resize both to 256x256
                img_sat = cv2.resize(img_sat, (256, 256))
                img_flood = cv2.resize(img_flood, (256, 256))

                # 2. Stack horizontally [Satellite | Flood]
                combined = np.hstack((img_sat, img_flood))

                # 3. Save to processed folder
                cv2.imwrite(os.path.join(output_dir, f"{name_only}.jpg"), combined)
                count += 1
    
    print(f"✅ Successfully created {count} paired images in {output_dir}")

if __name__ == "__main__":
    process_sentinel_data(
        sat_dir="data/raw/satellite",
        flood_dir="data/raw/flood_maps",
        output_dir="data/processed/train"
    )