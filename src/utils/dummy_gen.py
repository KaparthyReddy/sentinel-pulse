import numpy as np
import cv2
import os

# Create the training folder
out_dir = "data/processed/train"
os.makedirs(out_dir, exist_ok=True)

print("🛠 Generating 10 synthetic satellite/flood pairs...")
for i in range(10):
    # Create a 256x256 random "satellite" image
    sat = np.random.randint(50, 150, (256, 256, 3), dtype=np.uint8)
    
    # Create a "flood mask" (Blue block in the middle)
    flood = np.zeros((256, 256, 3), dtype=np.uint8)
    flood[100:200, 100:200] = [255, 0, 0] # Red/Blue mask
    
    # Stitch them: [Satellite | Flood]
    paired = np.hstack((sat, flood))
    cv2.imwrite(os.path.join(out_dir, f"synth_{i}.jpg"), paired)

print(f"✅ Success! {out_dir} now has images. You can run 'python3 src/engine.py' now.")