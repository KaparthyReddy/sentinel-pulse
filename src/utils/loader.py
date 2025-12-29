import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class PulseDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.files = sorted(os.listdir(root))
        self.transform = transform

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.files[index])
        img = Image.open(img_path).convert("RGB")
        
        # Split the 512x256 image into two 256x256 halves
        w, h = img.size
        img_a = img.crop((0, 0, w // 2, h))    # Left: Satellite (Input)
        img_b = img.crop((w // 2, 0, w, h))    # Right: Flood (Target)

        if self.transform:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)

        return {"A": img_a, "B": img_b}

    def __len__(self):
        return len(self.files)