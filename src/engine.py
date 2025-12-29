import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
import yaml
import os

# Absolute imports from your project structure
from models.generator import UNetGenerator
from models.discriminator import Discriminator
from utils.loader import PulseDataset

# 1. Load Configuration
with open("config/settings.yaml", "r") as f:
    config = yaml.safe_load(f)

# 2. Setup Hardware Acceleration (MPS for Mac, CUDA for Nvidia, or CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🚀 Initializing Sentinel-Pulse GAN on: {device}")

# 3. Initialize Models (Pix2Pix Architecture)
net_G = UNetGenerator().to(device) # The "Artist" (creates flood maps)
net_D = Discriminator().to(device) # The "Critic" (detects fakes)

# 4. Losses & Optimizers
criterion_GAN = nn.MSELoss() # Least Squares GAN loss for stability
criterion_L1 = nn.L1Loss()   # L1 Loss for pixel-perfect accuracy
optimizer_G = torch.optim.Adam(net_G.parameters(), lr=config['training']['lr'], betas=(config['training']['beta1'], 0.999))
optimizer_D = torch.optim.Adam(net_D.parameters(), lr=config['training']['lr'], betas=(config['training']['beta1'], 0.999))

def train():
    # 5. Data Pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Ensure data exists before loading
    train_path = "data/processed/train"
    if not os.path.exists(train_path):
        print(f"❌ Error: {train_path} not found. Ensure your preprocessing script ran.")
        return

    dataset = PulseDataset(train_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)

    print(f"📈 Training started for {config['training']['epochs']} epochs...")

    for epoch in range(config['training']['epochs']):
        for i, batch in enumerate(dataloader):
            real_A = batch["A"].to(device) # Input Satellite Imagery
            real_B = batch["B"].to(device) # Target Flood Ground Truth

            # --- A. Train Discriminator (Critic) ---
            optimizer_D.zero_grad()
            fake_B = net_G(real_A)
            
            # Real pair (Satellite + Real Flood Map)
            pred_real = net_D(real_A, real_B)
            loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
            
            # Fake pair (Satellite + AI-Generated Flood Map)
            pred_fake = net_D(real_A, fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()

            # --- B. Train Generator (Artist) ---
            optimizer_G.zero_grad()
            pred_fake = net_D(real_A, fake_B)
            
            # Adv Loss: Try to make the Critic think the fake is real
            loss_G_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
            # Content Loss: Ensure the fake map actually matches the real flood map pixels
            loss_G_L1 = criterion_L1(fake_B, real_B) * config['training']['lambda_L1']
            
            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            optimizer_G.step()

            if i % 10 == 0:
                print(f"Epoch [{epoch}/{config['training']['epochs']}] Batch {i} | Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f}")

        # --- C. Checkpointing & Visualization ---
        os.makedirs("outputs", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        
        # Save visual comparison: [Satellite | AI Generated | True Ground Truth]
        save_image(torch.cat((real_A, fake_B, real_B), -1), f"outputs/epoch_{epoch}.png", normalize=True)
        
        # Save the primary model for testing
        torch.save(net_G.state_dict(), "models/sentinel_v1.pth")
        
    print(f"✅ Training Complete. Weights saved to models/sentinel_v1.pth")

if __name__ == "__main__":
    train()