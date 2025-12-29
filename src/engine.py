import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
import yaml
import os

from models.generator import UNetGenerator
from models.discriminator import Discriminator
from utils.loader import PulseDataset

# Load Config
with open("config/settings.yaml", "r") as f:
    config = yaml.safe_load(f)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Initialize Models
net_G = UNetGenerator().to(device)
net_D = Discriminator().to(device)

# Loss & Optimizers
criterion_GAN = nn.MSELoss() # Least Squares GAN
criterion_L1 = nn.L1Loss()
optimizer_G = torch.optim.Adam(net_G.parameters(), lr=config['training']['lr'], betas=(config['training']['beta1'], 0.999))
optimizer_D = torch.optim.Adam(net_D.parameters(), lr=config['training']['lr'], betas=(config['training']['beta1'], 0.999))

def train():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = PulseDataset("data/processed/train", transform=transform)
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)

    for epoch in range(config['training']['epochs']):
        for i, batch in enumerate(dataloader):
            real_A = batch["A"].to(device) # Satellite
            real_B = batch["B"].to(device) # Flood Ground Truth

            # --- Train Discriminator ---
            optimizer_D.zero_grad()
            fake_B = net_G(real_A)
            
            # Real pair
            pred_real = net_D(real_A, real_B)
            loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
            
            # Fake pair
            pred_fake = net_D(real_A, fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()

            # --- Train Generator ---
            optimizer_G.zero_grad()
            pred_fake = net_D(real_A, fake_B)
            
            loss_G_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
            loss_G_L1 = criterion_L1(fake_B, real_B) * config['training']['lambda_L1']
            
            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            optimizer_G.step()

            if i % 10 == 0:
                print(f"Epoch [{epoch}/{config['training']['epochs']}] Batch {i} | Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f}")

        # Save Sample every epoch
        os.makedirs("outputs", exist_ok=True)
        save_image(torch.cat((real_A, fake_B, real_B), -1), f"outputs/epoch_{epoch}.png", normalize=True)
        
        # Save Weights
        torch.save(net_G.state_dict(), "models/generator_latest.pth")

if __name__ == "__main__":
    train()