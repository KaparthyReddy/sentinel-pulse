import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from src.models.generator import UNetGenerator
from src.utils.loader import PulseDataset

# For advanced metrics
from skimage.metrics import structural_similarity as ssim

def calculate_metrics(real_img, fake_img):
    """Calculates professional CV metrics between two images."""
    real_np = real_img.cpu().numpy().transpose(1, 2, 0)
    fake_np = fake_img.cpu().numpy().transpose(1, 2, 0)
    
    # Normalize to [0, 1] for metric calculation
    real_np = (real_np + 1) / 2
    fake_np = (fake_np + 1) / 2
    
    mse = np.mean((real_np - fake_np) ** 2)
    # Calculate SSIM (Structural Similarity)
    score, _ = ssim(real_np, fake_np, full=True, channel_axis=2, data_range=1.0)
    
    # Simple Pixel Accuracy (within a small tolerance)
    acc = np.mean(np.abs(real_np - fake_np) < 0.1) 
    
    return mse, score, acc

def run_sentinel_test():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🛰  Advanced Sentinel-Pulse Validation on: {device}")

    # 1. Load the Trained Generator
    model = UNetGenerator().to(device)
    model_path = "models/sentinel_v1.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ FATAL ERROR: Model weights not found at {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 2. Data Pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = PulseDataset("data/processed/train", transform=transform)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

    # 3. Process a Batch for Visualization & Metrics
    batch = next(iter(dataloader))
    real_A = batch["A"].to(device)
    real_B = batch["B"].to(device)

    with torch.no_grad():
        fake_B = model(real_A)

    # 4. Professional Visualization (3 Samples)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.4)
    
    total_mse, total_ssim, total_acc = 0, 0, 0

    for i in range(3):
        # Calculate individual metrics
        mse, s_score, acc = calculate_metrics(real_B[i], fake_B[i])
        total_mse += mse
        total_ssim += s_score
        total_acc += acc

        # Display Logic
        imgs = [real_A[i], real_B[i], fake_B[i]]
        titles = [f"Input Sat {i+1}", f"Ground Truth {i+1}", f"AI Predicted {i+1}\n(SSIM: {s_score:.4f})"]
        
        for j in range(3):
            img_disp = imgs[j].cpu().permute(1, 2, 0).numpy()
            img_disp = (img_disp + 1) / 2
            axes[i, j].imshow(np.clip(img_disp, 0, 1))
            axes[i, j].set_title(titles[j], fontsize=10)
            axes[i, j].axis('off')

    # 5. Summary Report
    print("\n" + "="*30)
    print("📊 VALIDATION SUMMARY")
    print("="*30)
    print(f"Avg Pixel Accuracy: {total_acc/3:.2%}")
    print(f"Avg SSIM Score:     {total_ssim/3:.4f} (Ideal: 1.0)")
    print(f"Avg MSE:            {total_mse/3:.4f} (Ideal: 0.0)")
    print("="*30)

    # Save to path defined in .gitignore exceptions
    os.makedirs("outputs/plots", exist_ok=True)
    save_path = "outputs/plots/validation_comparison.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\n✅ COMPLETED: Results saved to {save_path}")

if __name__ == "__main__":
    run_sentinel_test()