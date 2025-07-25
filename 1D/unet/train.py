import os
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging
from tqdm import tqdm
import numpy as np
from unet import UNet
import math
import torch.nn.functional as F

# =========================
# Configuration
# =========================
CONFIG = {
    'data_path': '/work-old/yz886/ANFWI_HNO/data/1D/1e6',
    'hidden_channels': [48, 96, 192, 384, 768],
    'hidden_blocks': [3, 3, 3, 3, 3],
    'kernel_size': 3,
    'activation': 'SiLU',
    'padding_mode': 'circular',
    'learning_rate': 2e-4,
    'weight_decay': 1e-5,
    'batch_size': 32,
    'epochs': 100,
    'ckpt_dir': 'checkpoints',
    'in_channels': 3,
    'out_channels': 1,
    'spatial': 1,
    'embedding': 0
}

# =========================
# Device Setup
# =========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class FluidDataset(Dataset):
    def __init__(self, file_path, num_data=None):
        with h5py.File(file_path, 'r') as f:
            data = torch.from_numpy(f['x'][()])  # shape: (N, 3, 256)
            targets = torch.from_numpy(f['y'][()])  # shape: (N, 1, 256)

        if num_data is not None:
            data = data[:num_data]
            targets = targets[:num_data]

        self.inputs = data.float()     # Already includes [sound map, mask, position encoding]
        self.targets = targets.float()

        print(f"✅ Loaded dataset from {file_path}:")
        print(f"   Inputs shape:  {self.inputs.shape}")
        print(f"   Targets shape: {self.targets.shape}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# =========================
# Training Function
# =========================
import matplotlib.pyplot as plt  # <-- Add this import at the top

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    ckpt_path = Path(CONFIG['ckpt_dir'])
    ckpt_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading datasets...")
    train_dataset = FluidDataset(os.path.join(CONFIG['data_path'], "train.h5"), num_data=10000)
    valid_dataset = FluidDataset(os.path.join(CONFIG['data_path'], "valid.h5"))  # default: all (or change to num_data=10000)
    
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)

    logger.info("Creating model...")
    model = UNet(
        in_channels=CONFIG['in_channels'],
        out_channels=CONFIG['out_channels'],
        hidden_channels=CONFIG['hidden_channels'],
        hidden_blocks=CONFIG['hidden_blocks'],
        kernel_size=CONFIG['kernel_size'],
        activation=CONFIG['activation'],
        spatial=CONFIG['spatial'],
        embedding=CONFIG['embedding'],
        padding_mode=CONFIG['padding_mode']
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'], eta_min=1e-6)

    logger.info("Starting training...")
    best_valid_loss = float('inf')
    train_losses = []
    valid_losses = []

    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss = 0.0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                valid_loss += loss.item()
        valid_loss /= len(valid_loader)

        scheduler.step()
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        logger.info(f"Epoch {epoch+1}/{CONFIG['epochs']} - Train Loss: {train_loss:.6f} - Valid Loss: {valid_loss:.6f} - LR: {scheduler.get_last_lr()[0]:.6e}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), ckpt_path / 'best_model_1D_1e6_10k.pth')
            logger.info(f"Best model saved at epoch {epoch+1} with val loss {valid_loss:.6f}")

    # === Plotting loss (log scale) ===
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(ckpt_path / 'loss_log_plot_1D_1e6_10k.png')
    plt.close()

if __name__ == "__main__":
    main()
