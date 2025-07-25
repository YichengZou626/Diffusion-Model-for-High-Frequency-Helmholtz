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
    'data_path': '/work-old/yz886/ANFWI_HNO/data/1D/1.5e5',
    'hidden_channels': [48, 96, 192, 384, 768],
    'hidden_blocks': [3, 3, 3, 3, 3],
    'kernel_size': 3,
    'activation': 'SiLU',
    'padding_mode': 'circular',
    'learning_rate': 2e-4,
    'weight_decay': 1e-5,
    'batch_size': 32,
    'epochs': 100,
    'ckpt_path': 'checkpoints/best_model_1D_1.5e5_100k.pth',
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
# Evaluation
# =========================
def evaluate():
    print("Preparing test dataset...")
    dataset = FluidDataset(os.path.join(CONFIG['data_path'], "perturbation.h5"))
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)

    print("Loading model...")
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

    model.load_state_dict(torch.load(CONFIG['ckpt_path'], map_location=device))
    model.eval()

    all_preds = []

    print("Running inference...")
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = outputs.cpu().numpy().squeeze(1)  # [B, H, W]
            all_preds.append(preds)

    # === Predictions from your evaluation loop ===
    predictions = np.concatenate(all_preds, axis=0)  # [N, H, W]

    # === Hardcoded output directory and file path ===
    output_dir = Path('/work-old/yz886/unet_no_mid/unet_results_1D_1.5e5')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'perturbation100k.npy'
    np.save(output_file, predictions)

    print(f"Saved predictions to {output_file} with shape {predictions.shape}")

if __name__ == "__main__":
    evaluate()
