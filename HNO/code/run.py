import numpy as np 
import torch
import matplotlib.pyplot as plt
import os
import copy
import glob
import obspy

from util import HNO
from joblib import load
from obspy.geodetics.base import gps2dist_azimuth
from torch.nn import DataParallel
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from scipy.interpolate import interp1d 

# Automatically select GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

from timeit import default_timer
from util.loss_func import LpLoss
from util.HNO import UNO2D
from util.load_data import load_2d_data_from_h5

# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Hyperparameters
nx, ny = 64, 64
width = 32
batch_size = 32
in_channels = 2
out_channels = 1
epochs = 1000

# Initialize model and optimizer
model = UNO2D(in_channels, out_channels, width, pad=0)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

# Apply DataParallel only if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
    model = DataParallel(model)

# Load data
train_in, train_out = load_2d_data_from_h5('/work/yz886/ANFWI_HNO/data/train.h5')
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(train_in, train_out),
    batch_size=batch_size,
    shuffle=True
)

valid_in, valid_out = load_2d_data_from_h5('/work/yz886/ANFWI_HNO/data/valid.h5')
valid_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(valid_in, valid_out),
    batch_size=batch_size,
    shuffle=False
)

# Loss functions
L2 = LpLoss(p=2, size_average=False)
L1 = LpLoss(p=1, size_average=False)

# Training loop
for ep in range(epochs):
    t1 = default_timer()

    model.train()
    train_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        L2_loss = L2(out, y)
        L1_loss = L1(out, y)
        loss = 0.9 * L1_loss + 0.1 * L2_loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= train_out.size(0)

    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            L2_loss = L2(out, y)
            L1_loss = L1(out, y)
            loss = 0.9 * L1_loss + 0.1 * L2_loss
            valid_loss += loss.item()

    valid_loss /= valid_out.size(0)
    scheduler.step()

    t2 = default_timer()

    print(ep, f"Time: {(t2 - t1) / 3600:.2f}h", f"Train Loss: {train_loss:.6f}", f"Valid Loss: {valid_loss:.6f}", flush=True)

    torch.cuda.empty_cache()

# === Run model on test data ===
import os

# Load test data
test_in, test_out = load_2d_data_from_h5('/work/yz886/ANFWI_HNO/data/test.h5')
test_dataset = torch.utils.data.TensorDataset(test_in, test_out)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# Set model to evaluation mode
model.eval()

# Directory to save results
results_dir = "/work/yz886/ANFWI_HNO/results"
os.makedirs(results_dir, exist_ok=True)

# Loop through test samples
for sample_idx, (x, y) in enumerate(test_loader):
    x = x.to(device)
    y = y.to(device)
    with torch.no_grad():
        out = model(x).cpu().detach().numpy()

    # Convert input and ground truth to numpy
    #x_np = x.cpu().numpy().squeeze()
    #y_np = y.cpu().numpy().squeeze()

    # Save files
    #np.save(os.path.join(results_dir, f"x_sample_{sample_idx + 1}.npy"), x_np)
    #np.save(os.path.join(results_dir, f"y_sample_{sample_idx + 1}.npy"), y_np)
    np.save(os.path.join(results_dir, f"out_sample_{sample_idx + 1}.npy"), out.squeeze())

    print(f"Saved sample {sample_idx + 1}", flush=True)

print(f"\nx, y, and out saved as .npy files in {results_dir}", flush=True)

# === Sensitivity Analysis ===
model.eval()
sensitivity_sum = np.zeros((64, 64))

for idx, (x, y) in enumerate(test_loader):
    if idx >= 500:  # Limit to first 500 samples
        break
        
    x = x.to(device)
    y = y.to(device)

    # Enable gradients on input
    x.requires_grad_(True)
    
    # Forward pass
    out = model(x)
    misfit = torch.nn.functional.mse_loss(out, y)

    # Backward pass
    misfit.backward()

    # Extract gradient w.r.t. the sound map (channel 0)
    grad = x.grad.detach().cpu().numpy().squeeze()  # shape: (64, 64, 2) or (1, 64, 64, 2)
    if grad.ndim == 4:
        grad = grad[0]  # remove batch dimension

    #print(grad.shape)
    sensitivity = np.mean(np.abs(grad), axis=-1) # shape: [64, 64]
    sensitivity_sum += sensitivity

# Normalize
relative_sensitivity = sensitivity_sum / np.max(sensitivity_sum)

# === Plot the Sensitivity Map ===
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 5))
plt.imshow(relative_sensitivity, cmap='hot', origin='lower')
plt.colorbar(label='Relative Sensitivity')
plt.title("Gradient-Based Sensitivity Map")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "sensitivity_map.png"), dpi=300)

