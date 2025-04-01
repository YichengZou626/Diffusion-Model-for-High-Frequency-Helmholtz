import torch
import h5py

def load_2d_data_from_h5(filepath):
    with h5py.File(filepath, 'r') as f:
        data = f['x'][()]

    # Input: first 2 channels, frame 0
    data_in = data[:, :2, 0, :, :]           # Shape: [N, 2, 64, 64]
    data_in = torch.from_numpy(data_in).permute(0, 2, 3, 1)  # [N, 64, 64, 2]

    # Output: third channel, frame 0
    data_out = data[:, 2, 0, :, :]           # Shape: [N, 64, 64]
    data_out = torch.from_numpy(data_out).unsqueeze(-1)      # [N, 64, 64, 1]

    return data_in, data_out
