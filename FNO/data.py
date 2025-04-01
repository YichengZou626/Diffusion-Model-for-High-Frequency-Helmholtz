import h5py
import torch
import os

def load_h5_data(file_path):
    with h5py.File(file_path, 'r') as f:
        data = torch.from_numpy(f['x'][()])  # Shape: (N, 3, 2, 64, 64)

    # Extract inputs (2 channels from first 3) and targets (1 channel from third)
    inputs = data[:, :2, 0]  # Shape: (N, 2, 64, 64)
    targets = data[:, 2, 0].unsqueeze(1)  # Shape: (N, 1, 64, 64)

    return inputs, targets


def prepare_and_save_data(train_file, test_file, save_dir):
    # Load train and test data
    train_x, train_y = load_h5_data(train_file)
    test_x, test_y = load_h5_data(test_file)

    # Save as .pt files
    train_dict = {'x': train_x, 'y': train_y}
    test_dict = {'x': test_x, 'y': test_y}

    os.makedirs(save_dir, exist_ok=True)
    torch.save(train_dict, os.path.join(save_dir, 'helmholtz_train_64.pt'))
    torch.save(test_dict, os.path.join(save_dir, 'helmholtz_test_64.pt'))

    print("ata saved successfully!")
    print(f"Train x shape: {train_x.shape}, Train y shape: {train_y.shape}")
    print(f"Test x shape: {test_x.shape}, Test y shape: {test_y.shape}")


# File paths and directory setup
train_file = "/work/yz886/neuraloperator/train.h5"
test_file = "/work/yz886/neuraloperator/test.h5"
save_dir = "/work/yz886/neuraloperator/neuralop/data/datasets/data"

# Execute data preparation and sanity check
prepare_and_save_data(train_file, test_file, save_dir)

