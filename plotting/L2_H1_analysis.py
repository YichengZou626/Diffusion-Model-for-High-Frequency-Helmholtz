import numpy as np
import h5py
import matplotlib.pyplot as plt
import os

# Define the base directory and frequency folders
base_dir = "/work/yz886/results"
folders = [
    "1.5e5_64x64", "2.5e5_64x64", "5e5_64x64", "1.5e6_64x64",
    "2.5e6_64x64", "5e6_64x64"
]
frequencies = [1.5e5, 2.5e5, 5e5, 1.5e6, 2.5e6, 5e6]  # Corresponding frequencies

# Initialize lists to store relative error values
l2_errors = []
h1_errors = []

# Compute relative errors for each folder
for folder, freq in zip(folders, frequencies):
    folder_path = os.path.join(base_dir, folder)
    test_h5_path = os.path.join(folder_path, "test.h5")  # Adjusted to read from each folder

    # Check if test.h5 exists
    if not os.path.exists(test_h5_path):
        print(f"Skipping {folder} as test.h5 is missing.")
        continue

    # Read ground truth values
    with h5py.File(test_h5_path, 'r') as f:
        ground_truth_values = f['x'][:500, 2, 0]  # Expected shape (500, 256, 256)

    fno_results_dir = os.path.join(folder_path, "fno_results")
    
    # Check if test.h5 exists
    if not os.path.exists(fno_results_dir):
        print(f"Skipping {folder} as fno_results is missing.")
        continue

    l2_sum = 0
    h1_sum = 0
    valid_count = 0  # Count valid samples
    
    for i in range(1, 501):
        npy_file = os.path.join(fno_results_dir, f"out_sample_{i}.npy")
        if os.path.exists(npy_file):
            out_sample = np.load(npy_file).squeeze()
            ground_truth = ground_truth_values[i - 1]  # Corresponding ground truth
           
            # Compute L2 norm of ground truth
            gt_norm = np.sqrt(np.mean(ground_truth**2)) + 1e-8  # Avoid division by zero

            # Compute relative L2 loss (Mean Squared Error normalized)
            l2_error = np.sqrt(np.mean((out_sample - ground_truth) ** 2)) / gt_norm
            l2_sum += l2_error
            
            # Compute gradients for H1 error
            grad_sample_x, grad_sample_y = np.gradient(out_sample)
            grad_truth_x, grad_truth_y = np.gradient(ground_truth)

            grad_diff_x = grad_sample_x - grad_truth_x
            grad_diff_y = grad_sample_y - grad_truth_y

            # Compute relative H1 loss (L2 norm of gradient difference normalized)
            grad_norm = np.sqrt(np.mean(grad_truth_x**2 + grad_truth_y**2)) + 1e-8  # Normalize by GT gradient norm
            h1_error = np.sqrt(l2_error**2 + np.mean(grad_diff_x**2 + grad_diff_y**2)) / grad_norm
            h1_sum += h1_error

            valid_count += 1  # Increment valid sample count

    # Normalize errors over valid samples
    if valid_count > 0:
        l2_errors.append(l2_sum / valid_count)
        h1_errors.append(h1_sum / valid_count)
    else:
        l2_errors.append(None)
        h1_errors.append(None)

# Convert frequencies to string labels for x-axis
frequency_labels = [f"{freq:.1e}" for freq in frequencies]

# Plot Relative L2 and H1 Errors
plt.figure(figsize=(8, 6))
plt.plot(frequency_labels, l2_errors, marker='o', label="Relative L2 Error")
plt.plot(frequency_labels, h1_errors, marker='s', label="Relative H1 Error")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Error Value (0-1)")
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.legend()
plt.title("FNO Performance over Different Frequencies (Relative Errors)")
plt.grid(True)

# Save the figure
plot_path = os.path.join(base_dir, "Relative_L2_H1_Errors_vs_Frequency.png")
plt.savefig(plot_path, dpi=300)
plt.show()

# Selected indices for visualization
indices = [1, 100, 200, 300, 400, 500]

# Iterate over each frequency folder to create separate figures
for folder, freq in zip(folders, frequencies):
    folder_path = os.path.join(base_dir, folder)
    test_h5_path = os.path.join(folder_path, "test.h5")
    fno_results_dir = os.path.join(folder_path, "fno_results")

    if not os.path.exists(test_h5_path):
        print(f"Skipping {folder} as test.h5 is missing.")
        continue
        
    # Check if test.h5 exists
    if not os.path.exists(fno_results_dir):
        print(f"Skipping {folder} as fno_results is missing.")
        continue
    
    # Read ground truth values
    with h5py.File(test_h5_path, 'r') as f:
        ground_truth_values = f['x'][:500, 2, 0, :, :]  # Shape (500, 256, 256)
    
    # Create figure
    fig, axes = plt.subplots(2, 6, figsize=(15, 6))  # 2 rows, 6 columns (GT and FNO outputs)
    fig.suptitle(f'Comparison at {freq:.1e} Hz', fontsize=14)
    
    for col, idx in enumerate(indices):
        # Plot ground truth
        axes[0, col].imshow(ground_truth_values[idx - 1], cmap='seismic')
        axes[0, col].set_title(f'GT {idx}')
        axes[0, col].axis('off')
        
        # Plot model predictions
        npy_file = os.path.join(fno_results_dir, f"out_sample_{idx}.npy")
        if os.path.exists(npy_file):
            out_sample = np.load(npy_file).squeeze()
            axes[1, col].imshow(out_sample, cmap='seismic')
            axes[1, col].set_title(f'FNO {idx}')
        else:
            axes[1, col].axis('off')
        
        axes[1, col].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, f"GT_vs_FNO_{freq:.1e}Hz.png"), dpi=300)
    plt.show()
