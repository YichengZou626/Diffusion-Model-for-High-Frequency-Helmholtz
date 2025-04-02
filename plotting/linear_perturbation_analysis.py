import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

# Define base directories for all models
base_paths = {
    'attention_conditional': '/work/yz886/results/1.5e6_256x256/attention_conditional/1/trajectory_outputs/',
    'conditional': '/work/yz886/results/1.5e6_256x256/conditional/1/trajectory_outputs/',
    'joint': '/work/yz886/results/1.5e6_256x256/joint/1/trajectory_outputs/',
    'fno': '/work/yz886/results/1.5e6_256x256/fno_results/'
}

# Define the path to test.h5 (Ground Truth)
ground_truth_path = '/work/yz886/results/1.5e6_256x256/test.h5'

# Create a directory for results
output_dir = "ux"
os.makedirs(output_dir, exist_ok=True)

# Define shifted boundary points (2 pixels inward)
boundary_points = [(10, 10), (10, 245), (245, 10), (245, 245), (10, 128), (128, 10), (245, 128), (128, 245)]

# Define center points near (32, 32)
center_points = [(120, 120), (120, 128), (120, 136), (128, 120), (128, 136), (136, 120), (136, 128), (136, 136)]

# Combine selected points
selected_points = boundary_points + center_points

# Generate 1000 equispaced points from 0 to 1
s = np.linspace(0, 1, 1000)

# Store ux values for each model and selected point
ux_values = {model: {point: [] for point in selected_points} for model in base_paths.keys()}

# Iterate through each model
for model, base_dir in base_paths.items():
    for i in range(1, 1001):
        file_path = (os.path.join(base_dir, f'out_sample_{500+i}.npy')
                     if model == 'fno' else os.path.join(base_dir, f'batch_{500+i}', 'timestep_3.npy'))
        
        if os.path.exists(file_path):
            data = np.load(file_path).squeeze()
            for x, y in selected_points:
                ux_values[model][(x, y)].append(data[x, y])
        else:
            print(f'Warning: {file_path} not found in {model}')

# Read Ground Truth (test.h5)
ground_truth_values = []
ground_truth_wavefields = []
if os.path.exists(ground_truth_path):
    with h5py.File(ground_truth_path, 'r') as f:
        ground_truth_values = f['x'][500:, 2, 0]
        print(f'Loaded Ground Truth data with shape {ground_truth_values.shape}')
else:
    print(f'Warning: {ground_truth_path} not found')

# Create and save figures per point
colors = {'attention_conditional': 'r', 'conditional': 'b', 'joint': 'g', 'fno': 'm', 'Ground Truth': 'k'}
for point in selected_points:
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in base_paths.keys():
        ax.plot(s, ux_values[model][point], color=colors[model], alpha=0.7, label=model)
    if len(ground_truth_values) == 1000:
        ax.plot(s, ground_truth_values[:, point[0], point[1]], color=colors['Ground Truth'], linestyle='dashed', label='Ground Truth')
    ax.set_title(f's vs ux for Point {point}')
    ax.set_xlabel('s')
    ax.set_ylabel('ux')
    ax.legend()
    plt.savefig(os.path.join(output_dir, f's_vs_ux_point_{point[0]}_{point[1]}.png'))
    plt.close()

# Generate wavefield plots
wavefield_indices = np.arange(1, 1001, 100)
n_cols, n_rows = 5, len(wavefield_indices) // 5 + (len(wavefield_indices) % 5 > 0)

for model, base_dir in base_paths.items():
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    axes = axes.flatten()
    for ax, idx in zip(axes, wavefield_indices):
        file_path = (os.path.join(base_dir, f'out_sample_{500+idx}.npy')
                     if model == 'fno' else os.path.join(base_dir, f'batch_{500+idx}', 'timestep_3.npy'))
        if os.path.exists(file_path):
            data = np.load(file_path).squeeze()
            ax.imshow(data, cmap='seismic')
            ax.set_title(f'Sample {idx}')
            ax.axis('off')
    plt.savefig(os.path.join(output_dir, f'wavefield_summary_{model}_shifted.png'))
    plt.close()

# Ground Truth wavefield plots
if len(ground_truth_values) == 1000:
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    axes = axes.flatten()
    for ax, idx in zip(axes, wavefield_indices):
        ax.imshow(ground_truth_values[idx - 1], cmap='seismic')
        ax.set_title(f'Sample {idx}')
        ax.axis('off')
    plt.savefig(os.path.join(output_dir, 'wavefield_summary_Ground_Truth.png'))
    plt.close()

# Example wavefield plot
example_index = 100
example_wavefield = ground_truth_values[example_index]
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(example_wavefield, cmap='seismic')
for point in selected_points:
    ax.scatter(point[1], point[0], color='yellow', edgecolor='black', s=100, label="Selected Points" if point == selected_points[0] else "")
ax.set_title(f"Selected Points on Ground Truth Wavefield (Sample {example_index})")
ax.set_xlabel("X")
ax.set_ylabel("Y")
plt.savefig(os.path.join(output_dir, 'selected_points_visualization.png'))
plt.close()

