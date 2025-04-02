import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

# Define base directories for all models
base_paths = {
    'attention_conditional': '/work/yz886/results/1.5e6_256x256/attention_conditional/',
    'conditional': '/work/yz886/results/1.5e6_256x256/conditional/',
    'joint': '/work/yz886/results/1.5e6_256x256/joint/',
    'fno': '/work/yz886/results/1.5e6_256x256/fno_results/'
}

# Path to ground truth file
ground_truth_path = '/work/yz886/results/1.5e6_256x256/test.h5'

# Read Ground Truth
if os.path.exists(ground_truth_path):
    with h5py.File(ground_truth_path, 'r') as f:
        ground_truth_values = f['x'][:500, 2, 0, :, :]  # Expected shape (500, 256, 256)
else:
    raise FileNotFoundError(f"Ground truth file not found at {ground_truth_path}")

# Initialize error storage
l2_errors = {model: [] for model in base_paths}
h1_errors = {model: [] for model in base_paths}

# Compute errors for each model
for model, base_dir in base_paths.items():
    
    model_l2_errors = []
    model_h1_errors = []
    
    # Handle models with multiple folders
    if model in ['attention_conditional', 'conditional', 'joint']:
        folders = [os.path.join(base_dir, str(i), "trajectory_outputs") for i in range(1, 4)]
    else:
        folders = [base_dir]
    
    for folder in folders:
        if not os.path.exists(folder):
            print(f"Skipping {folder} as it does not exist.")
            continue
        
        folder_l2_errors = []
        folder_h1_errors = []
        
        for i in range(1, 501):
            file_path = os.path.join(folder, f'out_sample_{i}.npy') if model == 'fno' else os.path.join(folder, f'batch_{i}', 'timestep_3.npy')
            
            if os.path.exists(file_path):
                pred = np.load(file_path).squeeze()
                gt = ground_truth_values[i - 1]
                
                # Compute relative L2 error
                gt_norm = np.sqrt(np.mean(gt**2)) + 1e-8
                l2_error = np.sqrt(np.mean((pred - gt) ** 2)) / gt_norm
                
                # Compute relative H1 error
                grad_pred_x, grad_pred_y = np.gradient(pred)
                grad_gt_x, grad_gt_y = np.gradient(gt)
                grad_norm = np.sqrt(np.mean(grad_gt_x**2 + grad_gt_y**2)) + 1e-8
                h1_error = np.sqrt(l2_error**2 + np.mean((grad_pred_x - grad_gt_x)**2 + (grad_pred_y - grad_gt_y)**2)) / grad_norm
                
                folder_l2_errors.append(l2_error)
                folder_h1_errors.append(h1_error)
            else:
                print(f'Warning: {file_path} not found in {model}')
        
        if folder_l2_errors:
            model_l2_errors.append(np.mean(folder_l2_errors))
        if folder_h1_errors:
            model_h1_errors.append(np.mean(folder_h1_errors))
    
    # Store mean and variance
    l2_errors[model] = (np.mean(model_l2_errors), np.std(model_l2_errors)) if model in ['attention_conditional', 'conditional', 'joint'] else (np.mean(folder_l2_errors), 0)
    h1_errors[model] = (np.mean(model_h1_errors), np.std(model_h1_errors)) if model in ['attention_conditional', 'conditional', 'joint'] else (np.mean(folder_h1_errors), 0)

# Plot results
models = list(base_paths.keys())
l2_means = [l2_errors[m][0] for m in models]
l2_stds = [l2_errors[m][1] for m in models]
h1_means = [h1_errors[m][0] for m in models]
h1_stds = [h1_errors[m][1] for m in models]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, l2_means, width, yerr=l2_stds, label='L2 Error', capsize=5)
rects2 = ax.bar(x + width/2, h1_means, width, yerr=h1_stds, label='H1 Error', capsize=5)

ax.set_xlabel('Models')
ax.set_ylabel('Relative Error')
ax.set_title('Relative L2 and H1 Errors Across Models')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=0, fontsize=6)
ax.legend()
ax.grid(True)

plt.savefig(os.path.join('/work/yz886/results/1.5e6_256x256/', "Relative_L2_H1_Errors.png"), dpi=300)
plt.show()



# Visualization of ground truth vs model outputs
indices = [1, 100, 200, 300, 400, 500]
fig, axes = plt.subplots(len(indices), len(models) + 1, figsize=(12, 10))

for row, idx in enumerate(indices):
    axes[row, 0].imshow(ground_truth_values[idx - 1], cmap='seismic')
    axes[row, 0].set_title(f'GT {idx}')
    axes[row, 0].axis('off')

    for col, model in enumerate(models):
        pred_path = os.path.join(base_paths[model], f'out_sample_{idx}.npy') if model == 'fno' else os.path.join(base_paths[model], '1', 'trajectory_outputs', f'batch_{idx}', 'timestep_3.npy')

        if os.path.exists(pred_path):
            pred = np.load(pred_path).squeeze()
            axes[row, col + 1].imshow(pred, cmap='seismic')
            axes[row, col + 1].set_title(f'{model} {idx}')
        else:
            axes[row, col + 1].axis('off')

        axes[row, col + 1].axis('off')

plt.tight_layout()
plt.savefig(os.path.join('/work/yz886/results/1.5e6_256x256/', "GT_vs_Model_Outputs.png"), dpi=300)
plt.show()

