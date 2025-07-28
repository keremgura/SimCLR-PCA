#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# ─── Configuration ─────────────────────────────────────────────────────────────
resize = 32
output_dir = os.path.expanduser("~/SimCLR/outputs")
os.makedirs(output_dir, exist_ok=True)

# Shared transform
transform = transforms.Compose([
    transforms.Resize((resize, resize)),
    transforms.ToTensor(),
])

# ─── CIFAR-10 PCA ───────────────────────────────────────────────────────────────
# Load entire CIFAR-10 train split
cifar_ds = torchvision.datasets.CIFAR10(
    root="./data/cifar10",
    train=True,
    download=False,
    transform=transform
)
cifar_loader = DataLoader(cifar_ds, batch_size=len(cifar_ds), shuffle=False)
cifar_imgs, _ = next(iter(cifar_loader))
# Flatten to (N, D)
cifar_np = cifar_imgs.numpy().reshape(len(cifar_ds), -1)

# Standardize
c_mean, c_std = cifar_np.mean(axis=0), cifar_np.std(axis=0)
c_flat = (cifar_np - c_mean) / (c_std + 1e-8)

# Fit PCA
pca_cifar = PCA()
pca_cifar.fit(c_flat)

# Save PCA results
np.save(os.path.join(output_dir, f"pc_matrix_ipca_cifar10_{resize}.npy"),
        pca_cifar.components_)
np.save(os.path.join(output_dir, f"eigenvalues_ipca_cifar10_{resize}.npy"),
        pca_cifar.explained_variance_)
np.save(os.path.join(output_dir, f"eigen_ratio_ipca_cifar10_{resize}.npy"),
        pca_cifar.explained_variance_ratio_)

cum_ratio_cifar = np.cumsum(pca_cifar.explained_variance_ratio_)

# ─── STL-10 PCA ────────────────────────────────────────────────────────────────
stl_ds = torchvision.datasets.STL10(
    root="./data/stl10",
    split="train",
    download=False,
    transform=transform
)
stl_loader = DataLoader(stl_ds, batch_size=len(stl_ds), shuffle=False)
stl_imgs, _ = next(iter(stl_loader))
stl_np = stl_imgs.numpy().reshape(len(stl_ds), -1)

# Standardize
s_mean, s_std = stl_np.mean(axis=0), stl_np.std(axis=0)
s_flat = (stl_np - s_mean) / (s_std + 1e-8)

# Fit PCA
pca_stl = PCA()
pca_stl.fit(s_flat)

# Save PCA results
np.save(os.path.join(output_dir, f"pc_matrix_ipca_stl10_{resize}.npy"),
        pca_stl.components_)
np.save(os.path.join(output_dir, f"eigenvalues_ipca_stl10_{resize}.npy"),
        pca_stl.explained_variance_)
np.save(os.path.join(output_dir, f"eigen_ratio_ipca_stl10_{resize}.npy"),
        pca_stl.explained_variance_ratio_)

cum_ratio_stl = np.cumsum(pca_stl.explained_variance_ratio_)

# ─── Plot & Save Comparison ───────────────────────────────────────────────────
plt.figure()
plt.plot(cum_ratio_cifar[:160], label="CIFAR10")
plt.plot(cum_ratio_stl[:160],   label="STL10")
plt.ylabel("Cumulative Variance Ratio")

# x-ticks at multiples of 16 (16, 32, ..., 160)
ticks = list(range(16, 161, 16))
positions = [t-1 for t in ticks]  # zero-based indices
plt.xticks(positions, ticks)

# Annotate cumulative-variance values at the selected PCs
for t in ticks:
    pos = t - 1  # zero-based index
    # CIFAR10 value
    val_c = cum_ratio_cifar[pos]
    plt.text(pos, val_c, f"{val_c:.2f}", ha='center', va='bottom', fontsize='small')
    # STL10 value
    val_s = cum_ratio_stl[pos]
    plt.text(pos, val_s, f"{val_s:.2f}", ha='center', va='top', fontsize='small')


plt.legend()
plt.tight_layout()

plot_path = os.path.join(output_dir, f"pca_cumvar_compare_{resize}.png")
plt.savefig(plot_path, bbox_inches="tight")
plt.close()
print(f"PCA components and eigenvalues saved in {output_dir}")
print(f"Cumulative‐variance comparison plot saved to {plot_path}")