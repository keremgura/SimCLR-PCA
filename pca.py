#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import argparse

parser = argparse.ArgumentParser(description="Fit PCA/IPCA for CIFAR-10, STL-10, and Tiny-ImageNet")
parser.add_argument('--resize', type=int, default=32, help='Image side length after resize')
args = parser.parse_args()


# configuration
output_dir = os.path.expanduser("~/SimCLR/outputs")
os.makedirs(output_dir, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((args.resize, args.resize)),
    transforms.ToTensor(),])

# CIFAR10
cifar_ds = torchvision.datasets.CIFAR10(
    root="./data/cifar10",
    train=True,
    download=False,
    transform=transform)
cifar_loader = DataLoader(cifar_ds, batch_size=len(cifar_ds), shuffle=False)
cifar_imgs, _ = next(iter(cifar_loader))

cifar_np = cifar_imgs.cpu().numpy().reshape(len(cifar_ds), -1)


c_mean, c_std = cifar_np.mean(axis=0), cifar_np.std(axis=0)
c_flat = (cifar_np - c_mean) / (c_std + 1e-8)

# Fit PCA
pca_cifar = PCA()
pca_cifar.fit(c_flat)

# Save PCA results
np.save(os.path.join(output_dir, f"pc_matrix_ipca_cifar10_{args.resize}.npy"),
        pca_cifar.components_)
np.save(os.path.join(output_dir, f"eigenvalues_ipca_cifar10_{args.resize}.npy"),
        pca_cifar.explained_variance_)
np.save(os.path.join(output_dir, f"eigen_ratio_ipca_cifar10_{args.resize}.npy"),
        pca_cifar.explained_variance_ratio_)

cum_ratio_cifar = np.cumsum(pca_cifar.explained_variance_ratio_)

# STL10
stl_ds = torchvision.datasets.STL10(
    root="./data/stl10",
    split="train",
    download=False,
    transform=transform)

stl_loader = DataLoader(stl_ds, batch_size=len(stl_ds), shuffle=False)
stl_imgs, _ = next(iter(stl_loader))
stl_np = stl_imgs.cpu().numpy().reshape(len(stl_ds), -1)


s_mean, s_std = stl_np.mean(axis=0), stl_np.std(axis=0)
s_flat = (stl_np - s_mean) / (s_std + 1e-8)

# Fit PCA
pca_stl = PCA()
pca_stl.fit(s_flat)

# Save PCA results
np.save(os.path.join(output_dir, f"pc_matrix_ipca_stl10_{args.resize}.npy"),
        pca_stl.components_)
np.save(os.path.join(output_dir, f"eigenvalues_ipca_stl10_{args.resize}.npy"),
        pca_stl.explained_variance_)
np.save(os.path.join(output_dir, f"eigen_ratio_ipca_stl10_{args.resize}.npy"),
        pca_stl.explained_variance_ratio_)

cum_ratio_stl = np.cumsum(pca_stl.explained_variance_ratio_)

# Tiny ImageNet
tiny_root = "./data/tiny-imagenet-200"
tiny_train_dir = os.path.join(tiny_root, "train")
if os.path.isdir(tiny_train_dir):
    tiny_ds = torchvision.datasets.ImageFolder(
        root=tiny_train_dir,
        transform=transform)
    tiny_loader = DataLoader(tiny_ds, batch_size=2048, shuffle=False, num_workers=4)

    
    N = len(tiny_ds)
    D = 3 * args.resize * args.resize
    sum_vec = np.zeros(D, dtype=np.float64)
    sumsq_vec = np.zeros(D, dtype=np.float64)

    for imgs, _ in tiny_loader:
        arr = imgs.cpu().numpy().reshape(imgs.size(0), -1)  # (B, D)
        sum_vec += arr.sum(axis=0)
        sumsq_vec += (arr ** 2).sum(axis=0)

    mean_vec = sum_vec / N
    var_vec = (sumsq_vec / N) - (mean_vec ** 2)
    std_vec = np.sqrt(np.maximum(var_vec, 1e-12))

    ipca = IncrementalPCA(n_components=D)
    for imgs, _ in tiny_loader:
        arr = imgs.cpu().numpy().reshape(imgs.size(0), -1)
        arr_std = (arr - mean_vec) / (std_vec + 1e-8)
        ipca.partial_fit(arr_std)

    # Save PCA results
    np.save(os.path.join(output_dir, f"pc_matrix_ipca_tinyimagenet_{args.resize}.npy"),
            ipca.components_)
    np.save(os.path.join(output_dir, f"eigenvalues_ipca_tinyimagenet_{args.resize}.npy"),
            ipca.explained_variance_)
    np.save(os.path.join(output_dir, f"eigen_ratio_ipca_tinyimagenet_{args.resize}.npy"),
            ipca.explained_variance_ratio_)

    cum_ratio_tiny = np.cumsum(ipca.explained_variance_ratio_)
else:
    cum_ratio_tiny = None