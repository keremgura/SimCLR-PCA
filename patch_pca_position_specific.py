#!/usr/bin/env python3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from sklearn.decomposition import IncrementalPCA
import argparse

parser = argparse.ArgumentParser(description="Fit position-specific patch PCA")
parser.add_argument('--dataset', choices=['stl10','cifar10', 'tiny_imagenet'], default='cifar10',
                    help="Which dataset to use")
parser.add_argument('--resize',    type=int, default=32,
                    help="Image side length after resize")
parser.add_argument('--patch_size',type=int, default=16,
                    help="Patch height & width")
parser.add_argument('--n_components', type=int, default=100,
                    help="Number of PCA components per patch")
parser.add_argument('--batch_size',   type=int, default=1024 * 3,
                    help="Loader batch size")
parser.add_argument('--output_dir',   type=str,
                    default=os.path.expanduser('~/SimCLR/outputs/patch_pca_pos'),
                    help="Where to save PCA files")
args = parser.parse_args()


transform = transforms.Compose([
    transforms.Resize((args.resize, args.resize)),
    transforms.ToTensor(),])

if args.dataset == 'stl10':
    ds = torchvision.datasets.STL10(
        root='./data/stl10', split='train', download=False, transform=transform)
elif args.dataset == 'tiny_imagenet':
    ds = torchvision.datasets.ImageFolder(
        root='./data/tiny-imagenet-200/train', transform=transform)
elif args.dataset == 'cifar10':
    ds = torchvision.datasets.CIFAR10(
        root='./data/cifar10', train=True, download=False, transform=transform)
loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)


H_p = W_p = args.resize // args.patch_size
d  = 3 * args.patch_size * args.patch_size
pca_grid = [
    [IncrementalPCA(n_components=d) for _ in range(W_p)]
    for _ in range(H_p)]

unfold = torch.nn.Unfold(kernel_size=args.patch_size, stride=args.patch_size)


for images, _ in loader:
    B = images.size(0)
    patches = unfold(images)
    patches = patches.permute(0, 2, 1)   
    
    for idx in range(H_p * W_p):
        i, j = divmod(idx, W_p)
        cell_patches = patches[:, idx, :].reshape(-1, d).numpy()
        pca_grid[i][j].partial_fit(cell_patches)


os.makedirs(args.output_dir, exist_ok=True)
for i in range(H_p):
    for j in range(W_p):
        base = f"pos_{i}_{j}_{args.dataset}_{args.resize}_{args.patch_size}"
        np.save(
            os.path.join(args.output_dir, f"patch_pc_matrix_{base}.npy"),
            pca_grid[i][j].components_)
        np.save(
            os.path.join(args.output_dir, f"patch_eigenvalues_{base}.npy"),
            pca_grid[i][j].explained_variance_)
        np.save(
            os.path.join(args.output_dir, f"patch_eigen_ratio_{base}.npy"),
            pca_grid[i][j].explained_variance_ratio_)


for i in range(H_p):
    for j in range(W_p):
        cell_patches = patches[:, idx, :].reshape(-1, d).numpy()
        
        # Compute mean & std on raw patches
        mean_vec = cell_patches.mean(axis=0)       
        std_vec  = cell_patches.std(axis=0) + 1e-8    
        
        # Standardize for PCA
        cell_patches = (cell_patches - mean_vec) / std_vec
        
        # Incremental PCA
        pca_grid[i][j].partial_fit(cell_patches)
        
        # Save mean/std for this cell
        np.save(os.path.join(args.output_dir, f"patch_mean_pos_{i}_{j}_{args.dataset}_{args.resize}_{args.patch_size}.npy"), mean_vec)
        np.save(os.path.join(args.output_dir, f"patch_std_pos_{i}_{j}_{args.dataset}_{args.resize}_{args.patch_size}.npy"), std_vec)