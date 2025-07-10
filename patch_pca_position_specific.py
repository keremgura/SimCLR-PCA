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
parser.add_argument('--dataset', choices=['stl10','cifar10'], default='stl10',
                    help="Which dataset to use")
parser.add_argument('--resize',    type=int, default=32,
                    help="Image side length after resize")
parser.add_argument('--patch_size',type=int, default=8,
                    help="Patch height & width")
parser.add_argument('--n_components', type=int, default=100,
                    help="Number of PCA components per patch")
parser.add_argument('--batch_size',   type=int, default=64,
                    help="Loader batch size")
parser.add_argument('--output_dir',   type=str,
                    default=os.path.expanduser('~/SimCLR/outputs/patch_pca_pos'),
                    help="Where to save PCA files")
args = parser.parse_args()

# —– Prepare dataset —–
transform = transforms.Compose([
    transforms.Resize((args.resize, args.resize)),
    transforms.ToTensor(),
])
if args.dataset == 'stl10':
    ds = torchvision.datasets.STL10(
        root='./data/stl10', split='train', download=False, transform=transform
    )
else:
    ds = torchvision.datasets.CIFAR10(
        root='./data/cifar10', train=True, download=False, transform=transform
    )
loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

# —– Prepare PCA grid —–
H_p = W_p = args.resize // args.patch_size
d     = 3 * args.patch_size * args.patch_size
pca_grid = [
    [IncrementalPCA() for _ in range(W_p)]
    for _ in range(H_p)
]

unfold = torch.nn.Unfold(kernel_size=args.patch_size, stride=args.patch_size)

print("Fitting position-specific patch PCA…")
for images, _ in loader:
    # images: [B, C, H, W]
    B = images.size(0)
    patches = unfold(images)            # [B, d, H_p*W_p]
    patches = patches.permute(0, 2, 1)   # [B, H_p*W_p, d]
    # for each spatial cell, collect the B patches and update its PCA
    for idx in range(H_p * W_p):
        i, j = divmod(idx, W_p)
        cell_patches = patches[:, idx, :].reshape(-1, d).numpy()
        pca_grid[i][j].partial_fit(cell_patches)

# —– Save out all position-specific bases —–
os.makedirs(args.output_dir, exist_ok=True)
for i in range(H_p):
    for j in range(W_p):
        base = f"pos_{i}_{j}_{args.dataset}_{args.resize}_{args.patch_size}"
        np.save(
            os.path.join(args.output_dir, f"patch_pc_matrix_{base}.npy"),
            pca_grid[i][j].components_
        )
        np.save(
            os.path.join(args.output_dir, f"patch_eigenvalues_{base}.npy"),
            pca_grid[i][j].explained_variance_
        )
        np.save(
            os.path.join(args.output_dir, f"patch_eigen_ratio_{base}.npy"),
            pca_grid[i][j].explained_variance_ratio_
        )
print("Done. PCA files in", args.output_dir)
