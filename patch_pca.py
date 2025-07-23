import os
# Disable GPU so everything runs on CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from sklearn.decomposition import IncrementalPCA

import argparse

parser = argparse.ArgumentParser(description="Fit patch-level PCA")
parser.add_argument('--dataset', choices=['stl10', 'cifar10'], default='cifar10',
                    help="Dataset to use for PCA (stl10 or cifar10)")
args = parser.parse_args()

 # —— Configurable settings —— 

# —— Configurable settings —— 
resize = 32             # image resize
patch_size = 8          # patch height & width
n_components = 100      # number of PCA components per patch


if args.dataset == 'stl10':
    dataset_name = 'stl10'
    data_root = "./data/stl10"
    data_fn = torchvision.datasets.STL10
    data_kwargs = {'split': 'train', 'download': False}
else:
    dataset_name = 'cifar10'
    data_root = "./data/cifar10"
    data_fn = torchvision.datasets.CIFAR10
    data_kwargs = {'train': True, 'download': False}

output_dir = os.path.expanduser("~/SimCLR/outputs/patch_pca")

os.makedirs(output_dir, exist_ok=True)

# —— Data loading —— 
transform = transforms.Compose([
    transforms.Resize((resize, resize)),
    transforms.ToTensor(),])
trainset = data_fn(root=data_root, transform=transform, **data_kwargs)
loader = DataLoader(trainset, batch_size=64, shuffle=False, num_workers=2)

# —— Initialize Incremental PCA —— 
d = 3 * patch_size * patch_size
pca = IncrementalPCA()

# —— Patch extraction helper —— 
unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)


sum_patches = np.zeros(d, dtype=np.float64)
sum_sq_patches = np.zeros(d, dtype=np.float64)
count = 0
for images, _ in loader:
    patches = unfold(images)  # [B, d, num_patches]
    patches = patches.permute(0, 2, 1).contiguous().view(-1, d)
    arr = patches.numpy()
    sum_patches += arr.sum(axis=0)
    sum_sq_patches += np.square(arr).sum(axis=0)
    count += arr.shape[0]
mean = sum_patches / count
var = sum_sq_patches / count - mean ** 2
std = np.sqrt(var + 1e-8)


np.save(os.path.join(output_dir, f"patch_mean_{dataset_name}_{resize}_{patch_size}.npy"), mean)
np.save(os.path.join(output_dir, f"patch_std_{dataset_name}_{resize}_{patch_size}.npy"), std)


for images, _ in loader:
    # images: [B, C, H, W]
    patches = unfold(images)  
    # patches: [B, C*patch_size*patch_size, num_patches]
    patches = patches.permute(0, 2, 1).contiguous().view(-1, d)  
    # Standardize per‐feature
    mean = patches.mean(dim=0, keepdim=True)
    std  = patches.std(dim=0, keepdim=True) + 1e-8
    patches = (patches - mean) / std
    pca.partial_fit(patches.numpy())


np.save(os.path.join(output_dir, f"patch_pc_matrix_{dataset_name}_{resize}_{patch_size}.npy"),
    pca.components_)
np.save(os.path.join(output_dir, f"patch_eigenvalues_{dataset_name}_{resize}_{patch_size}.npy"),
    pca.explained_variance_)
np.save(os.path.join(output_dir, f"patch_eigen_ratio_{dataset_name}_{resize}_{patch_size}.npy"),
    pca.explained_variance_ratio_)
    


