import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
import glob
import random

sys.path.append("../")

#import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
#import plotly.graph_objects as go
#from plotly.subplots import make_subplots
import sklearn
from sklearn.decomposition import PCA, IncrementalPCA
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms

random.seed(42)

"""
Performs PCA on a subset of ImageNet images
"""
resize = 32
name="stl10"
#data_fn = torchvision.datasets.ImageFolder
data_fn = torchvision.datasets.STL10
folder = "./data/stl10"



# apply transformations to loaded images
transform = transforms.Compose([
    transforms.Resize((resize, resize)),
    transforms.ToTensor(),])

trainset = data_fn(root=folder, download=False, transform=transform) # contains training images with applied transformations
num_samples = len(trainset)


# Create a DataLoader for the subset
trainloader = torch.utils.data.DataLoader(trainset, batch_size= num_samples, shuffle=False)
data_iter = iter(trainloader)
images_np, _ = next(data_iter) # image tensors

images_np = images_np.numpy()
pca_dim=500
pca = PCA()  # You can adjust the number of components


# Reshape the images to (num_samples, height * width * channels)
num_samples = images_np.shape[0]
original_shape = images_np.shape
images_np = images_np.reshape(num_samples, -1) # 2d array (num_samples * num_pixels)

# Standardize
mean, std   = np.mean(images_np, axis=0), np.std(images_np, axis=0)
images_flat = (images_np - mean) / std

# Step 4: Perform PCA
pca.fit(images_flat)


output_dir = os.path.expanduser("~/SimCLR/outputs")
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

# Save PCA results
suffix = str(resize)
np.save(os.path.join(output_dir, f"pc_matrix_ipca_stl_{suffix}.npy"), pca.components_)
np.save(os.path.join(output_dir, f"eigenvalues_ipca_stl_{suffix}.npy"), pca.explained_variance_)
np.save(os.path.join(output_dir, f"eigenvalues_ratio_ipca_stl_{suffix}.npy"), pca.explained_variance_ratio_)
