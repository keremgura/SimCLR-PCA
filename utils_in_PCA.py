import os
import numpy as np
import shutil

import torch
import yaml
from datetime import datetime
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from data_aug.view_generator import ContrastiveLearningViewGenerator, PCAAugmentorWrapper, PCAPlusTransformWrapper
from models.resnet_simclr import ResNetSimCLR
from PCAAugmentorSimCLR_in_PCA import PCAAugmentor
from data_aug.gaussian_blur import GaussianBlur
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader, random_split



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def get_linear_classifier(out_dim, num_classes=10, device='cuda'):
    classifier = nn.Linear(out_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    return classifier, optimizer, criterion


def generate_experiment_name(args, prefix="simclr_in"):
    """
    Generate a standardized experiment name using SimCLR config options.
    
    Format: {prefix}_pca_{pca_ratio}_{pca_flag}_{extra_flag}_{timestamp}
    Example: simclr_pca_07_pca_extra_Mar26_20-15-05
    """
    timestamp = datetime.now().strftime("%b%d_%H-%M-%S")
    dataset = str(args.dataset_name)
    # Format PCA ratio
    pca_ratio_str = str(args.pca_ratio).replace(".", "")
    drop_pc_ratio = "drop_ratio_" + str(args.drop_pc_ratio)
    
    # Flags
    pca_flag = "pca" if args.pca == 1 else "no_pca"
    extra_flag = "extra" + str(args.extra_transforms)
    double = "double" if args.double else ""
    shuffle_flag = "shuffle" if args.shuffle else "no_shuffle"
    interpolate_flag = "interpolate" if args.interpolate else ""
    pad = str(args.pad_strategy)
    
    # Final name
    experiment_name = f"{prefix}_pca_{dataset}_{pca_ratio_str}_{pca_flag}_{double}_{extra_flag}_{shuffle_flag}_{interpolate_flag}_{pad}_{drop_pc_ratio}_{timestamp}"
    return experiment_name

# function to compute the minimum and maximum considering every image
def compute_dataset_min_max(dataset):
    """Compute global min and max pixel values across dataset for normalization."""
    global_min, global_max = float('inf'), float('-inf')

    for img_sample, _ in dataset:
        # If dataset returns multiple views per sample
        if isinstance(img_sample, (list, tuple)):
            images = img_sample
        else:
            images = [img_sample]

        for img in images:
            # Only apply ToTensor if it's a PIL Image or ndarray
            if not isinstance(img, torch.Tensor):
                img = transforms.ToTensor()(img)
            global_min = min(global_min, img.min().item())
            global_max = max(global_max, img.max().item())

    return global_min, global_max

def setup_pca(args, dataset):
    if args.pca != 1:
        return None, None

    if args.dataset_name == "cifar10":
        #pca_matrix = torch.tensor(np.load("/Users/keremgura/Desktop/SimCLR/outputs/pc_matrix_ipca.npy"), dtype=torch.float32, device=args.device)
        pca_matrix = torch.tensor(np.load("/cluster/home/kguera/SimCLR/outputs/pc_matrix_ipca.npy"), dtype=torch.float32, device=args.device)
        #eigenvalues = torch.tensor(np.load("/Users/keremgura/Desktop/SimCLR/outputs/eigenvalues_ratio_ipca.npy"), dtype=torch.float32, device=args.device)
        eigenvalues = torch.tensor(np.load("/cluster/home/kguera/SimCLR/outputs/eigenvalues_ratio_ipca.npy"), dtype=torch.float32, device=args.device)
    else:
        #pca_matrix = torch.tensor(np.load("/Users/keremgura/Desktop/SimCLR/outputs/pc_matrix_ipca_stl.npy"), dtype=torch.float32, device=args.device)
        pca_matrix = torch.tensor(np.load("/cluster/home/kguera/SimCLR/outputs/pc_matrix_ipca_stl.npy"), dtype=torch.float32, device=args.device)
        #eigenvalues = torch.tensor(np.load("/Users/keremgura/Desktop/SimCLR/outputs/eigenvalues_ratio_ipca_stl.npy"), dtype=torch.float32, device=args.device)
        eigenvalues = torch.tensor(np.load("/cluster/home/kguera/SimCLR/outputs/eigenvalues_ratio_ipca_stl.npy"), dtype=torch.float32, device=args.device)
    
    
    pca_augmentor = PCAAugmentor(
        pca_matrix.T,
        pca_ratio=args.pca_ratio,
        device=args.device,
        drop_ratio=args.drop_pc_ratio,
        shuffle=args.shuffle,
        interpolate=args.interpolate,
        pad_strategy=args.pad_strategy)

    return pca_augmentor, eigenvalues



def prepare_dataloaders(args, dataset, pca_augmentor, eigenvalues):
    if args.dataset_name == 'stl10':
        train_dataset = dataset.get_dataset(
            name='stl10',
            n_views=args.n_views,
            pca_augmentor=pca_augmentor,
            eigenvalues=eigenvalues,
            split='unlabeled')

        # Validate on labeled set
        val_dataset = dataset.get_dataset(
            name='stl10',
            n_views=args.n_views,
            pca_augmentor=pca_augmentor,
            eigenvalues=eigenvalues,
            split='train')

    else:
        full_dataset = dataset.get_dataset(
            args.dataset_name,
            n_views=args.n_views,
            pca_augmentor=pca_augmentor,
            eigenvalues=eigenvalues)
    

        train_size = int((1 - args.validation_size) * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    def simclr_collate_fn(batch):
        view1 = torch.stack([example[0][0] for example in batch], dim=0)  # First view
        view2 = torch.stack([example[0][1] for example in batch], dim=0)  # Second view
        labels = torch.tensor([example[1] for example in batch])          # Labels
        return [view1, view2], labels

    

    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, drop_last=True, collate_fn=simclr_collate_fn)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, drop_last=True, collate_fn=simclr_collate_fn)

    

    return train_loader, val_loader, train_dataset


def visualize_views(train_dataset, original_dataset, args):
    pca_ratio_str = str(args.pca_ratio).replace(".", "")
    save_folder = f"views/simclr_pca_{pca_ratio_str}"
    os.makedirs(save_folder, exist_ok=True)

    for i in range(10):
        img_views, label = train_dataset[i]
        img1, img2 = img_views[0].cpu(), img_views[1].cpu()
        original_img, _ = original_dataset[train_dataset.indices[i]]
        original_tensor = transforms.ToTensor()(original_img)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(F.to_pil_image(img1)); axes[0].set_title("Augmented View 1")
        axes[1].imshow(F.to_pil_image(img2)); axes[1].set_title("Augmented View 2")
        axes[2].imshow(F.to_pil_image(original_tensor)); axes[2].set_title("Original")
        for ax in axes: ax.axis('off')
        plt.savefig(os.path.join(save_folder, f"sample_{i}.png"))
        plt.close(fig)

    print(f"[✓] Visualizations saved to: {save_folder}")