import os
import numpy as np
import shutil

import torch
import yaml
from datetime import datetime
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from data_aug.view_generator import ContrastiveLearningViewGenerator, PCAAugmentorWrapper, PCAPlusTransformWrapper
from models.resnet_simclr import ResNetSimCLR, SimCLRViTModel
from PCAAugmentorSimCLR import PCAAugmentor
from data_aug.gaussian_blur import GaussianBlur
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import lr_scheduler  # ensure this is at the top of the file




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


def generate_experiment_name(args, prefix="simclr"):
    """
    Generate a standardized experiment name using SimCLR config options.
    
    Format: {prefix}_pca_{pca_ratio}_{pca_flag}_{extra_flag}_{timestamp}
    Example: simclr_pca_07_pca_extra_Mar26_20-15-05
    """
    timestamp = datetime.now().strftime("%b%d_%H-%M-%S")

    # Base components
    dataset = str(args.dataset_name)
    pca_ratio_str = str(args.pca_ratio).replace(".", "")
    masking = args.masking_method
    drop_ratio_str = f"drop_ratio_{args.drop_pc_ratio}"
    pca_flag = "pca" if args.pca == 1 else "no_pca"
    extra_flag = f"extra{args.extra_transforms}"
    shuffle_flag = "shuffle" if args.shuffle else "no_shuffle"
    min_scale_str = f"min_scale{args.min_crop_scale}"
    interpolate_flag = "interpolate" if args.interpolate else ""
    drop_strategy = args.drop_strategy
    pad = str(args.pad_strategy)
    resize=str(args.stl_resize)

    # Optional flags
    double_flag = "double" if getattr(args, "double", False) else ""
    vit_flag = "vit" if getattr(args, "vit", False) else ""

    batch_size = str(args.batch_size)
    temp = f"temp{args.temperature}"
    lr = f"lr{args.lr}"
    wd = f"wd{args.weight_decay}"
    patch_size = str(args.patch_size)

    # Assemble parts
    parts = [
        prefix,
        dataset,
        f"pca_{pca_ratio_str}",
        masking,
        pca_flag,
        double_flag,
        extra_flag,
        min_scale_str,
        shuffle_flag,
        drop_ratio_str,
        drop_strategy,
        interpolate_flag,
        pad,
        temp,
        lr,
        wd,
        resize,
        batch_size,
        patch_size,
        timestamp,
        vit_flag
    ]

    # Filter out empty strings (e.g., when double or vit is not active)
    experiment_name = "_".join(filter(None, parts))
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
    if args.patch_pca_agnostic:
        # Determine resize based on dataset
        resize_str = str(args.stl_resize) if args.dataset_name == "stl10" else "32"
        # Base directory where patch PCA files are saved
        patch_dir = "/cluster/home/kguera/SimCLR/outputs/patch_pca"
        # Construct file paths
        pca_matrix_path = os.path.join(
            patch_dir,
            f"patch_pc_matrix_{args.dataset_name}_{resize_str}_{args.patch_size}.npy"
        )
        eigenvalues_path = os.path.join(
            patch_dir,
            f"patch_eigenvalues_{args.dataset_name}_{resize_str}_{args.patch_size}.npy"
        )
        # Load patch PCA results
        pca_matrix = torch.tensor(
            np.load(pca_matrix_path), dtype=torch.float32, device=args.device
        )
        eigenvalues = torch.tensor(
            np.load(eigenvalues_path), dtype=torch.float32, device=args.device
        )
        # Build augmentor using the patch PCA basis
        pca_augmentor = PCAAugmentor(
            pca_matrix.T,
            pca_ratio=args.pca_ratio,
            device=args.device,
            drop_ratio=args.drop_pc_ratio,
            shuffle=args.shuffle,
            base_fractions=args.base_fractions,
            patch_size=args.patch_size,
            drop_strategy=args.drop_strategy,
            double=args.double,
            interpolate=args.interpolate,
            pad_strategy=args.pad_strategy,
            mean=None,
            std=None
        )
        return pca_augmentor, eigenvalues

    if args.patch_pca_specific:
        patch_dir = "/cluster/home/kguera/SimCLR/outputs/patch_pca_pos"
        # Position-specific patch PCA loading
        resize = args.stl_resize if args.dataset_name == "stl10" else 32
        H_p = W_p = resize // args.patch_size
        # Prepare grids to hold per-cell bases and eigenvalues
        pca_matrix_grid = [[None for _ in range(W_p)] for _ in range(H_p)]
        eigenvalues_grid = [[None for _ in range(W_p)] for _ in range(H_p)]
        for i in range(H_p):
            for j in range(W_p):
                base = f"pos_{i}_{j}_{args.dataset_name}_{resize}_{args.patch_size}"
                matrix_path = os.path.join(patch_dir, f"patch_pc_matrix_{base}.npy")
                eig_path    = os.path.join(patch_dir, f"patch_eigenvalues_{base}.npy")
                # Load into tensors on the correct device
                pca_matrix_grid[i][j]  = torch.tensor(np.load(matrix_path), dtype=torch.float32, device=args.device)
                eigenvalues_grid[i][j] = torch.tensor(np.load(eig_path), dtype=torch.float32, device=args.device)
        # Build augmentor in position-specific mode
        pca_augmentor = PCAAugmentor(
            masking_fn_=pca_matrix_grid,
            pca_ratio=args.pca_ratio,
            device=args.device,
            drop_ratio=args.drop_pc_ratio,
            shuffle=args.shuffle,
            base_fractions=args.base_fractions,
            drop_strategy=args.drop_strategy,
            double=args.double,
            interpolate=args.interpolate,
            pad_strategy=args.pad_strategy,
            mean=None,
            std=None,
            patch_size=args.patch_size,
            patch_specific=True
        )
        return pca_augmentor, eigenvalues_grid

        
    if args.dataset_name == "cifar10":
        pca_matrix = torch.tensor(np.load("/cluster/home/kguera/SimCLR/outputs/imagenet32_cifar10_pc_matrix_flipped.npy"), dtype=torch.float32, device=args.device).T
        eigenvalues = torch.tensor(np.load("/cluster/home/kguera/SimCLR/outputs/imagenet32_cifar10_eigenvalues_ratio.npy"), dtype=torch.float32, device=args.device)


        #pca_matrix = torch.tensor(np.load("/cluster/home/kguera/SimCLR/outputs/pc_matrix_ipca.npy"), dtype=torch.float32, device=args.device)
        #eigenvalues = torch.tensor(np.load("/cluster/home/kguera/SimCLR/outputs/eigenvalues_ratio_ipca.npy"), dtype=torch.float32, device=args.device)
    else:
        resize = str(args.stl_resize)
        pca_matrix_path = f"/cluster/home/kguera/SimCLR/outputs/pc_matrix_ipca_stl_{resize}.npy"
        eigenvalues_path = f"/cluster/home/kguera/SimCLR/outputs/eigenvalues_ratio_ipca_stl_{resize}.npy"
        pca_matrix = torch.tensor(np.load(pca_matrix_path), dtype=torch.float32, device=args.device)
        eigenvalues = torch.tensor(np.load(eigenvalues_path), dtype=torch.float32, device=args.device)

   

    eigenvalues = eigenvalues.cpu()
    pca_matrix = pca_matrix.cpu()

    # Compute global mean and std for normalization before PCA
    all_images = []
    for img, _ in dataset.get_dataset(args.dataset_name, n_views=1, augmentations=False):
        if isinstance(img, (list, tuple)):
            img = img[0]
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)
        all_images.append(img.view(-1))
    all_images = torch.stack(all_images)
    mean = all_images.mean(dim=0).to(args.device) # move mean, std to gpu
    std = all_images.std(dim=0).to(args.device)

    


    pca_augmentor = PCAAugmentor(pca_matrix.T, pca_ratio=args.pca_ratio, 
                                device=args.device, drop_ratio = args.drop_pc_ratio, shuffle = args.shuffle, base_fractions = args.base_fractions,
                                drop_strategy = args.drop_strategy, double = args.double, 
                                interpolate= args.interpolate, pad_strategy = args.pad_strategy,
                                mean = mean, std = std)

    return pca_augmentor, eigenvalues


class DatasetWithIndex(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, label = self.dataset[index]
        return data, label, index

    def __len__(self):
        return len(self.dataset)

def prepare_dataloaders(args, dataset, pca_augmentor, eigenvalues):
    image_size = args.stl_resize if args.dataset_name == 'stl10' else 32

    if args.dataset_name == 'stl10':
        train_dataset = dataset.get_dataset(
            name='stl10',
            n_views=args.n_views,
            pca_augmentor=pca_augmentor,
            eigenvalues=eigenvalues,
            augmentations=True,
            extra_augmentations=False,
            split='unlabeled')

        # Validate on labeled set
        val_dataset = dataset.get_dataset(
            name='stl10',
            n_views=args.n_views,
            pca_augmentor=pca_augmentor,
            eigenvalues=eigenvalues,
            augmentations=True,
            extra_augmentations=False,
            split='train')

    else:
        full_dataset = dataset.get_dataset(
            args.dataset_name,
            n_views=args.n_views,
            pca_augmentor=pca_augmentor,
            eigenvalues=eigenvalues,
            augmentations=True,
            extra_augmentations=False)
    

        train_size = int((1 - args.validation_size) * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    """# light augmentations
    extra_augmentations = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomResizedCrop(size=image_size, scale = (args.min_crop_scale, 1)),
        transforms.ToTensor()])"""
    extra_aug_list = []
    if not (args.dataset_name == 'stl10' and args.stl_resize == 96):
        extra_aug_list.append(transforms.Resize((image_size, image_size)))
    extra_aug_list.append(transforms.RandomResizedCrop(size=image_size, scale=(args.min_crop_scale, 1)))
    extra_aug_list.append(transforms.ToTensor())
    extra_augmentations = transforms.Compose(extra_aug_list)

    s = 1
    size = args.stl_resize if args.dataset_name == 'stl10' else 32
    
    if pca_augmentor and args.extra_transforms == 0 and args.vit:
        train_dataset.dataset.transform = PCAAugmentorWrapper(
            pca_augmentor=pca_augmentor,
            eigenvalues=eigenvalues,
            masking_method=args.masking_method,
            patch_size=args.patch_size)

    if args.extra_transforms == 1: # apply a light version of augmentations
        if args.dataset_name == 'stl10':
            """train_dataset.transform = transforms.Compose([
                transforms.Resize((size, size)),
                PCAPlusTransformWrapper(pca_augmentor=pca_augmentor,
                eigenvalues=eigenvalues,
                extra_augmentations=extra_augmentations,
                n_views=args.n_views)])"""

            transform_list = []
            if args.stl_resize != 96:
                transform_list.append(transforms.Resize((size, size)))
            transform_list.append(PCAPlusTransformWrapper(pca_augmentor=pca_augmentor,
                eigenvalues=eigenvalues,
                extra_augmentations=extra_augmentations,
                masking_method=args.masking_method,
                patch_size=args.patch_size,
                n_views=args.n_views))
            train_dataset.transform = transforms.Compose(transform_list)
            
        else:
            train_dataset.dataset.transform = PCAPlusTransformWrapper(
                pca_augmentor=pca_augmentor,
                eigenvalues=eigenvalues,
                masking_method=args.masking_method,
                patch_size=args.patch_size,
                extra_augmentations=extra_augmentations,
                n_views=args.n_views
            )

    #train_dataset = DatasetWithIndex(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, drop_last=True, pin_memory = False)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, drop_last=True, pin_memory = False)

    return train_loader, val_loader, train_dataset


def visualize_views(train_dataset, original_dataset, args):
    pca_ratio_str = str(args.pca_ratio).replace(".", "")
    save_folder = f"views/simclr_pca_{pca_ratio_str}"
    os.makedirs(save_folder, exist_ok=True)

    for i in range(15):
        img_views, label = train_dataset[i]
        """img1, img2 = img_views[0].cpu(), img_views[1].cpu()"""

        img1, img2 = img_views[0], img_views[1]
        # Reshape flattened images if necessary (e.g., for flattened image-space transformer)
        if img1.ndim == 1:
            img1 = img1.view(3, 32, 32)
        if img2.ndim == 1:
            img2 = img2.view(3, 32, 32)
        if hasattr(train_dataset, 'indices'):
            original_index = train_dataset.indices[i]
        else:
            original_index = i

        original_img, _ = original_dataset[original_index]
        original_tensor = original_img if isinstance(original_img, torch.Tensor) else transforms.ToTensor()(original_img)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(F.to_pil_image(img1)); axes[0].set_title("Augmented View 1")
        axes[1].imshow(F.to_pil_image(img2)); axes[1].set_title("Augmented View 2")
        axes[2].imshow(F.to_pil_image(original_tensor)); axes[2].set_title("Original")
        for ax in axes: ax.axis('off')
        plt.savefig(os.path.join(save_folder, f"sample_{i}.png"))
        plt.close(fig)

    print(f"[✓] Visualizations saved to: {save_folder}")

class LinearWarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, target_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.target_lr = target_lr
        self.base_lr = 0.0
        self.annealing_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, total_epochs - warmup_epochs, eta_min=0)

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr + (self.target_lr - self.base_lr) * (epoch / self.warmup_epochs)
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            self.annealing_scheduler.step(epoch - self.warmup_epochs)