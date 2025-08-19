import argparse
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import random_split
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from data_aug.view_generator import ContrastiveLearningViewGenerator, PCAAugmentorWrapper, PCAPlusTransformWrapper
from models.resnet_simclr import ResNetSimCLR, ViTSimCLR
from simclr import SimCLR
from PCAAugmentorSimCLR import PCAAugmentor
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
from utils import (
    compute_dataset_min_max,
    setup_pca,
    prepare_dataloaders,
    visualize_views,
    LinearWarmupScheduler)
import wandb
from wandb.sdk.wandb_settings import Settings

os.environ["WANDB_MODE"] = "offline"


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default='./data',
                    help='path to dataset')
parser.add_argument('--dataset_name', default='tiny_imagenet',
                    help='dataset name', choices=['stl10', 'cifar10', 'tiny_imagenet'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', # 12 normally
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--hidden_dim', default=512, type=int,
                    help='hidden dimension for SimCLR in PC space (default: 512)')
parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=250, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument("--pca_ratio", default = 0.7, type = float, help = "pca masking ratio")
parser.add_argument("--global_scaling", default = 0, type = int, choices = [1, 0], help = "whether scaling of images are done image-based (0) or based on the min / max of the whole batch (1)")
parser.add_argument("--pca", default = 1, type = int, choices = [1, 0], help = "1 if pca is applied in the augmentations")
parser.add_argument("--extra_transforms", default = 0, type = int, choices = [2, 1, 0], help = '2: all simclr augmentations, 1: only cropping and flipping, 0: no extra augmentations')
parser.add_argument('--dropout', type=float, default=0.05,
                    help='Dropout rate to apply in the projection head. Set to 0.0 to disable.')
parser.add_argument('--validation_size', default = 0.1, type = float)
parser.add_argument('--shuffle', action='store_true', default = True, help='Enable PCA component shuffling')
parser.add_argument('--drop_pc_ratio', default = 0, type = float, help = 'ratio of variance explained to be dropped')
parser.add_argument("--drop_strategy", default = "random", choices = ["random", "low", "middle", "arbitrary"], help = "determines which principal components are dropped")
parser.add_argument('--vit', action='store_true', help = 'whether a vision transformer is used instead of the ResNet')
parser.add_argument('--min_crop_scale', default = 0.6, type = float, help = 'minimum scale for cropping if extra transformations are used')
parser.add_argument('--double', action='store_true', help = 'enables double shuffling')
parser.add_argument('--interpolate', action='store_true', help = 'enables interpolating')
parser.add_argument('--pad_strategy', default = "random", choices = ["hybrid", "pad", "mean", "gaussian", "random"])
parser.add_argument('--stl_resize', default = 32, type = int, help = "Resizing applied to stl10 image data")
parser.add_argument('--masking_method', default = "global", choices = ["global", "stochastic", "cyclical", "auto", "combined", "patch_agnostic", "patch_specific"], help = "method of masking to use while generating pca-based augmentations")
parser.add_argument("--base_fractions", type=float, nargs=2, default=[0.1, 0.3], help="Two base fractions for cyclic PCA masking shift per view")
parser.add_argument("--patch_size", default = 8, type = int, help = "patch size in 2D or patchified masking")
parser.add_argument('--patch_pca_agnostic', action='store_true', help = "whether to use position-agnostic patchified pca")
parser.add_argument('--patch_pca_specific', action = 'store_true', help = "whether to use position-sepcific patchified pca")

parser.add_argument("--warmup_epochs", default = 10, type = int, help = "number of warmup epochs for the linear warmup scheduler")

parser.add_argument('--min_crop_scale_spatial', default = 0.08, type = float, help = 'minimum scale for cropping if extra transformations are used')
parser.add_argument("--color_jitter_prob", default = 0.8, type = float)
parser.add_argument("--gray_scale_prob", default = 0.2, type = float)

parser.add_argument('--subset_size', default = 10000)


#ViT parameters
parser.add_argument('--vit_patch_size', type=int, default=8, help='ViT patch size (e.g., 4 or 8)')
parser.add_argument('--vit_hidden_size', type=int, default=256, help='ViT hidden size')
parser.add_argument('--vit_layers', type=int, default=8, help='Number of transformer layers in ViT')
parser.add_argument('--vit_heads', type=int, default=4, help='Number of attention heads in ViT')
parser.add_argument('--vit_intermediate_size', type=int, default=None, help='Optional intermediate size (defaults to 4x hidden size if None)')
parser.add_argument('--vit_pooling', type=str, choices=['cls', 'mean', 'both'], default='both', help='Pooling strategy: CLS token or mean of patch tokens')
parser.add_argument('--proj_hidden_dim', type=int, default=512, help='ViT projector hidden dim')
parser.add_argument('--proj_num_layers', type=int, default=2)

parser.add_argument('--debug_timing', default = 1)
parser.add_argument('--debug_every', default = 10)

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

def main():
    args = parser.parse_args()

    
    if getattr(args, "patch_pca_specific", False):
        args.masking_method = "patch_specific"
    elif getattr(args, "patch_pca_agnostic", False):
        args.masking_method = "patch_agnostic"

    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
        
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    
    # Data & augmentor setup
    dataset = ContrastiveLearningDataset(args)
    pca_augmentor, eigenvalues = setup_pca(args, dataset)

    
    
    # Prepare training and validation dataloaders
    train_loader, val_loader, train_dataset = prepare_dataloaders(args, dataset, pca_augmentor, eigenvalues)
    

    visualization_base_dataset = dataset.get_dataset(
        args.dataset_name,
        n_views=1,
        eigenvalues=None,
        pca_augmentor=None,
        augmentations=False,
        split='unlabeled')

    probe_train_dataset = dataset.get_dataset(
            args.dataset_name,
            n_views=1,
            eigenvalues=None,
            pca_augmentor=None,
            augmentations=False,
            split = 'train',
            train = True)

    probe_test_dataset = dataset.get_dataset(
            args.dataset_name,
            n_views=1,
            eigenvalues=None,
            pca_augmentor=None,
            augmentations=False,
            split = 'test',
            train = False)
    

        


    # Visualize a few samples for sanity checking
    visualize_views(train_dataset, visualization_base_dataset, args)


    resize = args.stl_resize if args.dataset_name != "cifar10" else 32
    transform_list = []
    
    transform_list.append(transforms.Resize((resize, resize)))
    transform_list.append(transforms.ToTensor())

    probe_train_dataset.transform = transforms.Compose(transform_list)
    probe_test_dataset.transform = transforms.Compose(transform_list)

    
    if args.vit:
        model = ViTSimCLR(args)
    else:
        model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim, dropout = args.dropout)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)
    scheduler = LinearWarmupScheduler(
        optimizer=optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        target_lr=args.lr)


    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args, pca_augmentor = pca_augmentor, eigenvalues = eigenvalues)
        simclr.train(train_loader, val_loader)
        simclr.linear_probe_full(probe_train_dataset, probe_test_dataset)




import torch.multiprocessing as mp
if __name__ == "__main__":
    main()

    
