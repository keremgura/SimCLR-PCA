import argparse
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import random_split
from torchvision import models
from data_aug.contrastive_learning_dataset_in_PCA import ContrastiveLearningDataset
from data_aug.view_generator import ContrastiveLearningViewGenerator, PCAAugmentorWrapper, PCAPlusTransformWrapper
from models.resnet_simclr import ResNetSimCLR, PCASimCLR, PCATransformerSimCLR
from simclr_in_PCA import SimCLR
from PCAAugmentorSimCLR_in_PCA import PCAAugmentor
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
from utils_in_PCA import (
    compute_dataset_min_max,
    setup_pca,
    prepare_dataloaders,
    visualize_views)


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default='./data',
                    help='path to dataset')
parser.add_argument('--dataset-name', default='cifar10',
                    help='dataset name', choices=['stl10', 'cifar10'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', # 12 normally
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=250, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
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
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument("--pca_ratio", default = 0.5, type = float, help = "pca masking ratio")
parser.add_argument("--global_scaling", default = 0, type = int, choices = [1, 0])
parser.add_argument("--pca", default = 1, type = int, choices = [1, 0])
parser.add_argument("--extra_transforms", default = 0, type = int, choices = [2, 1, 0], help = '2: heavy augmentations, 1: light augmentations, 0: no extra augmentations')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='Dropout rate to apply in the projection head. Set to 0.0 to disable.')
parser.add_argument('--validation_size', default = 0.2, type = float)
parser.add_argument('--drop_pc_ratio', default = 0.0, type = float)
parser.add_argument('--shuffle', action='store_true', help='Enable PCA component shuffling')
parser.add_argument("--drop_strategy", default = "middle", choices = ["random", "low", "middle", "high"], help = "determines which principal components are dropped")
parser.add_argument('--vit', action='store_true', help = 'whether a vision transformer is used')
parser.add_argument('--min_crop_scale', default = 0.6, type = float, help = 'minimum scale for cropping')
parser.add_argument('--double', action='store_true', help = 'enables double shuffling')
parser.add_argument('--interpolate', action='store_true', help = 'enables interpolating', default = True)
parser.add_argument('--pad_strategy', default = "hybrid", choices = ["random", "pad", "mean", "gaussian", "hybrid"])
parser.add_argument('--stl_resize', default = 48)

#ViT parameters
parser.add_argument('--vit_patch_size', type=int, default=16, help='ViT patch size (e.g., 4 or 8)') # try it
parser.add_argument('--vit_hidden_size', type=int, default=512, help='ViT hidden size') # only increase makes it worse
parser.add_argument('--vit_layers', type=int, default=4, help='Number of transformer layers in ViT') # looks problematic to increase above 8, try 6 and 8
parser.add_argument('--vit_heads', type=int, default=2, help='Number of attention heads in ViT') # remain same, must divide hidden size
parser.add_argument('--vit_intermediate_size', type=int, default=None, help='Optional intermediate size (defaults to 4x hidden size if None)')
parser.add_argument('--vit_pooling', type=str, choices=['cls', 'mean'], default='mean', help='Pooling strategy: CLS token or mean of patch tokens') # try both







def main():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1
    
    if args.dataset_name == "cifar10":
        #pca_matrix = torch.tensor(np.load("/Users/keremgura/Desktop/SimCLR/outputs/pc_matrix_ipca.npy"), dtype=torch.float32, device=args.device)
        pca_matrix = torch.tensor(np.load("/cluster/home/kguera/SimCLR/outputs/pc_matrix_ipca.npy"), dtype=torch.float32, device=args.device)
    else:
        #pca_matrix = torch.tensor(np.load("/Users/keremgura/Desktop/SimCLR/outputs/pc_matrix_ipca_stl.npy"), dtype=torch.float32, device=args.device)
        pca_matrix = torch.tensor(np.load("/cluster/home/kguera/SimCLR/outputs/pc_matrix_ipca_stl.npy"), dtype=torch.float32, device=args.device)

    # Data & augmentor setup
    dataset = ContrastiveLearningDataset(args.data, args.stl_resize)
    pca_augmentor, eigenvalues = setup_pca(args, dataset)

    
    
    # Prepare training and validation dataloaders
    train_loader, val_loader, train_dataset = prepare_dataloaders(args, dataset, pca_augmentor, eigenvalues)

    

    # for final encoder linear probing
    #probe_augmentor = PCAAugmentor(pca_matrix.T, pca_ratio=args.pca_ratio, device=args.device, normalize=False, shuffle=args.shuffle, interpolate=args.interpolate)
    probe_augmentor = PCAAugmentor(pca_matrix.T, pca_ratio=1, device=args.device, normalize=False, shuffle=False, interpolate=False)
    full_base_dataset = dataset.get_dataset(
        args.dataset_name,
        n_views=1,
        eigenvalues=eigenvalues,
        pca_augmentor=probe_augmentor,
        split = 'test',
        train = False)

    probe_train_dataset = dataset.get_dataset(
        args.dataset_name,
        n_views=1,
        eigenvalues=eigenvalues,
        pca_augmentor=probe_augmentor,
        split = 'train',
        train = True)

    probe_test_dataset = dataset.get_dataset(
        args.dataset_name,
        n_views=1,
        eigenvalues=eigenvalues,
        pca_augmentor=probe_augmentor,
        split = 'test',
        train = False)
    

    # Visualize a few samples for sanity checking
    #visualize_views(train_dataset, full_base_dataset, args)
    

    pca_input_dim = pca_augmentor.masking_fn_.shape[0] // 2
    
    if args.vit:  # If you want to use ViT-style encoder in PCA space
        model = PCATransformerSimCLR(
            input_dim=pca_matrix.shape[0],
            out_dim=args.out_dim,
            hidden_dim=args.vit_hidden_size,
            dropout=args.dropout)
    else:
        model = PCASimCLR(
                input_dim=pca_matrix.shape[0],
                out_dim=args.out_dim,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout)

    #model = MinimalPCATransformerSimCLR(input_dim=pca_matrix.shape[0], out_dim=args.out_dim)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        simclr.train(train_loader, val_loader)
        
        simclr.linear_probe_full(probe_train_dataset, probe_test_dataset)




import torch.multiprocessing as mp
if __name__ == "__main__":
    main()

    
