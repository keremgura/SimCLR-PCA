from torchvision.transforms import transforms
import torch
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator, PCAAugmentorWrapper, PCAPlusTransformWrapper
from exceptions.exceptions import InvalidDatasetSelection


class SimCLRPcaDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]  # Ignore label
        views = self.transform(img)  # This returns [view1, view2] in PCA space
        return views, label

class ContrastiveLearningDataset:
    def __init__(self, root_folder, resize):
        self.root_folder = root_folder
        self.resize = resize

    
    def get_dataset(self, name, n_views, pca_augmentor=None, eigenvalues=None, split="train", train = True):
        
        dataset_paths = {
            'cifar10': f"{self.root_folder}/cifar10",  # Use manually downloaded dataset
            'stl10': f"{self.root_folder}/stl10"
        }

        if name not in dataset_paths:
            raise ValueError(f"Dataset {name} is not supported!")
        dataset_root = dataset_paths[name]
        #dataset_root = './data/cifar10'

        print("Dataset root:", dataset_root)

        if name == 'cifar10':
            dataset_class = datasets.CIFAR10
            dataset_kwargs = {'train': train}
            img_size = 32
        elif name == 'stl10':
            dataset_class = datasets.STL10
            dataset_kwargs = {'split': split}
            img_size = 96
        else:
            raise ValueError(f"Dataset {name} is not supported!")

        resize_transform = transforms.Resize((self.resize, self.resize)) if img_size == 96 else transforms.Resize((32, 32))
        base_dataset = dataset_class(dataset_root, transform=None, download=False, **dataset_kwargs)
        if pca_augmentor is not None and eigenvalues is not None:
            transform = transforms.Compose([
                resize_transform,
                PCAAugmentorWrapper(pca_augmentor, eigenvalues)])

            return SimCLRPcaDataset(base_dataset, transform)

        raise ValueError("PCA augmentor and eigenvalues must be provided for PCA-based dataset.")