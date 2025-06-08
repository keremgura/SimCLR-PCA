from torchvision.transforms import transforms
import torch
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator, PCAAugmentorWrapper, PCAPlusTransformWrapper
from exceptions.exceptions import InvalidDatasetSelection


class ContrastiveLearningDataset:
    def __init__(self, root_folder, resize, masking_method, patch_size):
        self.root_folder = root_folder
        self.resize = resize
        self.masking_method = masking_method
        self.patch_size = patch_size

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()]) # convert to tensor
        return data_transforms # transformation function

    def get_dataset(self, name, n_views, pca_augmentor=None, eigenvalues=None, augmentations=True, extra_augmentations=False, split='train', train = True):
        dataset_paths = {
            'cifar10': f"{self.root_folder}/cifar10",
            'stl10': f"{self.root_folder}/stl10"
        }

        if name not in dataset_paths:
            raise ValueError(f"Dataset {name} is not supported!")
        
        dataset_root = dataset_paths[name]

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

        ### 🛠️ ADD A RESIZE TRANSFORM FIRST
        if name == 'stl10' and self.resize != 96:
            resize_transform = transforms.Resize((self.resize, self.resize))
            
        else:
            resize_transform = transforms.Lambda(lambda x: x)  # no-op

       

        ### 🧩 CASE 1: PCA augmentor
        if pca_augmentor is not None and eigenvalues is not None:
            if extra_augmentations:
                transform = transforms.Compose([
                    resize_transform,
                    PCAPlusTransformWrapper(pca_augmentor, eigenvalues, self.masking_method, self.patch_size, extra_augmentations)
                ])
            else:
                transform = transforms.Compose([
                    resize_transform,
                    PCAAugmentorWrapper(pca_augmentor, eigenvalues, self.masking_method, self.patch_size)
                ])
        
        ### 🧩 CASE 2: SimCLR augmentations
        elif augmentations:
            transform = transforms.Compose([
                resize_transform,
                ContrastiveLearningViewGenerator(self.get_simclr_pipeline_transform(size=self.resize if img_size == 96 else 32), n_views)
            ])

        ### 🧩 CASE 3: No augmentations
        else:
            transform = resize_transform

        

        return dataset_class(dataset_root, transform=transform, download=False, **dataset_kwargs)
