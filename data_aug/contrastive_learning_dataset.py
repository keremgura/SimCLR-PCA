from torchvision.transforms import transforms
import torch
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator, PCAAugmentorWrapper, PCAPlusTransformWrapper
from exceptions.exceptions import InvalidDatasetSelection

class IdentityTransform:
    def __call__(self, x):
        return x

class ContrastiveLearningDataset:
    def __init__(self, args):
        self.args = args

    @staticmethod
    def get_simclr_pipeline_transform(size, min_crop_scale, color_jitter_prob, gray_scale_prob, jitter_strength=1.0, blur_kernel_scale=0.1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * jitter_strength, 0.8 * jitter_strength, 0.8 * jitter_strength, 0.2 * jitter_strength)
        data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=(min_crop_scale, 1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=color_jitter_prob),
            transforms.RandomGrayscale(p=gray_scale_prob),
            GaussianBlur(kernel_size=int(blur_kernel_scale * size)),
            transforms.ToTensor()]) # convert to tensor

        return data_transforms # transformation function

    def get_dataset(self, name, n_views, pca_augmentor=None, eigenvalues=None, augmentations=True, extra_augmentations=False, split='train', train = True):
        dataset_paths = {
            'cifar10': f"{self.args.data}/cifar10",
            'stl10': f"{self.args.data}/stl10"}

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


        resize_transform = transforms.Resize((self.args.stl_resize, self.args.stl_resize)) if name == 'stl10' and self.args.stl_resize != 96 else IdentityTransform()

        ### 🧩 CASE 1: PCA augmentor
        if pca_augmentor is not None and eigenvalues is not None:
            if extra_augmentations:
                transform = transforms.Compose([
                    resize_transform,
                    PCAPlusTransformWrapper(pca_augmentor, eigenvalues, self.args.masking_method, self.args.patch_size, extra_augmentations)])
            else:
                transform = transforms.Compose([
                    resize_transform,
                    PCAAugmentorWrapper(pca_augmentor, eigenvalues, self.args.masking_method, self.args.patch_size)])
        
        ### 🧩 CASE 2: SimCLR augmentations
        elif augmentations:
            transform = transforms.Compose([
                resize_transform,
                ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(
                        size=self.args.stl_resize if img_size == 96 else 32,
                        min_crop_scale=self.args.min_crop_scale_spatial,
                        color_jitter_prob=self.args.color_jitter_prob,
                        gray_scale_prob=self.args.gray_scale_prob), n_views)])
        ### 🧩 CASE 3: No augmentations
        else:
            transform = resize_transform

        return dataset_class(dataset_root, transform=transform, download=False, **dataset_kwargs)
