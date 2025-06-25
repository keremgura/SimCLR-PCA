import numpy as np
import torchvision.transforms as T
import torch
from torchvision.transforms import functional as TF

np.random.seed(0)


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]
    
# replace view generator
class PCAAugmentorWrapper:
    def __init__(self, pca_augmentor, eigenvalues, masking_method, patch_size):
        """
        Wrapper to use PCAAugmentor in SimCLR.
        Args:
            pca_augmentor (PCAAugmentor): Instance of PCAAugmentor.
            eigenvalues (Tensor): Precomputed PCA eigenvalues.
        """
        self.pca_augmentor = pca_augmentor
        self.eigenvalues = eigenvalues
        self.masking_method = masking_method
        self.patch_size = patch_size
        

    def __call__(self, img):
        """
        Apply PCAAugmentor and return two PCA-masked views.
        """
        method = self.masking_method
        if method == "auto": # randomly select between the two methods
            method = np.random.choice(["stochastic", "cyclical"])
        elif method == "combined": # one view with each method
            img1 = self.pca_augmentor.stochastic_patchwise_masking(img, self.eigenvalues, patch_size=self.patch_size)[0]
            img2 = self.pca_augmentor.patchwise_cyclic_masking(img, self.eigenvalues, patch_size=self.patch_size)[1]
            if img1.dim() == 2:
                img1, img2 = img1.squeeze(0), img2.squeeze(0)
            return [img1, img2]

        
        if method == "global":
            img1, img2 = self.pca_augmentor.extract_views(img, self.eigenvalues)
        elif method == "stochastic":
            img1, img2 = self.pca_augmentor.stochastic_patchwise_masking(img, self.eigenvalues, patch_size = self.patch_size)
        elif method == "cyclical":
            img1, img2 = self.pca_augmentor.patchwise_cyclic_masking(img, self.eigenvalues, patch_size=self.patch_size)
        else:
            raise Exception("Invalid masking method.")

        # for simclr in pca
        if img1.dim() == 2:
            img1, img2 = img1.squeeze(0), img2.squeeze(0)

        

        
        return [img1, img2]
        """return img""" # for batch extract views
        
        


class PCAPlusTransformWrapper:
    def __init__(self, pca_augmentor, eigenvalues, masking_method, patch_size, extra_augmentations, n_views=2):
        self.pca_augmentor = pca_augmentor
        self.eigenvalues = eigenvalues
        self.extra_augmentations = extra_augmentations
        self.n_views = n_views
        self.masking_method = masking_method
        self.patch_size = patch_size

    def __call__(self, img):
        # Apply PCA masking
        method = self.masking_method
        if method == "auto": # randomly select between the two methods
            method = np.random.choice(["stochastic", "cyclical"])
        elif method == "combined": # one view with each method
            view1 = self.pca_augmentor.stochastic_patchwise_masking(img, self.eigenvalues, patch_size=self.patch_size)[0]
            view2 = self.pca_augmentor.patchwise_cyclic_masking(img, self.eigenvalues, patch_size=self.patch_size)[1]

            view1 = TF.to_pil_image(view1)
            view2 = TF.to_pil_image(view2)

            # Apply extra augmentations to both views
            view1 = self.extra_augmentations(view1)
            view2 = self.extra_augmentations(view2)

            if not isinstance(view1, torch.Tensor):
                view1 = TF.to_tensor(view1)
            if not isinstance(view2, torch.Tensor):
                view2 = TF.to_tensor(view2)

            return [view1, view2]
        
        
        if method == "global":
            view1, view2 = self.pca_augmentor.extract_views(img, self.eigenvalues)
        elif method == "stochastic":
            view1, view2 = self.pca_augmentor.stochastic_patchwise_masking(img, self.eigenvalues, patch_size = self.patch_size)
        elif method == "cyclical":
            view1, view2 = self.pca_augmentor.patchwise_cyclic_masking(img, self.eigenvalues, patch_size=self.patch_size)
        else:
            raise Exception("Invalid masking method.")

        view1 = TF.to_pil_image(view1)
        view2 = TF.to_pil_image(view2)

        # Apply extra augmentations to both views
        view1 = self.extra_augmentations(view1)
        view2 = self.extra_augmentations(view2)

        if not isinstance(view1, torch.Tensor):
            view1 = TF.to_tensor(view1)
        if not isinstance(view2, torch.Tensor):
            view2 = TF.to_tensor(view2)

        return [view1, view2]
