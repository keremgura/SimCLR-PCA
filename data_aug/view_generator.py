import numpy as np
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
        """Apply PCA masking and return two views."""
        method = self.masking_method

        if method in ("patch_agnostic", "patch_specific", "global"):
            img1, img2 = self.pca_augmentor.extract_views(img, self.eigenvalues)
        elif method == "auto":
            # randomly pick between stochastic or cyclical
            method = np.random.choice(["stochastic", "cyclical"])
            if method == "stochastic":
                img1, img2 = self.pca_augmentor.stochastic_patchwise_masking(
                    img, self.eigenvalues, patch_size=self.patch_size)
            else:
                img1, img2 = self.pca_augmentor.patchwise_cyclic_masking(
                    img, self.eigenvalues, patch_size=self.patch_size)

        elif method == "combined":
            img1 = self.pca_augmentor.stochastic_patchwise_masking(
                img, self.eigenvalues, patch_size=self.patch_size)[0]

            img2 = self.pca_augmentor.patchwise_cyclic_masking(
                img, self.eigenvalues, patch_size=self.patch_size)[1]
        elif method == "stochastic":
            img1, img2 = self.pca_augmentor.stochastic_patchwise_masking(
                img, self.eigenvalues, patch_size=self.patch_size)
        elif method == "cyclical":
            img1, img2 = self.pca_augmentor.patchwise_cyclic_masking(
                img, self.eigenvalues, patch_size=self.patch_size)
        else:
            raise ValueError("Invalid masking method.")

        if hasattr(img1, "dim") and img1.dim() == 2:
            img1 = img1.squeeze(0)
        if hasattr(img2, "dim") and img2.dim() == 2:
            img2 = img2.squeeze(0)

        return [img1, img2]

class PCAPlusTransformWrapper:
    def __init__(self, pca_augmentor, eigenvalues, masking_method, patch_size, extra_augmentations, n_views=2):
        self.pca_augmentor = pca_augmentor
        self.eigenvalues = eigenvalues
        self.extra_augmentations = extra_augmentations
        self.n_views = n_views
        self.masking_method = masking_method
        self.patch_size = patch_size

    def __call__(self, img):
        """Apply PCA masking, then extra augmentations, and return two views."""

        def _apply_extra(view): # apply extra transformations
            view = TF.to_pil_image(view)
            view = self.extra_augmentations(view)
            return TF.to_tensor(view) if not isinstance(view, torch.Tensor) else view

        method = self.masking_method
        if method == "auto":
            method = np.random.choice(["stochastic", "cyclical"])

        if method == "combined":  # one view with each method
            view1 = self.pca_augmentor.stochastic_patchwise_masking(
                img, self.eigenvalues, patch_size=self.patch_size)[0]
            view2 = self.pca_augmentor.patchwise_cyclic_masking(
                img, self.eigenvalues, patch_size=self.patch_size)[1]
        elif method == "global":
            view1, view2 = self.pca_augmentor.extract_views(img, self.eigenvalues)
        elif method == "stochastic":
            view1, view2 = self.pca_augmentor.stochastic_patchwise_masking(
                img, self.eigenvalues, patch_size=self.patch_size)
        elif method == "cyclical":
            view1, view2 = self.pca_augmentor.patchwise_cyclic_masking(
                img, self.eigenvalues, patch_size=self.patch_size)
        else:
            raise ValueError("Invalid masking method.")

        return [_apply_extra(view1), _apply_extra(view2)]
