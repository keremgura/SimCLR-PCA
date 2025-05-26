import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

class PCAAugmentor:
    def __init__(self, masking_fn_, pca_ratio, global_min = None, global_max = None, device="cpu", img_size=32, normalize = True, drop_ratio = 0, shuffle = True, interpolate = True, pad_strategy = "pad"):
        """
        Initializes the PCA-based augmentor.
        
        Args:
            masking_fn_: The PCA transformation matrix (shape: num_pixels x num_pcs).
            pca_ratio: The fraction of PCA components to mask.
            device: Device to run computations on.
        """
        self.masking_fn_ = masking_fn_.to(device)  # PCA transformation matrix
        self.pca_ratio = pca_ratio  # How much variance to retain
        self.device = device
        self.img_size = img_size
        self.num_pixels = img_size * img_size * 3
        self.normalize = normalize
        self.global_min = global_min
        self.global_max = global_max
        self.drop_ratio = drop_ratio
        self.shuffle = shuffle
        self.interpolate = interpolate
        self.pad_strategy = pad_strategy
        

        self.to_tensor = transforms.ToTensor()

    

    def compute_pc_mask(self, eigenvalues, drop_ratio = None):
        m = self.pca_ratio
        d = self.drop_ratio if drop_ratio is None else drop_ratio
        """
        Compute the PCA mask (which components to mask) based on cumulative variance.
        """
        
        eigenvalues_np = eigenvalues.cpu().numpy()
        def get_view_mask():
            index = torch.randperm(len(eigenvalues)).cpu().numpy()
            eigvals_shuffled = eigenvalues_np[index]
            cumsum = np.cumsum(eigvals_shuffled)

            drop_cutoff = np.argmin(np.abs(cumsum - d))
            cumsum_after_drop = np.cumsum(eigvals_shuffled[drop_cutoff:])
            retain_thresh = np.argmin(np.abs(cumsum_after_drop - m * (1 - d)))

            selected = index[drop_cutoff:drop_cutoff + retain_thresh]

            return torch.tensor(selected.copy(), dtype=torch.long, device=self.device)

        """pc_mask_input = get_view_mask()
        pc_mask_target = get_view_mask()

        return pc_mask_input, pc_mask_target, torch.randperm(len(eigenvalues)).cpu().numpy()"""

            
        if self.shuffle:
            index = torch.randperm(eigenvalues.shape[0]).cpu().numpy()
        else:
            index = np.arange(eigenvalues.shape[0])  # no shuffle

        
        cumsum = np.cumsum(eigenvalues.cpu().numpy()[index])

        drop_cutoff = np.argmin(np.abs(cumsum - d))
        first_view_cutoff = np.argmin(np.abs(cumsum - (d + (1 - d) * m)))

        pc_mask_input = torch.tensor(index[drop_cutoff:first_view_cutoff].copy(), dtype=torch.long, device=self.device)
        
        pc_mask_target = torch.tensor(index[first_view_cutoff:].copy(), dtype=torch.long, device=self.device)

        return pc_mask_input, pc_mask_target, index
        

    def extract_views(self, img, eigenvalues):

        def pad_matrix(P_full, keep_indices, strategy="pad", target_dim=None, std=0.01):
            device = P_full.device
            D = P_full.shape[1]
            target_dim = target_dim or D
            P_padded = torch.zeros_like(P_full)
            P_padded[:, keep_indices] = P_full[:, keep_indices]

            all_indices = torch.arange(target_dim, device=device)
            mask = torch.ones(target_dim, dtype=torch.bool, device=device)
            mask[keep_indices] = False
            drop_indices = all_indices[mask]

            if strategy == "mean":
                mean_vector = P_full[:, keep_indices].mean(dim=1, keepdim=True)
                P_padded[:, drop_indices] = mean_vector
            elif strategy == "gaussian":
                noise = torch.randn((P_full.shape[0], len(drop_indices)), device=device) * std
                P_padded[:, drop_indices] = noise
            elif strategy == "hybrid":
                rand_vals = torch.rand(len(drop_indices), device=device)
                mean_mask = rand_vals < 0.4
                gaussian_mask = (rand_vals >= 0.4) & (rand_vals < 0.6)
                mean_indices = drop_indices[mean_mask]
                gaussian_indices = drop_indices[gaussian_mask]
                if len(mean_indices) > 0:
                    mean_vector = P_full[:, keep_indices].mean(dim=1, keepdim=True)
                    P_padded[:, mean_indices] = mean_vector
                if len(gaussian_indices) > 0:
                    noise = torch.randn((P_full.shape[0], len(gaussian_indices)), device=device) * std
                    P_padded[:, gaussian_indices] = noise
            return P_padded


        if not isinstance(img, torch.Tensor):
            img = self.to_tensor(img).cpu()
        #img = self.to_tensor(img).cpu()
        img = img.to(self.device)  # Move image to device
        img_flat = img.view(1, -1)  # Flatten image
        
        # Compute which PCA components to mask
        pc_mask, pc_mask_input, index = self.compute_pc_mask(eigenvalues)

        

        D = self.masking_fn_.shape[1]  # e.g., 3072 or 12288

        import random
        if self.pad_strategy == "random":
            strategy_input = random.choice(["pad", "mean", "gaussian", "hybrid"])
            strategy_target = random.choice(["pad", "mean", "gaussian", "hybrid"])
        else:
            strategy_input, strategy_target = self.pad_strategy, self.pad_strategy

        
        p_input_full = pad_matrix(self.masking_fn_, pc_mask, strategy_input, D)
        p_target_full = pad_matrix(self.masking_fn_, pc_mask_input, strategy_target, D)

        """
        if self.pad_strategy == "pad":
            p_input_full = pad_pca_matrix_with_mask(self.masking_fn_, pc_mask)
            p_target_full = pad_pca_matrix_with_mask(self.masking_fn_, pc_mask_input)
        elif self.pad_strategy == "mean":
            p_input_full = mean_pad_pca_matrix(self.masking_fn_, pc_mask, D)
            p_target_full = mean_pad_pca_matrix(self.masking_fn_, pc_mask_input, D)
        elif self.pad_strategy == "gaussian":
            p_input_full = gaussian_pad_pca_matrix(self.masking_fn_, pc_mask, D)
            p_target_full = gaussian_pad_pca_matrix(self.masking_fn_, pc_mask_input, D)
        else:
            p_input_full = interpolate_pca_matrix(self.masking_fn_[:, pc_mask], self.masking_fn_[:, pc_mask].shape[0], D)
            p_target_full = interpolate_pca_matrix(self.masking_fn_[:, pc_mask_input], self.masking_fn_[:, pc_mask_input].shape[0], D)"""

        

        # === Projection & Reconstruction ===
        view1 = img_flat @ p_input_full
        view2 = img_flat @ p_target_full

        """view1 = F.normalize(view1, dim=1)
        view2 = F.normalize(view2, dim=1)"""

        return view1.cpu(), view2.cpu()

