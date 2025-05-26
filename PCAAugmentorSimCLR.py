import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

class PCAAugmentor:
    def __init__(self, masking_fn_, pca_ratio, shuffle, global_min = None, global_max = None, device="cpu", img_size=32, normalize = True, drop_ratio = 0, drop_strategy = "random", double = False, interpolate = False, pad_strategy = "pad"):
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
        self.drop_strategy = drop_strategy
        self.double = double
        self.interpolate = interpolate
        self.pad_strategy = pad_strategy
        

        self.to_tensor = transforms.ToTensor()

    def compute_pc_mask(self, eigenvalues, drop_ratio=None):
        double_shuffle = self.double
        m = self.pca_ratio
        d = self.drop_ratio if drop_ratio is None else drop_ratio

        eigenvalues_np = eigenvalues.cpu().numpy()
        total_variance = np.sum(eigenvalues_np)


        if double_shuffle:

            def get_view_mask():
                if self.drop_strategy == "random":
                    index = torch.randperm(len(eigenvalues)).cpu().numpy()
                    eigvals_shuffled = eigenvalues_np[index]
                    cumsum = np.cumsum(eigvals_shuffled)

                    drop_cutoff = np.argmin(np.abs(cumsum - d))
                    cumsum_after_drop = np.cumsum(eigvals_shuffled[drop_cutoff:])
                    retain_thresh = np.argmin(np.abs(cumsum_after_drop - m * (1 - d)))

                    selected = index[drop_cutoff:drop_cutoff + retain_thresh]

                elif self.drop_strategy == "low":
                    sorted_indices = np.argsort(eigenvalues_np)[::-1]  # Descending
                    sorted_eigvals = eigenvalues_np[sorted_indices]
                    cumsum = np.cumsum(sorted_eigvals)

                    if d == 0:
                        retain_indices = sorted_indices
                    else:
                        drop_threshold = np.argmax(cumsum >= (1 - d) * total_variance)
                        retain_indices = sorted_indices[:drop_threshold]

                    if self.shuffle:
                        np.random.shuffle(retain_indices)

                    retained_eigvals = eigenvalues_np[retain_indices]
                    cumsum_ret = np.cumsum(retained_eigvals)
                    threshold = np.argmin(np.abs(cumsum_ret - m * (1 - d)))
                    selected = retain_indices[:threshold]

                elif self.drop_strategy == "middle":
                    sorted_indices = np.argsort(eigenvalues_np)[::-1]
                    sorted_eigvals = eigenvalues_np[sorted_indices]
                    cumsum = np.cumsum(sorted_eigvals)
                    


                    drop_lower = (0.5 - d / 2) * total_variance
                    drop_upper = (0.5 + d / 2) * total_variance

                    start_idx = np.searchsorted(cumsum, drop_lower)
                    end_idx   = np.searchsorted(cumsum, drop_upper)

                    # Drop the middle range
                    dropped_indices = sorted_indices[start_idx:end_idx]

                    # Retain the rest
                    retain_indices = np.concatenate([
                        sorted_indices[:start_idx],
                        sorted_indices[end_idx:]])

                    if self.shuffle:
                        np.random.shuffle(retain_indices)

                    retained_eigvals = eigenvalues_np[retain_indices]

                    cumsum_ret = np.cumsum(retained_eigvals)
                    threshold = np.argmin(np.abs(cumsum_ret - m * (1 - d)))
                    selected = retain_indices[:threshold]
                    
                else:
                    raise ValueError(f"Unsupported drop strategy: {self.drop_strategy}")

                return torch.tensor(selected.copy(), dtype=torch.long, device=self.device)

            pc_mask_input = get_view_mask()
            pc_mask_target = get_view_mask()

            return pc_mask_input, pc_mask_target, torch.randperm(len(eigenvalues)).cpu().numpy()



        # no double shuffling
        if self.shuffle:
            index = torch.randperm(eigenvalues.shape[0]).cpu().numpy()  # Random shuffle
        else:
            index = np.arange(eigenvalues.shape[0])

        

        if self.drop_strategy == "random":
            cumsum = np.cumsum(eigenvalues.cpu().numpy()[index])

            drop_cutoff = np.argmin(np.abs(cumsum - d))
            first_view_cutoff = np.argmin(np.abs(cumsum - (d + (1 - d) * m)))

            pc_mask_input = torch.tensor(index[drop_cutoff:first_view_cutoff].copy(), dtype=torch.long, device=self.device)
            pc_mask_target = torch.tensor(index[first_view_cutoff:].copy(), dtype=torch.long, device=self.device)

            return pc_mask_input, pc_mask_target, index
        
        
        

        total_variance = torch.sum(eigenvalues).item()
        eigenvalues_np = eigenvalues.cpu().numpy()
        total_pc = len(eigenvalues_np)

        # Step 1: Drop components based on variance contribution
        sorted_indices = np.argsort(eigenvalues_np)[::-1]  # Descending by variance
        sorted_eigvals = eigenvalues_np[sorted_indices]
        cumsum = np.cumsum(sorted_eigvals)

        if self.drop_strategy == "low":
            # Drop PCs from the tail of explained variance (low variance)
            drop_threshold = np.argmax(cumsum >= (1 - d) * total_variance)
            retain_indices = sorted_indices[:drop_threshold]
        elif self.drop_strategy == "middle":
            # Identify both start and end of the "middle" d% of explained variance to drop
            drop_lower = (0.5 - d / 2) * total_variance
            drop_upper = (0.5 + d / 2) * total_variance

            start_idx = np.searchsorted(cumsum, drop_lower)
            end_idx   = np.searchsorted(cumsum, drop_upper)

            retain_indices = np.concatenate((
                sorted_indices[:start_idx],
                sorted_indices[end_idx:]))

        elif self.drop_strategy == "high":
            drop_threshold = np.argmax(cumsum >= d * total_variance)
            
            retain_indices = sorted_indices[drop_threshold:]
        

        retained_eigvals = eigenvalues_np[retain_indices]
        if self.shuffle:
            idx = torch.randperm(retained_eigvals.shape[0])
            retained_eigvals = retained_eigvals[idx]
            retain_indices = retain_indices[idx]


        # Step 3: Select m% of retained variance for View 1
        
        retained_total_var = np.sum(retained_eigvals)
        retained_cumsum = np.cumsum(retained_eigvals)

        visible_threshold = np.argmin(np.abs(retained_cumsum - m * retained_total_var))

        pc_mask_input = retain_indices[:visible_threshold]
        pc_mask_target = retain_indices[visible_threshold:]
        
        pc_mask_input = torch.tensor(pc_mask_input, dtype=torch.long, device=self.device)
        pc_mask_target = torch.tensor(pc_mask_target, dtype=torch.long, device=self.device)


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
        img = img.to(self.device)  # Move image to device
        img_flat = img.view(1, -1)  # Flatten image

        pc_mask, pc_mask_input, index = self.compute_pc_mask(eigenvalues)

        D = self.masking_fn_.shape[1]
        
        
        
        if self.interpolate:
            import random
            if self.pad_strategy == "random":
                strategy_input = random.choice(["pad", "mean", "gaussian", "hybrid"])
                strategy_target = random.choice(["pad", "mean", "gaussian", "hybrid"])
            else:
                strategy_input, strategy_target = self.pad_strategy, self.pad_strategy

            
            p_input_full = pad_matrix(self.masking_fn_, pc_mask, strategy_input, D)
            p_target_full = pad_matrix(self.masking_fn_, pc_mask_input, strategy_target, D)
            img_reconstructed = img_flat @ p_input_full @ p_input_full.T
            target = img_flat @ p_target_full @ p_target_full.T
        else:
            # Compute which PCA components to mask
            p_input = self.masking_fn_[:, pc_mask_input]
            p_target = self.masking_fn_[:, pc_mask]
            
            target = (img_flat @ p_target) @ p_target.T
            img_reconstructed = (img_flat @ p_input) @ p_input.T
        


        # **Normalize both views to [0,1]**
        if self.normalize:
            img_reconstructed = img_reconstructed - img_reconstructed.min()
            img_reconstructed = img_reconstructed / (img_reconstructed.max() - img_reconstructed.min())

            target = target - target.min()
            target = target / (target.max() - target.min())

        
        return img_reconstructed.view(img.shape).cpu(), target.view(img.shape).cpu()

