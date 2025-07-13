import numpy as np
import torch
import random
import torchvision.transforms as transforms
import torch.nn.functional as F

class PCAAugmentor:
        def __init__(self, masking_fn_, pca_ratio, shuffle, base_fractions,
                 global_min=None, global_max=None, device="cpu",
                 img_size=32, patch_size=None, patch_specific=False,
                 normalize=True, drop_ratio=0, drop_strategy="random",
                 double=False, interpolate=False, pad_strategy="pad",
                 mean=None, std=None):
        
        
        self.masking_fn_ = masking_fn_.to(device)  # PCA transformation matrix     
        self.pca_ratio = pca_ratio  # How much variance to retain
        self.device = device
        self.img_size = img_size
        # Determine whether using patch PCA (patch_size != None)
        self.use_patch = patch_size is not None
        self.patch_specific = patch_specific
        self.patch_size = patch_size

        if self.patch_specific:
            # masking_fn_ is now a 2D list: pca_matrix_grid[i][j]
            self.pca_matrix_grid = masking_fn_
        else:
            # single basis (global or agnostic-patch)
            self.masking_fn_ = masking_fn_.to(device)

        if self.use_patch:
            # dimension of each patch vector
            self.patch_dim = 3 * patch_size * patch_size
            self.num_pixels = self.patch_dim
        else:
            self.num_pixels = img_size * img_size * 3
        self.normalize = normalize
        self.global_min = global_min
        self.global_max = global_max
        self.drop_ratio = drop_ratio
        self.shuffle = shuffle
        self.base_fractions = base_fractions
        self.drop_strategy = drop_strategy
        self.double = double
        self.interpolate = interpolate
        self.pad_strategy = pad_strategy
        self.precomputed_masks = None

        
        self.mean = mean.clone().detach().to(dtype=torch.float32, device=device)
        self.std = std.clone().detach().to(dtype=torch.float32, device=device)
        """self.mean = torch.tensor(mean, dtype=torch.float32, device=device) if mean is not None else None
        self.std = torch.tensor(std, dtype=torch.float32, device=device) if std is not None else None"""
        

        self.to_tensor = transforms.ToTensor()

    def compute_pc_mask(self, eigenvalues, drop_ratio=None):
        double_shuffle = self.double
        m = self.pca_ratio
        d = self.drop_ratio if drop_ratio is None else drop_ratio

        """eigenvalues_np = eigenvalues.cpu().numpy()
        total_variance = np.sum(eigenvalues_np)"""

        eigenvalues = eigenvalues.to(self.device)
        total_variance = eigenvalues.sum()

        def sample_view_mask(strategy, drop_ratio, retain_ratio):
            #sorted_indices = np.argsort(eigenvalues_np)[::-1]
            sorted_indices = torch.argsort(eigenvalues, descending=True)
            #sorted_eigvals = eigenvalues_np[sorted_indices]
            #cumsum = np.cumsum(sorted_eigvals)
            sorted_eigvals = eigenvalues[sorted_indices]
            cumsum = torch.cumsum(sorted_eigvals, dim=0)

            

            if strategy == "low":
                mask = cumsum >= (1 - drop_ratio) * total_variance
                """if not np.any(mask):
                    drop_threshold = len(sorted_indices)
                else:
                    drop_threshold = np.argmax(mask)"""

                if not torch.any(mask):
                    drop_threshold = len(sorted_indices)
                else:
                    drop_threshold = torch.argmax(mask)

                retain_indices = sorted_indices[:drop_threshold]
                if len(retain_indices) == 0 or drop_ratio == 0:
                    retain_indices = sorted_indices

                

            elif strategy == "middle":
                drop_lower = (0.5 - drop_ratio / 2) * total_variance
                drop_upper = (0.5 + drop_ratio / 2) * total_variance
                """start_idx = np.searchsorted(cumsum, drop_lower)
                end_idx = np.searchsorted(cumsum, drop_upper)"""
                start_idx = torch.searchsorted(cumsum, torch.tensor(drop_lower, device=self.device))
                end_idx = torch.searchsorted(cumsum, torch.tensor(drop_upper, device=self.device))
                #retain_indices = np.concatenate([sorted_indices[:start_idx], sorted_indices[end_idx:]])
                retain_indices = torch.cat([sorted_indices[:start_idx], sorted_indices[end_idx:]])

                if len(retain_indices) == 0 or end_idx <= start_idx:
                    retain_indices = sorted_indices

            else:  # "random"
                """index = torch.randperm(len(eigenvalues)).cpu().numpy()
                eigvals_shuffled = eigenvalues_np[index]
                cumsum = np.cumsum(eigvals_shuffled)"""

                index = torch.randperm(len(eigenvalues), device=self.device)
                eigvals_shuffled = eigenvalues[index]
                cumsum = torch.cumsum(eigvals_shuffled, dim=0)

                """drop_cutoff = np.argmin(np.abs(cumsum - d))
                cumsum_after_drop = np.cumsum(eigvals_shuffled[drop_cutoff:])
                retain_thresh = np.argmin(np.abs(cumsum_after_drop - m * (1 - d)))"""

                drop_cutoff = torch.argmin(torch.abs(cumsum - d))
                #cumsum_after_drop = torch.cumsum(eigvals_shuffled[drop_cutoff:])
                cumsum_after_drop = torch.cumsum(eigvals_shuffled[drop_cutoff:], dim=0)
                retain_thresh = torch.argmin(torch.abs(cumsum_after_drop - m * (1 - d)))

                selected = index[drop_cutoff:drop_cutoff + retain_thresh]
                #retain_indices = np.arange(len(eigenvalues_np))

            

            if strategy != "random":
                """if self.shuffle:
                    np.random.shuffle(retain_indices)
                retained_eigvals = eigenvalues_np[retain_indices]
                cumsum_ret = np.cumsum(retained_eigvals)

                diffs = np.abs(cumsum_ret - retain_ratio * (1 - drop_ratio))
                if len(diffs) == 0:
                    selected = retain_indices
                else:
                    threshold = np.argmin(diffs)
                    selected = retain_indices if threshold == 0 else retain_indices[:threshold]"""

                retain_indices = retain_indices[torch.randperm(len(retain_indices))]
                retained_eigvals = eigenvalues[retain_indices]
                cumsum_ret = torch.cumsum(retained_eigvals, dim=0)
                diffs = torch.abs(cumsum_ret - retain_ratio * (1 - drop_ratio))
                threshold = torch.argmin(diffs)

            #return torch.tensor(selected.copy(), dtype=torch.long, device=self.device)
            return selected.clone().detach().long().to(self.device)
            #return torch.tensor(selected.copy(), dtype=torch.long)


        # Apply slight randomization to drop and retain ratios for both input and target
        drop_randomize = 0.1 if d > 0 else 0
        mask_randomize = 0.1
        """drop_input = d + np.random.uniform(-drop_randomize, drop_randomize)
        retain_input = m + np.random.uniform(-mask_randomize, mask_randomize)
        drop_target = d + np.random.uniform(-drop_randomize, drop_randomize)
        retain_target = m + np.random.uniform(-mask_randomize, mask_randomize)"""

        drop_input = d + (torch.rand(1, device=self.device).item() * 2 * drop_randomize - drop_randomize)
        retain_input = m + (torch.rand(1, device=self.device).item() * 2 * mask_randomize - mask_randomize)
        drop_target = d + (torch.rand(1, device=self.device).item() * 2 * drop_randomize - drop_randomize)
        retain_target = m + (torch.rand(1, device=self.device).item() * 2 * mask_randomize - mask_randomize)

        # Weighted randomization of drop_strategy per call if requested
        if self.drop_strategy == "arbitrary":
            strategy_pool = ["random", "low", "middle"]
            strategy_weights = [0.7, 0.15, 0.15]
            """selected_strategy_input = np.random.choice(strategy_pool, p=strategy_weights)
            selected_strategy_target = np.random.choice(strategy_pool, p=strategy_weights)"""

            selected_strategy_input = random.choices(strategy_pool, weights=strategy_weights, k=1)[0]
            selected_strategy_target = random.choices(strategy_pool, weights=strategy_weights, k=1)[0]
        else:
            selected_strategy_input = selected_strategy_target = self.drop_strategy

            

        if double_shuffle:
            pc_mask_input = sample_view_mask(
                selected_strategy_input,
                np.clip(drop_input, 0.0, 0.5),
                np.clip(retain_input, 0.0, 1.0))
            pc_mask_target = sample_view_mask(
                selected_strategy_target,
                np.clip(drop_target, 0.0, 0.5),
                np.clip(retain_target, 0.0, 1.0))

            
        else:
            # Use a shared retained set, then split into disjoint parts
            base_mask = sample_view_mask(selected_strategy_input, d, 1.0)
            if self.shuffle:
                base_mask = base_mask[torch.randperm(base_mask.shape[0])]
            split_idx = int(m * base_mask.shape[0])
            pc_mask_input = base_mask[:split_idx]
            pc_mask_target = base_mask[split_idx:]

            """eigvals = eigenvalues[base_mask.cpu()].cpu().numpy()
            if self.shuffle:
                # Shuffle eigenvalues and base_mask together
                indices = np.arange(len(base_mask))
                np.random.shuffle(indices)
                eigvals = eigvals[indices]
                base_mask = base_mask[indices]

            cumsum = np.cumsum(eigvals)
            total = cumsum[-1]
            midpoint = np.searchsorted(cumsum, total * m)
            pc_mask_input = base_mask[:midpoint]
            pc_mask_target = base_mask[midpoint:]"""


        return pc_mask_input, pc_mask_target

    def precompute_masks(self, eigenvalues_tensor: torch.Tensor):
        """
        Precomputes and stores PCA masks for all samples in the dataset.
        Should be called once per epoch.
        """
        self.precomputed_masks = []
        for i in range(eigenvalues_tensor.size(0)):
            eigval_sample = eigenvalues_tensor[i]
            if eigval_sample.dim() == 0:
                eigval_sample = eigval_sample.unsqueeze(0)
            pc_mask_input, pc_mask_target = self.compute_pc_mask(eigval_sample)
            self.precomputed_masks.append((pc_mask_input, pc_mask_target))

    def extract_views(self, img, eigenvalues, index = None):
        
        def pad_matrix(P_full, keep_indices, strategy="pad", target_dim=None, std=0.01):
            device = P_full.device
            #device = torch.device("cpu")
            
            D = P_full.shape[1]
            target_dim = target_dim or D
            P_padded = torch.zeros_like(P_full)
            P_padded[:, keep_indices] = P_full[:, keep_indices]

            all_indices = torch.arange(target_dim, device=device)
            mask = torch.ones(target_dim, dtype=torch.bool, device=device)
            mask[keep_indices] = False
            drop_indices = all_indices[mask]

            mean_by_own = True

            if strategy == "mean":
                if mean_by_own:
                    mean_values = P_full[:, drop_indices].mean(dim=0, keepdim=True)  # Shape: (1, num_drop)
                    # Expand to batch size and assign to padded matrix
                    P_padded[:, drop_indices] = mean_values.expand(P_full.shape[0], -1)
                else:
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
                    if mean_by_own:
                        mean_values = P_full[:, mean_indices].mean(dim=0, keepdim=True)
                        P_padded[:, mean_indices] = mean_values.expand(P_full.shape[0], -1)
                    else:
                        mean_vector = P_full[:, keep_indices].mean(dim=1, keepdim=True)
                        P_padded[:, mean_indices] = mean_vector
                if len(gaussian_indices) > 0:
                    noise = torch.randn((P_full.shape[0], len(gaussian_indices)), device=device) * std
                    P_padded[:, gaussian_indices] = noise
            return P_padded

        """if self.patch_specific:
            # assume img.shape == [C,H,W] and H,W divisible by patch_size
            C, H, W = img.shape
            H_p = W_p = H // self.patch_size

            # unfold into [1, patch_dim, H_p*W_p]
            patches = F.unfold(
                img.unsqueeze(0),
                kernel_size=self.patch_size,
                stride=self.patch_size
            )
            # → [num_patches, patch_dim]
            P = patches.squeeze(0).transpose(0,1)

            recon_in, recon_tg = [], []
            for idx in range(H_p * W_p):
                i, j = divmod(idx, W_p)
                Pmat = self.pca_matrix_grid[i][j]      # [patch_dim, K]
                eigs = eigenvalues[i][j]               # [K]
                pc_in, pc_tg = self.compute_pc_mask(eigs)

                pv = P[idx:idx+1]                       # [1, patch_dim]
                Pin = Pmat[:, pc_in]                   # [patch_dim, #in]
                Pt  = Pmat[:, pc_tg]                   # [patch_dim, #tg]

                recon_in.append((pv @ Pin) @ Pin.T)    # [1, patch_dim]
                recon_tg.append((pv @ Pt)  @ Pt.T)     # [1, patch_dim]

            # fold back into images
            stack_in = torch.cat(recon_in, dim=0).transpose(0,1).unsqueeze(0)
            stack_tg = torch.cat(recon_tg, dim=0).transpose(0,1).unsqueeze(0)

            img1 = F.fold(
                stack_in, output_size=(H,W),
                kernel_size=self.patch_size, stride=self.patch_size
            ).squeeze(0)
            img2 = F.fold(
                stack_tg, output_size=(H,W),
                kernel_size=self.patch_size, stride=self.patch_size
            ).squeeze(0)

            # normalize to [0,1]
            def norm(x):
                x = x - x.min()
                return x / (x.max() - x.min() + 1e-6)

            return norm(img1), norm(img2)"""

        if not isinstance(img, torch.Tensor):
            img = self.to_tensor(img)
        img = img.to(self.device)  # Move image to device

        
        """if not self.use_patch:
            # --- Global PCA path ---
            img_flat = img.view(1, -1)  # Flatten image
            if self.mean is not None and self.std is not None:
                img_flat = (img_flat - self.mean) / (self.std + 1e-6)
            if self.precomputed_masks is not None and index is not None:
                pc_mask_input, pc_mask = self.precomputed_masks[index]
            else:
                pc_mask, pc_mask_input = self.compute_pc_mask(eigenvalues)

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
                denom = (img_reconstructed.max() - img_reconstructed.min())
                if denom < 1e-4:
                    denom = 1e-4
                img_reconstructed = (img_reconstructed - img_reconstructed.min()) / denom

                denom_target = target.max() - target.min()
                if denom_target < 1e-4:
                    denom_target = 1e-4

                target = (target - target.min()) / denom_target
                

            
            #return img_reconstructed.view(img.shape).cpu(), target.view(img.shape).cpu()
            return img_reconstructed.view(img.shape), target.view(img.shape) # no switching to cpu

        
        C, H, W = img.shape
        # 1) Unfold into non-overlapping patches
        patches = F.unfold(
            img.unsqueeze(0),
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        # [1, patch_dim, num_patches] → [num_patches, patch_dim]
        P = patches.squeeze(0).transpose(0, 1)

        # 2) For each patch, compute input/target projections
        recon_patches_in = []
        recon_patches_tg = []
        for i in range(P.size(0)):
            pv = P[i : i + 1]             # [1, patch_dim]
            pc_in, pc_tg = self.compute_pc_mask(eigenvalues)
            Pmat = self.masking_fn_       # [patch_dim, num_pcs]
            Pin = Pmat[:, pc_in]          # select cols for view1
            Pt  = Pmat[:, pc_tg]          # select cols for view2
            recon_patches_in.append((pv @ Pin) @ Pin.T)
            recon_patches_tg.append((pv @ Pt)  @ Pt.T)

        # 3) Fold back into two full images
        stack_in = torch.cat(recon_patches_in, dim=0).transpose(0, 1).unsqueeze(0)
        stack_tg = torch.cat(recon_patches_tg, dim=0).transpose(0, 1).unsqueeze(0)
        img_in = F.fold(
            stack_in,
            output_size=(H, W),
            kernel_size=self.patch_size,
            stride=self.patch_size
        ).squeeze(0)
        img_tg = F.fold(
            stack_tg,
            output_size=(H, W),
            kernel_size=self.patch_size,
            stride=self.patch_size
        ).squeeze(0)

        # 4) Normalize both views to [0,1]
        def norm(x):
            x = x - x.min()
            return x / (x.max() - x.min() + 1e-6)

        return norm(img_in), norm(img_tg)"""
            



        img_flat = img.view(1, -1)  # Flatten image

        if self.mean is not None and self.std is not None:
            img_flat = (img_flat - self.mean) / (self.std + 1e-6)

        if self.precomputed_masks is not None and index is not None:
            pc_mask_input, pc_mask = self.precomputed_masks[index]
        else:
            pc_mask, pc_mask_input = self.compute_pc_mask(eigenvalues)

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
            denom = (img_reconstructed.max() - img_reconstructed.min())
            if denom < 1e-4:
                denom = 1e-4
            img_reconstructed = (img_reconstructed - img_reconstructed.min()) / denom

            denom_target = target.max() - target.min()
            if denom_target < 1e-4:
                denom_target = 1e-4

            target = (target - target.min()) / denom_target
            

        
        #return img_reconstructed.view(img.shape).cpu(), target.view(img.shape).cpu()
        return img_reconstructed.view(img.shape), target.view(img.shape) # no switching to cpu


    """def extract_views_batch(self, imgs: torch.Tensor, eigenvalues: torch.Tensor):
        
        Batched PCA-based augmentation. Operates on GPU.
        
        Args:
            imgs: Tensor of shape (B, C, H, W)
            eigenvalues: Tensor of shape (B, D)
        
        Returns:
            Two views (img1, img2) of shape (B, C, H, W), projected via masked PCA bases.
        
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

            mean_by_own = True

            if strategy == "mean":
                if mean_by_own:
                    mean_values = P_full[:, drop_indices].mean(dim=0, keepdim=True)
                    P_padded[:, drop_indices] = mean_values.expand(P_full.shape[0], -1)
                else:
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
                    if mean_by_own:
                        mean_values = P_full[:, mean_indices].mean(dim=0, keepdim=True)
                        P_padded[:, mean_indices] = mean_values.expand(P_full.shape[0], -1)
                    else:
                        mean_vector = P_full[:, keep_indices].mean(dim=1, keepdim=True)
                        P_padded[:, mean_indices] = mean_vector
                if len(gaussian_indices) > 0:
                    noise = torch.randn((P_full.shape[0], len(gaussian_indices)), device=device) * std
                    P_padded[:, gaussian_indices] = noise
            return P_padded

        # imgs: (B, C, H, W), eigenvalues: (B, D)
        B, C, H, W = imgs.shape
        imgs = imgs.to(self.device)
        img_flat = imgs.view(B, -1)

        if self.mean is not None and self.std is not None:
            img_flat = (img_flat - self.mean) / (self.std + 1e-6)

        # For each sample in batch, compute pc masks
        pc_masks = [self.compute_pc_mask(eigenvalues[i]) for i in range(B)]

        # Only support non-interpolate mode for batch for now (for simplicity)
        img_reconstructed = []
        target = []
        for i in range(B):
            pc_mask_input, pc_mask = pc_masks[i]
            p_input = self.masking_fn_[:, pc_mask_input]
            p_target = self.masking_fn_[:, pc_mask]
            v1 = (img_flat[i:i+1] @ p_input) @ p_input.T
            v2 = (img_flat[i:i+1] @ p_target) @ p_target.T
            img_reconstructed.append(v1)
            target.append(v2)
        img_reconstructed = torch.cat(img_reconstructed, dim=0)
        target = torch.cat(target, dim=0)

        # Normalize both views to [0,1] per sample in batch
        if self.normalize:
            img_reconstructed = (img_reconstructed - img_reconstructed.min(dim=1, keepdim=True)[0]) / \
                                (img_reconstructed.max(dim=1, keepdim=True)[0] - img_reconstructed.min(dim=1, keepdim=True)[0] + 1e-4)
            target = (target - target.min(dim=1, keepdim=True)[0]) / \
                    (target.max(dim=1, keepdim=True)[0] - target.min(dim=1, keepdim=True)[0] + 1e-4)

        return img_reconstructed.view(B, C, H, W).cpu(), target.view(B, C, H, W).cpu()"""


    


    def get_patch_row_indices(self, i, j, patch_size, img_size):
        indices = []
        for c in range(3):  # R,G,B
            base = c * img_size * img_size
            for dy in range(patch_size):
                for dx in range(patch_size):
                    y = i + dy
                    x = j + dx
                    indices.append(base + y*img_size + x)
        return indices

    def stochastic_patchwise_masking(self, img, eigenvalues, patch_size=8):
        """
        Apply stochastic PCA masking per patch by randomly shuffling components
        and retaining enough components to meet the variance threshold.
        """
        if not isinstance(img, torch.Tensor):
            img = self.to_tensor(img)
        img = img.to(self.device)
        img_flat = img.view(-1)

        mask_randomize = 0.2
        
        def apply_patchwise_masking():
            C, H, W = img.shape
            P = self.masking_fn_
            eigvals = eigenvalues.cpu().numpy()
            total_components = P.shape[1]
            total_variance = np.sum(eigvals)

            patches = []
            for i in range(0, H, patch_size):
                for j in range(0, W, patch_size):
                    patch = img[:, i:i+patch_size, j:j+patch_size]
                    patch_flat = patch.contiguous().view(-1)

                    variance_ratio = self.pca_ratio + np.random.uniform(-mask_randomize, mask_randomize)
                    

                    # shuffle PCA components
                    indices = np.random.permutation(total_components)
                    eigvals_shuffled = eigvals[indices]
                    cumsum = np.cumsum(eigvals_shuffled)
                    num_to_retain = np.argmax(cumsum >= variance_ratio * total_variance) + 1

                    retained_indices = indices[:num_to_retain]
                    # For now, assume PCA basis rows are consistent with flattened patch layout
                    # (More robust row slicing logic can be added later)
                    #P_patch = P[:patch_flat.shape[0], :]
                    sparse_flat = torch.zeros_like(img_flat)
                    row_ids = self.get_patch_row_indices(i, j, patch_size, H)

                    """P_patch = P[row_ids, :] 
                    P_subset = P_patch[:, retained_indices]
                    
                    projected = patch_flat @ P_subset
                    reconstructed = projected @ P_subset.T

                    patch_recon = reconstructed.view(C, patch_size, patch_size)"""

                    sparse_flat[row_ids] = patch_flat
                    P_subset = P[:, retained_indices]
                    projected = sparse_flat @ P_subset
                    reconstructed = projected @ P_subset.T
                    
                    # Extract the reconstructed patch region from the reconstructed full image
                    patch_recon = reconstructed[row_ids].view(C, patch_size, patch_size)

                    patches.append(patch_recon)

            # stitch patches back
            recon_img = torch.zeros_like(img)
            count = 0
            for i in range(0, H, patch_size):
                for j in range(0, W, patch_size):
                    recon_img[:, i:i+patch_size, j:j+patch_size] = patches[count]
                    count += 1

            recon_img -= recon_img.min()
            denom = recon_img.max() - recon_img.min()
            if denom < 1e-4:
                denom = 1e-4
            recon_img /= denom

            #return recon_img.cpu()
            return recon_img

        view1 = apply_patchwise_masking()
        view2 = apply_patchwise_masking()
        return view1, view2
        

    
    def patchwise_cyclic_masking(self, img, eigenvalues, patch_size=8):
        """
        Apply cyclic PCA masking per patch based on explained variance.
        Each patch selects a different subset of components starting from a cyclically shifted
        variance threshold. If self.double is True, return two such views.
        """
        if not isinstance(img, torch.Tensor):
            img = self.to_tensor(img)
        img = img.to(self.device)

        P = self.masking_fn_
        C, H, W = img.shape
        eigvals = eigenvalues.cpu().numpy()
        sorted_indices = np.argsort(-eigvals)
        sorted_eigvals = eigvals[sorted_indices]
        cumsum = np.cumsum(sorted_eigvals)
        total_variance = cumsum[-1]

        img_flat = img.view(-1)

        def get_indices_for_patch(i, j, base_fraction, window_fraction):
            patch_index = (i // patch_size) * (W // patch_size) + (j // patch_size)
            offset = patch_index * base_fraction
            start_var = (offset % 1.0) * total_variance
            end_var = min(start_var + window_fraction * total_variance, total_variance)


            if end_var <= total_variance:
                mask = (cumsum >= start_var) & (cumsum <= end_var)
                selected = np.where(mask)[0]
            else:
                mask1 = (cumsum >= start_var)
                mask2 = (cumsum <= (end_var - total_variance))
                selected = np.concatenate([np.where(mask1)[0], np.where(mask2)[0]])
                
            if len(selected) == 0:
                selected = np.where(cumsum >= start_var)[0][:1]
            return sorted_indices[selected]

        def apply_patchwise_cyclic(base_fraction):
            patches = []
            for i in range(0, H, patch_size):
                for j in range(0, W, patch_size):
                    mask_randomize = 0.2
                    variance_ratio = self.pca_ratio + np.random.uniform(-mask_randomize, mask_randomize)
                    
                    
                    patch = img[:, i:i+patch_size, j:j+patch_size]
                    patch_flat = patch.contiguous().view(-1)
                    indices = get_indices_for_patch(i, j, base_fraction, variance_ratio)
                    row_ids = self.get_patch_row_indices(i, j, patch_size, H)

                    # Instead of slicing P_patch = P[row_ids, :] and P_subset = P_patch[:, indices],
                    # create a sparse full-image vector with only the patch region retaining values
                    sparse_flat = torch.zeros_like(img_flat)
                    sparse_flat[row_ids] = patch_flat
                    P_subset = P[:, indices]
                    projected = sparse_flat @ P_subset
                    reconstructed = projected @ P_subset.T

                    patch_recon = reconstructed[row_ids].view(C, patch_size, patch_size)

                    """P_patch = P[row_ids, :] 
                    P_subset = P_patch[:, indices]

                    projected = patch_flat @ P_subset
                    reconstructed = projected @ P_subset.T
                    patch_recon = reconstructed.view(C, patch_size, patch_size)"""
                    patches.append(patch_recon)

            recon_img = torch.zeros_like(img)
            count = 0
            for i in range(0, H, patch_size):
                for j in range(0, W, patch_size):
                    recon_img[:, i:i+patch_size, j:j+patch_size] = patches[count]
                    count += 1

            if self.normalize:
                recon_img -= recon_img.min()
                denom = recon_img.max() - recon_img.min()
                if denom < 1e-4:
                    denom = 1e-4
                recon_img /= denom
            #return recon_img.cpu()
            return recon_img

        view1 = apply_patchwise_cyclic(base_fraction=self.base_fractions[0])
        view2 = apply_patchwise_cyclic(base_fraction=self.base_fractions[1])
        return view1, view2
        



