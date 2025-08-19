import numpy as np
import torch
import random
import torchvision.transforms as transforms
import torch.nn.functional as F

class PCAAugmentor:
    def __init__(self, masking_fn_, pca_ratio, shuffle, base_fractions,
                 global_min=None, global_max=None, device="cpu",
                 img_size=32, patch_size=None, patch_specific=False, mean_grid=None, std_grid=None,
                 normalize=True, drop_ratio=0, drop_strategy="random",
                 double=False, interpolate=False, pad_strategy="pad",
                 mean=None, std=None):
                 
        self.pca_ratio = pca_ratio
        self.device = device
        #self.device = 'cpu'
        self.img_size = img_size
        
        self.use_patch = patch_size is not None
        self.patch_specific = patch_specific
        self.patch_size = patch_size

        if self.patch_specific:
            self.pca_matrix_grid = masking_fn_
            self.mean_grid = mean_grid
            self.std_grid = std_grid
        else:
            self.masking_fn_ = masking_fn_.to(self.device)

        self.num_pixels = 3 * patch_size * patch_size if self.use_patch else img_size * img_size * 3
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
        
        self.mean = torch.tensor(mean, dtype=torch.float32, device=self.device) if mean is not None else None
        self.std = torch.tensor(std, dtype=torch.float32, device=self.device) if std is not None else None

        self.to_tensor = transforms.ToTensor()

    def compute_pc_mask(self, eigenvalues, drop_ratio=None):
        m = self.pca_ratio
        d = self.drop_ratio if drop_ratio is None else drop_ratio


        eigenvalues = eigenvalues.to(self.device)
        total_variance = eigenvalues.sum()
        double_shuffle = self.double

        def sample_view_mask(strategy, drop_ratio, retain_ratio):
            sorted_indices = torch.argsort(eigenvalues, descending=True)
            sorted_eigvals = eigenvalues[sorted_indices]
            cumsum = torch.cumsum(sorted_eigvals, dim=0)

            rel_cumsum = cumsum / total_variance

            if strategy == "low":
                threshold_mask = cumsum < (1 - drop_ratio) * total_variance

                selected = sorted_indices[: torch.nonzero(threshold_mask, as_tuple=False)[-1].item() + 1] if torch.any(threshold_mask) else sorted_indices

                

            elif strategy == "middle":
                drop_lower = (0.5 - drop_ratio / 2) * total_variance
                drop_upper = (0.5 + drop_ratio / 2) * total_variance

                start_idx = torch.searchsorted(cumsum, torch.tensor(drop_lower, device=self.device))
                end_idx = torch.searchsorted(cumsum, torch.tensor(drop_upper, device=self.device))
                
                selected = torch.cat([sorted_indices[:start_idx], sorted_indices[end_idx:]]) if end_idx > start_idx else sorted_indices

            else:  # "random"
                if self.shuffle:
                    index = torch.randperm(len(eigenvalues), device=self.device)
                    eigvals_shuffled = eigenvalues[index]
                else:
                    index = torch.arange(len(eigenvalues), device=self.device)
                    eigvals_shuffled = eigenvalues
                cumsum = torch.cumsum(eigvals_shuffled, dim=0)

                drop_cutoff = torch.argmin(torch.abs(cumsum - d))
                
                cumsum_after_drop = torch.cumsum(eigvals_shuffled[drop_cutoff:], dim=0)
                retain_thresh = torch.argmin(torch.abs(cumsum_after_drop - m * (1 - d)))

                selected = index[drop_cutoff:drop_cutoff + retain_thresh]
            
            return selected.clone().detach().long().to(self.device)


        # Apply slight randomization to drop and retain ratios for both input and target
        drop_randomize = 0 if d > 0 else 0
        mask_randomize = 0

        def randomized(r, eps):
            return r + (torch.rand(1, device=self.device).item() * 2 * eps - eps)

        drop_input = randomized(d, drop_randomize)
        retain_input = randomized(m, mask_randomize)
        drop_target = randomized(d, drop_randomize)
        retain_target = randomized(m, mask_randomize)

        # Weighted randomization of drop_strategy per call if requested
        if self.drop_strategy == "arbitrary":
            strategy_pool = ["random", "low", "middle"]
            strategy_weights = [0.7, 0.15, 0.15]

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
            index_based_approach = False
            
            n = eigenvalues.numel()
            order = torch.randperm(n, device=self.device) if self.shuffle else torch.arange(n, device=self.device)
            eigs_shuffled = eigenvalues[order]
            cumsum = torch.cumsum(eigs_shuffled, dim=0)
            total = cumsum[-1]

            
            d_clamped = float(np.clip(d, 0.0, 0.5))
            m_clamped = float(np.clip(m, 0.0, 1.0))
            drop_mass = torch.as_tensor(d_clamped, device=self.device) * total
            retain_mass = torch.as_tensor(m_clamped, device=self.device) * (total - drop_mass)

            # Find the smallest contiguous block [start:end) in the shuffled order that hits the variance target
            start_idx = torch.searchsorted(cumsum, drop_mass)
            target_mass = drop_mass + retain_mass
            end_idx = torch.searchsorted(cumsum, target_mass)

            # Guard rails
            start_idx = torch.clamp(start_idx, 0, n)
            end_idx = torch.clamp(end_idx, start_idx, n)

            pc_mask_input = order[start_idx:end_idx].long()

            # Complement goes to target
            if start_idx == 0:
                pc_mask_target = order[end_idx:].long()
            elif end_idx == n:
                pc_mask_target = order[:start_idx].long()
            else:
                pc_mask_target = torch.cat([order[:start_idx], order[end_idx:]], dim=0).long()

            if index_based_approach:
                base_mask = sample_view_mask(selected_strategy_input, d, 1.0)
                if self.shuffle:
                    base_mask = base_mask[torch.randperm(base_mask.shape[0])]
                split_idx = int(m * base_mask.shape[0])
                
                pc_mask_input, pc_mask_target = base_mask[:split_idx], base_mask[split_idx:]
            

        
        return pc_mask_input, pc_mask_target

    def extract_views(self, img, eigenvalues, index = None):
        def norm(x):
            x = x - x.min()
            return x / (x.max() - x.min() + 1e-6)
        
        def pad_matrix(P_full, keep_indices, strategy="pad", target_dim=None, std=0.01):
            device = P_full.device
            target_dim = target_dim or P_full.shape[1]
            P_padded = torch.zeros_like(P_full)
            P_padded[:, keep_indices] = P_full[:, keep_indices]

            all_indices = torch.arange(target_dim, device=device)
            mask = torch.ones(target_dim, dtype=torch.bool, device=device)
            mask[keep_indices] = False
            drop_indices = all_indices[mask]

            if strategy == "mean":
                mean_values = P_full[:, drop_indices].mean(dim=0, keepdim=True)
                P_padded[:, drop_indices] = mean_values.expand(P_full.shape[0], -1)
                
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
                    mean_values = P_full[:, mean_indices].mean(dim=0, keepdim=True)
                    P_padded[:, mean_indices] = mean_values.expand(P_full.shape[0], -1)
                if len(gaussian_indices) > 0:
                    noise = torch.randn((P_full.shape[0], len(gaussian_indices)), device=device) * std
                    P_padded[:, gaussian_indices] = noise
            return P_padded

        if not isinstance(img, torch.Tensor):
            img = self.to_tensor(img)
        img = img.to(self.device)

        if self.patch_specific:
            C, H, W = img.shape
            H_p = W_p = H // self.patch_size

            patches = F.unfold(
                img.unsqueeze(0),
                kernel_size=self.patch_size,
                stride=self.patch_size)
            
            P = patches.squeeze(0).transpose(0,1)
            

            recon_in, recon_tg = [], []
            for idx in range(H_p * W_p):
                i, j = divmod(idx, W_p)
                Pmat = self.pca_matrix_grid[i][j].T.to(P.dtype)
                
                eigs = eigenvalues[i][j].to(P.dtype)
                
                mean_v = self.mean_grid[i][j].unsqueeze(0).to(P.dtype)
                std_v = self.std_grid[i][j].unsqueeze(0).to(P.dtype)
                pv = (P[idx:idx + 1] - mean_v) / std_v        

                pc_in, pc_tg = self.compute_pc_mask(eigs)
                Pin = Pmat[:, pc_in]
                Pt = Pmat[:, pc_tg]

                r_in = (pv @ Pin) @ Pin.T
                r_tg = (pv @ Pt)  @ Pt.T

                r_in = r_in * std_v + mean_v
                r_tg = r_tg * std_v + mean_v
                recon_in.append(r_in)
                recon_tg.append(r_tg)

            # fold back into images
            stack_in = torch.cat(recon_in, dim=0).transpose(0,1).unsqueeze(0)
            stack_tg = torch.cat(recon_tg, dim=0).transpose(0,1).unsqueeze(0)

                        
            """sum_patches = []
            for idx in range(H_p * W_p):
                i, j = divmod(idx, W_p)
                mean_v = self.mean_grid[i][j].unsqueeze(0).to(P.dtype)
                std_v  = self.std_grid[i][j].unsqueeze(0).to(P.dtype)
                sum_patch = (recon_in[idx] + recon_tg[idx]) * std_v + mean_v  # back to pixel space
                sum_patches.append(sum_patch)

                        
            stack_sum = torch.cat(sum_patches, dim=0).transpose(0,1).unsqueeze(0)
            
                
            img_sum = F.fold(
                stack_sum, output_size=(H, W),
                kernel_size=self.patch_size, stride=self.patch_size).squeeze(0)"""

            img1 = F.fold(# fold back into images (use the same sum image for both outputs)
                stack_in, output_size=(H,W),
                kernel_size=self.patch_size, stride=self.patch_size).squeeze(0)
            img2 = F.fold(
                stack_tg, output_size=(H,W),
                kernel_size=self.patch_size, stride=self.patch_size).squeeze(0)


            #return norm(img1), norm(img2)
            return img1, img2

        if not self.use_patch:
            # --- Global PCA path ---
            img_flat = img.view(1, -1)
            if self.mean is not None and self.std is not None:
                img_flat = (img_flat - self.mean) / (self.std + 1e-6)
            if self.precomputed_masks is not None and index is not None:
                pc_mask_input, pc_mask = self.precomputed_masks[index]
            else:
                pc_mask, pc_mask_input = self.compute_pc_mask(eigenvalues)


            if self.interpolate:
                strategy_input = random.choice(["pad", "mean", "gaussian", "hybrid"]) if self.pad_strategy == "random" else self.pad_strategy
                strategy_target = random.choice(["pad", "mean", "gaussian", "hybrid"]) if self.pad_strategy == "random" else self.pad_strategy
                p_input_full = pad_matrix(self.masking_fn_, pc_mask, strategy_input)
                p_target_full = pad_matrix(self.masking_fn_, pc_mask_input, strategy_target)
                img_reconstructed = img_flat @ p_input_full @ p_input_full.T
                target = img_flat @ p_target_full @ p_target_full.T
            else:
                # Compute which PCA components to mask
                p_input = self.masking_fn_[:, pc_mask_input]
                p_target = self.masking_fn_[:, pc_mask]

                """proj_sum = p_input @ p_input.T + p_target @ p_target.T
                identity = torch.eye(proj_sum.shape[0], device=proj_sum.device)

                # Check deviation from identity
                print("Max abs diff from I:", torch.max(torch.abs(proj_sum - identity)).item())
                print("Frobenius norm diff:", torch.norm(proj_sum - identity, p='fro').item())"""
                
                target = (img_flat @ p_target) @ p_target.T
                img_reconstructed = (img_flat @ p_input) @ p_input.T
            
            if self.normalize:
                img_reconstructed = norm(img_reconstructed)
                target = norm(target)

            """img_reconstructed = (img_reconstructed + target) * self.std + self.mean
            target = (img_reconstructed + target) * self.std + self.mean"""
                
            return img_reconstructed.view(img.shape), target.view(img.shape) # no switching to cpu

        """Position-agnostic path"""
        C, H, W = img.shape
        
        patches = F.unfold(
            img.unsqueeze(0),
            kernel_size=self.patch_size,
            stride=self.patch_size)
        P = patches.squeeze(0).transpose(0, 1)

        
        recon_in = []
        recon_tg = []
        for i in range(P.size(0)):
            pv = (P[i:i + 1] - self.mean) / (self.std + 1e-6)
            pc_in, pc_tg = self.compute_pc_mask(eigenvalues)
            
            Pin = self.masking_fn_[:, pc_in]
            Pt = self.masking_fn_[:, pc_tg]
            recon_in.append((pv @ Pin) @ Pin.T)
            recon_tg.append((pv @ Pt) @ Pt.T)

        
        stack_in = torch.cat(recon_in, dim=0).transpose(0, 1).unsqueeze(0)
        stack_tg = torch.cat(recon_tg, dim=0).transpose(0, 1).unsqueeze(0)
        img_in = F.fold(stack_in, output_size=(H, W), kernel_size=self.patch_size, stride=self.patch_size).squeeze(0)
        img_tg = F.fold(stack_tg, output_size=(H, W), kernel_size=self.patch_size, stride=self.patch_size).squeeze(0)

        return norm(img_in), norm(img_tg)
            


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

        mask_randomize = 0.1
        
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
                    sparse_flat = torch.zeros_like(img_flat)
                    row_ids = self.get_patch_row_indices(i, j, patch_size, H)

                    

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
                    mask_randomize = 0.1
                    variance_ratio = self.pca_ratio + np.random.uniform(-mask_randomize, mask_randomize)
                    
                    
                    patch = img[:, i:i+patch_size, j:j+patch_size]
                    patch_flat = patch.contiguous().view(-1)
                    indices = get_indices_for_patch(i, j, base_fraction, variance_ratio)
                    row_ids = self.get_patch_row_indices(i, j, patch_size, H)

                    
                    sparse_flat = torch.zeros_like(img_flat)
                    sparse_flat[row_ids] = patch_flat
                    P_subset = P[:, indices]
                    projected = sparse_flat @ P_subset
                    reconstructed = projected @ P_subset.T

                    patch_recon = reconstructed[row_ids].view(C, patch_size, patch_size)

                    
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
            return recon_img

        view1 = apply_patchwise_cyclic(base_fraction=self.base_fractions[0])
        view2 = apply_patchwise_cyclic(base_fraction=self.base_fractions[1])
        return view1, view2