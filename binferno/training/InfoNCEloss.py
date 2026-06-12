import torch.nn as nn
import torch.nn.functional as F
import torch


class info_nce_loss_hard_weight(nn.Module):
    def __init__(self, temperature=0.15):
        super().__init__()
        self.temperature = temperature
        self._cache = {}

    def forward(self, features, device, batch_size=1024, n_views=6):
        """
        Ultra-fast multi-view InfoNCE loss matching the 'comebin' style.
        Computes (n_views - 1) independent losses per anchor view against all negatives,
        leveraging fused mathematical operators and memory-bound optimizations.
        """
        total = batch_size * n_views
        cache_key = (batch_size, n_views, str(device))

        # --- Efficient Persistent Caching ---
        if cache_key not in self._cache:
            sample_ids = torch.arange(batch_size, device=device).repeat(n_views)

            # Construct masks for structured extraction
            pos_mask = sample_ids.unsqueeze(0).eq(sample_ids.unsqueeze(1))
            diag_mask = torch.eye(total, dtype=torch.bool, device=device)
            pos_mask_no_diag = pos_mask & ~diag_mask

            # Fast index matrix for gathering positive scores: [total, n_views - 1]
            pos_indices = pos_mask_no_diag.nonzero()[:, 1].view(total, n_views - 1)

            # Complete negative mask to isolate true negative elements
            neg_mask = pos_mask

            self._cache[cache_key] = (pos_indices, neg_mask, sample_ids)

        pos_indices, neg_mask, sample_ids = self._cache[cache_key]

        # 1. High-speed Matrix Multiplication
        features = F.normalize(features, dim=1)
        sim = torch.matmul(features, features.T) / self.temperature

        # 2. Extract Positives and Flatten: Shape [total * (n_views - 1), 1]
        # This gathers exactly the 5 positive pairs per row
        positives = torch.gather(sim, dim=1, index=pos_indices).view(-1, 1)

        # 3. Mask out true positives and self-diagonal targets to isolate negatives
        negatives_only = sim.masked_fill(neg_mask, float("-inf"))

        # Virtual repetition to pair each row's negatives against each individual positive view
        # Shape: [total * (n_views - 1), total]
        negatives_repeated = torch.repeat_interleave(negatives_only, n_views - 1, dim=0)

        # 4. Fused Mathematical Log-Sum-Exp Reduction (Avoids huge torch.cat calls)
        max_val = torch.clamp(positives, min=0.0)
        pos_exp = torch.exp(positives - max_val)
        neg_sum_exp = torch.sum(
            torch.exp(negatives_repeated - max_val), dim=1, keepdim=True
        )

        # Final objective calculation matching comebin's multi-view behavior
        loss = -(positives - max_val - torch.log(pos_exp + neg_sum_exp)).mean()

        # 5. Optimized Metric Computation (Top-1 and Top-5 Accuracy)
        with torch.no_grad():
            # Extract pure negative scores across the batch
            # Shape: [total, total] -> max negative per row: [total, 1]
            max_negatives = negatives_only.max(dim=1, keepdim=True).values
            max_negatives_repeated = torch.repeat_interleave(
                max_negatives, n_views - 1, dim=0
            )

            # Top-1: Check if the individual positive view score beats the best negative score
            top1_acc = (
                positives > max_negatives_repeated
            ).float().mean().item() * 100.0

            # Top-5: Sort or partition the negative space to find the 5th highest negative baseline
            # topk(5) returns sorted negative tensor of shape [total, 5]. The 5th item (index 4) is our threshold.
            top5_neg_threshold = torch.topk(negatives_only, k=5, dim=1).values[:, 4:5]
            top5_neg_repeated = torch.repeat_interleave(
                top5_neg_threshold, n_views - 1, dim=0
            )

            # Top-5: True if the positive score beats the 5th highest negative score
            top5_acc = (positives > top5_neg_repeated).float().mean().item() * 100.0

        return loss, top1_acc, top5_acc


class info_nce_loss(nn.Module):
    def __init__(self, temperature=0.15):
        super().__init__()
        self.temperature = temperature
        self._cache = {}

    def forward(self, features, device, batch_size=1024, n_views=6):
        """
        Ultra-fast multi-view InfoNCE loss matching the 'comebin' style.
        Computes (n_views - 1) independent losses per anchor view against all negatives,
        leveraging fused mathematical operators and memory-bound optimizations.
        """
        total = batch_size * n_views
        cache_key = (batch_size, n_views, str(device))

        # --- Efficient Persistent Caching ---
        if cache_key not in self._cache:
            sample_ids = torch.arange(batch_size, device=device).repeat(n_views)

            # Construct masks for structured extraction
            pos_mask = sample_ids.unsqueeze(0).eq(sample_ids.unsqueeze(1))
            diag_mask = torch.eye(total, dtype=torch.bool, device=device)
            pos_mask_no_diag = pos_mask & ~diag_mask

            # Fast index matrix for gathering positive scores: [total, n_views - 1]
            pos_indices = pos_mask_no_diag.nonzero()[:, 1].view(total, n_views - 1)

            # Complete negative mask to isolate true negative elements
            neg_mask = pos_mask

            self._cache[cache_key] = (pos_indices, neg_mask, sample_ids)

        pos_indices, neg_mask, sample_ids = self._cache[cache_key]

        # 1. High-speed Matrix Multiplication
        features = F.normalize(features, dim=1)
        sim = torch.matmul(features, features.T) / self.temperature

        # 2. Extract Positives and Flatten: Shape [total * (n_views - 1), 1]
        # This gathers exactly the 5 positive pairs per row
        positives = torch.gather(sim, dim=1, index=pos_indices).view(-1, 1)

        # 3. Mask out true positives and self-diagonal targets to isolate negatives
        negatives_only = sim.masked_fill(neg_mask, float("-inf"))

        # Virtual repetition to pair each row's negatives against each individual positive view
        # Shape: [total * (n_views - 1), total]
        negatives_repeated = torch.repeat_interleave(negatives_only, n_views - 1, dim=0)

        # 4. Fused Mathematical Log-Sum-Exp Reduction (Avoids huge torch.cat calls)
        max_val = torch.clamp(positives, min=0.0)
        pos_exp = torch.exp(positives - max_val)
        neg_sum_exp = torch.sum(
            torch.exp(negatives_repeated - max_val), dim=1, keepdim=True
        )

        # Final objective calculation matching comebin's multi-view behavior
        loss = -(positives - max_val - torch.log(pos_exp + neg_sum_exp)).mean()

        # 5. Optimized Metric Computation (Top-1 and Top-5 Accuracy)
        with torch.no_grad():
            # Extract pure negative scores across the batch
            # Shape: [total, total] -> max negative per row: [total, 1]
            max_negatives = negatives_only.max(dim=1, keepdim=True).values
            max_negatives_repeated = torch.repeat_interleave(
                max_negatives, n_views - 1, dim=0
            )

            # Top-1: Check if the individual positive view score beats the best negative score
            top1_acc = (
                positives > max_negatives_repeated
            ).float().mean().item() * 100.0

            # Top-5: Sort or partition the negative space to find the 5th highest negative baseline
            # topk(5) returns sorted negative tensor of shape [total, 5]. The 5th item (index 4) is our threshold.
            top5_neg_threshold = torch.topk(negatives_only, k=5, dim=1).values[:, 4:5]
            top5_neg_repeated = torch.repeat_interleave(
                top5_neg_threshold, n_views - 1, dim=0
            )

            # Top-5: True if the positive score beats the 5th highest negative score
            top5_acc = (positives > top5_neg_repeated).float().mean().item() * 100.0

        return loss, top1_acc, top5_acc
