import torch.nn as nn
import torch.nn.functional as F
import torch

class info_nce_loss(nn.Module):
    def __init__(self, temperature: float = 0.15):
        super().__init__()
        self.temperature = temperature
        self._cache = {}

    def _masks(self, batch_size: int, n_views: int, device):
        key = (batch_size, n_views, str(device))
        if key not in self._cache:
            total = batch_size * n_views
            sample_ids = torch.arange(batch_size, device=device).repeat(n_views)
            same_contig = sample_ids.unsqueeze(0).eq(sample_ids.unsqueeze(1))
            diag = torch.eye(total, dtype=torch.bool, device=device)
            pos_idx = (same_contig & ~diag).nonzero()[:, 1].view(total, n_views - 1)
            self._cache[key] = (pos_idx, same_contig)
        return self._cache[key]

    def _neg_logpartition(self, neg, n_views):
        """log sum_k exp(s_k) over the negatives of each row -> [total, 1]."""
        return torch.logsumexp(neg, dim=1, keepdim=True)

    def forward(self, features, device=None, batch_size: int = 1024, n_views: int = 6):
        device = features.device if device is None else device
        pos_idx, same_contig = self._masks(batch_size, n_views, device)

        # Always score in fp32 even if the caller is inside autocast.
        features = F.normalize(features.float(), dim=1)
        sim = (features @ features.T) / self.temperature

        # [total * (n_views - 1), 1]
        positives = torch.gather(sim, 1, pos_idx).reshape(-1, 1)

        neg = sim.masked_fill(same_contig, float("-inf"))
        neg_lse = self._neg_logpartition(neg, n_views)              # [total, 1]
        neg_lse = neg_lse.repeat_interleave(n_views - 1, dim=0)     # [total*(V-1), 1]

        # -log( exp(pos) / (exp(pos) + sum_k exp(neg_k)) )
        loss = (torch.logaddexp(positives, neg_lse) - positives).mean()

        with torch.no_grad():
            max_neg = neg.max(dim=1, keepdim=True).values
            max_neg = max_neg.repeat_interleave(n_views - 1, dim=0)
            top1 = (positives > max_neg).float().mean().item() * 100.0

        return loss, top1

class InfoNCEHardNegative(info_nce_loss):
    """Actual hard-negative reweighting (Robinson et al., ICLR 2021).

    Your previous `info_nce_loss_hard_weight` was a byte-for-byte copy of
    `info_nce_loss`, so `hard_weighted_nce=True` was a no-op. Here negatives
    are reweighted by w_k ∝ exp(beta * s_k), normalised to sum to N_neg:

        log Z_hard = logsumexp((1+beta) * s) - logsumexp(beta * s) + log(N_neg)

    beta=0 recovers the plain loss. beta in [0.5, 1.0] is a sensible range;
    be careful in metagenomics, where the hardest negatives are frequently
    contigs from the *same* genome (i.e. false negatives).
    """

    def __init__(self, temperature: float = 0.15, beta: float = 0.5):
        super().__init__(temperature=temperature)
        self.beta = beta

    def _neg_logpartition(self, neg, n_views):
        if self.beta == 0.0:
            return torch.logsumexp(neg, dim=1, keepdim=True)
        n_neg = neg.shape[1] - n_views  # -inf entries are the same-contig block
        num = torch.logsumexp((1.0 + self.beta) * neg, dim=1, keepdim=True)
        den = torch.logsumexp(self.beta * neg, dim=1, keepdim=True)
        return num - den + torch.log(
            torch.tensor(float(n_neg), device=neg.device, dtype=neg.dtype)
        )

