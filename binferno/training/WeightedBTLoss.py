from __future__ import annotations

import itertools
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


class WeightedBTLoss(nn.Module):
    """Drop-in replacement for BarlowTwinsTrainer.BT_loss.

    Signature is backwards compatible: if `w` is None the behaviour is
    numerically identical to the unweighted loss, so you can flip the feature
    on and off without touching the training loop.

    The weighted cross-correlation is

        w_hat = w / sum(w)
        mu_k  = sum_i w_hat_i z_ik
        v_k   = sum_i w_hat_i (z_ik - mu_k)^2
        zn    = (z - mu) / sqrt(v + eps)
        C     = sum_i w_hat_i zn1_i zn2_i^T

    so diag(C) = 1 under perfect invariance, exactly as in the unweighted case.
    """

    def __init__(
        self,
        lambda_bt: float = 5e-3,
        alpha: float = 1.0,
        eps: float = 1e-6,
        min_n_eff_ratio: float = 0.25,
    ):
        super().__init__()
        self.lambda_bt = lambda_bt
        self.alpha = alpha
        self.eps = eps
        # If N_eff drops below min_n_eff_ratio * N the empirical correlation
        # matrix is being estimated from too few effective samples and the
        # off-diagonal term starts penalising sampling noise.  We do not raise
        # -- we surface it in stats so you can see it in the epoch log.
        self.min_n_eff_ratio = min_n_eff_ratio
        self._cache: Dict[Tuple[int, str], torch.Tensor] = {}

    @staticmethod
    def _weighted_standardise(
        z: torch.Tensor, w: torch.Tensor, eps: float
    ) -> torch.Tensor:
        mu = (w * z).sum(dim=0, keepdim=True)
        centred = z - mu
        var = (w * centred.pow(2)).sum(dim=0, keepdim=True)
        return centred / (var.sqrt() + eps)

    def forward(
        self,
        feature1: torch.Tensor,
        feature2: torch.Tensor,
        w: Optional[torch.Tensor] = None,
        return_stats: bool = False,
    ):
        N, D = feature1.size()
        device = feature1.device

        cache_key = (D, str(device))
        if cache_key not in self._cache:
            self._cache[cache_key] = torch.eye(D, device=device)
        eye = self._cache[cache_key]

        if w is None:
            w_hat = feature1.new_full((N, 1), 1.0 / N)
        else:
            w = w.to(device=device, dtype=feature1.dtype).reshape(N, 1)
            w = w.clamp_min(0.0)
            w_hat = w / (w.sum() + self.eps)

        z1 = self._weighted_standardise(feature1, w_hat, self.eps)
        z2 = self._weighted_standardise(feature2, w_hat, self.eps)

        C = (w_hat * z1).T @ z2

        diag = torch.diagonal(C)
        invariance_loss = (diag - 1.0).pow(2).sum()

        sq = C.pow(2)
        redundancy_loss = sq.sum() - torch.diagonal(sq).sum()

        loss = self.alpha * invariance_loss + self.lambda_bt * redundancy_loss

        if not return_stats:
            return loss

        with torch.no_grad():
            n_eff = 1.0 / (w_hat.pow(2).sum() + self.eps)
            off_diag_abs = C.abs().masked_fill(eye.to(torch.bool), 0.0)
            stats_tensor = torch.stack(
                [
                    diag.mean(),
                    diag.std(),
                    off_diag_abs.sum() / (D * D - D),
                    off_diag_abs.max(),
                    n_eff,
                ]
            )
            s = stats_tensor.cpu().tolist()
            stats = {
                "mean_diagonal": s[0],
                "std_diagonal": s[1],
                "mean_abs_off_diagonal": s[2],
                "max_abs_off_diagonal": s[3],
                "n_eff": s[4],
                "n_eff_ratio": s[4] / N,
                "n_eff_below_floor": bool(s[4] < self.min_n_eff_ratio * N),
                "invariance_loss": invariance_loss.item(),
                "redundancy_loss": redundancy_loss.item(),
            }
        return loss, stats
