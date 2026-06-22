"""
cvae_binning.py
================
A contrastive variational autoencoder (cVAE) for metagenomic binning, adapting
Abid & Zou (2019, arXiv:1902.04601) to the binferno setting and augmenting the
objective with an InfoNCE (SimCLR-style) term on the salient latent.

Conceptual mapping (paper -> binning)
-------------------------------------
  target  x  : real contigs, features = [TNF (k-mer composition) | coverage]
  background b: signal-poor contigs (e.g. GC-matched k-mer shuffles, or coverage
                profiles permuted across samples) -- genome-of-origin structure
                destroyed, marginal statistics preserved  ->  b ~ f_theta(0, z)
  salient  s : genome/bin-discriminative embedding you CLUSTER on
  irrelevant z: depth scale, GC bias, per-sample batch effects (nuisance)

Objective
---------
  L = -ELBO_target(x)            # recon(x | s, z) + KL(s) + KL(z)
      -ELBO_background(b)        # recon(b | 0, z) + KL(z)
      + gamma * TC(s, z)         # density-ratio total-correlation (decorrelate s _|_ z)
      + lambda * InfoNCE(s)      # pull augmented views of same contig together in s-space

The TC term follows the paper's density-ratio trick (Kim & Mnih 2018); InfoNCE
is the new contrastive signal on s, operating on the *salient* projection only,
so coverage/composition nuisance stays quarantined in z.

Notes
-----
- Zero-bias + ReLU constraint (paper Sec. 3) is supported via `zero_bias=True`
  so that feeding z=0 / s=0 truly removes a latent block's contribution.
- Coverage encoder uses LayerNorm (your established preference over BatchNorm1d)
  and column-wise max normalization with a small pseudocount is expected upstream.
- Single-file, framework-light. Swap the encoders for conv/attention as needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class CVAEConfig:
    tnf_dim: int = 103  # tetranucleotide-frequency feature width (canonical 4-mers)
    cov_dim: int = 1  # number of coverage samples (abundance across N samples)
    hidden: int = 512
    s_dim: int = 32  # salient latent (what you cluster on)
    z_dim: int = 32  # irrelevant latent (nuisance)
    proj_dim: int = 256  # InfoNCE projection head output (kept off the latent itself)
    proj_hidden: int = 512
    dropout: float = 0.1
    zero_bias: bool = False  # set True for clean s=0 / z=0 ablation (paper Sec. 3)
    # loss weights
    beta_s: float = 1.0  # KL weight on salient
    beta_z: float = 1.0  # KL weight on irrelevant
    gamma_tc: float = 1.0  # total-correlation weight
    lambda_nce: float = 1.0  # InfoNCE weight
    nce_temperature: float = 0.1  # NT-Xent temperature (your sweep range 0.07-0.2)


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def _linear(in_f: int, out_f: int, bias: bool) -> nn.Linear:
    return nn.Linear(in_f, out_f, bias=bias)


class MLPEncoder(nn.Module):
    """Shared-trunk encoder producing (mu, logvar) for a latent block.

    The paper uses two *shared* encoders q_{phi_s} and q_{phi_z}, each applied to
    both target and background. We keep them separate modules but identical in
    spirit: each maps the full feature vector -> its own latent block.
    """

    def __init__(
        self, in_dim: int, hidden: int, latent_dim: int, dropout: float, zero_bias: bool
    ):
        super().__init__()
        bias = not zero_bias
        self.net = nn.Sequential(
            _linear(in_dim, hidden, bias),
            nn.LayerNorm(hidden),  # LayerNorm over BatchNorm1d (binferno convention)
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            _linear(hidden, hidden, bias),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = _linear(hidden, latent_dim, bias)
        self.fc_logvar = _linear(hidden, latent_dim, bias)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """Single shared decoder f_theta([s, z]) -> reconstruction.

    For the background, s is replaced by zeros, so the decoder sees [0, z]
    (paper eq. 2 / Fig. 3a). Reconstructs concatenated [TNF | coverage].
    """

    def __init__(
        self,
        s_dim: int,
        z_dim: int,
        hidden: int,
        out_dim: int,
        dropout: float,
        zero_bias: bool,
    ):
        super().__init__()
        bias = not zero_bias
        self.net = nn.Sequential(
            _linear(s_dim + z_dim, hidden, bias),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            _linear(hidden, hidden, bias),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
        )
        self.out = _linear(hidden, out_dim, bias)

    def forward(self, s: Tensor, z: Tensor) -> Tensor:
        return self.out(self.net(torch.cat([s, z], dim=-1)))


class ProjectionHead(nn.Module):
    """SimCLR-style projection head for InfoNCE, applied to the salient mean.

    Kept *separate* from the latent so the contrastive geometry doesn't distort
    the clustered s-space directly (standard SimCLR practice). Final BN, no bias
    on the last layer -- mirrors your established projector recipe.
    """

    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, affine=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TCDiscriminator(nn.Module):
    """Logistic-regression-style discriminator for the density-ratio TC estimate
    (paper Appendix C). Distinguishes joint q(s,z) samples from the dimension-
    shuffled product q(s)q(z)."""

    def __init__(self, s_dim: int, z_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim + z_dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, 2),  # logits: [joint, product]
        )

    def forward(self, s: Tensor, z: Tensor) -> Tensor:
        return self.net(torch.cat([s, z], dim=-1))


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class ContrastiveVAE(nn.Module):
    def __init__(self, cfg: CVAEConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.tnf_dim + cfg.cov_dim

        # Two shared encoders: one for salient s, one for irrelevant z.
        self.enc_s = MLPEncoder(
            in_dim, cfg.hidden, cfg.s_dim, cfg.dropout, cfg.zero_bias
        )
        self.enc_z = MLPEncoder(
            in_dim, cfg.hidden, cfg.z_dim, cfg.dropout, cfg.zero_bias
        )

        # Single shared decoder.
        self.decoder = Decoder(
            cfg.s_dim, cfg.z_dim, cfg.hidden, in_dim, cfg.dropout, cfg.zero_bias
        )

        # InfoNCE projection head on the salient embedding.
        self.proj = ProjectionHead(cfg.s_dim, cfg.proj_hidden, cfg.proj_dim)

        # TC density-ratio discriminator.
        self.tc_disc = TCDiscriminator(cfg.s_dim, cfg.z_dim)

    # ---- core VAE ops ----------------------------------------------------- #
    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def encode(self, x: Tensor) -> dict[str, Tensor]:
        s_mu, s_logvar = self.enc_s(x)
        z_mu, z_logvar = self.enc_z(x)
        s = self.reparameterize(s_mu, s_logvar)
        z = self.reparameterize(z_mu, z_logvar)
        return dict(
            s=s, z=z, s_mu=s_mu, s_logvar=s_logvar, z_mu=z_mu, z_logvar=z_logvar
        )

    def encode_salient(self, x: Tensor) -> Tensor:
        """Inference path: the embedding you hand to the clusterer (uses s_mu)."""
        s_mu, _ = self.enc_s(x)
        return s_mu

    # ---- forward ---------------------------------------------------------- #
    def forward(self, x_target: Tensor, x_background: Optional[Tensor] = None) -> dict:
        out = {"target": self.encode(x_target)}
        t = out["target"]
        out["target"]["recon"] = self.decoder(t["s"], t["z"])

        if x_background is not None:
            b = self.encode(x_background)
            # Background: salient block forced to zero (paper eq. 2).
            zeros_s = torch.zeros_like(b["s"])
            b["recon"] = self.decoder(zeros_s, b["z"])
            out["background"] = b
        return out


# --------------------------------------------------------------------------- #
# Loss terms
# --------------------------------------------------------------------------- #
def kl_standard_normal(mu: Tensor, logvar: Tensor) -> Tensor:
    """KL(N(mu, sigma) || N(0, I)), summed over latent dim, mean over batch."""
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


def gaussian_nll(recon: Tensor, target: Tensor) -> Tensor:
    """Reconstruction as Gaussian NLL up to constant == MSE summed over features.
    Swap for a per-block loss (e.g. softmax-CE on TNF, MSE on coverage) if you
    model the two feature families with different likelihoods."""
    return torch.mean(torch.sum((recon - target) ** 2, dim=1))


def total_correlation_loss(disc: TCDiscriminator, s: Tensor, z: Tensor):
    """Density-ratio TC estimate (paper Appendix C, eq. 14).

    Returns (tc_penalty, disc_loss).
      tc_penalty : encourages independence of s and z in the VAE objective.
      disc_loss  : trains the discriminator to tell joint from shuffled-product.
    """
    # Joint samples [s, z]; product samples via independent row-permutation of z.
    logits_joint = disc(s, z)
    # E[log(D / (1-D))] over the "joint" class, estimated from logits.
    # With 2-way logits, log-ratio = logit_joint - logit_product.
    tc_penalty = torch.mean(logits_joint[:, 0] - logits_joint[:, 1])

    # Discriminator training targets: joint -> class 0, product -> class 1.
    perm = torch.randperm(z.size(0), device=z.device)
    z_perm = z[perm]
    logits_prod = disc(s.detach(), z_perm.detach())
    logits_joint_d = disc(s.detach(), z.detach())
    ones = torch.ones(s.size(0), dtype=torch.long, device=s.device)
    zeros = torch.zeros(s.size(0), dtype=torch.long, device=s.device)
    disc_loss = 0.5 * (
        F.cross_entropy(logits_joint_d, zeros) + F.cross_entropy(logits_prod, ones)
    )
    return tc_penalty, disc_loss


def info_nce_loss(z_i: Tensor, z_j: Tensor, temperature: float) -> Tensor:
    """NT-Xent / InfoNCE over two augmented views (SimCLR formulation).

    z_i, z_j : (B, D) L2-normalizable projections of the two views of B contigs.
    Positives are matched index pairs across the two views; everything else in
    the 2B batch is a negative.
    """
    B = z_i.size(0)
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    reps = torch.cat([z_i, z_j], dim=0)  # (2B, D)
    sim = reps @ reps.t() / temperature  # (2B, 2B)

    # Mask self-similarity.
    self_mask = torch.eye(2 * B, dtype=torch.bool, device=reps.device)
    sim.masked_fill_(self_mask, float("-inf"))

    # Positive index: i <-> i+B and i+B <-> i.
    targets = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(reps.device)
    return F.cross_entropy(sim, targets)


# --------------------------------------------------------------------------- #
# Combined objective
# --------------------------------------------------------------------------- #
def cvae_loss(
    model: ContrastiveVAE,
    x_target: Tensor,
    x_background: Tensor,
    x_target_view2: Optional[Tensor] = None,
    cfg: Optional[CVAEConfig] = None,
) -> dict[str, Tensor]:
    """Compute the full binferno-cVAE objective.

    x_target        : (B, F) features of the contig batch
    x_background     : (B, F) features of the signal-poor background batch
    x_target_view2   : (B, F) a second augmentation of the SAME contigs as
                       x_target (for InfoNCE). If None, InfoNCE is skipped.
    """
    cfg = cfg or model.cfg
    out = model(x_target, x_background)
    t, b = out["target"], out["background"]

    # --- ELBO (target): recon(x|s,z) + KL(s) + KL(z) ----------------------- #
    recon_t = gaussian_nll(t["recon"], x_target)
    kl_s = kl_standard_normal(t["s_mu"], t["s_logvar"])
    kl_z_t = kl_standard_normal(t["z_mu"], t["z_logvar"])

    # --- ELBO (background): recon(b|0,z) + KL(z) --------------------------- #
    recon_b = gaussian_nll(b["recon"], x_background)
    kl_z_b = kl_standard_normal(b["z_mu"], b["z_logvar"])

    elbo = recon_t + recon_b + cfg.beta_s * kl_s + cfg.beta_z * (kl_z_t + kl_z_b)

    # --- Total correlation: decorrelate s _|_ z on the target -------------- #
    tc_penalty, disc_loss = total_correlation_loss(model.tc_disc, t["s"], t["z"])

    # --- InfoNCE on salient projection ------------------------------------ #
    if x_target_view2 is not None and cfg.lambda_nce > 0:
        s1 = model.encode_salient(x_target)  # view 1 salient mean
        s2 = model.encode_salient(x_target_view2)  # view 2 salient mean
        p1, p2 = model.proj(s1), model.proj(s2)
        nce = info_nce_loss(p1, p2, cfg.nce_temperature)
    else:
        nce = torch.zeros((), device=x_target.device)

    total = elbo + cfg.gamma_tc * tc_penalty + cfg.lambda_nce * nce

    return dict(
        total=total,
        disc_loss=disc_loss,
        recon_target=recon_t.detach(),
        recon_background=recon_b.detach(),
        kl_s=kl_s.detach(),
        kl_z=(kl_z_t + kl_z_b).detach(),
        tc=tc_penalty.detach(),
        info_nce=nce.detach(),
    )


# --------------------------------------------------------------------------- #
# Augmentations (placeholder hooks -- wire to your real contig pipeline)
# --------------------------------------------------------------------------- #
def augment_contig(x: Tensor, cfg: CVAEConfig, noise_std: float = 0.05) -> Tensor:
    """Cheap, biology-agnostic augmentation for InfoNCE views. Replace with
    domain augmentations: fragment subsampling, coverage dropout across samples,
    Gaussian jitter on TNF, reverse-complement-equivalent k-mer perturbation.

    Here: feature jitter + random coverage dropout, leaving TNF block intact-ish.
    """
    x = x.clone()
    # Jitter everything slightly.
    x = x + noise_std * torch.randn_like(x)
    # Coverage dropout (last cov_dim columns).
    if cfg.cov_dim > 0:
        cov = x[:, cfg.tnf_dim :]
        drop = (torch.rand_like(cov) > 0.1).float()
        x[:, cfg.tnf_dim :] = cov * drop
    return x


def make_background(x_target: Tensor, cfg: CVAEConfig) -> Tensor:
    """Construct a signal-poor background by destroying genome-of-origin
    structure while preserving marginals:
      - TNF block: shuffle each column across the batch (breaks contig identity,
        keeps per-feature marginal distribution).
      - coverage block: shuffle each sample-column independently.
    Replace with your real background (GC-matched shuffles / cross-sample perms).
    """
    b = x_target.clone()
    B = b.size(0)
    for c in range(b.size(1)):
        b[:, c] = b[torch.randperm(B, device=b.device), c]
    return b


# --------------------------------------------------------------------------- #
# Minimal training step
# --------------------------------------------------------------------------- #
def train_step(
    model: ContrastiveVAE,
    x_target: Tensor,
    opt_model: torch.optim.Optimizer,
    opt_disc: torch.optim.Optimizer,
    cfg: CVAEConfig,
) -> dict[str, float]:
    model.train()
    x_bg = make_background(x_target, cfg)
    x_v2 = augment_contig(x_target, cfg)
    x_v1 = augment_contig(x_target, cfg)  # augment the "anchor" view too

    losses = cvae_loss(model, x_v1, x_bg, x_target_view2=x_v2, cfg=cfg)

    # Update VAE + encoders + projection (everything except discriminator).
    opt_model.zero_grad(set_to_none=True)
    losses["total"].backward(retain_graph=True)
    opt_model.step()

    # Update TC discriminator separately.
    opt_disc.zero_grad(set_to_none=True)
    losses["disc_loss"].backward()
    opt_disc.step()

    return {k: float(v.detach()) for k, v in losses.items()}


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = CVAEConfig(tnf_dim=103, cov_dim=8, s_dim=32, z_dim=32)
    model = ContrastiveVAE(cfg)

    # Two parameter groups: discriminator trained on its own loss.
    disc_params = list(model.tc_disc.parameters())
    disc_ids = {id(p) for p in disc_params}
    model_params = [p for p in model.parameters() if id(p) not in disc_ids]

    opt_model = torch.optim.Adam(model_params, lr=1e-3, betas=(0.9, 0.999))
    opt_disc = torch.optim.Adam(disc_params, lr=1e-3, betas=(0.5, 0.9))

    B, F_dim = 256, cfg.tnf_dim + cfg.cov_dim
    for step in range(5):
        x = torch.randn(B, F_dim)
        logs = train_step(model, x, opt_model, opt_disc, cfg)
        print(f"step {step}: " + "  ".join(f"{k}={v:.4f}" for k, v in logs.items()))

    # Inference: salient embedding for clustering.
    model.eval()
    with torch.no_grad():
        emb = model.encode_salient(torch.randn(B, F_dim))
    print("salient embedding for clustering:", tuple(emb.shape))
