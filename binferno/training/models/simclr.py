import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .cr_vae import VAE
from .taxonomy import TaxonomyEncoder


# The neuralnetwork orthologous to comebin's MLP, but we have to gain further modules for the third-modality encoding. 
class NeuralNetwork(nn.Module):
    def __init__(
        self,
        tnf_dim: int,
        coverage_dim: int,
        dropout: float = 0.05,
        output_dim: int = 128,
        kmer_dim: int = 136,
        tax_emb_dim: int = 0,
        proj_head_dim: int = 2048,
        proj: bool = False,
        hidden_dim: int = 1024,
    ):
        super(NeuralNetwork, self).__init__()
        self.coverage_dim = coverage_dim  # by default we ask users to provide coverage.
        self.output_dim = output_dim
        self.kmer_dim = kmer_dim
        self.proj_head_dim = proj_head_dim
        self.proj = proj
        self.tnf_dim = tnf_dim
        self.tnf_enc_dim = kmer_dim
        self.tax_emb_dim = tax_emb_dim # corgi or dnabert_s embedding dimension.
        self.hidden_dim = hidden_dim

        self.coverage_encoder = _build_coverage_encoder(
            coverage_dim=coverage_dim, output_dim=output_dim, dropout=dropout,
            hidden_dim = hidden_dim, n_hidden_layers=3
        )

        self.final_cov_dim = output_dim

        # taxonomy string feature prediction, requires hierarchical loss.
        if tax_emb_dim > 0:
            self.tnf_enc_dim += tax_emb_dim
            self.taxonomy_encoder = nn.Sequential(
                nn.Linear(tax_emb_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, tax_emb_dim),
                nn.BatchNorm1d(tax_emb_dim),
                nn.LeakyReLU(),
            )

        # embedding-based prediction, no hierarchical loss.
        elif self.tax_emb_dim > 0:
            self.tnf_enc_dim += self.output_dim 
            self.taxemb_encoder = nn.Sequential(
                nn.Linear(self.tax_emb_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, self.output_dim), 
                nn.BatchNorm1d(self.output_dim),
                nn.LeakyReLU(),
            )
        tnf_hidden = self.hidden_dim

        modules = []
        n_hidden_layers = 3
        input_dim = self.tnf_enc_dim
        for _ in range(n_hidden_layers):
            modules.append(nn.Linear(input_dim, tnf_hidden))
            modules.append(nn.BatchNorm1d(tnf_hidden))
            modules.append(nn.LeakyReLU())
            modules.append(nn.Dropout(0.1))
            input_dim = tnf_hidden
        modules.append(nn.Linear(tnf_hidden, output_dim))
        modules.append(nn.BatchNorm1d(output_dim))
        modules.append(nn.LeakyReLU())
        self.tnf_encoder = nn.Sequential(*modules)
            
        self.representation_dim = (
            self.output_dim
        )

        self.proj_head = nn.Sequential(
            nn.Linear(self.representation_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, proj_head_dim),
            nn.BatchNorm1d(proj_head_dim)
        )

    def _encode(self, x):
        # encode tnf and coverage features separately, then fuse them together.
        tnf_features = x[:, : self.kmer_dim]
        cov_features = x[:, self.kmer_dim : self.kmer_dim + self.coverage_dim]
        cov_encode = self.coverage_encoder(cov_features)  # not normalized anymore.
        fused_encode = self.tnf_encoder(torch.cat([tnf_features, F.normalize(cov_encode, dim=-1)], dim=-1))  # fuse tnf and coverage features together by concatenating them and feeding them into the tnf encoder. This is a bit hacky but it allows us to keep the fusion network simple. We can also try other ways to fuse them together later, but for now we just concatenate them and feed them into the fusion network.
        return fused_encode, cov_encode

    def get_embedding(self, x):
        return self._encode(x)[0]

    def forward(self, x):
        fused_emb, cov_emb = self._encode(x)
        if self.proj:
            proj_emb = self.proj_head(fused_emb)
            return proj_emb, cov_emb
        return fused_emb, cov_emb

def _build_coverage_encoder(
    coverage_dim: int, output_dim: int, dropout: float, hidden_dim: int = 512, n_hidden_layers: int = 3
):
    """
    Adaptive coverage encoder that scales capacity with input information content.

    coverage_dim is always 2 * n_samples (mean + std per sample).
    n_samples drives capacity, not coverage_dim directly.
    """
    encoder = []
    input_dim = coverage_dim
    for _ in range(n_hidden_layers):
        encoder.append(nn.Linear(input_dim, hidden_dim))
        encoder.append(nn.LayerNorm(hidden_dim))
        encoder.append(nn.LeakyReLU())
        encoder.append(nn.Dropout(dropout))
        input_dim = hidden_dim
    encoder.append(nn.Linear(hidden_dim, output_dim))
    encoder.append(nn.LayerNorm(output_dim))
    encoder.append(nn.LeakyReLU())
    return nn.Sequential(*encoder)