from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cr_vae import VAE
from .taxonomy import TaxonomyFusion

class NeuralNetwork(nn.Module):
    LATENT_DIM=32
    def __init__(
        self,
        tnf_dim: int,
        coverage_dim: int,
        dropout: float = 0.05,
        output_dim: int = 128,
        proj_head_dim: int = 2048,
        proj: bool = False,
        hidden_dim: int = 1024,
        tax_emb_dim: int = 0,
        hierarchical_taxonomy: bool = False,
        # for hierarchical taxonomy.
        parental_array: torch.Tensor = None,
        tax_nodes: list = None,
        node_indices: dict = None,
    ):
        super(NeuralNetwork, self).__init__()
        self.coverage_dim = coverage_dim  # by default we ask users to provide coverage.
        self.output_dim = output_dim
        self.tnf_dim = tnf_dim
        self.proj_head_dim = proj_head_dim
        self.proj = proj
        self.tax_emb_dim = tax_emb_dim
        self.hidden_dim = hidden_dim
        self.hierarchical_taxonomy = hierarchical_taxonomy

        self.tnf_enc_dim = tnf_dim

        self.coverage_encoder = _build_coverage_encoder(
            coverage_dim=coverage_dim,
            output_dim=output_dim,
            dropout=dropout,
            hidden_dim=hidden_dim,
            n_hidden_layers=3,
        )

        self.tnf_enc_dim += output_dim
        self.final_cov_dim = output_dim

        # taxonomy string feature prediction, requires hierarchical loss.
        if tax_emb_dim > 0:
            if not hierarchical_taxonomy:
                self.tnf_enc_dim += tax_emb_dim
            else:
                # hierarchical taxonomy.
                tax_emb_dim = output_dim
                self.taxonomy_encoder = TaxonomyFusion(
                    in_dim=tax_emb_dim,
                    latent_dim=self.LATENT_DIM,
                    table_parent=parental_array,
                    tax_nodes=tax_nodes,
                    node_indices=node_indices,
                    dropout=dropout,
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

        self.representation_dim = self.output_dim

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
            nn.BatchNorm1d(proj_head_dim),
        )

    def _encode(
        self, 
        x: torch.Tensor,
        tax_emb: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor|None|Dict]]:
        tnf_features = x[:, : self.tnf_dim]
        cov_features = x[:, self.tnf_dim : self.tnf_dim + self.coverage_dim]
        cov_encode = self.coverage_encoder(cov_features)  # not normalized anymore.
        if (not tax_emb is None): 
            if (not self.hierarchical_taxonomy):
                tax_emb = F.normalize(tax_emb, dim=-1)
                tnf_features = torch.cat([tnf_features,tax_emb], dim=-1)
                fused_encode = self.tnf_encoder(
                    torch.cat([tnf_features, F.normalize(cov_encode, dim=-1)], dim=-1)
                )
                return fused_encode, cov_encode, tax_emb # here it is a tensor.
            else:
                fused_encode = self.tnf_encoder(
                    torch.cat([tnf_features, F.normalize(cov_encode, dim=-1)], dim=-1)
                )
                tax_encode = self.taxonomy_encoder(fused_encode)
                return fused_encode, cov_encode, tax_encode
        else:
            fused_encode = self.tnf_encoder(
                torch.cat([tnf_features, F.normalize(cov_encode, dim=-1)], dim=-1)
            )
            return fused_encode, cov_encode, None

    def get_embedding(
        self, 
        x: torch.Tensor, 
        tax_emb: torch.Tensor = None
    ):
        return self._encode(x, tax_emb)[0]
    
    def get_separate_modality_embs(self, x: torch.Tensor, tax_emb: torch.Tensor = None):
        fused_enc, cov_enc, tax_enc = self._encode(x, tax_emb)
        if tax_enc is None:
            return fused_enc, cov_enc, None
        return fused_enc, cov_enc, tax_enc

    def forward(
        self, 
        x: torch.Tensor, 
        tax_emb: torch.Tensor = None
    ):
        fused_enc, cov_enc, tax_enc = self._encode(x, tax_emb)
        if self.proj:
            proj_emb = self.proj_head(fused_enc)
            return proj_emb, cov_enc, tax_enc
        return fused_enc, cov_enc, tax_enc

def _build_coverage_encoder(
    coverage_dim: int,
    output_dim: int,
    dropout: float,
    hidden_dim: int = 512,
    n_hidden_layers: int = 3,
) -> nn.Sequential:
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

# def _build_tax_emb_encoder(
#     tax_emb_dim: int, 
#     hidden_dim: int = 512, 
#     output_dim: int = 128, 
#     dropout: float = 0.1
# ):
#     n_layers = 3
#     modules = []
#     input_dim = tax_emb_dim
#     for _ in range(n_layers):
#         modules.append(nn.Linear(input_dim, hidden_dim))
#         modules.append(nn.BatchNorm1d(hidden_dim))
#         modules.append(nn.LeakyReLU())
#         modules.append(nn.Dropout(dropout))
#         input_dim = hidden_dim
#     modules.append(nn.Linear(hidden_dim, output_dim))
#     modules.append(nn.BatchNorm1d(output_dim))
#     modules.append(nn.LeakyReLU())
#     return nn.Sequential(*modules)