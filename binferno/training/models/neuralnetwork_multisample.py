import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AdaptiveDropout(nn.Module):
    def __init__(self, base_rate=0.1, max_rate=0.3, adaptation_factor=0.1):
        super().__init__()

        self.base_rate = base_rate
        self.max_rate = max_rate
        self.adaptation_factor = adaptation_factor

    def forward(self, x):
        if not self.training:
            return x

        input_var = torch.var(x, dim=-1, keepdim=True)

        normalized_var = torch.sigmoid(input_var * self.adaptation_factor)

        adaptive_rate = (
            self.base_rate + (self.max_rate - self.base_rate) * normalized_var
        )

        # change to rand_like.
        mask = torch.rand_like(x) > adaptive_rate

        return x * mask.float() / (1 - adaptive_rate + 1e-8)


class MultiScaleKmerEncoder(nn.Module):
    def __init__(self, kmer_dims, embedding_dim, num_heads=4, dropout=0.1):
        super().__init__()

        self.encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, embedding_dim),
                    nn.BatchNorm1d(embedding_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                for dim in kmer_dims
            ]
        )

        self.scale_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, kmer_features):
        """
        kmer_features: list of tensors
            [
                k4: [B, k4_dim],
                k5: [B, k5_dim],
                k6: [B, k6_dim],
            ]
        """

        encoded = [encoder(x) for encoder, x in zip(self.encoders, kmer_features)]

        tokens = torch.stack(encoded, dim=1)  # [B, num_kmer_scales, D]

        attended, _ = self.scale_attention(tokens, tokens, tokens)

        pooled = attended.mean(dim=1)

        return self.norm(pooled)


class MultiScaleCoverageEncoder(nn.Module):
    def __init__(
        self,
        summary_dim,
        sample_dim,
        embedding_dim,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()

        self.summary_encoder = nn.Sequential(
            nn.Linear(summary_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.sample_encoder = nn.Sequential(
            nn.Linear(sample_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.window_encoder = nn.GRU(
            input_size=sample_dim,
            hidden_size=embedding_dim,
            batch_first=True,
            bidirectional=False,
        )

        self.scale_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, cov_summary, cov_sample, cov_windows):
        z_summary = self.summary_encoder(cov_summary)
        z_sample = self.sample_encoder(cov_sample)

        _, h = self.window_encoder(cov_windows)
        z_window = h[-1]  # [B, D]

        tokens = torch.stack(
            [z_summary, z_sample, z_window],
            dim=1,
        )  # [B, 3, D]

        attended, _ = self.scale_attention(tokens, tokens, tokens)

        pooled = attended.mean(dim=1)

        return self.norm(pooled)


class NeuralNetwork(nn.Module):
    def __init__(
        self,
        kmer_dims,
        embedding_dim,
        summary_dim,
        sample_dim,
        num_heads=4,
        dropout=0.1,
        proj_head_dim=512,
    ):
        super().__init__()
        self.representation_dim = embedding_dim
        self.proj_head_dim = proj_head_dim
        self.kmers_encoder = MultiScaleKmerEncoder(
            kmer_dims, embedding_dim, num_heads=num_heads, dropout=dropout
        )
        self.covs_encoder = MultiScaleCoverageEncoder(
            summary_dim, sample_dim, embedding_dim, num_heads=num_heads, dropout=dropout
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.fusion_layer = BiologicalMultiScaleFusion(
            embedding_dim, num_heads=num_heads, dropout=dropout
        )
        self.proj_head = nn.Sequential(
            nn.Linear(self.representation_dim, self.proj_head_dim),
            nn.BatchNorm1d(self.proj_head_dim),
            nn.ReLU(),
            nn.Linear(
                self.proj_head_dim, self.proj_head_dim
            ),  # here we projects to a higher dimension to make the Barlow Twins loss work better, since the original paper shows that using a higher dimension for the projection head can improve the performance. We can also add a non-linear activation function here to make the projection head more powerful.
        )

    def _encode(
        self, kmer_features, coverage_summary, coverage_sample, coverage_windows
    ):
        z_kmer = self.kmers_encoder(kmer_features)
        z_coverage = self.covs_encoder(
            coverage_summary, coverage_sample, coverage_windows
        )
        return self.fusion_layer(z_kmer, z_coverage)

    def forward_one(
        self, kmer_features, coverage_summary, coverage_sample, coverage_windows
    ):
        encoded_feature = self._encode(
            kmer_features, coverage_summary, coverage_sample, coverage_windows
        )
        projected = self.proj_head(encoded_feature)
        return projected

    def forward(self, x1, x2):
        """
        x1, x2: dicts with keys:
            kmer_features: list of tensors
                [
                    k4: [B, k4_dim],
                    k5: [B, k5_dim],
                    k6: [B, k6_dim],
                ]
            coverage_summary: [B, summary_dim]
            coverage_sample: [B, sample_dim]
            coverage_windows: [B, num_windows, sample_dim]
        """

        z1 = self.forward_one(
            x1["kmer_features"],
            x1["coverage_summary"],
            x1["coverage_sample"],
            x1["coverage_windows"],
        )

        z2 = self.forward_one(
            x2["kmer_features"],
            x2["coverage_summary"],
            x2["coverage_sample"],
            x2["coverage_windows"],
        )

        return z1, z2

    def get_emb(self, x):
        return self._encode(
            x["kmer_features"],
            x["coverage_summary"],
            x["coverage_sample"],
            x["coverage_windows"],
        )


class BiologicalMultiScaleFusion(nn.Module):
    def __init__(self, embedding_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.kmer_gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Sigmoid(),
        )

        self.coverage_gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Sigmoid(),
        )

        self.interaction_module = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim * 3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 3, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

        self.adaptive_dropout = AdaptiveDropout(
            base_rate=dropout,
            max_rate=dropout * 2,
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 4, embedding_dim * 2),
            nn.BatchNorm1d(embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
        )

    def forward(self, z_kmer, z_coverage):
        """
        kmer_features:
            list of tensors, e.g.
            [k4, k5, k6]

        cov_summary:
            [B, cov_summary_dim]

        cov_sample:
            [B, cov_sample_dim]

        cov_windows:
            [B, num_windows, cov_sample_dim]
        """

        modality_tokens = torch.stack(
            [z_kmer, z_coverage],
            dim=1,
        )  # [B, 2, D]

        attended, _ = self.cross_attention(
            modality_tokens,
            modality_tokens,
            modality_tokens,
        )

        kmer_attended = attended[:, 0, :]
        cov_attended = attended[:, 1, :]

        combined = torch.cat([z_kmer, z_coverage], dim=1)

        kmer_gate = self.kmer_gate(combined)
        cov_gate = self.coverage_gate(combined)

        gated_kmer = kmer_gate * z_kmer + (1 - kmer_gate) * kmer_attended

        gated_cov = cov_gate * z_coverage + (1 - cov_gate) * cov_attended

        interaction = self.interaction_module(torch.cat([gated_kmer, gated_cov], dim=1))

        interaction = self.adaptive_dropout(interaction)

        alignment = kmer_attended + cov_attended

        final_features = torch.cat(
            [
                gated_kmer,
                gated_cov,
                interaction,
                alignment,
            ],
            dim=1,
        )

        return self.fusion_mlp(final_features)
