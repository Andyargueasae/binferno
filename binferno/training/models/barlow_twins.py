import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NeuralNetwork(nn.Module):
    def __init__(
        self,
        tnf_dim,
        coverage_dim,
        dropout=0.05,
        output_dim=128,
        corgi_dim=0,
        n_layers=2,
        kmer_dim=136,
        proj_head_dim=512,
    ):
        super(NeuralNetwork, self).__init__()
        self.coverage_dim = coverage_dim  # by default we ask users to provide coverage.
        self.output_dim = output_dim
        self.kmer_dim = kmer_dim
        self.proj_head_dim = proj_head_dim
        self.tnf_dim = tnf_dim
        self.tnf_enc_dim = kmer_dim
        self.corgi_dim = corgi_dim

        if corgi_dim > 0:  # concat kmer with corgi and run single encoder.
            self.tnf_enc_dim += corgi_dim

        # fix the number of layers to 3.
        tnf_hidden1 = 512
        tnf_hidden2 = 256
        tnf_hidden3 = output_dim
        self.tnf_encoder = nn.Sequential(
            nn.Linear(self.tnf_enc_dim, tnf_hidden1),
            nn.BatchNorm1d(tnf_hidden1),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(tnf_hidden1, tnf_hidden2),
            nn.BatchNorm1d(tnf_hidden2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(tnf_hidden2, tnf_hidden3),
            nn.BatchNorm1d(tnf_hidden3),
            nn.LeakyReLU(),
        )
        # else:  # tnf->256->128
        #     self.tnf_encoder = nn.Sequential(
        #         nn.Linear(self.tnf_enc_dim, 256),
        #         nn.BatchNorm1d(256),
        #         nn.LeakyReLU(),
        #         nn.Dropout(dropout),
        #         nn.Linear(256, output_dim),
        #         nn.BatchNorm1d(output_dim),
        #         nn.LeakyReLU(),  # the forgoten leakyrelu after batchnorm can cause the representation to have a lot of zeros, which can lead to worse performance. Adding it back can improve the performance.
        #     )

        # corgi treated as an individual modality and encode separately.

        # apply adaptive coverage dim choices.
        # n_samples = max(
        #     1, coverage_dim // 2
        # )  # one coverage mean and one coverage std, divide by 2.

        # if n_samples <= 2:
        #     cov_hidden1 = 32
        #     cov_hidden2 = 16

        # elif n_samples <= 5:
        #     cov_hidden1 = 64
        #     cov_hidden2 = 32

        # elif n_samples <= 10:
        #     cov_hidden1 = 128
        #     cov_hidden2 = 64
        # else:
        self.coverage_encoder = _build_coverage_encoder(
            coverage_dim=coverage_dim, output_dim=output_dim, dropout=dropout
        )

        self.final_cov_dim = output_dim

        self.fusion_network = fusion_network(
            emb_dim=output_dim,
            kmer_enc_dim=output_dim,
            coverage_dim=self.final_cov_dim,
        )
        self.representation_dim = output_dim
        self.proj_head = nn.Sequential(
            nn.Linear(self.representation_dim, self.proj_head_dim),
            nn.BatchNorm1d(self.proj_head_dim),
            nn.LeakyReLU(),
            nn.Linear(
                self.proj_head_dim, self.proj_head_dim
            ),  # here we projects to a higher dimension to make the Barlow Twins loss work better, since the original paper shows that using a higher dimension for the projection head can improve the performance. We can also add a non-linear activation function here to make the projection head more powerful.
            nn.BatchNorm1d(self.proj_head_dim),
        )

    def _encode(self, x):
        # encode tnf and coverage features separately, then fuse them together.
        tnf_features = x[:, : self.tnf_enc_dim]
        tnf_encode = F.normalize(
            self.tnf_encoder(tnf_features), dim=-1
        )  # 128, not normalized.
        cov_features = x[:, self.tnf_enc_dim :]
        cov_encode = F.normalize(
            self.coverage_encoder(cov_features), dim=-1
        )  # 128, not normalized.
        fused_encode = self.fusion_network(tnf_encode, cov_encode)  # 128
        return fused_encode

    def get_embedding(self, x):
        return self._encode(x)

    def get_separate_modality_embs(self, x):
        kmer_features = x[:, : self.tnf_enc_dim]
        kmer_encoded = self.tnf_encoder(kmer_features)
        coverage_features = x[:, self.tnf_enc_dim :]
        coverage_encoded = self.coverage_encoder(coverage_features)
        return kmer_encoded, coverage_encoded

    def forward_one(self, x):
        fused_emb = self._encode(x)
        proj_emb = self.proj_head(fused_emb)  # 512
        return proj_emb

    def forward(self, x1, x2):
        z1 = self.forward_one(x1)
        z2 = self.forward_one(x2)
        return z1, z2


# Then our model got an attention-based fusion network.
class attentionFusion(nn.Module):
    DEFAULT_NUM_HEADS = 4
    DEFAULT_DROPOUT = 0.1
    GATE_INPUT_MULTIPLIER = 2  # For concatenated features in gates
    MULTI_SCALE_CONCAT_MULTIPLIER = 6  # 3 scales * 2 (kmer+coverage (+corgi)) features each, so could be 9 if we include corgi as well. But for now we will keep it at 6 for kmer and coverage only.
    COMPRESSOR_HIDDEN_MULTIPLIER = 4
    COMPRESSOR_OUTPUT_MULTIPLIER = 2
    INTERACTION_HIDDEN_MULTIPLIER = 3
    INTERACTION_INTERMEDIATE_MULTIPLIER = 2
    FINAL_FUSION_INPUT_MULTIPLIER = 4  # gated_kmer + gated_coverage + interaction + alignment / / if we include corgi, then it will be gated_kmer + gated_coverage + gated_corgi + interaction + alignment, so the multiplier will be 5.
    FINAL_FUSION_HIDDEN_MULTIPLIER = 2
    RESIDUAL_WEIGHT_INIT = 0.5
    ADAPTIVE_DROPOUT_MAX_MULTIPLIER = 2.0

    def __init__(
        self,
        kmer_enc_dim,
        emb_dim,
        coverage_dim,
    ):
        super(attentionFusion, self).__init__()
        # we will implement the attention fusion later, for now we can simply concatenate the features.
        self.kmer_enc_dim = kmer_enc_dim
        self.emb_dim = emb_dim
        self.NUM_HEADS = self.DEFAULT_NUM_HEADS
        self.DROPOUT = self.DEFAULT_DROPOUT
        self.coverage_dim = coverage_dim
        self.RESIDUAL_WEIGHT_INIT = self.RESIDUAL_WEIGHT_INIT
        self.MULTI_SCALE_CONCAT_MULTIPLIER = (
            self.MULTI_SCALE_CONCAT_MULTIPLIER
        )  # add 3 more for corgi features at different scales.

        self.GATE_INPUT_MULTIPLIER = 2
        self.FUSE_INPUT_MULTIPLIER = self.FINAL_FUSION_INPUT_MULTIPLIER
        self.HIDDEN_MULTIPLIER = 2

        self.kmer_proj = nn.Linear(self.kmer_enc_dim, self.emb_dim)
        self.coverage_proj = nn.Linear(self.coverage_dim, self.emb_dim)
        self.fusion_dropout = nn.Dropout(self.DROPOUT)
        self.batch_norm1 = nn.BatchNorm1d(self.emb_dim)
        self.batch_norm2 = nn.BatchNorm1d(self.emb_dim)

        self.residual_weight = nn.Parameter(torch.tensor(self.RESIDUAL_WEIGHT_INIT))

        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.emb_dim,
            num_heads=self.NUM_HEADS,
            dropout=self.DROPOUT,
            batch_first=True,
        )
        # corgi was once treated as a separate modality and entered attention, but gave poor results.

        self.kmer_gate = nn.Sequential(
            nn.Linear(self.emb_dim * self.GATE_INPUT_MULTIPLIER, self.emb_dim),
            nn.Sigmoid(),
        )
        self.coverage_gate = nn.Sequential(
            nn.Linear(self.emb_dim * self.GATE_INPUT_MULTIPLIER, self.emb_dim),
            nn.Sigmoid(),
        )

        self.fused_proj = nn.Sequential(
            nn.Linear(
                self.emb_dim * self.FUSE_INPUT_MULTIPLIER,
                self.emb_dim * self.HIDDEN_MULTIPLIER,
            ),
            nn.BatchNorm1d(self.emb_dim * self.HIDDEN_MULTIPLIER),
            nn.LeakyReLU(),
            nn.Dropout(self.DROPOUT),
            nn.Linear(self.emb_dim * self.HIDDEN_MULTIPLIER, self.emb_dim),
            nn.BatchNorm1d(self.emb_dim),
            nn.LeakyReLU(),
        )

        # apply multi-scale transformation.
        self.multiscale_fusion = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(emb_dim, emb_dim // 2),
                    nn.ReLU(),
                    nn.Linear(emb_dim // 2, emb_dim),  # Project back to common dim
                ),
                nn.Linear(emb_dim, emb_dim),  # Identity scale
                nn.Sequential(
                    nn.Linear(emb_dim, emb_dim * 2),
                    nn.ReLU(),
                    nn.Linear(emb_dim * 2, emb_dim),  # Project back to common dim
                ),
            ]
        )
        self.multi_scale_compressor = nn.Sequential(
            nn.Linear(
                emb_dim * self.MULTI_SCALE_CONCAT_MULTIPLIER,
                emb_dim * self.COMPRESSOR_HIDDEN_MULTIPLIER,
            ),
            nn.BatchNorm1d(emb_dim * self.COMPRESSOR_HIDDEN_MULTIPLIER),
            nn.ReLU(),
            nn.Dropout(self.DROPOUT),
            nn.Linear(
                emb_dim * self.COMPRESSOR_HIDDEN_MULTIPLIER,
                emb_dim * self.COMPRESSOR_OUTPUT_MULTIPLIER,
            ),
        )

        self.interaction_module = nn.Sequential(
            nn.Linear(
                emb_dim * self.COMPRESSOR_OUTPUT_MULTIPLIER,
                emb_dim * self.INTERACTION_HIDDEN_MULTIPLIER,
            ),
            nn.ReLU(),
            nn.Dropout(self.DROPOUT),
            nn.Linear(
                emb_dim * self.INTERACTION_HIDDEN_MULTIPLIER,
                emb_dim * self.INTERACTION_INTERMEDIATE_MULTIPLIER,
            ),
            nn.ReLU(),
            nn.Linear(emb_dim * self.INTERACTION_INTERMEDIATE_MULTIPLIER, emb_dim),
        )
        self.token_type_emb = nn.Embedding(2, emb_dim)  # 0=kmer, 1=coverage

    def forward(self, tnf_feature, coverage_feature):
        kmer_emb = self.kmer_proj(tnf_feature).unsqueeze(1)  # ( batch_size, 1, emb_dim)
        cov_emb = self.coverage_proj(coverage_feature).unsqueeze(
            1
        )  # ( batch_size, 1, emb_dim)

        type_ids = torch.tensor([0, 1], device=kmer_emb.device)
        token_type = self.token_type_emb(type_ids).unsqueeze(0)  # (1, 2, emb_dim)
        tokens = torch.cat([kmer_emb, cov_emb], dim=1) + token_type
        attended, _ = self.self_attn(tokens, tokens, tokens)

        kmer_attended = attended[:, 0, :]  # ( batch_size, emb_dim)
        cov_attended = attended[:, 1, :]  # ( batch_size, emb_dim)

        attended_feature = kmer_attended + cov_attended  # ( batch_size, emb_dim)

        kmer_emb = kmer_emb.squeeze(1)
        cov_emb = cov_emb.squeeze(1)
        kmer_gate_weight = self.kmer_gate(torch.cat([kmer_emb, cov_emb], dim=1))
        coverage_gate_weight = self.coverage_gate(torch.cat([kmer_emb, cov_emb], dim=1))

        kmer_attended = kmer_attended.squeeze(1)
        cov_attended = cov_attended.squeeze(1)
        gated_kmer = (
            kmer_gate_weight * kmer_emb + (1 - kmer_gate_weight) * kmer_attended
        )
        gated_coverage = (
            coverage_gate_weight * cov_emb + (1 - coverage_gate_weight) * cov_attended
        )

        residual_weight_clamped = torch.clamp(self.residual_weight, min=0.1, max=0.9)

        gated_kmer = self.fusion_dropout(
            self.batch_norm1(
                (1 - residual_weight_clamped) * gated_kmer
                + residual_weight_clamped * kmer_emb
            )
        )
        gated_coverage = self.fusion_dropout(
            self.batch_norm2(
                (1 - residual_weight_clamped) * gated_coverage
                + residual_weight_clamped * cov_emb
            )
        )

        scale_features = []
        for scale_layer in self.multiscale_fusion:
            scale_kmer = scale_layer(gated_kmer)
            scale_coverage = scale_layer(gated_coverage)
            scale_features.append(torch.cat([scale_kmer, scale_coverage], dim=1))

        multi_scale_concat = torch.cat(scale_features, dim=1)
        multi_scale_compressed = self.multi_scale_compressor(multi_scale_concat)
        interaction_features = self.interaction_module(multi_scale_compressed)
        final_fused_features = torch.cat(
            [
                gated_kmer,
                gated_coverage,
                interaction_features,
                attended_feature,  # Cross-modal alignment signal
            ],
            dim=1,
        )

        output = self.fused_proj(final_fused_features)
        return output


def fusion_network(emb_dim=128, kmer_enc_dim=128, coverage_dim=128):
    return attentionFusion(
        kmer_enc_dim=kmer_enc_dim,
        emb_dim=emb_dim,
        coverage_dim=coverage_dim,
    )


def _build_coverage_encoder(coverage_dim: int, output_dim: int, dropout: float):
    """
    Adaptive coverage encoder that scales capacity with input information content.

    coverage_dim is always 2 * n_samples (mean + std per sample).
    n_samples drives capacity, not coverage_dim directly.
    """
    n_samples = max(1, coverage_dim // 2)

    if n_samples == 1:
        # Single-sample: 2D input, low complexity.
        # Small network with LayerNorm (BatchNorm is unstable for near-2D manifolds).
        return nn.Sequential(
            nn.Linear(coverage_dim, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
            nn.LayerNorm(output_dim),
            nn.LeakyReLU(),
        )

    elif n_samples <= 5:
        # Few samples: coverage starts carrying differential signal.
        # Transition to BatchNorm — enough input dimensions to be stable.
        hidden = 128
        return nn.Sequential(
            nn.Linear(coverage_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(),
        )

    elif n_samples <= 20:
        # Medium samples: genuine multi-dimensional coverage signal.
        hidden = 256
        return nn.Sequential(
            nn.Linear(coverage_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(),
        )

    else:
        # Many samples: full COMEBin-style wide encoder justified.
        hidden = 512
        return nn.Sequential(
            nn.Linear(coverage_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(),
        )
