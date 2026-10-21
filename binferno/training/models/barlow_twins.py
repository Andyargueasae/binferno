import torch
import torch.nn as nn
from .taxonomy_gate import bounded_taxonomy_residual

class NeuralNetwork(nn.Module):
    def __init__(
        self,
        tnf_dim,
        coverage_dim,
        dropout=0.05,
        output_dim=128,
        corgi_dim=0,
        kmer_dim=136,
        tax_emb_dim=0,
        proj_head_dim=512,
        h_tax=False,
        tax_model=None,
        h_tax_model=None,
        alpha_tax: float = 0.05,
        protein_encoder=None,
    ):
        super(NeuralNetwork, self).__init__()
        self.coverage_dim = coverage_dim  # by default we ask users to provide coverage.
        self.protein_encoder = protein_encoder
        self.output_dim = output_dim
        self.kmer_dim = kmer_dim
        self.proj_head_dim = proj_head_dim
        self.tnf_dim = tnf_dim
        self.tnf_enc_dim = kmer_dim
        self.corgi_dim = corgi_dim
        self.tax_emb_dim = tax_emb_dim
        self.hierarchical_taxonomy = h_tax
        # Keep the development-time name as a compatibility alias.
        self.tax_model = tax_model if tax_model is not None else h_tax_model
        self.uses_rank_taxonomy = self.tax_model is not None
        self.taxonomy_vocab_sizes = (
            [layer.num_embeddings for layer in self.tax_model.embeddings]
            if self.uses_rank_taxonomy
            else []
        )
        self.taxonomy_classifiers = nn.ModuleList(
            [nn.Linear(output_dim, size) for size in self.taxonomy_vocab_sizes]
        )
        if not 0.0 <= alpha_tax <= 0.2:
            raise ValueError("alpha_tax must be between 0 and 0.2.")
        self.alpha_tax = alpha_tax

        if corgi_dim > 0:  # concat kmer with corgi and run single encoder.
            self.tnf_enc_dim += corgi_dim
        
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
        

        # determined by coverage dim.
        self.coverage_encoder = _build_coverage_encoder(
            coverage_dim=coverage_dim, output_dim=output_dim, dropout=dropout
        )

        self.final_cov_dim = output_dim

        self.fusion_network = AttentionFusion(
            emb_dim=output_dim,
            kmer_enc_dim=output_dim,
            coverage_dim=self.final_cov_dim,
        )

        self.representation_dim = output_dim

        self.proj_head = _build_bt_projector(
            in_dim=self.representation_dim,
            proj_dim=self.proj_head_dim,
            n_layers=3,
            dropout=0.0,
        )

    def encode_canonical(self, x: torch.Tensor) -> torch.Tensor:
        """Encode composition and coverage without exposing taxonomy labels."""
        protein_ids = None
        if self.protein_encoder is not None:
            x, protein_ids = x[:, :-1], x[:, -1]
        tnf_features = x[:, : self.tnf_enc_dim]
        cov_features = x[:, self.tnf_enc_dim :]
        tnf_encode = self.tnf_encoder(tnf_features)
        cov_encode = self.coverage_encoder(cov_features)
        fused = self.fusion_network(tnf_encode, cov_encode)
        if self.protein_encoder is not None:
            fused = self.protein_encoder.fuse(fused, protein_ids)
        return fused

    def _add_taxonomy(
        self, fused_encode: torch.Tensor, tax_emb: torch.Tensor = None
    ) -> torch.Tensor:
        if self.uses_rank_taxonomy:
            if tax_emb is None:
                raise ValueError("Taxonomy features are required by this model.")
            tax_encode = self.tax_model(tax_emb)
            fused_encode = bounded_taxonomy_residual(
                fused_encode, tax_encode, tax_emb, self.alpha_tax
            )
        return fused_encode

    def _encode(self, x: torch.Tensor, tax_emb: torch.Tensor = None) -> torch.Tensor:
        return self._add_taxonomy(self.encode_canonical(x), tax_emb)

    def taxonomy_logits(self, canonical_embedding: torch.Tensor):
        """Predict every taxonomy rank from canonical features only."""
        if not self.uses_rank_taxonomy:
            raise RuntimeError("This model has no rank-taxonomy prediction heads.")
        return [head(canonical_embedding) for head in self.taxonomy_classifiers]

    def encode(self, x: torch.Tensor, tax_emb: torch.Tensor = None) -> torch.Tensor:
        """Return the fused, pre-projector representation used for clustering."""
        return self._encode(x, tax_emb)

    def get_embedding(self, x, tax_emb: torch.Tensor = None):
        return self.encode(x, tax_emb)

    def get_separate_modality_embs(self, x):
        if self.protein_encoder is not None:
            x = x[:, :-1]
        kmer_features = x[:, : self.tnf_enc_dim]
        kmer_encoded = self.tnf_encoder(kmer_features)
        coverage_features = x[:, self.tnf_enc_dim :]
        coverage_encoded = self.coverage_encoder(coverage_features)
        return kmer_encoded, coverage_encoded

    # keep unchanged for forward functions.
    def forward_one(self, x: torch.Tensor, tax_emb: torch.Tensor = None):
        fused_emb = self.encode(x, tax_emb)
        proj_emb = self.proj_head(fused_emb)
        return proj_emb

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        tax_emb1: torch.Tensor = None,
        tax_emb2: torch.Tensor = None,
        return_canonical: bool = False,
    ):
        canonical1 = self.encode_canonical(x1)
        canonical2 = self.encode_canonical(x2)
        z1 = self.proj_head(self._add_taxonomy(canonical1, tax_emb1))
        z2 = self.proj_head(self._add_taxonomy(canonical2, tax_emb2))
        if return_canonical:
            return z1, z2, canonical1, canonical2
        return z1, z2

# The projection head: which directly gives barlow twin loss the projected embeddings, should be looked with careful caution.
def _build_bt_projector(in_dim, proj_dim=768, n_layers=3, dropout=0.0):
    """
    Barlow Twins projection head.

    BT rewards a wide, non-linear projector: each hidden block is
    Linear -> BN -> ReLU, and the final layer is Linear -> BN (no activation,
    no bias on the last linear since BN recenters). Loss is computed on the
    BN'd output. Width matters most; depth (3 layers) follows the BT recipe.

    in_dim:    representation_dim (fused embedding size, e.g. 128)
    proj_dim:  projector width — sweep this (512/768/1024), because output dim is only 128, we just bump it to 3-4x of the output dim, which is 512 or 768, to give the projector enough capacity to learn a good representation for the Barlow Twins loss optimization. The original paper uses 2048 for ImageNet, but our input dimension is much smaller, so we can use a smaller projector dimension.
    n_layers:  total linear layers (BT uses 3)
    dropout:   keep 0.0 to match the BT recipe; exposed only for experiments
    """
    assert n_layers >= 2, "projector needs at least 2 layers"
    layers = []
    d = in_dim
    # hidden blocks: Linear -> BN -> ReLU  (optional dropout)
    for _ in range(n_layers - 1):
        layers.append(nn.Linear(d, proj_dim, bias=False))
        layers.append(nn.BatchNorm1d(proj_dim))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = proj_dim
    # final block: Linear -> BN  (no activation; BT loss reads this)
    layers.append(nn.Linear(d, proj_dim, bias=False))
    layers.append(nn.BatchNorm1d(proj_dim))
    return nn.Sequential(*layers)


# Then our model got an attention-based fusion network.
class AttentionFusion(nn.Module):
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
        super(AttentionFusion, self).__init__()
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
