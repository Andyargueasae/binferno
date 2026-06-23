import torch
import torch.nn as nn

from .taxonomy import HierarchicalTaxonomyEncoder, _build_tax_emb_encoder

class NeuralNetwork(nn.Module):
    def __init__(
        self,
        tnf_dim,
        coverage_dim,
        dropout=0.05,
        output_dim=128,
        kmer_dim=136,
        proj_head_dim=512,
        tax_emb_dim: int = 0,  # corgi or dnabert_s
        hierarchical_taxonomy: bool = False,
    ):
        super(NeuralNetwork, self).__init__()
        self.coverage_dim = coverage_dim  # by default we ask users to provide coverage.
        self.output_dim = output_dim
        self.kmer_dim = kmer_dim
        self.proj_head_dim = proj_head_dim
        self.tnf_dim = tnf_dim
        self.tnf_enc_dim = kmer_dim
        self.tax_emb_dim = tax_emb_dim
        self.hierarchical_taxonomy = hierarchical_taxonomy  # used to determine later in the forward: whether to feed hierarchical taxonomy logits into the fusion network or an embedding

        if self.tax_emb_dim > 0:
            if not hierarchical_taxonomy:
                self.tnf_enc_dim += output_dim
                self.taxonomy_encoder = _build_tax_emb_encoder(
                    tax_emb_dim=tax_emb_dim,
                    hidden_dim=tax_emb_dim,
                    output_dim=output_dim,
                    dropout=dropout,
                )
            else:
                self.taxonomy_encoder = HierarchicalTaxonomyEncoder(
                    input_dim=tax_emb_dim,
                    hidden_dim=tax_emb_dim,
                    output_dim=output_dim,
                    dropout=dropout,
                )

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
            kmer_enc_dim=output_dim,  # modality 1
            coverage_dim=self.final_cov_dim,  # modality 2
            tax_emb_dim=tax_emb_dim,  # modality 3, if we include taxonomy embeddings as a separate modality. If we directly concatenate taxonomy embeddings with kmer features and feed into the same encoder, then we don't need this separate modality dimension for taxonomy.
        )

        self.representation_dim = output_dim

        self.proj_head = _build_bt_projector(
            in_dim=self.representation_dim,
            proj_dim=self.proj_head_dim,
            n_layers=3,
            dropout=0.0,
        )

    # TODO: embed the taxonomy features in model June 23 2026.
    def _encode(self, x: torch.Tensor, tax_emb: torch.Tensor = None):
        # encode tnf and coverage features separately, then fuse them together.
        tnf_features = x[:, : self.tnf_enc_dim]
        tnf_encode = self.tnf_encoder(tnf_features)
        cov_features = x[:, self.tnf_enc_dim :]
        cov_encode = self.coverage_encoder(cov_features)
        if not tax_emb is None:
            tax_encode = self.taxonomy_encoder(tax_emb)
            # here two diverged choices: merge with tnf or used alone. 
            fused_encode = self.fusion_network(tnf_encode, cov_encode, tax_encode=tax_encode)  # 128
        else:
            fused_encode = self.fusion_network(tnf_encode, cov_encode)  # 128
        return fused_encode

    def get_embedding(self, x: torch.Tensor, tax_emb: torch.Tensor = None):
        return self._encode(x, tax_emb)

    def get_separate_modality_embs(self, x: torch.Tensor, tax_emb: torch.Tensor = None):
        kmer_features = x[:, : self.tnf_enc_dim]
        kmer_encoded = self.tnf_encoder(kmer_features)
        coverage_features = x[:, self.tnf_enc_dim :]
        coverage_encoded = self.coverage_encoder(coverage_features)
        if not tax_emb is None:
            tax_encoded = self.taxonomy_encoder(tax_emb)
            return kmer_encoded, coverage_encoded, tax_encoded
        return kmer_encoded, coverage_encoded

    def forward_one(self, x: torch.Tensor, tax_emb: torch.Tensor = None):
        fused_emb = self._encode(x, tax_emb)
        proj_emb = self.proj_head(fused_emb)
        return proj_emb

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        tax_emb1: torch.Tensor = None,
        tax_emb2: torch.Tensor = None,
    ):
        z1 = self.forward_one(x1, tax_emb1)
        z2 = self.forward_one(x2, tax_emb2)
        return z1, z2

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
    GATE_INPUT_MULTIPLIER = 2
    MULTI_SCALE_CONCAT_MULTIPLIER = 6 
    COMPRESSOR_HIDDEN_MULTIPLIER = 4
    COMPRESSOR_OUTPUT_MULTIPLIER = 2
    INTERACTION_HIDDEN_MULTIPLIER = 3
    INTERACTION_INTERMEDIATE_MULTIPLIER = 2
    FINAL_FUSION_INPUT_MULTIPLIER = 4
    FINAL_FUSION_HIDDEN_MULTIPLIER = 2
    RESIDUAL_WEIGHT_INIT = 0.5
    ADAPTIVE_DROPOUT_MAX_MULTIPLIER = 2.0

    def __init__(
        self,
        kmer_enc_dim,
        emb_dim,
        coverage_dim,
        tax_emb_dim=0,
    ):
        super(AttentionFusion, self).__init__()
        self.kmer_enc_dim = kmer_enc_dim
        self.emb_dim = emb_dim
        self.tax_emb_dim = tax_emb_dim
        self.coverage_dim = coverage_dim
        
        self.kmer_proj = nn.Linear(self.kmer_enc_dim, self.emb_dim)
        self.coverage_proj = nn.Linear(self.coverage_dim, self.emb_dim)
        self.fusion_dropout = nn.Dropout(self.DEFAULT_DROPOUT)
        self.batch_norm = nn.BatchNorm1d(self.emb_dim)
        self.residual_weight = nn.Parameter(torch.tensor(self.RESIDUAL_WEIGHT_INIT))

        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.emb_dim,
            num_heads=self.DEFAULT_NUM_HEADS,
            dropout=self.DEFAULT_DROPOUT,
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
        if self.tax_emb_dim > 0:
            self.tax_emb_proj = nn.Linear(self.tax_emb_dim, self.emb_dim)
            self.taxonomy_gate = nn.Sequential(
                nn.Linear(self.emb_dim * self.GATE_INPUT_MULTIPLIER, self.emb_dim),
                nn.Sigmoid(),
            )
            self.FINAL_FUSION_INPUT_MULTIPLIER += 1  # add one more for taxonomy features in the final fusion.
            self.MULTI_SCALE_CONCAT_MULTIPLIER += 3 # add three more for taxonomy features in the multi-scale fusion (three scales).

        self.fused_proj = nn.Sequential(
            nn.Linear(
                self.emb_dim * self.FINAL_FUSION_INPUT_MULTIPLIER,
                self.emb_dim * self.FINAL_FUSION_HIDDEN_MULTIPLIER,
            ),
            nn.BatchNorm1d(self.emb_dim * self.FINAL_FUSION_HIDDEN_MULTIPLIER),
            nn.LeakyReLU(),
            nn.Dropout(self.DEFAULT_DROPOUT),
            nn.Linear(self.emb_dim * self.FINAL_FUSION_HIDDEN_MULTIPLIER, self.emb_dim),
            nn.BatchNorm1d(self.emb_dim),
            nn.LeakyReLU(),
        )

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
            nn.Dropout(self.DEFAULT_DROPOUT),
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
            nn.Dropout(self.DEFAULT_DROPOUT),
            nn.Linear(
                emb_dim * self.INTERACTION_HIDDEN_MULTIPLIER,
                emb_dim * self.INTERACTION_INTERMEDIATE_MULTIPLIER,
            ),
            nn.ReLU(),
            nn.Linear(emb_dim * self.INTERACTION_INTERMEDIATE_MULTIPLIER, emb_dim),
        )
        self.token_type_emb = nn.Embedding(2, emb_dim)  # 0=kmer, 1=coverage

    def forward(self, 
        tnf_feature: torch.Tensor, 
        coverage_feature: torch.Tensor, 
        tax_feature: torch.Tensor = None
    ):
        kmer_emb = self.kmer_proj(tnf_feature).unsqueeze(1)
        cov_emb = self.coverage_proj(coverage_feature).unsqueeze(1)
        
        if tax_feature is not None and self.tax_emb_dim > 0:
            tax_emb = self.tax_emb_proj(tax_feature).unsqueeze(1)
            type_ides = torch.tensor([0, 1, 2], device=kmer_emb.device)
            token_type = self.token_type_emb(type_ides).unsqueeze(0)  # (1, 3, emb_dim)
            tokens = torch.cat([kmer_emb, cov_emb, tax_emb], dim=1) + token_type # add the third token.
        else:
            type_ids = torch.tensor([0, 1], device=kmer_emb.device)
            token_type = self.token_type_emb(type_ids).unsqueeze(0)  # (1, 2, emb_dim)
            tokens = torch.cat([kmer_emb, cov_emb], dim=1) + token_type

        attended, _ = self.self_attn(tokens, tokens, tokens)

        kmer_attended = attended[:, 0, :]  # ( batch_size, emb_dim)
        cov_attended = attended[:, 1, :]  # ( batch_size, emb_dim)
        if tax_feature is not None:
            tax_attended = attended[:, 2, :]  # ( batch_size, emb_dim)

        attended_feature = kmer_attended + cov_attended  # ( batch_size, emb_dim)
        if tax_feature is not None:
            attended_feature += tax_attended # ( batch_size, emb_dim)

        kmer_emb = kmer_emb.squeeze(1)
        cov_emb = cov_emb.squeeze(1)
        gated_features = [kmer_emb, cov_emb]
        if tax_feature is not None and self.tax_emb_dim > 0:
            gated_features.append(tax_emb.squeeze(1))

        kmer_gate_weight = self.kmer_gate(torch.cat(gated_features, dim=1))
        coverage_gate_weight = self.coverage_gate(torch.cat(gated_features, dim=1))
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
            self.batch_norm(
                (1 - residual_weight_clamped) * gated_kmer
                + residual_weight_clamped * kmer_emb
            )
        )
        gated_coverage = self.fusion_dropout(
            self.batch_norm(
                (1 - residual_weight_clamped) * gated_coverage
                + residual_weight_clamped * cov_emb
            )
        )
        if tax_feature is not None and self.tax_emb_dim > 0:
            tax_gate_weight = self.taxonomy_gate(torch.cat(gated_features, dim=1))
            tax_attended = tax_attended.squeeze(1)
            gated_tax = (
                tax_gate_weight * tax_emb + (1 - tax_gate_weight) * tax_attended
            )
            gated_tax = self.fusion_dropout(
                self.batch_norm(
                    (1 - residual_weight_clamped) * gated_tax
                    + residual_weight_clamped * tax_emb
                )
            )

        scale_features = []
        for scale_layer in self.multiscale_fusion:
            scale_kmer = scale_layer(gated_kmer)
            scale_coverage = scale_layer(gated_coverage)
            if tax_feature is not None and self.tax_emb_dim > 0:
                scale_tax = scale_layer(gated_tax)
                scale_features.append(torch.cat([scale_kmer, scale_coverage, scale_tax], dim=1)) # if (batch_size, emb_dim) -> (batch_size, emb_dim * 3) after concatenation.
            else:
                scale_features.append(torch.cat([scale_kmer, scale_coverage], dim=1)) # if (batch_size, emb_dim) -> (batch_size, emb_dim * 2) after concatenation.

        multi_scale_concat = torch.cat(scale_features, dim=1) # if we have 3 scales and no taxonomy, this will be (batch_size, emb_dim * 2 * 3); if we have taxonomy, this will be (batch_size, emb_dim * 3 * 3)
        multi_scale_compressed = self.multi_scale_compressor(multi_scale_concat)
        interaction_features = self.interaction_module(multi_scale_compressed)

        # The last split if taxonomy is included. 
        final_features_to_fuse = [gated_kmer, gated_coverage, interaction_features, attended_feature]
        if tax_feature is not None and self.tax_emb_dim > 0:
            final_features_to_fuse.append(gated_tax)
        final_fused_features = torch.cat(final_features_to_fuse, dim=1)
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
