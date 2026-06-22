# # An old try to use fusion network for simclr, did not turn to be good. May be deprecated.
# class OldNeuralNetwork(nn.Module):
#     def __init__(
#         self,
#         tnf_dim: int,
#         coverage_dim: int,
#         dropout: float = 0.05,
#         output_dim: int = 128,
#         kmer_dim: int = 136,
#         corgi_dim: int = 0,
#         proj_head_dim: int = 128,
#         proj: bool = False,
#         gated: bool = False,
#         attention: bool = False,
#         dense_mode: bool = False,
#     ):
#         super(OldNeuralNetwork, self).__init__()
#         self.coverage_dim = coverage_dim  # by default we ask users to provide coverage.
#         self.output_dim = output_dim
#         self.kmer_dim = kmer_dim
#         self.proj_head_dim = proj_head_dim
#         self.proj = proj
#         self.tnf_dim = tnf_dim
#         self.tnf_enc_dim = kmer_dim
#         self.corgi_dim = corgi_dim

#         self.coverage_encoder = _build_coverage_encoder(
#             coverage_dim=coverage_dim, output_dim=output_dim, dropout=dropout
#         )

#         self.final_cov_dim = output_dim

#         if corgi_dim > 0:
#             self.tnf_enc_dim += corgi_dim

#         # was using 512.
#         if not dense_mode:
#             tnf_hidden1 = 512
#             tnf_hidden2 = 512
#             tnf_hidden3 = 512
#             tnf_hidden3 = output_dim

#             self.tnf_encoder = nn.Sequential(
#                 nn.Linear(self.tnf_enc_dim, tnf_hidden1),
#                 nn.BatchNorm1d(tnf_hidden1),
#                 nn.LeakyReLU(),
#                 nn.Dropout(0.1),
#                 nn.Linear(tnf_hidden1, tnf_hidden2),
#                 nn.BatchNorm1d(tnf_hidden2),
#                 nn.LeakyReLU(),
#                 nn.Dropout(0.1),
#                 nn.Linear(tnf_hidden2, tnf_hidden3),
#                 nn.BatchNorm1d(tnf_hidden3),
#                 nn.LeakyReLU(),
#                 nn.Dropout(0.1),
#                 nn.Linear(tnf_hidden3, output_dim),
#                 nn.BatchNorm1d(output_dim),
#                 nn.LeakyReLU(),
#             )
#         else:
#             tnf_hidden1 = 1024
#             tnf_hidden2 = 1024
#             tnf_hidden3 = 1024
#             tnf_hidden4 = output_dim
#             self.tnf_encoder = nn.Sequential(
#                 nn.Linear(self.tnf_enc_dim, tnf_hidden1),
#                 nn.BatchNorm1d(tnf_hidden1),
#                 nn.LeakyReLU(),
#                 nn.Dropout(0.1),
#                 nn.Linear(tnf_hidden1, tnf_hidden2),
#                 nn.BatchNorm1d(tnf_hidden2),
#                 nn.LeakyReLU(),
#                 nn.Dropout(0.1),
#                 nn.Linear(tnf_hidden2, tnf_hidden3),
#                 nn.BatchNorm1d(tnf_hidden3),
#                 nn.LeakyReLU(),
#                 nn.Dropout(0.1),
#                 nn.Linear(tnf_hidden3, tnf_hidden4),
#                 nn.BatchNorm1d(tnf_hidden4),
#                 nn.LeakyReLU(),
#             )

#         self.fusion_network = fusion_network(
#             emb_dim=output_dim,
#             gated=gated,
#             attention=attention,
#         )

#         self.representation_dim = (
#             self.fusion_network.out_dim
#         )  # this is the output dim of the fusion network, which is the representation dimension. By default it is 256. We can change it later if needed.

#         self.proj_head = nn.Sequential(
#             nn.Linear(self.representation_dim, 2048),
#             nn.BatchNorm1d(2048),
#             nn.ReLU(),
#             nn.Linear(2048, 2048),
#             nn.BatchNorm1d(2048),
#         )

#     def _encode(self, x):
#         # encode tnf and coverage features separately, then fuse them together.
#         tnf_features = x[:, : self.tnf_enc_dim]
#         tnf_encode = self.tnf_encoder(tnf_features)  # not normalized anymore.
#         cov_features = x[:, self.tnf_enc_dim :]
#         cov_encode = self.coverage_encoder(cov_features)  # not normalized anymore.
#         fused_encode = self.fusion_network(tnf_encode, cov_encode)
#         return fused_encode, tnf_encode, cov_encode

#     def get_embedding(self, x):
#         return self._encode(x)[0]

#     def get_separate_modality_embs(self, x):
#         kmer_features = x[:, : self.tnf_enc_dim]
#         kmer_encoded = F.normalize(self.tnf_encoder(kmer_features), dim=-1)
#         coverage_features = x[:, self.tnf_enc_dim :]
#         coverage_encoded = F.normalize(self.coverage_encoder(coverage_features), dim=-1)
#         return kmer_encoded, coverage_encoded

#     def forward_one(self, x):
#         fused_emb, tnf_emb, cov_emb = self._encode(x)
#         proj_emb = self.proj_head(fused_emb)  # 2048-projection head.
#         return proj_emb, tnf_emb, cov_emb


# def fusion_network(
#     emb_dim=128,
#     gated=False,
#     attention=False,
# ):
#     if gated:
#         return gated_fusion(in_dim=emb_dim, fused_dim=emb_dim*2, out_dim=emb_dim)
#     if attention:
#         return AttentionFusion(
#             emb_dim=emb_dim,
#             out_dim=emb_dim,
#             num_heads=4,
#             dropout=0.1,
#         )

# class AttentionFusion(nn.Module):
#     """
#     Transformer-style cross-modal fusion over [kmer, coverage] tokens.

#     Inputs:
#         kmer_emb: (B, emb_dim)
#         cov_emb:  (B, emb_dim)
#     Output:
#         fused:    (B, out_dim)   default out_dim = 256

#     Design notes vs. the original:
#       - Pre-norm attention + FFN sublayers, each with a residual, so the
#         raw token signal is never lost and the block trains stably.
#       - LayerNorm (not BatchNorm) inside the token block: it operates on the
#         (B, 2, emb_dim) sequence per-token and is batch-size robust, which
#         matters for SimCLR where effective batch composition varies.
#       - Readout concatenates mean-pooled and both attended tokens, giving the
#         final projection a richer view than flatten() alone.
#       - out_dim widened to 256 to relieve the downstream bottleneck.
#     """

#     def __init__(self, emb_dim=128, num_heads=4, out_dim=128, ffn_mult=2, dropout=0.1):
#         super().__init__()
#         self.emb_dim = emb_dim
#         self.token_type_emb = nn.Embedding(2, emb_dim)  # 0=kmer, 1=coverage
#         self.out_dim = out_dim

#         # Attention sublayer (pre-norm)
#         self.norm_attn = nn.LayerNorm(emb_dim)
#         self.self_attn = nn.MultiheadAttention(
#             embed_dim=emb_dim,
#             num_heads=num_heads,
#             dropout=dropout,
#             batch_first=True,
#         )
#         self.drop_attn = nn.Dropout(dropout)

#         # Feed-forward sublayer (pre-norm)
#         self.norm_ffn = nn.LayerNorm(emb_dim)
#         self.ffn = nn.Sequential(
#             nn.Linear(emb_dim, emb_dim * ffn_mult),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(emb_dim * ffn_mult, emb_dim),
#         )
#         self.drop_ffn = nn.Dropout(dropout)

#         # Readout: [kmer_token ; cov_token ; mean_pool] -> out_dim
#         self.readout = nn.Sequential(
#             nn.Linear(emb_dim * 3, emb_dim * 2),
#             nn.LayerNorm(emb_dim * 2),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(emb_dim * 2, out_dim),
#         )
#         self.out_norm = nn.LayerNorm(out_dim)

#     def forward(self, kmer_emb, cov_emb):
#         B = kmer_emb.size(0)

#         type_ids = torch.tensor([0, 1], device=kmer_emb.device)
#         token_type = self.token_type_emb(type_ids).unsqueeze(0)  # (1, 2, D)
#         tokens = torch.stack([kmer_emb, cov_emb], dim=1) + token_type  # (B, 2, D)

#         # Pre-norm attention with residual
#         x = self.norm_attn(tokens)
#         attended, _ = self.self_attn(x, x, x, need_weights=False)
#         tokens = tokens + self.drop_attn(attended)

#         # Pre-norm FFN with residual
#         x = self.norm_ffn(tokens)
#         tokens = tokens + self.drop_ffn(self.ffn(x)) # residual connection after FFN, so the original token signal is preserved and the block trains stably.

#         # Readout
#         kmer_t = tokens[:, 0]  # (B, D)
#         cov_t = tokens[:, 1]  # (B, D)
#         pooled = tokens.mean(dim=1)  # (B, D)
#         fused = self.readout(torch.cat([kmer_t, cov_t, pooled], dim=-1))
#         return self.out_norm(fused)


# class gated_fusion(nn.Module):
#     """
#     Residual gated fusion with concat path and widened output.

#     Inputs:
#         kmer: (B, 128)  from tnf_encoder
#         cov:  (B, 128)  from coverage_encoder
#     Output:
#         fused: (B, out_dim)   default out_dim=128

#     Changes vs. the multiplicative-gate version:
#       - Gates modulate a residual branch: signal = base + gate * delta,
#         so the raw modality features can never be zeroed out.
#       - Both modalities are concatenated (256-d) and projected, preserving
#         cross-modal interactions the old gate discarded.
#       - Output dim is configurable (128) to relieve the 128-d bottleneck.
#     """

#     def __init__(self, in_dim=128, fused_dim=256, out_dim=128, dropout=0.1):
#         super().__init__()
#         cat_dim = in_dim * 2  # kmer + cov concatenated
#         self.out_dim = out_dim

#         # Gates see the full concatenated context, output per-modality gates.
#         self.kmer_gate = nn.Sequential(
#             nn.LayerNorm(cat_dim),
#             nn.Linear(cat_dim, in_dim),
#             nn.Sigmoid(),
#         )
#         self.coverage_gate = nn.Sequential(
#             nn.LayerNorm(cat_dim),
#             nn.Linear(cat_dim, in_dim),
#             nn.Sigmoid(),
#         )

#         # Residual transforms: the "delta" each gate modulates.
#         self.kmer_transform = nn.Linear(in_dim, in_dim)
#         self.coverage_transform = nn.Linear(in_dim, in_dim)

#         # Fuse the concatenated, residually-gated features into fused_dim,
#         # then to out_dim. This is where the bottleneck is widened.
#         self.fuse = nn.Sequential(
#             nn.Linear(cat_dim, fused_dim),
#             nn.LayerNorm(fused_dim),
#             nn.LeakyReLU(0.01),
#             nn.Dropout(dropout),
#             nn.Linear(fused_dim, out_dim),
#         )
#         self.out_norm = nn.LayerNorm(out_dim)

#     def forward(self, kmer, cov):
#         cat = torch.cat([kmer, cov], dim=-1)  # (B, 256)

#         # Residual gating: base feature is always preserved.
#         kmer_g = kmer + self.kmer_gate(cat) * self.kmer_transform(kmer)
#         cov_g = cov + self.coverage_gate(cat) * self.coverage_transform(cov)

#         fused = self.fuse(torch.cat([kmer_g, cov_g], dim=-1))  # (B, out_dim)
#         return self.out_norm(fused)
