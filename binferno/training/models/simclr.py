import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .cr_vae import VAE

# Try for one thing: contrastive multimodal vae for using tnf and coverage, 
# then you can also carry on Corgi features (PCAed). 

# Reference for contrastive vae: https://arxiv.org/abs/1902.04601.
class cVAENetWork(nn.Module):
    def __init__(
        self,
        tnf_dim: int,
        coverage_dim: int,
        dropout: float = 0.05,
        output_dim: int = 128,
        kmer_dim: int = 136,
        corgi_dim: int = 0,
        proj_head_dim: int = 128,
        proj: bool = False,
        latent_dim: int = 32,
    ):
        super(cVAENetWork, self).__init__()
        self.coverage_dim = coverage_dim  # by default we ask users to provide coverage.
        self.output_dim = output_dim
        self.kmer_dim = kmer_dim
        self.proj_head_dim = proj_head_dim
        self.proj = proj
        self.tnf_dim = tnf_dim
        self.tnf_enc_dim = kmer_dim
        self.corgi_dim = corgi_dim
        self.hidden_dim = 512
        self.latent_dim = latent_dim

        self.vae_cov = _build_coverage_encoder(
            coverage_dim=coverage_dim, output_dim=output_dim, dropout=dropout)

        modules = []
        n_hidden_layers = 3
        input_dim = self.tnf_enc_dim
        for _ in range(n_hidden_layers):
            modules.append(nn.Linear(input_dim, self.hidden_dim))
            modules.append(nn.BatchNorm1d(self.hidden_dim))
            modules.append(nn.LeakyReLU())
            modules.append(nn.Dropout(0.1))
            input_dim = self.hidden_dim
        modules.append(nn.Linear(self.hidden_dim, self.latent_dim * 2))  # output both mean and logvar for VAE
        self.tnf_encoder = nn.Sequential(*modules)

        decode_modules = []
        decode_hidden = 512
        n_decode_hidden_layers = 3
        input_dim = self.latent_dim
        for _ in range(n_decode_hidden_layers):
            decode_modules.append(nn.Linear(input_dim, decode_hidden))
            decode_modules.append(nn.BatchNorm1d(decode_hidden))
            decode_modules.append(nn.LeakyReLU())
            decode_modules.append(nn.Dropout(0.1))
            input_dim = decode_hidden
        decode_modules.append(nn.Linear(decode_hidden, self.tnf_enc_dim))  # decode back to original tnf dimension
        self.tnf_decoder = nn.Sequential(*decode_modules)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        tnf_features = x[:, : self.tnf_enc_dim]
        cov_features = x[:, self.tnf_enc_dim :]
        cov_encoded = self.vae_cov(cov_features)
        tnf_encoded = self.tnf_encoder(tnf_features)
        mu, logvar = torch.chunk(tnf_encoded, 2, dim=-1)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar, cov_encoded
    
    def forward(self, x):
        z, mu, logvar, cov_encoded = self.encode(x)
        tnf_decoded = self.tnf_decoder(z)
        return tnf_decoded, cov_encoded, mu, logvar

class NeuralNetwork(nn.Module):
    def __init__(
        self,
        tnf_dim: int,
        coverage_dim: int,
        dropout: float = 0.05,
        output_dim: int = 128,
        kmer_dim: int = 136,
        corgi_dim: int = 0,
        proj_head_dim: int = 128,
        proj: bool = False,
        dense_mode: bool = False,
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
        self.corgi_dim = corgi_dim
        self.hidden_dim = hidden_dim

        self.coverage_encoder = _build_coverage_encoder(
            coverage_dim=coverage_dim, output_dim=output_dim, dropout=dropout,
            hidden_dim = hidden_dim, n_hidden_layers=3
        )

        self.final_cov_dim = output_dim

        if corgi_dim > 0:
            self.tnf_enc_dim += corgi_dim
        self.tnf_enc_dim += self.output_dim # add the coverage encoding dimension to the tnf encoding dimension, so that we can fuse them together later. This is a bit hacky but it allows us to keep the fusion network simple. We can also try other ways to fuse them together later, but for now we just concatenate them and feed them into the fusion network.

        # was using 512.
        if not dense_mode:
            tnf_hidden = self.hidden_dim
        else:
            tnf_hidden = 1024

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
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
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
        proj_emb = self.proj_head(fused_emb)
        return proj_emb, cov_emb


class OldNeuralNetwork(nn.Module):
    def __init__(
        self,
        tnf_dim: int,
        coverage_dim: int,
        dropout: float = 0.05,
        output_dim: int = 128,
        kmer_dim: int = 136,
        corgi_dim: int = 0,
        proj_head_dim: int = 128,
        proj: bool = False,
        gated: bool = False,
        attention: bool = False,
        dense_mode: bool = False,
    ):
        super(OldNeuralNetwork, self).__init__()
        self.coverage_dim = coverage_dim  # by default we ask users to provide coverage.
        self.output_dim = output_dim
        self.kmer_dim = kmer_dim
        self.proj_head_dim = proj_head_dim
        self.proj = proj
        self.tnf_dim = tnf_dim
        self.tnf_enc_dim = kmer_dim
        self.corgi_dim = corgi_dim

        # this is barlow-twins style.

        # if n_samples <= 2: # which is the case now.
        self.coverage_encoder = _build_coverage_encoder(
            coverage_dim=coverage_dim, output_dim=output_dim, dropout=dropout
        )

        self.final_cov_dim = output_dim

        if corgi_dim > 0:
            self.tnf_enc_dim += corgi_dim

        # was using 512.
        if not dense_mode:
            tnf_hidden1 = 512
            tnf_hidden2 = 512
            tnf_hidden3 = 512
            tnf_hidden3 = output_dim

            self.tnf_encoder = nn.Sequential(
                nn.Linear(self.tnf_enc_dim, tnf_hidden1),
                nn.BatchNorm1d(tnf_hidden1),
                nn.LeakyReLU(),
                nn.Dropout(0.1),
                nn.Linear(tnf_hidden1, tnf_hidden2),
                nn.BatchNorm1d(tnf_hidden2),
                nn.LeakyReLU(),
                nn.Dropout(0.1),
                nn.Linear(tnf_hidden2, tnf_hidden3),
                nn.BatchNorm1d(tnf_hidden3),
                nn.LeakyReLU(),
                nn.Dropout(0.1),
                nn.Linear(tnf_hidden3, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.LeakyReLU(),
            )
        else:
            tnf_hidden1 = 1024
            tnf_hidden2 = 1024
            tnf_hidden3 = 1024
            tnf_hidden4 = output_dim
            self.tnf_encoder = nn.Sequential(
                nn.Linear(self.tnf_enc_dim, tnf_hidden1),
                nn.BatchNorm1d(tnf_hidden1),
                nn.LeakyReLU(),
                nn.Dropout(0.1),
                nn.Linear(tnf_hidden1, tnf_hidden2),
                nn.BatchNorm1d(tnf_hidden2),
                nn.LeakyReLU(),
                nn.Dropout(0.1),
                nn.Linear(tnf_hidden2, tnf_hidden3),
                nn.BatchNorm1d(tnf_hidden3),
                nn.LeakyReLU(),
                nn.Dropout(0.1),
                nn.Linear(tnf_hidden3, tnf_hidden4),
                nn.BatchNorm1d(tnf_hidden4),
                nn.LeakyReLU(),
            )

        self.fusion_network = fusion_network(
            emb_dim=output_dim,
            gated=gated,
            attention=attention,
        )

        self.representation_dim = (
            self.fusion_network.out_dim
        )  # this is the output dim of the fusion network, which is the representation dimension. By default it is 256. We can change it later if needed.

        self.proj_head = nn.Sequential(
            nn.Linear(self.representation_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
        )

    def _encode(self, x):
        # encode tnf and coverage features separately, then fuse them together.
        tnf_features = x[:, : self.tnf_enc_dim]
        tnf_encode = self.tnf_encoder(tnf_features)  # not normalized anymore.
        cov_features = x[:, self.tnf_enc_dim :]
        cov_encode = self.coverage_encoder(cov_features)  # not normalized anymore.
        fused_encode = self.fusion_network(tnf_encode, cov_encode)
        return fused_encode, tnf_encode, cov_encode

    def get_embedding(self, x):
        return self._encode(x)[0]

    def get_separate_modality_embs(self, x):
        kmer_features = x[:, : self.tnf_enc_dim]
        kmer_encoded = F.normalize(self.tnf_encoder(kmer_features), dim=-1)
        coverage_features = x[:, self.tnf_enc_dim :]
        coverage_encoded = F.normalize(self.coverage_encoder(coverage_features), dim=-1)
        return kmer_encoded, coverage_encoded

    def forward_one(self, x):
        fused_emb, tnf_emb, cov_emb = self._encode(x)
        proj_emb = self.proj_head(fused_emb)  # 2048-projection head.
        return proj_emb, tnf_emb, cov_emb


def fusion_network(
    emb_dim=128,
    gated=False,
    attention=False,
):
    if gated:
        return gated_fusion(in_dim=emb_dim, fused_dim=emb_dim*2, out_dim=emb_dim)
    if attention:
        return AttentionFusion(
            emb_dim=emb_dim,
            out_dim=emb_dim,
            num_heads=4,
            dropout=0.1,
        )

# Just a single model cannot fit the case with concatenated MLP encodings, 
# as in vamb, the modalities collaborate in the latent space and the reconstruction
#  loss is computed on the latent space, which is not the case in our setting. 
# So we can just use a simple MLP encoder for coverage, and then fuse it with the
# tnf encoder output. We can also try a VAE encoder for coverage to see if it can 
# capture the distribution of coverage features better.


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


    # n_samples = max(1, coverage_dim // 2)
    # orig_cov_encoder = nn.Sequential(
    #         nn.Linear(coverage_dim, 64),
    #         nn.LayerNorm(64),
    #         nn.LeakyReLU(),
    #         nn.Dropout(dropout),
    #         nn.Linear(64, 128),
    #         nn.LayerNorm(128),
    #         nn.LeakyReLU(),
    #         nn.Dropout(dropout),
    #         nn.Linear(128, output_dim),
    #         nn.LayerNorm(output_dim),
    #         nn.LeakyReLU(),
    #     )
    # elif n_samples <= 5:
    #     # Few samples: coverage starts carrying differential signal.
    #     # Transition to BatchNorm — enough input dimensions to be stable.
    #     hidden = 128
    #     return nn.Sequential(
    #         nn.Linear(coverage_dim, hidden),
    #         nn.BatchNorm1d(hidden),
    #         nn.LeakyReLU(),
    #         nn.Dropout(dropout),
    #         nn.Linear(hidden, hidden),
    #         nn.BatchNorm1d(hidden),
    #         nn.LeakyReLU(),
    #         nn.Dropout(dropout),
    #         nn.Linear(hidden, output_dim),
    #         nn.BatchNorm1d(output_dim),
    #         nn.LeakyReLU(),
    #     )

    # elif n_samples <= 20:
    #     # Medium samples: genuine multi-dimensional coverage signal.
    #     hidden = 256
    #     return nn.Sequential(
    #         nn.Linear(coverage_dim, hidden),
    #         nn.BatchNorm1d(hidden),
    #         nn.LeakyReLU(),
    #         nn.Dropout(dropout),
    #         nn.Linear(hidden, hidden),
    #         nn.BatchNorm1d(hidden),
    #         nn.LeakyReLU(),
    #         nn.Dropout(dropout),
    #         nn.Linear(hidden, output_dim),
    #         nn.BatchNorm1d(output_dim),
    #         nn.LeakyReLU(),
    #     )

    # else:
    #     # Many samples: full COMEBin-style wide encoder justified.
    #     hidden = 512
    #     return nn.Sequential(
    #         nn.Linear(coverage_dim, hidden),
    #         nn.BatchNorm1d(hidden),
    #         nn.LeakyReLU(),
    #         nn.Dropout(dropout),
    #         nn.Linear(hidden, hidden),
    #         nn.BatchNorm1d(hidden),
    #         nn.LeakyReLU(),
    #         nn.Dropout(dropout),
    #         nn.Linear(hidden, output_dim),
    #         nn.BatchNorm1d(output_dim),
    #         nn.LeakyReLU(),
    #     )

class AttentionFusion(nn.Module):
    """
    Transformer-style cross-modal fusion over [kmer, coverage] tokens.

    Inputs:
        kmer_emb: (B, emb_dim)
        cov_emb:  (B, emb_dim)
    Output:
        fused:    (B, out_dim)   default out_dim = 256

    Design notes vs. the original:
      - Pre-norm attention + FFN sublayers, each with a residual, so the
        raw token signal is never lost and the block trains stably.
      - LayerNorm (not BatchNorm) inside the token block: it operates on the
        (B, 2, emb_dim) sequence per-token and is batch-size robust, which
        matters for SimCLR where effective batch composition varies.
      - Readout concatenates mean-pooled and both attended tokens, giving the
        final projection a richer view than flatten() alone.
      - out_dim widened to 256 to relieve the downstream bottleneck.
    """

    def __init__(self, emb_dim=128, num_heads=4, out_dim=128, ffn_mult=2, dropout=0.1):
        super().__init__()
        self.emb_dim = emb_dim
        self.token_type_emb = nn.Embedding(2, emb_dim)  # 0=kmer, 1=coverage
        self.out_dim = out_dim

        # Attention sublayer (pre-norm)
        self.norm_attn = nn.LayerNorm(emb_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop_attn = nn.Dropout(dropout)

        # Feed-forward sublayer (pre-norm)
        self.norm_ffn = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * ffn_mult, emb_dim),
        )
        self.drop_ffn = nn.Dropout(dropout)

        # Readout: [kmer_token ; cov_token ; mean_pool] -> out_dim
        self.readout = nn.Sequential(
            nn.Linear(emb_dim * 3, emb_dim * 2),
            nn.LayerNorm(emb_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, out_dim),
        )
        self.out_norm = nn.LayerNorm(out_dim)

    def forward(self, kmer_emb, cov_emb):
        B = kmer_emb.size(0)

        type_ids = torch.tensor([0, 1], device=kmer_emb.device)
        token_type = self.token_type_emb(type_ids).unsqueeze(0)  # (1, 2, D)
        tokens = torch.stack([kmer_emb, cov_emb], dim=1) + token_type  # (B, 2, D)

        # Pre-norm attention with residual
        x = self.norm_attn(tokens)
        attended, _ = self.self_attn(x, x, x, need_weights=False)
        tokens = tokens + self.drop_attn(attended)

        # Pre-norm FFN with residual
        x = self.norm_ffn(tokens)
        tokens = tokens + self.drop_ffn(self.ffn(x)) # residual connection after FFN, so the original token signal is preserved and the block trains stably.

        # Readout
        kmer_t = tokens[:, 0]  # (B, D)
        cov_t = tokens[:, 1]  # (B, D)
        pooled = tokens.mean(dim=1)  # (B, D)
        fused = self.readout(torch.cat([kmer_t, cov_t, pooled], dim=-1))
        return self.out_norm(fused)


class gated_fusion(nn.Module):
    """
    Residual gated fusion with concat path and widened output.

    Inputs:
        kmer: (B, 128)  from tnf_encoder
        cov:  (B, 128)  from coverage_encoder
    Output:
        fused: (B, out_dim)   default out_dim=128

    Changes vs. the multiplicative-gate version:
      - Gates modulate a residual branch: signal = base + gate * delta,
        so the raw modality features can never be zeroed out.
      - Both modalities are concatenated (256-d) and projected, preserving
        cross-modal interactions the old gate discarded.
      - Output dim is configurable (128) to relieve the 128-d bottleneck.
    """

    def __init__(self, in_dim=128, fused_dim=256, out_dim=128, dropout=0.1):
        super().__init__()
        cat_dim = in_dim * 2  # kmer + cov concatenated
        self.out_dim = out_dim

        # Gates see the full concatenated context, output per-modality gates.
        self.kmer_gate = nn.Sequential(
            nn.LayerNorm(cat_dim),
            nn.Linear(cat_dim, in_dim),
            nn.Sigmoid(),
        )
        self.coverage_gate = nn.Sequential(
            nn.LayerNorm(cat_dim),
            nn.Linear(cat_dim, in_dim),
            nn.Sigmoid(),
        )

        # Residual transforms: the "delta" each gate modulates.
        self.kmer_transform = nn.Linear(in_dim, in_dim)
        self.coverage_transform = nn.Linear(in_dim, in_dim)

        # Fuse the concatenated, residually-gated features into fused_dim,
        # then to out_dim. This is where the bottleneck is widened.
        self.fuse = nn.Sequential(
            nn.Linear(cat_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, out_dim),
        )
        self.out_norm = nn.LayerNorm(out_dim)

    def forward(self, kmer, cov):
        cat = torch.cat([kmer, cov], dim=-1)  # (B, 256)

        # Residual gating: base feature is always preserved.
        kmer_g = kmer + self.kmer_gate(cat) * self.kmer_transform(kmer)
        cov_g = cov + self.coverage_gate(cat) * self.coverage_transform(cov)

        fused = self.fuse(torch.cat([kmer_g, cov_g], dim=-1))  # (B, out_dim)
        return self.out_norm(fused)
