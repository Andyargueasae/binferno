import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# engineering ideas: Kmer set to 4 first. (index represent priority)
# 1. auxiliary losses on separate modality encoders.
# 2. residual connection in fusion network after fusion, add the learned to the original tnf features. For sample coverage, if n_sample is small, don't add;
# 3. new embedding network architecture resembling COMEBin. Please at least have a god damn try!
# 4. weighted gate fusion is a bottleneck and should not be used in long terms.
# 5. Maybe borrow one Vae from taxvamb/vamb to do a reconstruction here? Maybe not, as tnf is 103-dim in vamb's context. We may could try coverage by vae.


class NeuralNetwork(nn.Module):
    def __init__(
        self,
        tnf_dim,
        coverage_dim,
        dropout=0.05,
        output_dim=128,
        n_layers=3,
        kmer_dim=136,
        corgi_dim=0,
        proj_head_dim=128,
        proj=False,
        gated=False,
        attention=False,
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

        # this is barlow-twins style.

        # if n_samples <= 2: # which is the case now.
        self.coverage_encoder = _build_coverage_encoder(
            coverage_dim=coverage_dim, output_dim=output_dim, dropout=dropout
        )

        self.final_cov_dim = output_dim

        if corgi_dim > 0:
            self.tnf_enc_dim += corgi_dim

        # was using 512.
        tnf_hidden1 = 512
        tnf_hidden2 = 256
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
        )
        # Run A: 4-mer + correct architecture (your previous run, already done — ~71%)
        # Run B: 5-mer + same architecture structure (just change input dim)

        # leave fusion network to be implemented later.
        self.fusion_network = fusion_network(
            emb_dim=output_dim,
            gated=gated,
            attention=attention,
        )

        self.representation_dim = output_dim

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
        return fused_encode

    def get_embedding(self, x):
        return self._encode(x)

    def get_separate_modality_embs(self, x):
        kmer_features = x[:, : self.tnf_enc_dim]
        kmer_encoded = F.normalize(self.tnf_encoder(kmer_features), dim=-1)
        coverage_features = x[:, self.tnf_enc_dim :]
        coverage_encoded = F.normalize(self.coverage_encoder(coverage_features), dim=-1)
        return kmer_encoded, coverage_encoded

    def forward_one(self, x):
        fused_emb = self._encode(x)
        proj_emb = self.proj_head(fused_emb)  # 512
        return proj_emb

    def forward(self, x1, x2):
        z1 = self.forward_one(x1)
        z2 = self.forward_one(x2)
        return z1, z2


def fusion_network(
    emb_dim=128,
    gated=False,
    attention=False,
):
    if gated:
        return gated_fusion(emb_dim=emb_dim)
    if attention:
        return AttentionFusion(
            emb_dim=emb_dim,
            num_heads=4,
            dropout=0.1,
        )
    return ConcatFusion(emb_dim=emb_dim)


class vae_for_coverage(nn.Module):  # Look at vamb to borrow some thoughts.
    def __init__(self, coverage_dim, output_dim, dropout):
        super(vae_for_coverage, self).__init__()
        hidden = 128
        self.encoder = nn.Sequential(
            nn.Linear(coverage_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )
        self.mu_layer = nn.Linear(hidden, output_dim)
        self.logvar_layer = nn.Linear(hidden, output_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


def _build_coverage_encoder(
    coverage_dim: int, output_dim: int, dropout: float, vae_encoder=False
):
    """
    Adaptive coverage encoder that scales capacity with input information content.

    coverage_dim is always 2 * n_samples (mean + std per sample).
    n_samples drives capacity, not coverage_dim directly.
    """
    n_samples = max(1, coverage_dim // 2)

    if vae_encoder:
        return vae_for_coverage(coverage_dim, output_dim, dropout)

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


# Then our model got an attention-based fusion network.
class AttentionFusion(nn.Module):
    def __init__(self, emb_dim=128, num_heads=4, dropout=0.1):
        super().__init__()

        self.emb_dim = emb_dim

        self.self_attn = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.fused_proj = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim * 2),
            nn.BatchNorm1d(emb_dim * 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.LeakyReLU(),
        )
        # in __init__
        self.token_type_emb = nn.Embedding(2, emb_dim)  # 0=kmer, 1=coverage

    def forward(self, kmer_emb, cov_emb):
        # in forward
        type_ids = torch.tensor([0, 1], device=kmer_emb.device)
        token_type = self.token_type_emb(type_ids).unsqueeze(0)  # (1, 2, emb_dim)
        tokens = torch.stack([kmer_emb, cov_emb], dim=1) + token_type
        attended, _ = self.self_attn(tokens, tokens, tokens)

        fused = attended.flatten(1)
        fused = self.fused_proj(fused)

        return fused


class gated_fusion(nn.Module):
    # gate fusion uses layernorm.
    def __init__(self, emb_dim):
        super(gated_fusion, self).__init__()
        self.emb_dim = emb_dim

        self.kmer_gate = nn.Sequential(
            nn.LayerNorm(emb_dim * 2), nn.Linear(emb_dim * 2, emb_dim), nn.Sigmoid()
        )
        self.coverage_gate = nn.Sequential(
            nn.LayerNorm(emb_dim * 2), nn.Linear(emb_dim * 2, emb_dim), nn.Sigmoid()
        )
        self.out_norm = nn.LayerNorm(self.emb_dim)

    def forward(self, kmer_emb, coverage_emb):
        # gate_weight = self.gate(torch.cat([kmer_emb, coverage_emb], dim=1))
        kmer_gate = self.kmer_gate(torch.cat([kmer_emb, coverage_emb], dim=1))
        coverage_gate = self.coverage_gate(torch.cat([kmer_emb, coverage_emb], dim=1))
        fused = kmer_gate * kmer_emb + coverage_gate * coverage_emb
        return self.out_norm(fused)


class ConcatFusion(nn.Module):
    def __init__(self, emb_dim=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim * 2, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.LeakyReLU(),
        )

    def forward(self, kmer_emb, cov_emb):
        return self.net(torch.cat([kmer_emb, cov_emb], dim=-1))


# The raw COMEBin's embedding network is a simple MLP with batch normalization and dropout. We can use it as a baseline model for our experiments. We can also implement a more complex fusion network later to see if we can improve the performance.
class EmbeddingNet(nn.Module):
    # Useful code from fast.ai tabular model
    # https://github.com/fastai/fastai/blob/3b7c453cfa3845c6ffc496dd4043c07f3919270e/fastai/tabular/models.py#L6
    def __init__(self, in_sz, out_sz, emb_szs, ps, use_bn=True, actn=nn.ReLU()):
        super(EmbeddingNet, self).__init__()
        self.in_sz = in_sz
        self.out_sz = out_sz
        self.n_embs = len(emb_szs) - 1
        if ps == 0:
            ps = np.zeros(self.n_embs)
        # input layer
        layers = [nn.Linear(self.in_sz, emb_szs[0]), actn]
        # hidden layers
        for i in range(self.n_embs):
            layers += self.bn_drop_lin(
                n_in=emb_szs[i], n_out=emb_szs[i + 1], bn=use_bn, p=ps[i], actn=actn
            )
        # output layer
        layers.append(nn.Linear(emb_szs[-1], self.out_sz))
        self.fc = nn.Sequential(*layers)

    def bn_drop_lin(
        self,
        n_in: int,
        n_out: int,
        bn: bool = True,
        p: float = 0.0,
        actn: nn.Module = None,
    ):
        # https://github.com/fastai/fastai/blob/3b7c453cfa3845c6ffc496dd4043c07f3919270e/fastai/layers.py#L44
        "Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."
        layers = [nn.BatchNorm1d(n_in)] if bn else []
        if p != 0:
            layers.append(nn.Dropout(p))
        layers.append(nn.Linear(n_in, n_out))
        if actn is not None:
            layers.append(actn)
        return layers

    def forward(self, x):
        output = self.fc(x)
        return output  # F.normalize(output) #output #/ torch.linalg.vector_norm(output, dim=-1, keepdim=True)


class SimCLRNet(nn.Module):
    # Useful code from fast.ai tabular model
    # https://github.com/fastai/fastai/blob/3b7c453cfa3845c6ffc496dd4043c07f3919270e/fastai/tabular/models.py#L6
    def __init__(
        self,
        in_sz,
        out_sz,
        emb_szs,
        ps,
        use_bn=True,
        actn=nn.ReLU(),
        pretrained_model=None,
        cov_model=None,
        covmodel_notl2normalize=False,
    ):
        super(SimCLRNet, self).__init__()
        self.pretrained_model = pretrained_model
        self.cov_model = cov_model
        self.in_sz = in_sz
        self.out_sz = out_sz
        self.n_embs = len(emb_szs) - 1
        self.covmodel_notl2normalize = covmodel_notl2normalize
        if ps == 0:
            ps = np.zeros(self.n_embs)
        # input layer
        layers = [nn.Linear(self.in_sz, emb_szs[0]), actn]
        # hidden layers
        for i in range(self.n_embs):
            layers += self.bn_drop_lin(
                n_in=emb_szs[i], n_out=emb_szs[i + 1], bn=use_bn, p=ps[i], actn=actn
            )
        # output layer
        layers.append(nn.Linear(emb_szs[-1], self.out_sz))
        self.fc = nn.Sequential(*layers)
        project_layer = [actn, nn.Linear(self.out_sz, self.out_sz)]
        # project_layer= [nn.Linear(self.out_sz,self.out_sz),actn,nn.Linear(self.out_sz,self.out_sz)]
        self.fc2 = nn.Sequential(*project_layer)

    def bn_drop_lin(
        self,
        n_in: int,
        n_out: int,
        bn: bool = True,
        p: float = 0.0,
        actn: nn.Module = None,
    ):
        # https://github.com/fastai/fastai/blob/3b7c453cfa3845c6ffc496dd4043c07f3919270e/fastai/layers.py#L44
        "Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."
        layers = [nn.BatchNorm1d(n_in)] if bn else []
        if p != 0:
            layers.append(nn.Dropout(p))
        layers.append(nn.Linear(n_in, n_out))
        if actn is not None:
            layers.append(actn)
        return layers

    def forward(self, x, x2=None):
        if self.pretrained_model is not None and self.cov_model is None:
            kmeremb = self.pretrained_model(x)
            x = torch.cat([F.normalize(self.pretrained_model(x)), x2], dim=-1)
        if self.pretrained_model is not None and self.cov_model is not None:
            kmeremb = self.pretrained_model(x)
            if self.covmodel_notl2normalize:
                x = torch.cat(
                    [F.normalize(self.pretrained_model(x)), self.cov_model(x2)], dim=-1
                )
            else:
                x = torch.cat(
                    [
                        F.normalize(self.pretrained_model(x)),
                        F.normalize(self.cov_model(x2)),
                    ],
                    dim=-1,
                )

        # this is the case.
        if self.pretrained_model is None and self.cov_model is not None:
            if self.covmodel_notl2normalize:
                x = torch.cat([x, self.cov_model(x2)], dim=-1)
            else:
                # coverage model feature is normalized by default and concatenated with original features.
                x = torch.cat([x, F.normalize(self.cov_model(x2))], dim=-1)

        output = self.fc(x)

        if self.cov_model is not None and self.pretrained_model is not None:
            return output, self.cov_model(x2), kmeremb
        elif self.cov_model is not None and self.pretrained_model is None:
            return output, self.cov_model(x2)
        else:
            return output  # F.normalize(output) #output #/ torch.linalg.vector_norm(output, dim=-1, keepdim=True)


class EmbNet2(nn.Module):
    def __init__(
        self,
        composition_dim,
        coverage_dim,
        dropout=0.1,
        output_dim=128,
        kmer_dim=136,
    ):
        super(EmbNet2, self).__init__()
        self.tnf_dim = composition_dim
        self.coverage_dim = coverage_dim
        self.dropout = dropout
        self.output_dim = output_dim
        self.kmer_dim = kmer_dim

        self.coverage_encoder = _build_coverage_encoder(
            self, coverage_dim=coverage_dim, output_dim=output_dim, dropout=dropout
        )

        self.fuse_encoder = nn.Sequential(
            nn.Linear(self.tnf_dim + self.coverage_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(),
        )  # 3 internal layers plus in-projection and out-projection, all with batchnorm and leakyrelu.

    def forward_one(self, x):
        tnf_features = x[:, : self.tnf_dim]
        cov_features = x[:, self.tnf_dim :]
        cov_encode = F.normalize(self.coverage_encoder(cov_features), dim=-1)
        fused_input = torch.cat([tnf_features, cov_encode], dim=-1)
        fused_encode = self.fuse_encoder(fused_input)
        return fused_encode

    def get_embedding(self, x):
        return self.forward_one(x)
