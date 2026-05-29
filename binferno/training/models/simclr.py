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
        n_layers=3,
        kmer_dim=136,
        corgi_dim=0,
        proj_head_dim=128,
        proj=False,
        gated=False,
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
        
        if corgi_dim > 0:
            self.tnf_enc_dim += corgi_dim  # update tnf_enc_dim to include corgi features.
            # tnf_encoder can then take concatenated tnf features and corgi features.
            self.tnf_encoder = nn.Sequential(
                nn.Linear(self.tnf_enc_dim, 512),
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
            )
        else:  # tnf->256->128
            self.tnf_encoder = nn.Sequential(
                nn.Linear(self.tnf_enc_dim, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.LeakyReLU(),  # the forgoten leakyrelu after batchnorm can cause the representation to have a lot of zeros, which can lead to worse performance. Adding it back can improve the performance.
            )

        # set up corgi encoder if corgi_dim > 0.
        # apply adaptive coverage dim choices.
        n_samples = max(
            1, coverage_dim // 2
        )  # one coverage mean and one coverage std, divide by 2.

        cov_hidden1 = 256
        cov_hidden2 = 128

        self.coverage_encoder = nn.Sequential(
            nn.Linear(coverage_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, cov_hidden1),
            nn.BatchNorm1d(cov_hidden1),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(cov_hidden1, cov_hidden2),
            nn.BatchNorm1d(cov_hidden2),
            nn.LeakyReLU(),
        )

        self.final_cov_dim = cov_hidden2  # 128?

        # leave fusion network to be implemented later.
        self.fusion_network = fusion_network(
            emb_dim=output_dim,
            gated=gated,
        )
        self.representation_dim = output_dim
        self.proj_head = nn.Sequential(
            nn.Linear(self.representation_dim, self.proj_head_dim),
            nn.BatchNorm1d(self.proj_head_dim),
            nn.ReLU(),
            nn.Linear(self.proj_head_dim, self.proj_head_dim),
        )

    def _encode(self, x):
        # encode tnf and coverage features separately, then fuse them together.
        tnf_features = x[:, : self.tnf_enc_dim]
        tnf_encode = F.normalize(self.tnf_encoder(tnf_features), dim=-1)
        cov_features = x[:, self.tnf_enc_dim :]
        cov_encode = F.normalize(self.coverage_encoder(cov_features), dim=-1)
        fused_encode = self.fusion_network(tnf_encode, cov_encode)  # 128-dim fused representation.
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
        if not self.proj:
            return fused_emb
        proj_emb = self.proj_head(fused_emb)  # 512
        return proj_emb

    def forward(self, x1, x2):
        z1 = self.forward_one(x1)
        z2 = self.forward_one(x2)
        return z1, z2


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

    def forward(self, kmer_emb, cov_emb):
        # kmer_emb = F.normalize(kmer_emb, dim=-1)
        # cov_emb = F.normalize(cov_emb, dim=-1)

        tokens = torch.stack([kmer_emb, cov_emb], dim=1)
        attended, _ = self.self_attn(tokens, tokens, tokens)

        fused = attended.flatten(1)
        fused = self.fused_proj(fused)

        return fused

def fusion_network(
    emb_dim=128,
    gated=False,
):
    if gated:
        return gated_fusion(emb_dim=emb_dim)
    return AttentionFusion(
        emb_dim=emb_dim,
        num_heads=4,
        dropout=0.1,
    )


class gated_fusion(nn.Module):
    def __init__(self, emb_dim):
        super(gated_fusion, self).__init__()
        self.emb_dim = emb_dim
        self.gate = nn.Sequential(
            nn.LayerNorm(self.emb_dim * 2),
            # LayerNorm can help stabilize the training of the gating mechanism,
            # especially when the input features have different distributions.
            # It can also help mitigate the issue of one modality dominating the
            # other by normalizing the concatenated features before applying the
            # linear transformation and sigmoid activation.
            nn.Linear(self.emb_dim * 2, self.emb_dim),
            nn.Sigmoid(),
        )
        # symmetric gating for both modalities, can be easily extended to
        # asymmetric gating if needed. The gate will learn to weigh the
        # importance of each modality for the final fused representation,
        # allowing the model to adaptively focus on the more informative
        # modality for each sample.
        self.out_norm = nn.LayerNorm(self.emb_dim)

    def forward(self, kmer_emb, coverage_emb):
        gate_weight = self.gate(torch.cat([kmer_emb, coverage_emb], dim=1))
        fused = gate_weight * kmer_emb + (1 - gate_weight) * coverage_emb
        return self.out_norm(fused)


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
