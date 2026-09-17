import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


# Shared COMEBin-style block builder (kept behavior-identical).
def _bn_drop_lin(
    n_in: int,
    n_out: int,
    bn: bool = True,
    p: float = 0.0,
    actn: Optional[nn.Module] = None,
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


def _bn_drop_cnn(
    n_in: int,
    n_out: int,
    bn: bool = True,
    p: float = 0.0,
    actn: Optional[nn.Module] = None,
):
    layers = [nn.BatchNorm1d(n_in)] if bn else []
    if p != 0:
        layers.append(nn.Dropout(p))
    layers.append(nn.Conv1d(n_in, n_out, kernel_size=3, padding=1))
    if actn is not None:
        layers.append(actn)
    return layers


def _bn_drop_lstm(
    n_in: int,
    n_out: int,
    bn: bool = True,
    p: float = 0.0,
    actn: Optional[nn.Module] = None,
):
    layers = [nn.BatchNorm1d(n_in)] if bn else []
    if p != 0:
        layers.append(nn.Dropout(p))
    layers.append(nn.LSTM(n_in, n_out, batch_first=True))
    if actn is not None:
        layers.append(actn)
    return layers


# Shared COMEBin MLP stack builder for EmbeddingNet/SimCLRNet.
def _build_embedding_stack(
    in_sz,
    out_sz,
    emb_szs,
    ps,
    use_bn=True,
    actn=nn.ReLU(),
):
    n_embs = len(emb_szs) - 1
    if ps == 0:
        ps = np.zeros(n_embs)

    # input layer
    layers = [nn.Linear(in_sz, emb_szs[0]), actn]

    # hidden layers
    for i in range(n_embs):
        layers += _bn_drop_lin(
            n_in=emb_szs[i],
            n_out=emb_szs[i + 1],
            bn=use_bn,
            p=ps[i],
            actn=actn,
        )

    # output layer
    layers.append(nn.Linear(emb_szs[-1], out_sz))
    return nn.Sequential(*layers)


class CNNEmbeddingStack(nn.Module):
    """CNN alternative with the same [B, D] -> [B, out_sz] contract as the MLP."""

    def __init__(self, in_sz, out_sz, ps=0.0, actn=None):
        super().__init__()
        p = float(ps if isinstance(ps, (int, float)) else (ps[0] if len(ps) else 0.0))
        # Capacity-matched to the COMEBin SimCLR MLP used with
        # emb_szs=[2048, 2048, 2048] (~8.94M parameters for 136 -> 128).
        # Global pooling prevents the parameter explosion that would occur
        # if all 136 spatial positions were flattened before projection.
        self.net = nn.Sequential(
            nn.Conv1d(1, 1024, kernel_size=3, padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU() if actn is None else actn,
            nn.Dropout(p),
            nn.Conv1d(1024, 1280, kernel_size=3, padding=1),
            nn.BatchNorm1d(1280),
            nn.ReLU() if actn is None else actn,
            nn.Dropout(p),
            nn.Conv1d(1280, 1280, kernel_size=3, padding=1),
            nn.BatchNorm1d(1280),
            nn.ReLU() if actn is None else actn,
            nn.AdaptiveAvgPool1d(1),
        )
        self.out = nn.Linear(1280, out_sz)

    def forward(self, x):
        if x.ndim != 2:
            raise ValueError(f"CNNEmbeddingStack expects [B, D], got {tuple(x.shape)}")
        x = self.net(x.unsqueeze(1)).squeeze(-1)
        return self.out(x)


class LSTMEmbeddingStack(nn.Module):
    """LSTM alternative with the same [B, D] -> [B, out_sz] contract as the MLP."""

    def __init__(self, in_sz, out_sz, ps=0.0):
        super().__init__()
        p = float(ps if isinstance(ps, (int, float)) else (ps[0] if len(ps) else 0.0))
        # hidden_size=864 gives ~9.09M parameters for a 2-layer LSTM
        # plus the final 864 -> out_sz projection, closely matching the
        # ~8.94M-parameter 3x2048 SimCLR MLP baseline.
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=864,
            num_layers=2,
            batch_first=True,
            dropout=p,
        )
        self.out = nn.Linear(864, out_sz)

    def forward(self, x):
        if x.ndim != 2:
            raise ValueError(f"LSTMEmbeddingStack expects [B, D], got {tuple(x.shape)}")
        _, (h_n, _) = self.lstm(x.unsqueeze(-1))
        return self.out(h_n[-1])


def build_embedding_stack_cnn(
    in_sz,
    out_sz,
    emb_szs,
    ps,
    use_bn=True,
    actn=nn.ReLU(),
):
    # emb_szs/use_bn are kept in the signature so this remains a drop-in
    # builder alongside _build_embedding_stack.
    return CNNEmbeddingStack(in_sz=in_sz, out_sz=out_sz, ps=ps, actn=actn)


def build_embedding_stack_lstm(
    in_sz,
    out_sz,
    emb_szs,
    ps,
    use_bn=True,
    actn=nn.ReLU(),
):
    # LSTM is implemented as a dedicated module rather than nn.Sequential,
    # because nn.LSTM returns (output, (h_n, c_n)).
    return LSTMEmbeddingStack(in_sz=in_sz, out_sz=out_sz, ps=ps)


# The raw COMEBin's embedding network is a simple MLP with batch normalization and dropout. We can use it as a baseline model for our experiments. We can also implement a more complex fusion network later to see if we can improve the performance.
class EmbeddingNet(nn.Module):
    # Useful code from fast.ai tabular model
    # https://github.com/fastai/fastai/blob/3b7c453cfa3845c6ffc496dd4043c07f3919270e/fastai/tabular/models.py#L6
    def __init__(self, in_sz, out_sz, emb_szs, ps, use_bn=True, actn=nn.ReLU()):
        super(EmbeddingNet, self).__init__()
        self.in_sz = in_sz
        self.out_sz = out_sz
        self.n_embs = len(emb_szs) - 1
        self.fc = _build_embedding_stack(
            in_sz=self.in_sz,
            out_sz=self.out_sz,
            emb_szs=emb_szs,
            ps=ps,
            use_bn=use_bn,
            actn=actn,
        )

    def bn_drop_lin(
        self,
        n_in: int,
        n_out: int,
        bn: bool = True,
        p: float = 0.0,
        actn: nn.Module = None,
    ):
        return _bn_drop_lin(n_in=n_in, n_out=n_out, bn=bn, p=p, actn=actn)

    def forward(self, x):
        output = self.fc(x)
        return output  # F.normalize(output) #output #/ torch.linalg.vector_norm(output, dim=-1, keepdim=True)


class RankTaxonomyEncoder(nn.Module):
    """Encode rank-specific categorical IDs and their missing-value masks."""

    def __init__(self, vocab_sizes, out_sz=128, embedding_dim=16, dropout=0.2):
        super().__init__()
        if not vocab_sizes or any(size < 1 for size in vocab_sizes):
            raise ValueError("Taxonomy vocabulary sizes must be positive.")
        self.n_ranks = len(vocab_sizes)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(size, embedding_dim, padding_idx=0) for size in vocab_sizes]
        )
        self.projection = nn.Sequential(
            nn.Linear(self.n_ranks * embedding_dim + self.n_ranks, 256),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, out_sz),
        )

    def forward(self, features):
        expected = self.n_ranks * 2
        if features.ndim != 2 or features.shape[1] != expected:
            raise ValueError(
                f"Expected taxonomy tensor [B, {expected}], got {tuple(features.shape)}"
            )
        taxon_ids = features[:, : self.n_ranks].long()
        masks = features[:, self.n_ranks :].float().clamp(0.0, 1.0)
        encoded = []
        for rank, embedding in enumerate(self.embeddings):
            ids = taxon_ids[:, rank]
            if ids.numel() and (ids.min() < 0 or ids.max() >= embedding.num_embeddings):
                raise ValueError(f"Taxonomy ID is out of range at rank {rank}.")
            encoded.append(embedding(ids) * masks[:, rank : rank + 1])
        return self.projection(torch.cat([*encoded, masks], dim=1))


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
        tax_model=None,
        covmodel_notl2normalize=False,
        cnn=False,
        lstm=False,
        get_taxout=False,
    ):
        super(SimCLRNet, self).__init__()
        self.pretrained_model = pretrained_model
        self.cov_model = cov_model
        self.tax_model = tax_model
        self.in_sz = in_sz
        self.out_sz = out_sz
        self.n_embs = len(emb_szs) - 1
        self.covmodel_notl2normalize = covmodel_notl2normalize

        assert not (
            cnn is True and lstm is True
        ), "Only one of cnn or lstm can be True."

        if not cnn and not lstm:
            self.fc = _build_embedding_stack(
                in_sz=self.in_sz,
                out_sz=self.out_sz,
                emb_szs=emb_szs,
                ps=ps,
                use_bn=use_bn,
                actn=actn,
            )
        elif cnn:
            self.fc = build_embedding_stack_cnn(
                in_sz=self.in_sz,
                out_sz=self.out_sz,
                emb_szs=emb_szs,
                ps=ps,
                use_bn=use_bn,
                actn=actn,
            )
        elif lstm:
            self.fc = build_embedding_stack_lstm(
                in_sz=self.in_sz,
                out_sz=self.out_sz,
                emb_szs=emb_szs,
                ps=ps,
                use_bn=use_bn,
                actn=actn,
            )

    def bn_drop_lin(
        self,
        n_in: int,
        n_out: int,
        bn: bool = True,
        p: float = 0.0,
        actn: nn.Module = None,
    ):
        return _bn_drop_lin(n_in=n_in, n_out=n_out, bn=bn, p=p, actn=actn)

    def forward(self, x, x2=None, x3=None):
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

        if self.tax_model is not None:
            if x3 is None:
                raise ValueError(
                    "Taxonomic features are required when SimCLRNet has a tax_model."
                )
            x = torch.cat([x, F.normalize(self.tax_model(x3))], dim=-1)

        output = self.fc(x)

        if self.cov_model is not None:
            if self.pretrained_model is not None:
                return output, self.cov_model(x2), kmeremb
            elif self.pretrained_model is None and self.tax_model is not None:
                return output, self.cov_model(x2), self.tax_model(x3)
        else:
            return output  # F.normalize(output) #output #/ torch.linalg.vector_norm(output, dim=-1, keepdim=True)
