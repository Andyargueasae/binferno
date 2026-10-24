import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from typing import Optional
from .taxonomy_gate import bounded_taxonomy_residual
from ..protein_batch import unpack_features


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
        alpha_tax: float = 0.05,
        protein_encoder=None,
    ):
        super(SimCLRNet, self).__init__()
        self.protein_encoder = protein_encoder
        self.pretrained_model = pretrained_model
        self.cov_model = cov_model
        self.tax_model = tax_model
        self.in_sz = in_sz
        self.out_sz = out_sz
        self.n_embs = len(emb_szs) - 1
        self.covmodel_notl2normalize = covmodel_notl2normalize
        if not 0.0 <= alpha_tax <= 0.2:
            raise ValueError("alpha_tax must be between 0 and 0.2.")
        self.alpha_tax = alpha_tax
 
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

    def forward(self, x, x2=None, x3=None, contig_ids=None, view_ids=None):
        x, protein_ids, view_ids = unpack_features(x, contig_ids, view_ids)
        x2, _, _ = unpack_features(x2)
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
        output = self.fc(x)

        if self.tax_model is not None:
            tax_output = self.tax_model(x3)
            output = bounded_taxonomy_residual(
                output, tax_output, x3, self.alpha_tax
            )

        if self.protein_encoder is not None:
            output = self.protein_encoder.fuse(output, protein_ids, view_ids)

        if self.cov_model is not None:
            if self.pretrained_model is not None:
                return output, self.cov_model(x2), kmeremb
            elif self.pretrained_model is None and self.tax_model is not None:
                return output, self.cov_model(x2), tax_output
            return output, self.cov_model(x2)
        else:
            return output  # F.normalize(output) #output #/ torch.linalg.vector_norm(output, dim=-1, keepdim=True)
