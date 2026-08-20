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