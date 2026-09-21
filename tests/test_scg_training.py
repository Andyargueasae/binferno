import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from binferno.training.scg import SCGMarginLoss, prepare_scg_loader
from binferno.training.simclr import SimCLRTrainer
from binferno.training.models.COMEBin_models import SimCLRNet


@pytest.mark.parametrize("suffix", [".0", ".original"])
def test_scg_loader_is_augmentation_agnostic(suffix):
    index = [f"contig_{i}{suffix}" for i in range(3)]
    features = pd.DataFrame(np.arange(12).reshape(3, 4), index=index)
    taxonomy = pd.DataFrame(np.arange(6).reshape(3, 2), index=index)
    loader = prepare_scg_loader(
        data_df=features,
        auxiliary_df=taxonomy,
        marker_groups={"marker_a": ["contig_0", "contig_1", "contig_2"]},
        original_suffix=suffix,
        batch_size=8,
        seed=42,
    )

    batch = next(iter(loader))
    assert len(loader.dataset) == 3
    assert len(batch) == 8
    assert batch[0].shape[1] == features.shape[1]
    assert batch[2].shape[1] == taxonomy.shape[1]


def test_scg_margin_loss_penalizes_only_pairs_above_margin():
    loss = SCGMarginLoss(margin=0.4)
    left = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    right = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    result = loss.forward_pairs(left, right)
    assert result.item() == pytest.approx((0.6**2) / 2)


def test_simclrnet_returns_clustering_representation_without_taxonomy():
    model = SimCLRNet(
        in_sz=4,
        out_sz=2,
        emb_szs=[4, 4],
        ps=[0.0],
        cov_model=nn.Linear(1, 2),
    ).eval()
    encoded = model(torch.randn(3, 2), torch.randn(3, 1))
    assert isinstance(encoded, tuple)
    assert encoded[0].shape == (3, 2)


class _SimpleRepresentation(nn.Module):
    def get_embedding(self, features):
        return features


class _SimCLRNetRepresentation(nn.Module):
    def forward(self, composition, coverage, taxonomy=None):
        pieces = [composition, coverage]
        if taxonomy is not None:
            pieces.append(taxonomy)
        return torch.cat(pieces, dim=1), None


@pytest.mark.parametrize("simclrnet", [False, True])
def test_simclr_scg_loss_uses_clustering_representation(simclrnet):
    trainer = SimCLRTrainer()
    features1 = torch.tensor([[1.0, 0.0, 1.0]])
    features2 = torch.tensor([[1.0, 0.0, 1.0]])
    taxonomy1 = torch.tensor([[1.0]])
    taxonomy2 = torch.tensor([[1.0]])
    batch = (
        features1,
        features2,
        taxonomy1,
        taxonomy2,
        torch.tensor([0]),
        torch.tensor([0]),
        torch.tensor([1]),
        torch.tensor([1.0]),
    )
    model = _SimCLRNetRepresentation() if simclrnet else _SimpleRepresentation()

    loss = trainer._scg_pair_loss(
        model=model,
        batch=batch,
        criterion=SCGMarginLoss(margin=0.4),
        device=torch.device("cpu"),
        n_cov=1,
        simclrnet=simclrnet,
    )
    assert loss.item() == pytest.approx(0.6**2)
