import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from binferno.training.barlowtwins import BarlowTwinsTrainer, PositivePairDataset
from binferno.training.scg import (
    SCGMarginLoss,
    SCGMarkerGroupDataset,
    prepare_scg_group_loader,
)


def test_positive_pairs_gather_from_one_feature_matrix():
    names = ["a.original", "a.1", "b.original", "b.1"]
    features = pd.DataFrame(
        np.arange(16, dtype=np.float32).reshape(4, 4), index=names
    )
    taxonomy = pd.DataFrame(
        np.arange(8, dtype=np.float32).reshape(4, 2), index=names
    )
    builder = BarlowTwinsTrainer.BuildDataset(
        data_df=features, taxonomy_features=taxonomy, max_positive_pairs=10
    )
    dataset = builder.construct_tensordatasets_from_pairs()
    assert isinstance(dataset, PositivePairDataset)
    assert dataset.features.shape == (4, 4)
    assert dataset.taxonomy.shape == (4, 2)
    assert dataset.pair_indices.shape == (2, 2)
    for sample, (left, right) in zip(dataset, dataset.pair_indices.tolist()):
        assert len(sample) == 6
        assert torch.equal(sample[0], dataset.features[left])
        assert torch.equal(sample[1], dataset.features[right])
        assert torch.equal(sample[2], dataset.taxonomy[left])
        assert torch.equal(sample[3], dataset.taxonomy[right])
    collated = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
    assert len(collated) == 6
    assert torch.equal(collated[0], dataset.features[dataset.pair_indices[:, 0].long()])


def test_scg_group_batches_store_no_pairs_and_weight_membership():
    names = [f"c{i}.original" for i in range(4)]
    features = pd.DataFrame(np.eye(4, dtype=np.float32), index=names)
    taxonomy = pd.DataFrame(np.arange(8).reshape(4, 2), index=names)
    loader = prepare_scg_group_loader(
        data_df=features,
        auxiliary_df=taxonomy,
        marker_groups={"a": ["c0", "c1", "c2"], "b": ["c2", "c3"]},
        original_suffix=".original",
        batch_size=3,
        seed=7,
    )
    assert isinstance(loader.dataset, SCGMarkerGroupDataset)
    assert loader.dataset.features.shape == (4, 4)
    assert not hasattr(loader.dataset, "pairs")
    batches = list(loader)
    assert len(batches) == 2
    for batch in batches:
        data, tax, marker_ids, contig_ids, weights = batch
        assert len(contig_ids) == len(set(contig_ids.tolist()))
        assert marker_ids.unique().numel() == 1
        assert torch.equal(data, loader.dataset.features[contig_ids])
        assert torch.equal(tax, loader.dataset.auxiliary_features[contig_ids])
        for contig_id, weight in zip(contig_ids.tolist(), weights.tolist()):
            assert weight == pytest.approx(0.5 if contig_id == 2 else 1.0)


def test_scg_matrix_loss_selects_hardest_same_marker_pairs():
    embeddings = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.8, 0.6], [-1.0, 0.0]],
        requires_grad=True,
    )
    markers = torch.zeros(4, dtype=torch.long)
    contigs = torch.arange(4)
    weights = torch.tensor([1.0, 1.0, 0.5, 1.0])
    loss = SCGMarginLoss(margin=0.4)(
        embeddings, markers, contigs, pair_weights=weights, max_hard_pairs=1
    )
    assert loss.item() == pytest.approx(0.6**2)
    loss.backward()
    assert embeddings.grad is not None

    # A shared contig in another marker must not create cross-marker pairs.
    split_markers = torch.tensor([0, 0, 1, 1])
    result = SCGMarginLoss(margin=0.4)(
        embeddings.detach(), split_markers, contigs
    )
    expected = (0.6**2 + 0.0) / 2
    assert result.item() == pytest.approx(expected)


def test_large_scg_group_samples_unique_contigs_reproducibly():
    names = [f"c{i}.original" for i in range(8)]
    features = pd.DataFrame(np.eye(8, dtype=np.float32), index=names)
    kwargs = dict(
        data_df=features,
        marker_groups={"m": [f"c{i}" for i in range(8)]},
        original_suffix=".original",
        batch_size=3,
        seed=11,
    )
    first = prepare_scg_group_loader(**kwargs)
    second = prepare_scg_group_loader(**kwargs)
    assert len(first.dataset) == 3
    first_ids = [batch[2].tolist() for batch in first]
    second_ids = [batch[2].tolist() for batch in second]
    assert first_ids == second_ids
    assert all(len(ids) == len(set(ids)) == 3 for ids in first_ids)
