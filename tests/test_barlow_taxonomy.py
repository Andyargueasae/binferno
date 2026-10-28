import numpy as np
import pandas as pd
import pytest
import torch

from binferno.training.barlowtwins import BarlowTwinsTrainer
from binferno.training.models.barlow_twins import NeuralNetwork
from binferno.training.models.new_htax_encoder import HierarchicalTaxonomyEncoder


def _taxonomy_features(index):
    return pd.DataFrame(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1, 0],
            [1, 2, 1, 1, 1, 1],
            [1, 2, 0, 1, 1, 0],
        ],
        index=index,
        columns=[
            "tax_domain",
            "tax_phylum",
            "tax_species",
            "mask_domain",
            "mask_phylum",
            "mask_species",
        ],
        dtype=np.float32,
    )


def _taxonomy_model():
    encoder = HierarchicalTaxonomyEncoder(
        vocab_sizes=[2, 3, 2], embedding_dim=4, output_dim=8
    )
    return NeuralNetwork(
        tnf_dim=4,
        coverage_dim=2,
        kmer_dim=4,
        output_dim=8,
        proj_head_dim=16,
        tax_emb_dim=6,
        tax_model=encoder,
    )


def test_hierarchical_taxonomy_encoder_accepts_combined_or_split_features():
    encoder = HierarchicalTaxonomyEncoder(
        vocab_sizes=[2, 3, 2], embedding_dim=4, output_dim=8
    ).eval()
    features = torch.tensor(
        [[1, 2, 1, 1, 1, 1], [1, 0, 0, 1, 0, 0]], dtype=torch.float32
    )

    combined = encoder(features)
    split = encoder(features[:, :3], features[:, 3:])

    assert combined.shape == (2, 8)
    assert torch.allclose(combined, split)


def test_barlow_rank_taxonomy_is_opt_in():
    canonical_model = NeuralNetwork(
        tnf_dim=4,
        coverage_dim=2,
        kmer_dim=4,
        output_dim=8,
        proj_head_dim=16,
    ).eval()
    taxonomy_model = _taxonomy_model().eval()
    features = torch.randn(4, 6)
    taxonomy = torch.tensor(
        [[1, 1, 1, 1, 1, 1]] * 4, dtype=torch.float32
    )

    canonical_output = canonical_model.get_embedding(features)
    taxonomy_output = taxonomy_model.get_embedding(features, taxonomy)

    assert canonical_model.uses_rank_taxonomy is False
    assert canonical_output.shape == taxonomy_output.shape == (4, 8)
    with pytest.raises(ValueError, match="Taxonomy features are required"):
        taxonomy_model.get_embedding(features)


def test_barlow_taxonomy_auxiliary_loss_updates_canonical_encoder():
    trainer = BarlowTwinsTrainer(lambda_taxonomy=0.25)
    model = _taxonomy_model().train()
    criterion = trainer.BT_loss(lambda_bt=1e-3)
    features1 = torch.randn(4, 6)
    features2 = torch.randn(4, 6)
    taxonomy = torch.as_tensor(
        _taxonomy_features(["a", "b", "c", "d"]).to_numpy(),
        dtype=torch.float32,
    )

    loss, stats = trainer.calculate_loss(
        criterion,
        model,
        features1,
        features2,
        tax_1=taxonomy,
        tax_2=taxonomy,
        last_batch=True,
        w=torch.ones(4),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert stats["taxonomy_loss"] > 0.0
    assert stats["weighted_taxonomy_loss"] == pytest.approx(
        0.25 * stats["taxonomy_loss"]
    )
    assert all(head.weight.grad is not None for head in model.taxonomy_classifiers)
    assert model.tnf_encoder[0].weight.grad is not None


def test_barlow_taxonomy_auxiliary_loss_ignores_unknown_ranks():
    trainer = BarlowTwinsTrainer()
    model = _taxonomy_model().eval()
    canonical = torch.randn(4, 8)
    taxonomy = torch.zeros(4, 6)

    loss = trainer.rank_taxonomy_loss(model, canonical, taxonomy)

    assert loss.item() == 0.0


def test_barlow_dataset_and_embeddings_keep_taxonomy_separate(tmp_path):
    index = ["c1.original", "c1.1", "c2.original", "c2.1"]
    canonical = pd.DataFrame(
        np.random.default_rng(42).normal(size=(4, 6)).astype(np.float32),
        index=index,
    )
    taxonomy = _taxonomy_features(index)
    trainer = BarlowTwinsTrainer()
    dataset = trainer.BuildDataset(
        data_df=canonical,
        taxonomy_features=taxonomy,
        max_positive_pairs=10,
    ).construct_tensordatasets_from_pairs()

    first_batch = dataset[0]
    assert len(first_batch) == 6
    assert first_batch[0].shape == (6,)
    assert first_batch[2].shape == (6,)

    embeddings = trainer.get_embeddings(
        model=_taxonomy_model(),
        data_df=canonical,
        taxonomy_features=taxonomy,
        output_path=tmp_path / "embeddings.tsv",
        batch_size=2,
    )

    assert embeddings.index.tolist() == ["c1", "c2"]
    assert embeddings.shape == (2, 8)
    assert np.isfinite(embeddings.to_numpy()).all()
