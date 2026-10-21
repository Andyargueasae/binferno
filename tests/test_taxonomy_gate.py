import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from binferno.training.models.simclr import RankTaxonomyEncoder, SimCLRNet
from binferno.training.models.barlow_twins import NeuralNetwork
from binferno.training.models.new_htax_encoder import HierarchicalTaxonomyEncoder
from binferno.training.models.taxonomy_gate import (
    bounded_taxonomy_residual,
    validate_rank_taxonomy_table,
)


def test_gate_bounds_taxonomy_and_ignores_missing_lineage():
    base = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    taxonomy = torch.tensor([[0.0, 1.0]] * 3)
    features = torch.tensor([
        [1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 1],
    ], dtype=torch.float32)
    output = bounded_taxonomy_residual(base, taxonomy, features, 0.2)
    assert torch.allclose(output[1], F.normalize(base[1], dim=0))
    assert torch.allclose(output[0], F.normalize(base[0] + 0.2 * taxonomy[0], dim=0))
    assert output[2, 1] < output[0, 1]
    assert torch.allclose(
        bounded_taxonomy_residual(base, taxonomy, features, 0),
        F.normalize(base, dim=1),
    )


def test_barlow_gate_uses_alpha_and_preserves_projector_shape():
    torch.manual_seed(1)
    encoder = HierarchicalTaxonomyEncoder([2, 2], embedding_dim=4, output_dim=8)
    model = NeuralNetwork(
        tnf_dim=4, coverage_dim=2, kmer_dim=4, output_dim=8,
        proj_head_dim=16, tax_model=encoder, alpha_tax=0.1,
    ).eval()
    features = torch.randn(4, 6)
    taxonomy = torch.tensor([[1, 1, 1, 1], [0, 0, 0, 0]] * 2).float()
    with torch.no_grad():
        gated = model.get_embedding(features, taxonomy)
        model.alpha_tax = 0
        base = model.get_embedding(features, taxonomy)
        projected = model.forward_one(features, taxonomy)
    assert gated.shape == (4, 8)
    assert projected.shape == (4, 16)
    assert torch.allclose(gated[1], base[1])
    assert not torch.allclose(gated[0], base[0])


def test_simclr_gate_preserves_auxiliary_outputs():
    torch.manual_seed(1)
    cov = torch.nn.Linear(2, 8)
    tax = RankTaxonomyEncoder([2, 2], out_sz=8, embedding_dim=4, dropout=0)
    model = SimCLRNet(
        in_sz=12, out_sz=8, emb_szs=[16, 16], ps=0,
        cov_model=cov, tax_model=tax, alpha_tax=0.1,
    ).eval()
    composition = torch.randn(4, 4)
    coverage = torch.randn(4, 2)
    taxonomy = torch.tensor([[1, 1, 1, 1], [0, 0, 0, 0]] * 2).float()
    with torch.no_grad():
        gated, cov_out, tax_out = model(composition, coverage, taxonomy)
        model.alpha_tax = 0
        base, _, _ = model(composition, coverage, taxonomy)
    assert gated.shape == cov_out.shape == tax_out.shape == (4, 8)
    assert torch.allclose(gated[1], base[1])
    assert not torch.allclose(gated[0], base[0])


def test_taxonomy_validation_runs_before_embedding_lookup():
    table = pd.DataFrame({
        "tax_domain": [0, 1],
        "tax_species": [0, 3],
        "mask_domain": [0, 1],
        "mask_species": [0, 1],
    })
    ids = ["tax_domain", "tax_species"]
    masks = ["mask_domain", "mask_species"]
    assert validate_rank_taxonomy_table(table, ids, masks) == [2, 4]

    invalid = table.copy()
    invalid.loc[1, "tax_species"] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        validate_rank_taxonomy_table(invalid, ids, masks)

    invalid = table.astype(float)
    invalid.loc[1, "tax_species"] = 1.5
    with pytest.raises(ValueError, match="integers"):
        validate_rank_taxonomy_table(invalid, ids, masks)

    invalid = table.astype(float)
    invalid.loc[1, "mask_species"] = 0.5
    with pytest.raises(ValueError, match="0 or 1"):
        validate_rank_taxonomy_table(invalid, ids, masks)
