from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from binferno.training import esmc, scg
from binferno.training.barlowtwins import BarlowTwinsTrainer
from binferno.training.simclr import SimCLRTrainer
from binferno.training.models.barlow_twins import NeuralNetwork as BarlowNetwork
from binferno.training.models.old_simclr import NeuralNetwork as LegacyNetwork
from binferno.training.models.simclr import SimCLRNet
from binferno.training.InfoNCEloss import info_nce_loss


@pytest.fixture
def protein_table():
    proteins = np.random.default_rng(3).normal(size=(6, 8)).astype(np.float32)
    # Four contigs: three, one, zero and two proteins respectively.
    return proteins, np.array([0, 3, 4, 4, 6])


class _FakeESMC:
    def tokenizer(self, sequences, **kwargs):
        # Special tokens and padding deliberately dominate residue values.
        tokens = torch.full((len(sequences), max(map(len, sequences)) + 2), 1000)
        for row, sequence in enumerate(sequences):
            tokens[row, 1 : len(sequence) + 1] = torch.tensor(
                [ord(aa) - ord('A') + 1 for aa in sequence]
            )
        return {"input_ids": tokens}

    def __call__(self, sequence_tokens):
        return SimpleNamespace(embeddings=sequence_tokens[..., None].float().expand(-1, -1, 8))


def test_extraction_masks_special_tokens_padding_and_weights_long_chunks(monkeypatch):
    monkeypatch.setattr(esmc, '_load_model', lambda *args: _FakeESMC())
    config = esmc.ESMCConfig(batch_size=2, max_residues=2)
    result = esmc._embed_proteins(['AAAAC', 'BC'], config, torch.device('cpu'))
    np.testing.assert_allclose(result[0], [1.4] * 8)
    np.testing.assert_allclose(result[1], [2.5] * 8)


def test_annotation_and_embedding_cache_reuse(tmp_path, monkeypatch):
    contigs = tmp_path / 'contigs.fna'
    contigs.write_text('>a.dot\nATGC\n>b\nATGC\n>empty\nATGC\n')
    calls = []

    def predict(contig_file, faa_path, threads):
        calls.append('predict')
        faa_path.write_text('>a.dot_1\nAC*\n>a.dot_2\nBB\n>b_1\nC\n')

    monkeypatch.setattr(scg, 'run_pyrodigal', predict)
    monkeypatch.setattr(esmc, '_load_model', lambda *args: _FakeESMC())
    config = esmc.ESMCConfig(enabled=True, device='cpu')
    proteins, offsets = esmc.build_protein_latents(
        contigs, ['b', 'empty', 'a.dot'], tmp_path / 'scg', 1, config
    )
    assert calls == ['predict']
    np.testing.assert_array_equal(offsets, [0, 1, 1, 3])
    np.testing.assert_allclose(proteins[:, 0], [3, 2, 2])
    assert scg.prepare_gene_annotations(contigs, tmp_path / 'scg', 1).exists()
    assert calls == ['predict']
    monkeypatch.setattr(esmc, '_load_model', lambda *args: pytest.fail('cache should skip model loading'))
    cached, _ = esmc.build_protein_latents(
        contigs, ['b', 'empty', 'a.dot'], tmp_path / 'scg', 1, config
    )
    np.testing.assert_array_equal(cached, proteins)


def test_empty_annotation_needs_no_esm(tmp_path, monkeypatch):
    contigs = tmp_path / 'empty.fna'
    contigs.write_text('>empty\nATGC\n')
    monkeypatch.setattr(scg, 'run_pyrodigal', lambda contig_file, faa_path, threads: faa_path.touch())
    monkeypatch.setattr(esmc, '_load_model', lambda *args: pytest.fail('no proteins to embed'))
    proteins, offsets = esmc.build_protein_latents(
        contigs, ['empty'], tmp_path / 'scg', 1, esmc.ESMCConfig(enabled=True)
    )
    assert proteins.shape == (0, 960)
    np.testing.assert_array_equal(offsets, [0, 0])


def test_view_alignment_including_long_contig_halves(monkeypatch, protein_table):
    monkeypatch.setattr(esmc, 'build_protein_latents', lambda *args: protein_table)
    names = ['a.dot', 'b', 'empty', 'd']
    df = pd.DataFrame(np.zeros((4, 2)), index=['a.dot.h2.0', 'b.original', 'a.dot.h1.1', 'a.dot.original'])
    fragments = {'a.dot': {'fragment_info': {'a.dot.h2.0': {}, 'a.dot.h1.1': {}}}}
    result, encoder = esmc.prepare_esmc_features(
        df, None, names, None, 1, esmc.ESMCConfig(enabled=True), 8,
        fragment_dict=fragments,
    )
    assert result.index.equals(df.index)
    np.testing.assert_array_equal(result['__esmc_contig_id'], [0, 1, 0, 0])
    assert encoder is not None
    assert df.shape[1] == 2


@pytest.mark.parametrize('pooling', ['mean', 'attention'])
def test_pooling_masks_missing_proteins_and_is_permutation_invariant(pooling, protein_table):
    proteins, offsets = protein_table
    config = esmc.ESMCConfig(pooling=pooling, num_heads=2)
    encoder = esmc.ContigProteinEncoder(proteins, offsets, 8, config).eval()
    permuted = esmc.ContigProteinEncoder(proteins[[2, 0, 1, 3, 5, 4]], offsets, 8, config).eval()
    permuted.load_state_dict(encoder.state_dict())
    ids = torch.tensor([3, 0, 2, 1, 0])
    torch.testing.assert_close(encoder(ids), permuted(ids))
    canonical = torch.randn(5, 8)
    torch.testing.assert_close(encoder.fuse(canonical, ids)[2], canonical[2])
    assert torch.isfinite(encoder(ids)).all()
    torch.testing.assert_close(encoder(ids)[1], encoder(ids)[4])


@pytest.mark.parametrize('pooling', ['mean', 'attention'])
@pytest.mark.parametrize('branch', ['barlow', 'simclrnet', 'legacy'])
def test_existing_branch_loss_trains_protein_pooling(branch, pooling, protein_table):
    proteins, offsets = protein_table
    config = esmc.ESMCConfig(pooling=pooling, num_heads=2)
    output_dim = 256 if branch == 'legacy' else 8
    encoder = esmc.ContigProteinEncoder(proteins, offsets, output_dim, config)
    features = torch.randn(8, 6)
    ids = torch.arange(4).repeat(2).float()[:, None]
    augmented = torch.cat([features, ids], dim=1)
    if branch == 'barlow':
        model = BarlowNetwork(tnf_dim=4, coverage_dim=2, kmer_dim=4,
                              output_dim=8, proj_head_dim=8, protein_encoder=encoder)
        first, second = model(augmented[:4], augmented[4:])
        loss = BarlowTwinsTrainer.BT_loss()(first, second)
        clustering = model.get_embedding(augmented)
    elif branch == 'simclrnet':
        model = SimCLRNet(in_sz=12, out_sz=8, emb_szs=[16, 16], ps=[0.0],
                          cov_model=nn.Linear(2, 8), protein_encoder=encoder)
        clustering = model(augmented[:, 2:], augmented[:, :2])[0]
        loss, _ = info_nce_loss(temperature=0.15)(
            clustering, batch_size=4, device=torch.device('cpu'), n_views=2
        )
    else:
        model = LegacyNetwork(tnf_dim=4, coverage_dim=2, kmer_dim=4, output_dim=8,
                              attention=True, protein_encoder=encoder)
        projected, _, _ = model.forward_one(augmented)
        loss, _ = info_nce_loss(temperature=0.15)(
            projected, batch_size=4, device=torch.device('cpu'), n_views=2
        )
        clustering = model.get_embedding(augmented)
    assert torch.isfinite(loss)
    loss.backward()
    assert encoder.projection.weight.grad is not None
    assert encoder.projection.weight.grad.norm() > 0
    if pooling == 'attention':
        assert encoder.attention.in_proj_weight.grad.norm() > 0
        assert encoder.query.grad.norm() > 0
    assert clustering.shape == (8, output_dim)
    # Protein inputs affect the exported representation too.
    model.eval()
    changed = augmented.clone()
    changed[:, -1] = 1
    with torch.no_grad():
        if branch == 'simclrnet':
            before = model(augmented[:, 2:], augmented[:, :2])[0]
            after = model(changed[:, 2:], changed[:, :2])[0]
        else:
            before, after = model.get_embedding(augmented), model.get_embedding(changed)
    assert not torch.allclose(before[0], after[0])


def test_esmc_disabled_by_default_and_nested_config_available():
    assert not BarlowTwinsTrainer().esmc.enabled
    assert not SimCLRTrainer().esmc.enabled
    trainer = SimCLRTrainer(esmc={'enabled': True, 'pooling': 'attention'})
    assert trainer.esmc.enabled and trainer.esmc.pooling == 'attention'
