from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from binferno.training import BinfernoTrainer, esmc, scg
from binferno.training.barlowtwins import BarlowTwinsTrainer
from binferno.training.simclr import SimCLRTrainer
from binferno.training.models.barlow_twins import NeuralNetwork as BarlowNetwork
from binferno.training.models.old_simclr import NeuralNetwork as LegacyNetwork
from binferno.training.models.simclr import SimCLRNet
from binferno.training.InfoNCEloss import info_nce_loss
from binferno.training.protein_batch import ProteinBatch, feature_tensor, cat_features


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
        df, None, names, None, 1, esmc.ESMCConfig(enabled=True, augmentation='shared'), 8,
        fragment_dict=fragments,
    )
    assert result.index.equals(df.index)
    np.testing.assert_array_equal(feature_tensor(result).contig_ids, [0, 1, 0, 0])
    assert result.columns.equals(df.columns)
    assert feature_tensor(result).view_ids.dtype == torch.long
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
    ids = torch.arange(4).repeat(2)
    augmented = ProteinBatch(features, ids, torch.zeros_like(ids))
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
    changed = ProteinBatch(features.clone(), torch.ones_like(ids), torch.zeros_like(ids))
    with torch.no_grad():
        if branch == 'simclrnet':
            before = model(augmented[:, 2:], augmented[:, :2])[0]
            after = model(changed[:, 2:], changed[:, :2])[0]
        else:
            before, after = model.get_embedding(augmented), model.get_embedding(changed)
    assert not torch.allclose(before[0], after[0])


def test_esmc_disabled_by_default_and_nested_config_available():
    trainers = dict(simclr=SimCLRTrainer(), barlowtwins=BarlowTwinsTrainer())
    assert not BinfernoTrainer(**trainers).esmc.enabled
    trainer = BinfernoTrainer(
        **trainers, esmc={'enabled': True, 'pooling': 'attention'}
    )
    assert trainer.esmc.enabled and trainer.esmc.pooling == 'attention'


@pytest.mark.parametrize('ids', [[3, 0, 2, 1, 0], [2, 2]])
@pytest.mark.parametrize('device', ['cpu', pytest.param(
    'cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
)])
def test_packed_attention_matches_padded_outputs_and_gradients(protein_table, ids, device):
    from copy import deepcopy
    from torch.nn.utils.rnn import pad_sequence

    proteins, offsets = protein_table
    encoder = esmc.ContigProteinEncoder(
        proteins, offsets, 8, esmc.ESMCConfig(pooling='attention', num_heads=2)
    ).to(device)
    reference = deepcopy(encoder)
    ids = torch.tensor(ids)
    unique, inverse = torch.unique(ids, return_inverse=True)
    lengths = np.diff(offsets)[unique.numpy()]
    sequences = [
        reference.proteins[offsets[index]:offsets[index + 1]]
        if length else reference.proteins.new_zeros((1, proteins.shape[1]))
        for index, length in zip(unique.tolist(), lengths)
    ]
    tokens = reference.projection(pad_sequence(sequences, batch_first=True).to(device))
    mask = torch.arange(tokens.shape[1], device=device)[None, :] >= torch.tensor(
        np.maximum(lengths, 1), device=device
    )[:, None]
    expected, _ = reference.attention(
        reference.query.expand(len(unique), -1, -1), tokens, tokens,
        key_padding_mask=mask, need_weights=False,
    )
    expected = (expected[:, 0] * torch.tensor(lengths > 0, device=device)[:, None])[inverse.to(device)]
    projected_shapes = []
    hook = encoder.projection.register_forward_pre_hook(
        lambda module, args: projected_shapes.append(args[0].shape)
    )
    actual = encoder(ids.to(device))
    hook.remove()
    # Duplicated views and padding must not add projection work.
    assert projected_shapes == [torch.Size((int(lengths.sum()), proteins.shape[1]))]
    torch.testing.assert_close(actual, expected)
    weights = torch.randn_like(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    for (name, parameter), (other_name, other) in zip(
        encoder.named_parameters(), reference.named_parameters()
    ):
        assert name == other_name
        torch.testing.assert_close(parameter.grad, other.grad, atol=1e-6, rtol=1e-5)


@pytest.fixture
def coordinate_table():
    # ORFs [0,9), [9,18), [18,27), [50000,50009), on both strands.
    return esmc.ProteinLatentTable(
        np.eye(4, dtype=np.float32), np.array([0, 4, 4]),
        np.array([0, 9, 18, 50000]), np.array([9, 18, 27, 50009]),
        np.array([1, -1, 1, -1]),
    )


@pytest.mark.parametrize('pooling', ['mean', 'attention'])
def test_coordinate_views_full_orfs_boundaries_and_empty(monkeypatch, coordinate_table, pooling):
    monkeypatch.setattr(esmc, 'build_protein_latents', lambda *args: coordinate_table)
    df = pd.DataFrame(np.zeros((7, 2)), index=[
        'a.dot.original', 'a.dot.0', 'a.dot.1', 'a.dot.h2.0',
        'a.dot.feature', 'a.dot.empty', 'empty.original',
    ])
    fragments = {'a.dot': {'fragment_info': {
        'a.dot.0': {'start_pos': 0, 'length': 9},
        'a.dot.1': {'start_pos': 8, 'length': 10},
        'a.dot.h2.0': {'start_pos': 50000, 'length': 9},
        'a.dot.feature': {'feature_only': True},
        'a.dot.empty': {'start_pos': 1, 'length': 7},
    }}}
    result, encoder = esmc.prepare_esmc_features(
        df, None, ['a.dot', 'empty'], None, 1,
        esmc.ESMCConfig(enabled=True, pooling=pooling, num_heads=2), 4,
        fragment_dict=fragments,
    )
    batch = feature_tensor(result)
    for row, expected in enumerate([[0, 1, 2, 3], [0], [1], [3], [0, 1, 2, 3], [], []]):
        pair = (batch.contig_ids[row].item(), batch.view_ids[row].item())
        np.testing.assert_array_equal(encoder.view_rows[pair], expected)
    canonical = torch.randn(7, 4)
    output = encoder(batch.contig_ids, batch.view_ids)
    assert not torch.allclose(output[1], output[2])
    torch.testing.assert_close(output[0], output[4])
    assert torch.equal(output[-2:], torch.zeros(2, 4))
    fused = encoder.fuse(canonical, batch.contig_ids, batch.view_ids)
    assert torch.equal(fused[-2:], canonical[-2:])
    output.square().sum().backward()
    assert encoder.projection.weight.grad.norm() > 0


def test_coordinate_cache_retains_prodigal_coordinates(tmp_path, monkeypatch):
    contigs = tmp_path / 'assembly.fna'
    contigs.write_text('>a\n' + 'A' * 100 + '\n')
    def predict(contig_file, faa_path, threads):
        faa_path.write_text('>a_1 # 2 # 10 # -1 # ID=1_1;partial=00\nAAA\n')
    monkeypatch.setattr(scg, 'run_pyrodigal', predict)
    monkeypatch.setattr(esmc, '_load_model', lambda *args: _FakeESMC())
    config = esmc.ESMCConfig(enabled=True, device='cpu')
    table = esmc.build_protein_latents(contigs, ['a'], tmp_path / 'scg', 1, config)
    np.testing.assert_array_equal(table.orf_starts, [1])
    np.testing.assert_array_equal(table.orf_ends, [10])
    np.testing.assert_array_equal(table.orf_strands, [-1])
    monkeypatch.setattr(esmc, '_embed_proteins', lambda *args: pytest.fail('cache must be reused'))
    cached = esmc.build_protein_latents(contigs, ['a'], tmp_path / 'scg', 1, config)
    for field in ('proteins', 'offsets', 'orf_starts', 'orf_ends', 'orf_strands'):
        np.testing.assert_array_equal(getattr(table, field), getattr(cached, field))


def test_coordinate_requires_known_intervals_and_coordinates(monkeypatch, coordinate_table):
    monkeypatch.setattr(esmc, 'build_protein_latents', lambda *args: coordinate_table)
    args = (pd.DataFrame([[0.]], index=['a.1']), None, ['a', 'empty'], None, 1,
            esmc.ESMCConfig(enabled=True), 4)
    with pytest.raises(ValueError, match='fragment metadata'):
        esmc.prepare_esmc_features(*args)
    coordinate_table.orf_starts[0] = -1
    with pytest.raises(ValueError, match='coordinate-bearing'):
        esmc.prepare_esmc_features(*args)


@pytest.mark.parametrize('pooling', ['mean', 'attention'])
def test_shared_residual_matches_original_formula_and_checkpoint(pooling, protein_table):
    from copy import deepcopy
    proteins, offsets = protein_table
    encoder = esmc.ContigProteinEncoder(
        proteins, offsets, 8, esmc.ESMCConfig(pooling=pooling, num_heads=2, augmentation='shared')
    ).eval()
    ids = torch.tensor([0, 1, 2, 3])
    protein = encoder(ids)
    canonical = torch.randn(4, 8)
    expected = canonical + encoder.weight * canonical.norm(dim=-1, keepdim=True).clamp_min(1e-6) * torch.nn.functional.normalize(protein, dim=-1)
    torch.testing.assert_close(encoder.fuse(canonical, ids), expected)
    reloaded = deepcopy(encoder)
    reloaded.load_state_dict(encoder.state_dict())
    torch.testing.assert_close(reloaded(ids), protein)
    # Shared mode ignores view identity, preserving the full pool for any view.
    torch.testing.assert_close(encoder(ids, torch.ones_like(ids)), protein)


@pytest.mark.parametrize('taxonomy', [False, True])
def test_simclr_coordinate_training_export_and_reload(tmp_path, monkeypatch, coordinate_table, taxonomy):
    from loguru import logger
    from binferno.training.models.simclr import RankTaxonomyEncoder
    monkeypatch.setattr(esmc, 'build_protein_latents', lambda *args: coordinate_table)
    names = ['a', 'empty']
    index = ['a.0', 'empty.0', 'a.1', 'empty.1']
    df = pd.DataFrame(np.random.default_rng(4).normal(size=(4, 6)), index=index)
    fragments = {name: {'fragment_info': {
        f'{name}.0': {'start_pos': 0, 'length': 60000},
        f'{name}.1': {'start_pos': 0, 'length': 9},
    }} for name in names}
    df, encoder = esmc.prepare_esmc_features(
        df, None, names, None, 1, esmc.ESMCConfig(enabled=True), 4,
        fragment_dict=fragments,
    )
    fasta = tmp_path / 'contigs.fna'
    fasta.write_text(''.join(f'>{name}\n' + 'A' * 60000 + '\n' for name in names))
    trainer = SimCLRTrainer(view_num=2)
    tax_df = pd.DataFrame([[1, 1], [0, 0]] * 2, index=index) if taxonomy else None
    model = SimCLRNet(in_sz=8, out_sz=4, emb_szs=[8, 8], ps=[0.],
                      cov_model=nn.Linear(2, 4), protein_encoder=encoder,
                      tax_model=RankTaxonomyEncoder([2], 4) if taxonomy else None)
    loader, optimizer, scheduler, inference = trainer.prepare_trainer(
        df, tax_df=tax_df, batch_size=2, contig_length=1000, model=model,
        n_views=2, simclrnet=True, contigs=fasta, num_workers=0, epochs=1,
    )
    seen = []
    hook = encoder.register_forward_pre_hook(
        lambda module, args: seen.append((args[0].clone(), args[1].clone()))
    )
    trainer.simclrnet_train(loader, optimizer, scheduler, torch.device('cpu'),
                            epochs=1, model=model, n_cov=2, n_views=2,
                            output_dir=tmp_path, logger=logger)
    assert set(seen[0][1].tolist()) == {0, 2}
    assert encoder.projection.weight.grad.norm() > 0
    exported = trainer.simclr_get_embs(
        model, inference, names, tmp_path / 'embeddings.tsv', torch.device('cpu'),
        kmer_len=4, n_cov=2, tax_emb_dim=2 if taxonomy else 0,
        batch_size=2, save_individual_embs=True,
    )
    # Inference pools the original view, never the fragment-specific subset.
    assert torch.equal(seen[-1][1], torch.zeros(2, dtype=torch.long))
    original = feature_tensor(df.iloc[:2])
    model.load_state_dict(torch.load(tmp_path / 'best_model.pt', weights_only=True))
    with torch.no_grad():
        expected = model(original[:, 2:], original[:, :2],
                         torch.tensor(tax_df.iloc[:2].values).float() if taxonomy else None)[0]
    np.testing.assert_allclose(exported.values, expected.numpy(), rtol=1e-6, atol=1e-6)
    hook.remove()


@pytest.mark.parametrize('workers', [0, 1])
def test_barlow_coordinate_loader_and_scg_metadata(tmp_path, monkeypatch, coordinate_table, workers):
    from binferno.training.scg import prepare_scg_group_loader
    monkeypatch.setattr(esmc, 'build_protein_latents', lambda *args: coordinate_table)
    df = pd.DataFrame(np.random.default_rng(5).normal(size=(4, 6)),
                      index=['a.original', 'a.0', 'empty.original', 'empty.0'])
    fragments = {name: {'fragment_info': {
        f'{name}.0': {'start_pos': 0, 'length': 9},
    }} for name in ['a', 'empty']}
    df, encoder = esmc.prepare_esmc_features(
        df, None, ['a', 'empty'], None, 1, esmc.ESMCConfig(enabled=True), 4,
        fragment_dict=fragments,
    )
    builder = BarlowTwinsTrainer.BuildDataset(data_df=df, max_positive_pairs=10)
    dataset = builder.construct_tensordatasets_from_pairs()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, num_workers=workers,
        **({'multiprocessing_context': 'spawn'} if workers else {}),
    )
    first, second, _, _ = next(iter(loader))
    assert isinstance(first, ProteinBatch)
    for batch in (first, second):
        assert batch.contig_ids.dtype == batch.view_ids.dtype == torch.long
        assert batch.features.shape == (2, 6)
    model = BarlowNetwork(tnf_dim=4, coverage_dim=2, kmer_dim=4, output_dim=4,
                          proj_head_dim=4, protein_encoder=encoder)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss, _ = BarlowTwinsTrainer().train_an_epoch(model, loader, optimizer, torch.device('cpu'))
    assert np.isfinite(loss)
    for factory, suffix in [(scg.prepare_scg_loader, '.original'),
                            (prepare_scg_group_loader, '.original')]:
        scg_loader = factory(df, {'m': ['a', 'empty']}, original_suffix=suffix,
                             batch_size=2, seed=42)
        scg_features = next(iter(scg_loader))[0]
        assert isinstance(scg_features, ProteinBatch)
        assert torch.count_nonzero(scg_features.view_ids) == 0
    model.eval()
    exported = BarlowTwinsTrainer().get_embeddings(
        model=model, data_df=df, output_path=tmp_path / 'embeddings.tsv', batch_size=2,
        save_individual_embs=True,
    )
    with torch.no_grad():
        expected = torch.nn.functional.normalize(model.get_embedding(feature_tensor(df.iloc[[0, 2]])), dim=-1)
    np.testing.assert_allclose(exported.values, expected.numpy(), rtol=1e-6, atol=1e-6)


def test_barlow_crop_zero_cannot_silently_use_original_proteins(monkeypatch, coordinate_table):
    monkeypatch.setattr(esmc, 'build_protein_latents', lambda *args: coordinate_table)
    df = pd.DataFrame([[0.], [0.]], index=['a.original', 'a.0'])
    with pytest.raises(ValueError, match='fragment metadata'):
        esmc.prepare_esmc_features(df, None, ['a', 'empty'], None, 1,
                                  esmc.ESMCConfig(enabled=True), 4)


def test_integer_metadata_survives_dataframe_reordering_and_batching():
    df = pd.DataFrame([[1., 2.], [3., 4.]], index=['a.1', 'a.0'])
    df.attrs['esmc_lookup'] = {'a.0': (2**40 + 1, 0), 'a.1': (2**40 + 1, 3)}
    batch = feature_tensor(df.iloc[[1, 0]])
    combined = cat_features([batch[:1], batch[1:]])
    assert combined.contig_ids.tolist() == [2**40 + 1] * 2
    assert combined.view_ids.tolist() == [0, 3]
    torch.testing.assert_close(combined.features, torch.tensor([[3., 4.], [1., 2.]]))
