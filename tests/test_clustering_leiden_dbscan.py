import numpy as np
import pandas as pd
import pytest

from loguru import logger as test_logger
from binferno.clustering.dbscan import (
    completeness_contamination,
    count_markers_saturated,
    dbscan_marker_sweep,
    derive_eps_grid,
    parse_contig_to_markers,
    score_count,
)
from binferno.training import BinfernoTrainer
from binferno.training.simclr import SimCLRTrainer
from binferno.training.barlowtwins import BarlowTwinsTrainer


def _fake_emb(index=("contig_1", "contig_2"), dims=4, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(len(index), dims)).astype("float32")
    return pd.DataFrame(x, index=list(index), columns=[f"e{i}" for i in range(dims)])


def test_trainer_requires_method_enabled(tmp_path):
    trainer = BinfernoTrainer(
        simclr=SimCLRTrainer(),
        use_simclr=False,
        barlowtwins=BarlowTwinsTrainer(),
        use_barlowtwins=False,
        concat=True,
    )

    with pytest.raises(ValueError):
        trainer(
            contigs=tmp_path / "contigs.fna",
            bam=tmp_path / "bam",
            output_dir=tmp_path / "out",
            features=None,
            threads=1,
            logger=None,
        )


def test_trainer_only_simclr_path_called(tmp_path, monkeypatch):
    called = {"simclr": 0, "barlow": 0}

    def fake_simclr(self, contigs, bam, output_dir, features, threads, logger):
        called["simclr"] += 1
        return _fake_emb(dims=3, seed=1)

    def fake_barlow(self, contigs, bam, output_dir, features, threads, logger):
        called["barlow"] += 1
        return _fake_emb(dims=3, seed=2)

    monkeypatch.setattr(SimCLRTrainer, "__call__", fake_simclr)
    monkeypatch.setattr(BarlowTwinsTrainer, "__call__", fake_barlow)

    trainer = BinfernoTrainer(
        simclr=SimCLRTrainer(),
        use_simclr=True,
        barlowtwins=BarlowTwinsTrainer(),
        use_barlowtwins=False,
        concat=True,
    )

    out = trainer(
        contigs=tmp_path / "contigs.fna",
        bam=tmp_path / "bam",
        output_dir=tmp_path / "out",
        features=None,
        threads=1,
        logger=test_logger,
    )

    assert called["simclr"] == 1
    assert called["barlow"] == 0
    assert out.shape == (2, 3)


def test_trainer_concat_false_normalizes(tmp_path, monkeypatch):
    def fake_simclr(self, contigs, bam, output_dir, features, threads, logger):
        return _fake_emb(dims=4, seed=10)

    def fake_barlow(self, contigs, bam, output_dir, features, threads, logger):
        return _fake_emb(dims=4, seed=11)

    monkeypatch.setattr(SimCLRTrainer, "__call__", fake_simclr)
    monkeypatch.setattr(BarlowTwinsTrainer, "__call__", fake_barlow)

    trainer = BinfernoTrainer(
        simclr=SimCLRTrainer(),
        use_simclr=True,
        barlowtwins=BarlowTwinsTrainer(),
        use_barlowtwins=True,
        concat=False,
    )

    out = trainer(
        contigs=tmp_path / "contigs.fna",
        bam=tmp_path / "bam",
        output_dir=tmp_path / "out",
        features=None,
        threads=1,
        logger=test_logger,
    )

    assert out.shape == (2, 4)
    norms = np.linalg.norm(out.values, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_completeness_contamination_basic():
    counts = np.array([1, 1, 2, 0], dtype=np.int32)
    comp, cont = completeness_contamination(counts, n_scgs=4)

    assert comp == pytest.approx(0.75)
    assert cont == pytest.approx(0.25)


def test_count_markers_saturated_counts_unique_and_duplicates():
    contig_to_markers = {"c1": [0, 0, 1], "c2": [0, 2]}
    counts = count_markers_saturated([0, 1], contig_to_markers, ["c1", "c2"], n_scgs=3)

    assert counts is not None
    assert counts.tolist() == [3, 1, 1]


def test_score_count_rewards_only_good_bins():
    namelist = ["c1", "c2", "c3", "c4"]
    contig_to_markers = {
        "c1": [0, 1, 2],
        "c2": [0, 1, 2],
        "c3": [0],
        "c4": [0],
    }

    bins = [[0, 1], [2, 3]]
    score = score_count(bins, contig_to_markers, namelist, n_scgs=3)

    assert score == pytest.approx(1.0)


def test_parse_contig_to_markers_handles_fraggenescan_ids(tmp_path):
    hmmout = tmp_path / "marker.hmmout"
    hmmout.write_text(
        "contigA_1_10_+ 0 HMM1 15 100 0.0 0.0 0.0\n"
        "contigA_2_20_+ 0 HMM2 15 100 0.0 0.0 0.0\n"
        "contigB_3_30_- 0 HMM1 15 100 0.0 0.0 0.0\n"
    )

    contig_to_markers, n_scgs = parse_contig_to_markers(str(hmmout))

    assert n_scgs == 2
    assert contig_to_markers["contigA"] == [0, 1]
    assert contig_to_markers["contigB"] == [0]


def test_derive_eps_grid_returns_sorted_positive_values():
    latent = np.array(
        [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [10.0, 10.0]],
        dtype=np.float32,
    )
    graph = __import__("sklearn.neighbors", fromlist=["kneighbors_graph"]).kneighbors_graph(
        latent, n_neighbors=2, mode="distance", metric="euclidean", n_jobs=-1
    )

    eps_grid = derive_eps_grid(graph, latent.shape[0])

    assert eps_grid
    assert eps_grid == sorted(eps_grid)
    assert all(e > 0 for e in eps_grid)


def test_dbscan_marker_sweep_rescue_path_runs_with_noise():
    latent = np.array(
        [
            [0.0, 0.0],
            [0.02, 0.0],
            [0.04, 0.0],
            [10.0, 10.0],
            [10.05, 10.0],
        ],
        dtype=np.float32,
    )
    contig_to_markers = {f"c{i}": [0] for i in range(1, 6)}
    contig_lengths = np.ones(len(latent), dtype=np.float32)

    bins, score, eps_grid = dbscan_marker_sweep(
        latent=latent,
        contig_lengths=contig_lengths,
        contig_to_markers=contig_to_markers,
        namelist=[f"c{i}" for i in range(1, 6)],
        n_scgs=1,
        min_samples=1,
        metric="cosine",
        scoring="count",
        do_rescue=True,
        logger=None,
    )

    assert len(eps_grid) > 0
    assert score >= 0.0
    assert len(bins) >= 1