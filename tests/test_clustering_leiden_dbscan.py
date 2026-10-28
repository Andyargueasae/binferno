import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from loguru import logger as test_logger
from binferno.clustering.dbscan_sweep import (
    SweepConfig,
    build_knn_graph,
    density_sweep,
    derive_eps_grid,
    score_bins,
)
from binferno.clustering.marker_quality import MarkerTable, build_marker_table
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


def test_marker_table_resolves_fraggenescan_ids(tmp_path):
    hmm = tmp_path / "marker.hmm"
    hmm.write_text("NAME HMM1\nNAME HMM2\n//\n")

    hmmout = tmp_path / "marker.hmmout"
    hmmout.write_text(
        "# a minimal domtblout-like file\n"
        "contigA_1_10_+ 0 10 HMM1 0 20 0.0 15 0.0\n"
        "contigA_2_20_+ 0 10 HMM2 0 20 0.0 12 0.0\n"
        "contigB_3_30_- 0 10 HMM1 0 20 0.0 10 0.0\n"
    )

    table, info = build_marker_table(
        str(hmmout),
        ["contigA", "contigB"],
        contig_lengths=[1000, 1000],
        hmm_files={"bacteria": str(hmm)},
        max_evalue=1e-5,
    )

    assert info["orf_id_style"] == "fraggenescan"
    assert table.n_markers == 2
    assert table.quality([0, 1]).n_unique == 2


def test_score_bins_rewards_only_good_bins():
    namelist = ["c1", "c2", "c3", "c4"]
    counts = csr_matrix(
        np.array(
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 0, 0],
                [1, 0, 0],
            ],
            dtype=np.int32,
        )
    )
    table = MarkerTable(
        namelist=namelist,
        counts=counts,
        marker_names=["m1", "m2", "m3"],
        marker_sets={"bacteria": np.array([0, 1, 2], dtype=np.int64)},
        contig_lengths=np.ones(4, dtype=np.int64),
    )
    cfg = SweepConfig(
        hq_completeness=0.90,
        hq_contamination=0.05,
        mq_completeness=0.50,
        mq_contamination=0.10,
    )

    score, quals = score_bins([[0, 1], [2, 3]], table, cfg)

    assert score[0] == 1.0
    assert score[1] == 2.0
    assert len(quals) == 2


def test_derive_eps_grid_returns_sorted_positive_values():
    latent = np.array(
        [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [10.0, 10.0]],
        dtype=np.float32,
    )
    cfg = SweepConfig(
        k=2, eps_k_values=(1, 2), eps_quantiles=(0.05, 0.50), metric="euclidean"
    )

    graph = build_knn_graph(latent, k=2, metric="euclidean", n_jobs=1, cfg=cfg)
    eps_grid = derive_eps_grid(latent, cfg, n_jobs=1)

    assert graph.nnz > 0
    assert eps_grid
    assert eps_grid == sorted(eps_grid)
    assert all(e > 0 for e in eps_grid)


def test_density_sweep_rescue_path_runs_with_noise():
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
    counts = csr_matrix(np.ones((5, 1), dtype=np.int32))
    table = MarkerTable(
        namelist=[f"c{i}" for i in range(1, 6)],
        counts=counts,
        marker_names=["m1"],
        marker_sets={"bacteria": np.array([0], dtype=np.int64)},
        contig_lengths=np.ones(5, dtype=np.int64),
    )
    cfg = SweepConfig(
        k=2,
        eps_k_values=(1, 2),
        eps_quantiles=(0.05, 0.50),
        min_bin_bp=0,
        min_bin_contigs=1,
        core_bp=None,
        min_samples=1,
        metric="cosine",
    )

    result = density_sweep(
        latent=latent,
        contig_lengths=np.ones(5, dtype=np.float64),
        table=table,
        cfg=cfg,
        use_hdbscan=False,
        do_rescue=True,
        n_jobs=1,
        logger=None,
    )

    assert len(result.audit) > 0
    assert result.score is not None
    assert len(result.bins) >= 1
