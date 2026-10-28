from pathlib import Path
import pandas as pd

from binferno.main import binferno
from binferno.clustering.base import ClusterResult

import numpy as np
import pandas as pd
import pytest

from loguru import logger as test_logger
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


class DummyTraining:
    use_barlowtwins = False

    def __init__(self):
        self.called = 0

    def __call__(self, contigs, bam, output_dir, features, threads, logger):
        self.called += 1
        return pd.DataFrame(
            [[0.1, 0.2], [0.3, 0.4]],
            index=["contig_1", "contig_2"],
            columns=["e1", "e2"],
        )


class DummyClustering:
    from types import SimpleNamespace
    cluster_method = SimpleNamespace(value="kmeans")

    def __init__(self):
        self.called = 0
        self.plastid = False

    def __call__(self, contigs, embeddings, taxonomy, output_path, threads, logger):
        self.called += 1
        df = pd.DataFrame(
            {"contig_id": ["contig_1", "contig_2"], "cluster_id": ["group0", "group1"]}
        )
        return ClusterResult(n_genome=2, clusters={"dummy.tsv": df})


class DummyPostprocessor:
    def __init__(self):
        self.called = 0

    def __call__(
        self, contigs, cluster_result, num_threads, output_path, plastid, logger
    ):
        self.called += 1


class DummyFeatures:
    use_taxonomy = False


class DummyReducer:
    def __call__(self, data, logger, threads):
        return data


def _write_contigs(path: Path):
    path.write_text(
        ">contig_1\n" + "ATGC" * 500 + "\n" ">contig_2\n" + "GCAT" * 500 + "\n"
    )


def test_binferno_train_only_skips_cluster_and_postprocess(tmp_path):
    contigs = tmp_path / "contigs.fna"
    _write_contigs(contigs)
    bam_dir = tmp_path / "bam"
    bam_dir.mkdir()
    out_dir = tmp_path / "out"

    trainer = DummyTraining()
    clustering = DummyClustering()
    postprocessor = DummyPostprocessor()

    binferno(
        contigs=contigs,
        bam_dir=bam_dir,
        output_dir=out_dir,
        features=DummyFeatures(),
        reducer=DummyReducer(),
        training=trainer,
        clustering=clustering,
        postprocessor=postprocessor,
        threads=1,
        train_only=True,
    )

    assert trainer.called == 1
    assert clustering.called == 0
    assert postprocessor.called == 0


def test_binferno_full_pipeline_runs_all_stages(tmp_path):
    contigs = tmp_path / "contigs.fna"
    _write_contigs(contigs)
    bam_dir = tmp_path / "bam"
    bam_dir.mkdir()
    out_dir = tmp_path / "out"

    trainer = DummyTraining()
    clustering = DummyClustering()
    postprocessor = DummyPostprocessor()

    binferno(
        contigs=contigs,
        bam_dir=bam_dir,
        output_dir=out_dir,
        features=DummyFeatures(),
        reducer=DummyReducer(),
        training=trainer,
        clustering=clustering,
        postprocessor=postprocessor,
        threads=1,
        train_only=False,
    )

    assert trainer.called == 1
    assert clustering.called == 1
    assert postprocessor.called == 1
    assert (out_dir / "cluster_res" / "cluster_results.pkl").exists()

