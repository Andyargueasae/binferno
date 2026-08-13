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