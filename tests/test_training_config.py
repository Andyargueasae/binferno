"""Shared configuration must reach each trainer's training and inference paths."""
from importlib import import_module
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import torch
from loguru import logger

from binferno.training import BinfernoTrainer
from binferno.training.barlowtwins import BarlowTwinsTrainer
from binferno.training.config import CommonTrainingConfig, SCGConfig
from binferno.training.simclr import SimCLRTrainer


@pytest.mark.parametrize("method", ["barlowtwins", "simclr", "legacy_simclr"])
def test_shared_config_reaches_training_and_embeddings(method, tmp_path, monkeypatch):
    is_barlow = method == "barlowtwins"
    module = import_module(
        "binferno.training.barlowtwins" if is_barlow else "binferno.training.simclr"
    )
    trainer_cls = BarlowTwinsTrainer if is_barlow else SimCLRTrainer
    trainer = (
        trainer_cls(view_num=2) if is_barlow
        else trainer_cls(view_num=2, simclrnet=method == "simclr", temperature=0.2)
    )
    common = CommonTrainingConfig(
        batch_size=7, epochs=3, output_dim=8, contig_length=1500,
        seed=19, get_individual_embs=True,
    )
    orchestrator = BinfernoTrainer(
        common=common, scg=SCGConfig(),
        simclr=SimCLRTrainer() if is_barlow else trainer,
        barlowtwins=trainer if is_barlow else BarlowTwinsTrainer(),
        use_simclr=not is_barlow, use_barlowtwins=is_barlow,
    )
    suffix = "original" if is_barlow else "0"
    index = [f"c{i}.{view}" for view in [suffix, "1"] for i in range(2)]
    frame = pd.DataFrame(
        np.ones((4, 3), dtype=np.float32), index=index,
        columns=["cov_mean", "tnf_0", "tnf_1"],
    )
    fragments = {f"c{i}": {"sequence": "A" * 1500} for i in range(2)}
    views = Mock(return_value=(fragments, [], ["c0", "c1"]))
    monkeypatch.setattr(trainer_cls, "gen_views", views)
    seed = Mock()
    monkeypatch.setattr(trainer_cls, "unify_seeds", seed)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    model = torch.nn.Linear(3, common.output_dim)
    model_factory = Mock(return_value=model)
    monkeypatch.setattr(module, "NeuralNetwork", model_factory)
    if not is_barlow:
        monkeypatch.setattr(module, "SimCLRNet", model_factory)
        monkeypatch.setattr(module, "EmbeddingNet", Mock(return_value=torch.nn.Linear(1, 8)))
    protein_features = Mock(return_value=(frame, None))
    monkeypatch.setattr(module, "prepare_esmc_features", protein_features)
    prepare = Mock(return_value=([], None, None) if is_barlow else ([], None, None, []))
    monkeypatch.setattr(trainer_cls, "prepare_trainer", prepare)
    train = Mock(return_value=model)
    train_method = "simclrnet_train" if method == "simclr" else "train"
    monkeypatch.setattr(trainer_cls, train_method, train)
    embeddings = pd.DataFrame(np.ones((2, common.output_dim)), index=["c0", "c1"])
    get_embs = Mock(return_value=embeddings)
    emb_method = "simclr_get_embs" if method == "simclr" else "get_embeddings"
    monkeypatch.setattr(trainer_cls, emb_method, get_embs)
    features = Mock(return_value=((frame, None), 2))

    result = orchestrator(
        contigs=tmp_path / "contigs.fna", bam=tmp_path / "bam",
        output_dir=tmp_path, features=features, threads=1, logger=logger,
    )

    assert trainer._common is common
    assert trainer._scg is orchestrator.scg
    assert views.call_args.kwargs["contig_length"] == 1500
    seed.assert_called_once_with(19)
    assert prepare.call_args.kwargs["batch_size"] == 7
    assert prepare.call_args.kwargs["epochs"] == 3
    assert train.call_args.kwargs["epochs"] == 3
    assert result.shape == (2, 8)
    assert get_embs.call_args.kwargs["batch_size"] == 7
    assert get_embs.call_args.kwargs["save_individual_embs"] is True
    if method == "simclr":
        assert model_factory.call_args.kwargs["out_sz"] == 8
        assert train.call_args.kwargs["seed"] == 19
    else:
        assert model_factory.call_args.kwargs["output_dim"] == 8


def test_standalone_trainers_have_independent_config_defaults():
    first, second = BarlowTwinsTrainer(), SimCLRTrainer()
    first._common.seed = 5
    first._scg.lambda_scg = 0.5
    assert second._common.seed == 42
    assert second._scg.lambda_scg == 0.0


def test_barlow_scg_loader_uses_shared_batch_size_and_seed(monkeypatch):
    module = import_module("binferno.training.barlowtwins")
    trainer = BarlowTwinsTrainer()
    trainer._common = CommonTrainingConfig(seed=23)
    trainer._scg = SCGConfig(scg_batch_size=9)
    loader = Mock()
    monkeypatch.setattr(module, "prepare_scg_group_loader", loader)
    trainer.prepare_scg_loader(data_df=pd.DataFrame(), marker_groups={})
    assert loader.call_args.kwargs["batch_size"] == 9
    assert loader.call_args.kwargs["seed"] == 23


def test_simclr_scheduler_uses_shared_epochs(tmp_path):
    trainer = SimCLRTrainer()
    trainer._common = CommonTrainingConfig(epochs=7)
    fasta = tmp_path / "contigs.fna"
    fasta.write_text(">c0\n" + "A" * 1500 + "\n>c1\n" + "A" * 1500 + "\n")
    frame = pd.DataFrame(
        np.ones((4, 3), dtype=np.float32),
        index=["c0.0", "c1.0", "c0.1", "c1.1"],
    )
    _, _, scheduler, _ = trainer.prepare_trainer(
        data_df=frame, model=torch.nn.Linear(3, 3), n_views=2,
        simclrnet=True, contigs=fasta, num_workers=0,
    )
    assert scheduler.T_max == 7


def test_hard_negative_loss_resolves_before_batch_validation():
    trainer = SimCLRTrainer(hard_weighted_nce=True)
    with pytest.raises(ValueError, match="n_cov"):
        trainer.train_an_epoch(
            model=torch.nn.Linear(3, 3), dataloader=[], optimizer=None,
            device=torch.device("cpu"), n_cov=0,
        )
