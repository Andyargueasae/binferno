from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import torch
import numpy as np
import torch
import pytest


from binferno.training.simclr import SimCLRTrainer
from binferno.training.barlowtwins import BarlowTwinsTrainer
from binferno.training.models.old_simclr import NeuralNetwork as SimCLRModel
from binferno.training.models.barlow_twins import NeuralNetwork as BarlowTwinsModel
from .conftest import dummy_train_simclr, dummy_train_barlowtwins_args, toy_features
from binferno.training.simclr import EarlyStoppingManagerSimCLR

"""
Mainly tests the SimCLR path: from embeddings to dataloading to training and encoding. 
Create dummy data to serve testing purposes.  

2026 June 12 to June 14. 
"""

N_COV = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prep_model_simclr():
    args = dummy_train_simclr()
    model = SimCLRModel(
        tnf_dim=args.get("tnf_dim", 136),
        coverage_dim=N_COV * 2,
    ).to(DEVICE)
    return model


def prep_model_barlowtwins():
    args = dummy_train_barlowtwins_args()
    model = BarlowTwinsModel(
        tnf_dim=args.get("tnf_dim", 136),
        coverage_dim=N_COV * 2,
        tax_emb_dim=0,
        proj_head_dim=128,
    ).to(DEVICE)
    return model


@pytest.mark.long
def test_simclr_train_an_epoch():
    args = dummy_train_simclr()
    trainer = SimCLRTrainer(
        learning_rate=args.get("learning_rate", 1e-3),
        batch_size=args.get("batch_size", 32),
        epochs=args.get("epochs", 10),
        temperature=args.get("temperature", 0.15),
        auxiliary_loss=args.get("auxiliary_loss", False),
        seed=args.get("seed", 42),
        tnf_dim=args.get("tnf_dim", 136),
        attention=args.get("attention", True),
        gated=args.get("gated", True),
        hard_weighted_nce=args.get("hard_weighted_nce", True),
        view_num=args.get("view_num", 6),
        contig_length=args.get("contig_length", 1000),
        simclrnet=args.get("simclrnet", False),
    )
    model = prep_model_simclr()
    data_df = toy_features(n_cov=N_COV, simclr=True)  # input features.

    # tempcontig file, with 5 contigs with length all over 1000.
    # write to an existing path, so that the trainer can read it. Then delete it after the test.
    temp_contig_file = Path(__file__).parent / "temp_contigs.fasta"
    with open(temp_contig_file, "w") as f:
        for i in range(5):
            f.write(f">contig_{i}\n")
            f.write("ATCG" * 1000 + "\n")

        # temp bam file, with 5 contigs with length all over 1000.
    train_loader, optimizer, scheduler = trainer.prepare_trainer(
        data_df=data_df,
        batch_size=args.get("batch_size", 32),
        contig_length=args.get("contig_length", 128),
        model=model,
        simclrnet=args.get("simclrnet", False),
        n_views=args.get("view_num", 6),
        contigs=temp_contig_file,
        num_workers=1,
    )

    n_epoch = args.get("epochs", 5)
    avg_losses = []
    for epoch in range(n_epoch):
        avg_loss, stats = trainer.train_an_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=DEVICE,
            n_cov=N_COV,
            temperature=args.get("temperature", 0.15),
        )
        avg_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{n_epoch}, Average Loss: {avg_loss}")
        assert avg_loss >= 0, "Loss should be non-negative."
        assert not torch.isnan(torch.tensor(avg_loss)), "Loss should not be NaN."
        assert not torch.isinf(torch.tensor(avg_loss)), "Loss should not be Inf."
        assert stats.get("top1_acc", 0), "Top-1 accuracy should be present in stats."
        assert stats.get("loss", 0), "Loss should be present in stats."

    print(f"Finished pytest training {model} for {n_epoch} epochs.")

    # delete the temp contig file after the test.
    temp_contig_file.unlink()

    emb_df = trainer.get_embeddings(
        model=model,
        data_df=data_df,
        output_path=Path(tempfile.gettempdir()) / "temp_gated_embeddings.npy",
        batch_size=args.get("batch_size", 32),
        save_individual_embs=False,
        taxonomy_features=None,
    )

    orig_contigs = len([i for i in data_df.index if i.endswith(".0")])

    assert emb_df is not None, "Embeddings DataFrame should not be None."
    assert (
        emb_df.shape[0] == orig_contigs
    ), f"Number of embeddings should match number of original contigs. Expected {orig_contigs}, got {emb_df.shape[0]}."
    assert (
        emb_df.shape[1] == model.output_dim
    ), "Embedding dimension should match model's output dimension."
    assert np.isfinite(
        emb_df.values
    ).all(), "All embedding values should be finite numbers."
    assert not np.isnan(emb_df.values).any(), "Embedding values should not contain NaN."
    assert not np.isinf(emb_df.values).any(), "Embedding values should not contain Inf."
    print(f"Finished pytest getting embeddings for {model}.")


@pytest.mark.long
def test_barlowtwins_train_an_epoch():
    args = dummy_train_barlowtwins_args()
    trainer = BarlowTwinsTrainer(
        learning_rate=args.get("learning_rate", 1e-3),
        batch_size=args.get("batch_size", 32),
        epochs=args.get("epochs", 10),
        seed=args.get("seed", 42),
        tnf_dim=args.get("tnf_dim", 136),
    )
    model = prep_model_barlowtwins()
    data_df = toy_features(n_cov=N_COV, simclr=False)
    dataloader, optimizer, scheduler = trainer.prepare_trainer(
        model=model,
        data_df=data_df,
        batch_size=args.get("batch_size", 32),
        epochs=args.get("epochs", 10),
        num_workers=1,
    )

    n_epoch = args.get("epochs", 5)
    avg_losses = []
    for epoch in range(n_epoch):
        avg_loss, stats = trainer.train_an_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=DEVICE,
        )

        print(f"Epoch {epoch+1}/{n_epoch}, Average Loss: {avg_loss}")
        avg_losses.append(avg_loss)
        assert avg_loss >= 0, "Loss should be non-negative."
        assert not torch.isnan(torch.tensor(avg_loss)), "Loss should not be NaN."
        assert not torch.isinf(torch.tensor(avg_loss)), "Loss should not be Inf."
        assert stats.get(
            "invariance_loss", 0
        ), "Invariance loss should be present in stats."
        assert stats.get(
            "redundancy_loss", 0
        ), "Redundancy loss should be present in stats."

    print(f"Finished pytest training {model} for {n_epoch} epochs.")

    emb_df = trainer.get_embeddings(
        model=model,
        data_df=data_df,
        output_path=Path(tempfile.gettempdir()) / "temp_gated_embeddings.npy",
        batch_size=args.get("batch_size", 32),
        save_individual_embs=False,
        taxonomy_features=None,
    )

    orig_contigs = len([i for i in data_df.index if i.endswith(".original")])

    assert emb_df is not None, "Embeddings DataFrame should not be None."
    assert (
        emb_df.shape[0] == orig_contigs
    ), f"Number of embeddings should match number of original contigs. Expected {orig_contigs}, got {emb_df.shape[0]}."
    assert (
        emb_df.shape[1] == model.fusion_network.emb_dim
    ), "Embedding dimension should match model's output dimension."
    assert np.isfinite(
        emb_df.values
    ).all(), "All embedding values should be finite numbers."
    assert not np.isnan(emb_df.values).any(), "Embedding values should not contain NaN."
    assert not np.isinf(emb_df.values).any(), "Embedding values should not contain Inf."
    print(f"Finished pytest getting embeddings for {model}.")


def test_sort_features_by_method_simclr_reorders_views_and_indices():
    trainer = SimCLRTrainer()
    contig_ids = ["c1", "c2"]

    df = pd.DataFrame(
        {
            "x": [10, 20, 30, 40],
            "y": [100, 200, 300, 400],
        },
        index=["c1.2", "c2.2", "c1.0", "c2.0"],
    )

    out = trainer.sort_features_by_method(df, simclr=True, contig_ids=contig_ids)

    assert list(out.index) == ["c1.0", "c2.0", "c1.2", "c2.2"]
    assert out.shape == (4, 2)
    assert "view" not in out.columns


def test_prepare_trainer_rejects_empty_length_weight(tmp_path):
    trainer = SimCLRTrainer()
    df = pd.DataFrame(np.ones((2, 3)), index=["c1.0", "c2.0"])

    with pytest.raises(ValueError, match="Length weight is empty"):
        trainer.prepare_trainer(
            data_df=df,
            batch_size=1,
            contig_length=1000,
            model=None,
            n_views=1,
            contigs=tmp_path / "missing.fasta",
            num_workers=0,
        )


def test_train_an_epoch_rejects_zero_cov():
    trainer = SimCLRTrainer()
    model = torch.nn.Linear(4, 4)

    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.randn(8, 4)),
        batch_size=2,
    )

    with pytest.raises(ValueError, match="n_cov must be non-negative"):
        trainer.train_an_epoch(
            model=model,
            dataloader=dl,
            optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
            device=torch.device("cpu"),
            n_cov=0,
        )


def test_simclr_augmentation_generates_expected_views(tmp_path):
    trainer = SimCLRTrainer()
    fasta = tmp_path / "toy.fasta"
    fasta.write_text(">c1\n" + ("ACGT" * 200) + "\n")

    frags = trainer.simclr_augmentation(
        seqs=("ACGT" * 200),
        clean_header="c1",
        length_threshold=100,
        num_views=4,
    )

    assert len(frags) == 4
    assert frags[0][0] == "c1.0"
    assert all(header.startswith("c1.") for header, *_ in frags)


def test_early_stopping_manager_tracks_improvement():
    mgr = EarlyStoppingManagerSimCLR(patience=2)

    mgr.check_improvement(loss=1.0, top_acc=0.5, model_state={"a": 1})
    assert mgr.should_stop() is False

    mgr.check_improvement(loss=1.2, top_acc=0.5, model_state={"a": 2})
    assert mgr.should_stop() is False

    mgr.check_improvement(loss=1.3, top_acc=0.5, model_state={"a": 3})
    assert mgr.should_stop() is True


def test_bt_loss_returns_stats_and_scalar_loss():
    trainer = BarlowTwinsTrainer()
    criterion = trainer.BT_loss(lambda_bt=5e-3, alpha=1.0)

    feature1 = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
        dtype=torch.float32,
    )
    feature2 = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
        dtype=torch.float32,
    )

    loss, stats = criterion(feature1=feature1, feature2=feature2, return_stats=True)

    assert torch.isfinite(loss)
    assert loss >= 0.0
    assert set(stats.keys()) >= {
        "mean_diagonal",
        "std_diagonal",
        "mean_abs_off_diagonal",
        "max_abs_off_diagonal",
        "invariance_loss",
        "redundancy_loss",
    }
    assert np.isfinite(stats["mean_diagonal"])
    assert np.isfinite(stats["max_abs_off_diagonal"])


def test_bt_loss_works_on_nontrivial_views():
    trainer = BarlowTwinsTrainer()
    criterion = trainer.BT_loss(lambda_bt=1e-3, alpha=0.5)

    feature1 = torch.randn(32, 10, dtype=torch.float32)
    feature2 = feature1 + 0.1 * torch.randn_like(feature1)

    loss = criterion(feature1=feature1, feature2=feature2)

    assert torch.isfinite(loss)
    assert loss >= 0.0


def test_barlow_build_dataset_generates_positive_pairs():
    trainer = BarlowTwinsTrainer()
    df = pd.DataFrame(
        {
            "f": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        },
        index=[
            "c1.0",
            "c1.1",
            "c2.0",
            "c2.1",
            "c3.0",
            "c3.1",
        ],
    )

    ds = trainer.BuildDataset(data_df=df, max_positive_pairs=100)

    assert len(ds.training_pairs) == 3
    assert all(len(pair) == 2 for pair in ds.training_pairs)

    tensor_ds = ds.construct_tensordatasets_from_pairs()
    assert len(tensor_ds) > 0
    first_features = tensor_ds[0][0]
    second_features = tensor_ds[0][1]
    assert first_features.shape[0] == df.shape[1]
    assert second_features.shape[0] == df.shape[1]


class DummyModelNoTax:
    hierarchical_taxonomy = False

    def __call__(self, x1, x2):
        return x1, x2


class DummyModelTax:
    hierarchical_taxonomy = True

    def __init__(self):
        self.taxonomy_encoder = type(
            "TaxEnc",
            (),
            {
                "loss": staticmethod(
                    lambda logits, labels, mask, weights=None: torch.tensor(0.5)
                )
            },
        )()

    def __call__(self, x1, x2, tax_emb1=None, tax_emb2=None):
        return x1, x2, {"logits": x1}, {"logits": x2}


def test_calculate_loss_no_taxonomy_last_batch_returns_stats():
    trainer = BarlowTwinsTrainer()
    model = DummyModelNoTax()
    criterion = trainer.BT_loss(lambda_bt=1e-3, alpha=0.5)

    x1 = torch.randn(4, 3, dtype=torch.float32)
    x2 = torch.randn(4, 3, dtype=torch.float32)

    loss, stats = trainer.calculate_loss(
        criterion=criterion,
        model=model,
        features1=x1,
        features2=x2,
        last_batch=True,
    )

    assert torch.isfinite(loss)
    assert stats is not None
    assert "mean_diagonal" in stats


def test_calculate_loss_taxonomy_branch_returns_total_loss():
    trainer = BarlowTwinsTrainer()
    model = DummyModelTax()
    criterion = trainer.BT_loss(lambda_bt=1e-3, alpha=0.5)

    x1 = torch.randn(4, 3, dtype=torch.float32)
    x2 = torch.randn(4, 3, dtype=torch.float32)
    label1 = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    label2 = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    mask = torch.ones(4, dtype=torch.bool)

    loss, stats = trainer.calculate_loss(
        criterion=criterion,
        model=model,
        features1=x1,
        features2=x2,
        tax_1=label1,
        tax_2=label2,
        mask=mask,
        last_batch=True,
    )

    assert torch.isfinite(loss)
    assert stats is not None


def test_barlow_twins_augmentation_generates_valid_fragment_headers():
    trainer = BarlowTwinsTrainer()

    seq = "A" * 5000
    frags = trainer.barlow_twins_augmentation(
        sequence=seq,
        header="contig1",
        min_contig_length=1000,
        num_augmentations=3,
        max_overlap=0.25,
    )

    assert len(frags) >= 2
    assert frags[0][0].startswith("contig1.original")
    for header, fragment_seq, start_pos, frag_len in frags:
        assert isinstance(header, str)
        assert len(fragment_seq) >= 1000
        assert frag_len == len(fragment_seq)
        assert start_pos >= 0


def test_prepare_trainer_reduces_batch_size_for_small_dataset():
    trainer = BarlowTwinsTrainer(batch_size=128, epochs=3)
    df = pd.DataFrame(
        np.random.randn(12, 4),
        index=[f"c{i}.0" for i in range(6)] + [f"c{i}.1" for i in range(6)],
    )

    dataloader, optimizer, scheduler = trainer.prepare_trainer(
        model=torch.nn.Linear(4, 4),
        data_df=df,
        batch_size=128,
        epochs=3,
        num_workers=0,
    )

    assert len(dataloader) > 0
    assert optimizer is not None
    assert scheduler is not None

