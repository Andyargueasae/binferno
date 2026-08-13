from pathlib import Path
import tempfile

import numpy as np
import torch

from binferno.training.simclr import SimCLRTrainer
from binferno.training.barlowtwins import BarlowTwinsTrainer
from binferno.training.models.simclr import NeuralNetwork as SimCLRModel
from binferno.training.models.barlow_twins import NeuralNetwork as BarlowTwinsModel
from .conftest import (
    dummy_train_simclr, dummy_train_barlowtwins_args,
    toy_features
)
import pytest

'''
Mainly tests the SimCLR path: from embeddings to dataloading to training and encoding. 
Create dummy data to serve testing purposes.  

2026 June 12 to June 14. 
'''

N_COV=5
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prep_model_simclr():
    args = dummy_train_simclr()
    model = SimCLRModel(tnf_dim=args.get("tnf_dim", 136), 
        coverage_dim=N_COV*2,
        tax_emb_dim=0,
        ).to(DEVICE)
    return model

def prep_model_barlowtwins():
    args = dummy_train_barlowtwins_args()
    model = BarlowTwinsModel(tnf_dim=args.get("tnf_dim", 136),
        coverage_dim=N_COV*2,
        tax_emb_dim=0,
        proj_head_dim=128,
        ).to(DEVICE)
    return model


@pytest.mark.long
def test_simclr_train_an_epoch():
    args = dummy_train_simclr()
    trainer = SimCLRTrainer(learning_rate=args.get("learning_rate", 1e-3),
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
    data_df = toy_features(n_cov=N_COV, simclr=True) # input features.
    
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
    assert emb_df.shape[0] == orig_contigs, f"Number of embeddings should match number of original contigs. Expected {orig_contigs}, got {emb_df.shape[0]}."
    assert emb_df.shape[1] == model.output_dim, "Embedding dimension should match model's output dimension."
    assert np.isfinite(emb_df.values).all(), "All embedding values should be finite numbers."
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
        assert stats.get("invariance_loss", 0), "Invariance loss should be present in stats."
        assert stats.get("redundancy_loss", 0), "Redundancy loss should be present in stats."
        
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
    assert emb_df.shape[0] == orig_contigs, f"Number of embeddings should match number of original contigs. Expected {orig_contigs}, got {emb_df.shape[0]}."
    assert emb_df.shape[1] == model.fusion_network.emb_dim, "Embedding dimension should match model's output dimension."
    assert np.isfinite(emb_df.values).all(), "All embedding values should be finite numbers."
    assert not np.isnan(emb_df.values).any(), "Embedding values should not contain NaN."
    assert not np.isinf(emb_df.values).any(), "Embedding values should not contain Inf."
    print(f"Finished pytest getting embeddings for {model}.")