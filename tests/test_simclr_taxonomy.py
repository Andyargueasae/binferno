import numpy as np
import pandas as pd
import torch

from binferno.training.models.COMEBin_models import RankTaxonomyEncoder
from binferno.training.simclr import SimCLRTrainer


def test_rank_taxonomy_encoder_handles_ids_and_missing_ranks():
    encoder = RankTaxonomyEncoder([2, 3, 4], out_sz=8, embedding_dim=4)
    features = torch.tensor(
        [
            [1, 2, 3, 1, 1, 1],
            [1, 0, 0, 1, 0, 0],
        ],
        dtype=torch.float32,
    )

    output = encoder(features)

    assert output.shape == (2, 8)
    assert torch.isfinite(output).all()


def test_simclr_prepare_trainer_keeps_small_dataset(tmp_path):
    n_contigs = 15
    n_views = 6
    contig_ids = [f"contig_{i}" for i in range(n_contigs)]
    index = [
        f"{contig_id}.{view}" for view in range(n_views) for contig_id in contig_ids
    ]
    data_df = pd.DataFrame(
        np.random.default_rng(42).normal(size=(len(index), 4)),
        index=index,
    )
    fasta = tmp_path / "contigs.fasta"
    fasta.write_text(
        "".join(f">{contig_id}\n{chr(65) * 1001}\n" for contig_id in contig_ids)
    )
    trainer = SimCLRTrainer(batch_size=1024, epochs=1)

    loader, _, _, _ = trainer.prepare_trainer(
        data_df=data_df,
        tax_df=None,
        batch_size=1024,
        contig_length=1000,
        model=torch.nn.Linear(4, 4),
        n_views=n_views,
        simclrnet=True,
        contigs=fasta,
        num_workers=0,
    )

    assert len(loader) == 1
    assert loader.batch_size == n_contigs


def test_taxonomy_builder_filters_to_fasta_and_orders_ranks(tmp_path):
    from loguru import logger

    from binferno.features.taxonomy import TAXONOMIC_RANKS, TaxonomyFeatureBuilder

    fasta = tmp_path / "contigs.fasta"
    fasta.write_text(">c1\nACGT\n>c2\nACGT\n")
    taxonomy = tmp_path / "taxonomy.tsv"
    pd.DataFrame(
        [
            ("c1", "d_Bacteria;p_One"),
            ("c2", "d_Bacteria;p_Two"),
            ("outside", "d_Archaea;p_Three"),
        ],
        columns=["contigs", "predictions"],
    ).to_csv(taxonomy, sep="\t", index=False)

    features, vocabularies = TaxonomyFeatureBuilder(taxonomy=taxonomy)(fasta, logger)

    assert features.index.tolist() == ["c1", "c2"]
    assert features.columns.tolist() == [
        *[f"tax_{rank}" for rank in TAXONOMIC_RANKS],
        *[f"mask_{rank}" for rank in TAXONOMIC_RANKS],
    ]
    assert "d_Archaea" not in vocabularies["domain"]
