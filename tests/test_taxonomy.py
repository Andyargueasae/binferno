from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

from binferno.features.taxonomy import TaxonomyFeatureBuilder


def _write_fasta(path: Path, contigs: dict):
    with open(path, "w") as f:
        for name, seq in contigs.items():
            f.write(f">{name}\n{seq}\n")


def _write_taxonomy_tsv(path: Path, rows: list):
    df = pd.DataFrame(rows, columns=["contigs", "predictions"])
    df.to_csv(path, sep="\t", index=False)


def test_taxonomy_builder_single_level(tmp_path):
    """Basic integration: single-level taxonomy, verifies nodes and parental array."""
    fasta = tmp_path / "contigs.fasta"
    tsv = tmp_path / "tax.tsv"
    _write_fasta(fasta, {"ctg1": "ATGCATGC", "ctg2": "TTTTGGGG"})
    _write_taxonomy_tsv(tsv, [("ctg1", "Bacteria"), ("ctg2", "Bacteria")])

    builder = TaxonomyFeatureBuilder(taxonomy=tsv)
    nodes, node_indices, parental_array, contig_df = builder(fasta, logger)

    assert nodes[0] == "root"
    assert "Bacteria" in node_indices
    assert node_indices["root"] == 0
    assert parental_array[0] == -1  # root has no parent
    assert parental_array[node_indices["Bacteria"]] == 0  # Bacteria's parent is root
    assert set(contig_df["contig_id"].tolist()) == {"ctg1", "ctg2"}
    assert all(contig_df["tax_node_idx"] == node_indices["Bacteria"])


def test_taxonomy_builder_multi_level(tmp_path):
    """Multi-level taxonomy: verifies deep graph structure and BFS ordering."""
    fasta = tmp_path / "contigs.fasta"
    tsv = tmp_path / "tax.tsv"
    _write_fasta(fasta, {"ctg1": "ATGCATGC", "ctg2": "TTTTGGGG"})
    _write_taxonomy_tsv(
        tsv,
        [
            ("ctg1", "Bacteria;Proteobacteria;Gammaproteobacteria"),
            ("ctg2", "Bacteria;Firmicutes"),
        ],
    )

    builder = TaxonomyFeatureBuilder(taxonomy=tsv)
    nodes, node_indices, parental_array, contig_df = builder(fasta, logger)

    for name in (
        "root",
        "Bacteria",
        "Proteobacteria",
        "Gammaproteobacteria",
        "Firmicutes",
    ):
        assert name in node_indices

    # Parent of Proteobacteria must be Bacteria
    assert parental_array[node_indices["Proteobacteria"]] == node_indices["Bacteria"]
    # Parent of Gammaproteobacteria must be Proteobacteria
    assert (
        parental_array[node_indices["Gammaproteobacteria"]]
        == node_indices["Proteobacteria"]
    )

    # BFS ordering invariant: every parent index precedes its child
    assert all(p < i for i, p in enumerate(parental_array) if p != -1)

    assert set(contig_df["contig_id"].tolist()) == {"ctg1", "ctg2"}
    # ctg1 maps to finest taxon Gammaproteobacteria
    assert (
        contig_df.loc[contig_df["contig_id"] == "ctg1", "tax_node_idx"].values[0]
        == node_indices["Gammaproteobacteria"]
    )


def test_taxonomy_builder_contig_filtering(tmp_path):
    """Contigs in TSV absent from FASTA must be excluded; their taxa not added."""
    fasta = tmp_path / "contigs.fasta"
    tsv = tmp_path / "tax.tsv"
    _write_fasta(fasta, {"ctg1": "ATGCATGC"})  # only ctg1
    _write_taxonomy_tsv(
        tsv,
        [
            ("ctg1", "Bacteria"),
            ("ctg2", "Archaea"),  # ctg2 not in FASTA
        ],
    )

    builder = TaxonomyFeatureBuilder(taxonomy=tsv)
    nodes, node_indices, parental_array, contig_df = builder(fasta, logger)

    assert "ctg2" not in contig_df["contig_id"].tolist()
    assert "ctg1" in contig_df["contig_id"].tolist()
    assert "Archaea" not in node_indices  # filtered-out taxon absent from graph
    assert "Bacteria" in node_indices
