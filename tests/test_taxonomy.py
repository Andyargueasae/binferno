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
    fasta, tsv = tmp_path / "contigs.fasta", tmp_path / "tax.tsv"
    _write_fasta(fasta, {"ctg1": "ATGC", "ctg2": "TTTT"})
    _write_taxonomy_tsv(tsv, [("ctg1", "Bacteria"), ("ctg2", "Bacteria")])
    frame, vocab = TaxonomyFeatureBuilder(taxonomy=tsv)(fasta, logger)
    assert frame.index.tolist() == ["ctg1", "ctg2"]
    assert vocab["domain"] == {"<UNK>": 0, "Bacteria": 1}
    assert (frame.tax_domain == 1).all()
    assert (frame.mask_domain == 1.).all()
    assert (frame.tax_phylum == 0).all()
    assert (frame.mask_phylum == 0.).all()


def test_taxonomy_builder_multi_level(tmp_path):
    fasta, tsv = tmp_path / "contigs.fasta", tmp_path / "tax.tsv"
    _write_fasta(fasta, {"ctg1": "ATGC", "ctg2": "TTTT"})
    _write_taxonomy_tsv(tsv, [
        ("ctg1", "Bacteria;Proteobacteria;Gammaproteobacteria"),
        ("ctg2", "Bacteria;Firmicutes"),
    ])
    frame, vocab = TaxonomyFeatureBuilder(taxonomy=tsv)(fasta, logger)
    assert frame.loc["ctg1", "tax_phylum"] == vocab["phylum"]["Proteobacteria"]
    assert frame.loc["ctg2", "tax_phylum"] == vocab["phylum"]["Firmicutes"]
    assert frame.loc["ctg1", "tax_class"] == vocab["class"]["Gammaproteobacteria"]
    assert frame.loc["ctg2", "mask_class"] == 0.
    assert frame.shape == (2, 14)


def test_taxonomy_builder_contig_filtering(tmp_path):
    fasta, tsv = tmp_path / "contigs.fasta", tmp_path / "tax.tsv"
    _write_fasta(fasta, {"ctg1": "ATGC"})
    _write_taxonomy_tsv(tsv, [("ctg1", "Bacteria"), ("ctg2", "Archaea")])
    frame, vocab = TaxonomyFeatureBuilder(taxonomy=tsv)(fasta, logger)
    assert frame.index.tolist() == ["ctg1"]
    assert "Archaea" not in vocab["domain"]
    assert "Bacteria" in vocab["domain"]
