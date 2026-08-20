from pathlib import Path
import tempfile
from binferno.features import BinfernoFeatureBuilder
from binferno.features.kmer import KmerFeatureBuilder
from binferno.features.coverage import CoverageFeatureBuilder
from binferno.features.corgi import CorgiFeatureBuilder
from binferno.features.dnabert_s import DnaBertSFeatureBuilder
from binferno.features.taxonomy import TaxonomyFeatureBuilder
import pandas as pd

import numpy as np
import pytest
from loguru import logger

from .conftest import CONTIG, BAM
from binferno.features.kmer import (
    KmerFeatureBuilder,
    _calculate_kmer_composition,
    generate_feature_mapping,
)
from binferno.features.coverage import (
    CoverageFeatureBuilder,
    _map_fasta_to_bam_refs,
    _normalise_coverage_dicts,
)
from binferno.training.simclr import SimCLRTrainer


def test_generate_feature_mapping_k4_dim():
    mapping, dim = generate_feature_mapping(4)
    assert dim == 136
    assert isinstance(mapping, dict)
    assert len(mapping) == 256


def test_kmer_composition_short_and_ambiguous_sequences():
    seqs = [
        ("contig_short", "AT"),      # shorter than k
        ("contig_ambig", "ATGCNNN"), # contains ambiguous chars
        ("contig_ok", "ATGCATGC"),
    ]
    df = _calculate_kmer_composition(seqs, kmer_len=4, pseudocount=1.0)

    assert df.shape[0] == 3
    assert df.shape[1] == 136
    row_sums = df.sum(axis=1).to_numpy()
    assert np.allclose(row_sums, 1.0, atol=1e-6)
    assert np.isfinite(df.values).all()


def test_kmer_features(dummy_all_fragments):
    _, all_frags, _ = dummy_all_fragments
    builder = KmerFeatureBuilder(kmer_len=4, pseudocount=1)
    features = builder(all_frags).values

    assert features.shape == (len(all_frags), 136)
    assert np.all(np.linalg.norm(features, axis=1) > 0)
    assert np.isfinite(features).all()


def test_coverage_features_empty_bam(bam_dir, dummy_all_fragments):
    builder = CoverageFeatureBuilder()
    fragment_dict, all_frags, _ = dummy_all_fragments

    cov_df = builder(
        fragment_dict=fragment_dict,
        bam_path=str(bam_dir),
        threads=1,
        logger=logger,
    )

    assert cov_df.shape[0] == len(all_frags)
    assert np.all(cov_df.values >= 0)
    assert np.all(cov_df.values == 0.0)


def test_coverage_features_nonnegative_with_tiny_real_bam():
    builder = CoverageFeatureBuilder()
    trainer = SimCLRTrainer()

    fragments_dict, all_frags, _ = trainer.gen_views(
        contigs=CONTIG,
        output_dir=Path(tempfile.mkdtemp()),
        view_num=6,
    )

    assert BAM.exists()
    cov_df = builder(
        fragment_dict=fragments_dict,
        bam_path=str(BAM),
        threads=1,
        logger=logger,
    )

    assert cov_df.shape[0] == len(all_frags)
    assert cov_df.shape[1] == 2  # one BAM => mean/std
    assert np.all(cov_df.values >= 0)
    assert np.isfinite(cov_df.values).all()


def test_map_fasta_to_bam_refs_match_variants():
    fragments_dict = {
        "contigA": {"fragments": [], "fragment_info": {}},
        "contigB some extra": {"fragments": [], "fragment_info": {}},
        "contigC.1": {"fragments": [], "fragment_info": {}},
        "contigD": {"fragments": [], "fragment_info": {}},
    }
    bam_refs = {"contigA", "contigB", "contigC", "other"}

    ref_map, unmapped = _map_fasta_to_bam_refs(
        fragments_dict=fragments_dict,
        bam_references=bam_refs,
        disable_progress=True,
    )

    assert ref_map["contigA"] == "contigA"
    assert ref_map["contigB some extra"] == "contigB"
    assert ref_map["contigC.1"] == "contigC"
    assert ref_map["contigD"] is None
    assert "contigD" in unmapped


def test_normalise_coverage_dicts_zero_total_mapped():
    coverage = {"f1": 10.0, "f2": 20.0}
    coverage_std = {"f1": 1.0, "f2": 2.0}
    headers = ["f1", "f2"]

    cov, std = _normalise_coverage_dicts(
        coverage=coverage,
        coverage_std=coverage_std,
        all_fragment_headers=headers,
        total_mapped=0,
        rpkm_normalize=True
    )

    assert cov == {"f1": 0.0, "f2": 0.0}
    assert std == {"f1": 0.0, "f2": 0.0}


def test_normalise_coverage_dicts_rpkm():
    coverage = {"f1": 100.0, "f2": 50.0}
    coverage_std = {"f1": 10.0, "f2": 5.0}
    headers = ["f1", "f2"]

    cov, std = _normalise_coverage_dicts(
        coverage=coverage,
        coverage_std=coverage_std,
        all_fragment_headers=headers,
        total_mapped=2_000_000,  # factor=2
        rpkm_normalize=True,
    )

    assert cov["f1"] == pytest.approx(50.0)
    assert cov["f2"] == pytest.approx(25.0)
    assert std["f1"] == pytest.approx(5.0)
    assert std["f2"] == pytest.approx(2.5)


import pickle

def _make_feature_builder(
    use_kmer=True,
    use_coverage=True,
    use_corgi=False,
    use_dnabert_s=False,
    use_taxonomy=False,
):
    return BinfernoFeatureBuilder(
        use_kmer=use_kmer,
        kmer=KmerFeatureBuilder(),
        use_coverage=use_coverage,
        coverage=CoverageFeatureBuilder(),
        use_corgi=use_corgi,
        corgi=CorgiFeatureBuilder(),
        use_dnabert_s=use_dnabert_s,
        dnabert_s=DnaBertSFeatureBuilder(),
        use_taxonomy=use_taxonomy,
        taxonomy=TaxonomyFeatureBuilder(),
        save_features=True,
    )


def test_feature_builder_requires_at_least_one_modality(tmp_path):
    builder = _make_feature_builder(
        use_kmer=False,
        use_coverage=False,
        use_corgi=False,
        use_dnabert_s=False,
        use_taxonomy=False,
    )

    with pytest.raises(ValueError):
        builder(
            FragMent_Dict={},
            All_Fragments=[],
            contigs=CONTIG,
            bam_path=BAM,
            threads=1,
            logger=logger,
            output_dir=tmp_path,
        )


def test_feature_builder_kmer_coverage_success(monkeypatch, tmp_path):
    builder = _make_feature_builder(
        use_kmer=True,
        use_coverage=True,
        use_corgi=False,
        use_dnabert_s=False,
        use_taxonomy=False,
    )

    all_frags = [("f1", "ATGCATGC"), ("f2", "ATGCATGC")]
    frag_dict = {
        "contig_1": {
            "fragments": ["f1"],
            "fragment_info": {"f1": {"start_pos": 0, "length": 8}},
            "sequence": "ATGCATGC",
        },
        "contig_2": {
            "fragments": ["f2"],
            "fragment_info": {"f2": {"start_pos": 0, "length": 8}},
            "sequence": "ATGCATGC",
        },
    }

    fake_kmer = pd.DataFrame(
        [[0.5] * 136, [0.5] * 136],
        index=["f1", "f2"],
        columns=[str(i) for i in range(136)],
    )
    fake_cov = pd.DataFrame(
        [[1.0, 0.1], [2.0, 0.2]],
        index=["f1", "f2"],
        columns=["sample_coverage", "sample_coverage_std"],
    )

    monkeypatch.setattr(builder.kmer, "__call__", lambda fragments: fake_kmer)
    monkeypatch.setattr(
        builder.coverage,
        "__call__",
        lambda fragment_dict, bam_path, threads, logger, max_norm_stacked=False: fake_cov,
    )

    (features, taxonomy), kmer_dim = builder(
        FragMent_Dict=frag_dict,
        All_Fragments=all_frags,
        contigs=CONTIG,
        bam_path=BAM,
        threads=1,
        logger=logger,
        output_dir=tmp_path,
    )

    assert features.shape == (2, 138)
    assert taxonomy is None
    assert kmer_dim == 136
    assert (tmp_path / "features.tsv").exists()


def test_feature_builder_coverage_row_mismatch_raises(monkeypatch, tmp_path):
    builder = _make_feature_builder(
        use_kmer=True,
        use_coverage=True,
        use_corgi=False,
        use_dnabert_s=False,
        use_taxonomy=False,
    )

    all_frags = [("f1", "ATGCATGC"), ("f2", "ATGCATGC")]
    frag_dict = {
        "contig_1": {
            "fragments": ["f1"],
            "fragment_info": {"f1": {"start_pos": 0, "length": 8}},
            "sequence": "ATGCATGC",
        },
        "contig_2": {
            "fragments": ["f2"],
            "fragment_info": {"f2": {"start_pos": 0, "length": 8}},
            "sequence": "ATGCATGC",
        },
    }

    fake_kmer = pd.DataFrame(
        [[0.5] * 136, [0.5] * 136],
        index=["f1", "f2"],
        columns=[str(i) for i in range(136)],
    )
    bad_cov = pd.DataFrame(
        [[1.0, 0.1]],
        index=["f1"],
        columns=["sample_coverage", "sample_coverage_std"],
    )

    monkeypatch.setattr(builder.kmer, "__call__", lambda fragments: fake_kmer)
    monkeypatch.setattr(
        builder.coverage,
        "__call__",
        lambda fragment_dict, bam_path, threads, logger, max_norm_stacked=False: bad_cov,
    )

    with pytest.raises(ValueError):
        builder(
            FragMent_Dict=frag_dict,
            All_Fragments=all_frags,
            contigs=CONTIG,
            bam_path=BAM,
            threads=1,
            logger=logger,
            output_dir=tmp_path,
        )


def test_feature_builder_loads_cached_features_without_recompute(monkeypatch, tmp_path):
    builder = _make_feature_builder(
        use_kmer=True,
        use_coverage=True,
        use_corgi=False,
        use_dnabert_s=False,
        use_taxonomy=False,
    )

    cached_df = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=["f1", "f2"],
        columns=["a", "b"],
    )
    cached_df.to_csv(tmp_path / "features.tsv", sep="\t")
    with open(tmp_path / "taxonomy_features.pkl", "wb") as handle:
        pickle.dump({"dummy": 1}, handle)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Feature branch should not be recomputed when cache exists")

    monkeypatch.setattr(builder.kmer, "__call__", fail_if_called)
    monkeypatch.setattr(builder.coverage, "__call__", fail_if_called)

    (features, taxonomy), _ = builder(
        FragMent_Dict={},
        All_Fragments=[],
        contigs=CONTIG,
        bam_path=BAM,
        threads=1,
        logger=logger,
        output_dir=tmp_path,
    )

    assert features.equals(cached_df)
    assert taxonomy == {"dummy": 1}


def test_feature_builder_taxonomy_branch(monkeypatch, tmp_path):
    builder = _make_feature_builder(
        use_kmer=True,
        use_coverage=False,
        use_corgi=False,
        use_dnabert_s=False,
        use_taxonomy=True,
    )

    all_frags = [("f1", "ATGCATGC"), ("f2", "ATGCATGC")]
    frag_dict = {
        "contig_1": {
            "fragments": ["f1"],
            "fragment_info": {"f1": {"start_pos": 0, "length": 8}},
            "sequence": "ATGCATGC",
        },
        "contig_2": {
            "fragments": ["f2"],
            "fragment_info": {"f2": {"start_pos": 0, "length": 8}},
            "sequence": "ATGCATGC",
        },
    }

    fake_kmer = pd.DataFrame(
        [[0.5] * 136, [0.5] * 136],
        index=["f1", "f2"],
        columns=[str(i) for i in range(136)],
    )
    fake_tax_df = pd.DataFrame(
        {
            "contig_id": ["contig_1", "contig_2"],
            "taxonomy": ["A", "B"],
        }
    )

    monkeypatch.setattr(builder.kmer, "__call__", lambda fragments: fake_kmer)
    monkeypatch.setattr(
        builder.taxonomy,
        "__call__",
        lambda contigs, logger: (
            ["root", "A", "B"],
            {"root": 0, "A": 1, "B": 2},
            np.array([-1, 0, 0], dtype=np.int64),
            fake_tax_df,
        ),
    )

    (features, taxonomy), _ = builder(
        FragMent_Dict=frag_dict,
        All_Fragments=all_frags,
        contigs=CONTIG,
        bam_path=BAM,
        threads=1,
        logger=logger,
        output_dir=tmp_path,
    )

    assert features.shape[0] == 2
    assert isinstance(taxonomy, dict)
    assert "tax_node_index" in taxonomy
    assert "contig_taxonomy_df" in taxonomy

def test_feature_builder_taxonomy_only_no_coverage_no_kmer(monkeypatch, tmp_path):
    """Regression: no UnboundLocalError when use_coverage=False and use_kmer=False."""
    builder = _make_feature_builder(
        use_kmer=False,
        use_coverage=False,
        use_corgi=False,
        use_dnabert_s=False,
        use_taxonomy=True,
    )

    frag_dict = {
        "contig_1": {
            "fragments": ["f1"],
            "fragment_info": {"f1": {"start_pos": 0, "length": 8}},
            "sequence": "ATGCATGC",
        },
        "contig_2": {
            "fragments": ["f2"],
            "fragment_info": {"f2": {"start_pos": 0, "length": 8}},
            "sequence": "ATGCATGC",
        },
    }

    fake_tax_df = pd.DataFrame(
        {"contig_id": ["contig_1", "contig_2"], "tax_node_idx": [1, 2]}
    )

    monkeypatch.setattr(
        builder.taxonomy,
        "__call__",
        lambda contigs, logger: (
            ["root", "A", "B"],
            {"root": 0, "A": 1, "B": 2},
            np.array([-1, 0, 0], dtype=np.int64),
            fake_tax_df,
        ),
    )

    (features, taxonomy), kmer_dim = builder(
        FragMent_Dict=frag_dict,
        All_Fragments=[("f1", "ATGCATGC"), ("f2", "ATGCATGC")],
        contigs=CONTIG,
        bam_path=BAM,
        threads=1,
        logger=logger,
        output_dir=tmp_path,
    )

    assert isinstance(taxonomy, dict)
    assert "tax_node_index" in taxonomy
    assert kmer_dim == 0
    assert features.shape[1] == 0  # no kmer, no coverage columns


def test_feature_builder_corgi_branch(monkeypatch, tmp_path):
    """Exercises the corgi-only branch, covering lines missed by existing tests."""
    builder = _make_feature_builder(
        use_kmer=False,
        use_coverage=False,
        use_corgi=True,
        use_dnabert_s=False,
        use_taxonomy=False,
    )

    fake_corgi = pd.DataFrame(
        [[0.1] * 4, [0.2] * 4],
        index=["contig_1", "contig_2"],
        columns=["dim0", "dim1", "dim2", "dim3"],
    )

    # The builder calls the CorgiFeatureBuilder instance, so we patch the class method
    # and seed the expected augmented FASTA files the code checks for before reading outputs.
    (tmp_path / "augmented_fragments_view0.fasta").write_text(
        ">contig_1\nATGC\n>contig_2\nATGC\n"
    )
    (tmp_path / "augmented_fragments_view1.fasta").write_text(
        ">contig_1\nATGC\n>contig_2\nATGC\n"
    )
    monkeypatch.setattr(
        CorgiFeatureBuilder,
        "__call__",
        lambda self, seq_dir, contigs, min_length=1000: fake_corgi,
    )

    (features, taxonomy), kmer_dim = builder(
        FragMent_Dict={},
        All_Fragments=[],
        contigs=CONTIG,
        bam_path=BAM,
        threads=1,
        logger=logger,
        output_dir=tmp_path,
    )

    assert isinstance(taxonomy, pd.DataFrame)
    assert taxonomy.shape == (2, 4)
    assert kmer_dim == 0
    assert (tmp_path / "corgi_features.tsv").exists()


def test_feature_builder_corgi_branch_loads_cached(monkeypatch, tmp_path):
    """Exercises the cached-corgi code path (corgi_data.exists() branch)."""
    builder = _make_feature_builder(
        use_kmer=False,
        use_coverage=False,
        use_corgi=True,
        use_dnabert_s=False,
        use_taxonomy=False,
    )

    cached_corgi = pd.DataFrame(
        [[0.5] * 3, [0.6] * 3],
        index=["contig_1", "contig_2"],
        columns=["d0", "d1", "d2"],
    )
    cached_corgi.to_csv(tmp_path / "corgi_features.tsv", sep="\t")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Corgi model should not be called when cache exists")

    monkeypatch.setattr(builder.corgi, "__call__", fail_if_called)

    (features, taxonomy), _ = builder(
        FragMent_Dict={},
        All_Fragments=[],
        contigs=CONTIG,
        bam_path=BAM,
        threads=1,
        logger=logger,
        output_dir=tmp_path,
    )

    assert isinstance(taxonomy, pd.DataFrame)
    assert taxonomy.shape[0] == 2
