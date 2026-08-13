import numpy as np
import pandas as pd
from loguru import logger
from unittest.mock import patch

from binferno.features.coverage import (
    _calculate_fragment_stats_vectorized,
    _process_contig_coverage_worker,
    calculate_bam_cov,
    calculate_fragment_coverage,
    max_normalize_stacked_coverage,
)


def _tiny_fragment_dict():
    return {
        "contig_1": {
            "sequence": "ATGC" * 100,
            "fragments": ["frag_1", "frag_2"],
            "fragment_info": {
                "frag_1": {"start_pos": 0, "length": 10},
                "frag_2": {"start_pos": 10, "length": 10},
            },
        }
    }


def test_calculate_fragment_stats_vectorized_small_branch():
    coverage = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    coords = [(0, 2), (2, 5)]  # n_fragments < 10 triggers small-fragment branch

    means, stds = _calculate_fragment_stats_vectorized(coverage, coords)

    assert means.shape == (2,)
    assert stds.shape == (2,)
    assert np.isclose(means[0], 1.5)
    assert np.isclose(means[1], 4.0)
    assert np.all(stds >= 0)


def test_calculate_fragment_stats_vectorized_large_branch():
    coverage = np.arange(100, dtype=np.float64)
    coords = [(i, i + 5) for i in range(0, 50, 5)] + [(50, 60), (60, 70)]  # 12 fragments

    means, stds = _calculate_fragment_stats_vectorized(coverage, coords)

    assert means.shape == (12,)
    assert stds.shape == (12,)
    assert np.all(np.isfinite(means))
    assert np.all(np.isfinite(stds))
    assert np.all(stds >= 0)


def test_calculate_fragment_stats_vectorized_fallback_on_bad_indices():
    coverage = np.arange(20, dtype=np.float64)
    coords = [(0, 5)] * 10
    coords[3] = (1000, 1005)  # out-of-bounds triggers vectorized try/except fallback

    means, stds = _calculate_fragment_stats_vectorized(coverage, coords)

    assert means.shape == (10,)
    assert stds.shape == (10,)
    assert np.all(np.isfinite(means))
    assert np.all(np.isfinite(stds))
    assert means[3] == 0.0
    assert stds[3] == 0.0


def test_process_contig_worker_none_coverage_returns_zeros():
    contig_data_list = [
        (
            "contig_1",
            {
                "fragments": ["frag_a", "frag_b"],
                "fragment_info": {},
            },
        )
    ]

    cov, cov_std, extra = _process_contig_coverage_worker(
        ("contig_1", contig_data_list, None, None)
    )

    assert cov == {"frag_a": 0.0, "frag_b": 0.0}
    assert cov_std == {"frag_a": 0.0, "frag_b": 0.0}
    assert extra == []


def test_process_contig_worker_handles_bad_fragment_positions():
    contig_data_list = [
        (
            "contig_1",
            {
                "fragments": ["missing_info", "neg_start", "beyond_end", "empty_range", "ok"],
                "fragment_info": {
                    "neg_start": {"start_pos": -5, "length": 10},
                    "beyond_end": {"start_pos": 500, "length": 5},
                    "empty_range": {"start_pos": 5, "length": 0},
                    "ok": {"start_pos": 2, "length": 4},
                },
            },
        )
    ]
    total_coverage = np.ones(20, dtype=np.float64)

    cov, cov_std, _ = _process_contig_coverage_worker(
        ("contig_1", contig_data_list, total_coverage, 20)
    )

    assert cov["missing_info"] == 0.0
    assert cov_std["missing_info"] == 0.0
    assert cov["neg_start"] == 0.0
    assert cov["beyond_end"] == 0.0
    assert cov["empty_range"] == 0.0
    assert cov["ok"] >= 0.0
    assert cov_std["ok"] >= 0.0


def test_calculate_fragment_coverage_corrupt_bam_returns_empty(tmp_path):
    bad_bam = tmp_path / "bad.bam"
    bad_bam.write_text("not-a-bam")

    cov, cov_std = calculate_fragment_coverage(
        bam_file=str(bad_bam),
        fragments_dict=_tiny_fragment_dict(),
        cores=1,
        disable_progress=True,
    )

    assert cov == {}
    assert cov_std == {}


def test_calculate_bam_cov_no_bams_returns_empty_df(tmp_path):
    fragments = _tiny_fragment_dict()
    df = calculate_bam_cov(
        bam_files=str(tmp_path),  # directory with no *.bam
        fragments_dict=fragments,
        threads=1,
    )

    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_calculate_bam_cov_empty_bam_returns_zero_df(tmp_path):
    bam_dir = tmp_path / "bam"
    bam_dir.mkdir()
    (bam_dir / "sample1.bam").write_text("")

    fragments = _tiny_fragment_dict()
    df = calculate_bam_cov(
        bam_files=str(bam_dir),
        fragments_dict=fragments,
        threads=1,
    )

    assert df.shape == (2, 2)
    assert list(df.columns) == ["sample1_coverage", "sample1_coverage_std"]
    assert np.all(df.values == 0.0)


def test_coverage_builder_max_norm_stacked_on_empty_bam_returns_ones(tmp_path):
    bam_dir = tmp_path / "bam"
    bam_dir.mkdir()
    (bam_dir / "sample1.bam").write_text("")

    from binferno.features.coverage import CoverageFeatureBuilder

    builder = CoverageFeatureBuilder()
    df = builder(
        fragment_dict=_tiny_fragment_dict(),
        bam_path=str(bam_dir),
        threads=1,
        logger=logger,
        max_norm_stacked=True,
    )

    assert df.shape == (2, 2)
    # zeros -> +pseudocount -> divide by max => 1.0
    assert np.allclose(df.values, 1.0, atol=1e-8)


def test_max_normalize_stacked_coverage_basic():
    df = pd.DataFrame(
        {
            "s1_cov": [0.0, 2.0, 4.0],
            "s1_std": [0.0, 1.0, 2.0],
        },
        index=["f1", "f2", "f3"],
    )

    out = max_normalize_stacked_coverage(df)

    assert out.shape == df.shape
    assert np.all(out.values > 0.0)
    assert np.isclose(out["s1_cov"].max(), 1.0)
    assert np.isclose(out["s1_std"].max(), 1.0)


def test_process_contig_worker_inner_exception_fallback():
    """Forces the inner except-Exception block (coverage.py lines 217-238) by
    injecting a RuntimeError from _calculate_fragment_stats_vectorized."""

    def _raise(*args, **kwargs):
        raise RuntimeError("injected error to exercise fallback path")

    coverage_array = np.arange(50, dtype=np.float64)
    contig_data_list = [
        (
            "contig_1",
            {
                "fragments": ["frag_a", "frag_b"],
                "fragment_info": {
                    "frag_a": {"start_pos": 0, "length": 10},
                    "frag_b": {"start_pos": 10, "length": 10},
                },
            },
        )
    ]

    with patch(
        "binferno.features.coverage._calculate_fragment_stats_vectorized", _raise
    ):
        frag_cov, frag_cov_std, _ = _process_contig_coverage_worker(
            ("contig_1", contig_data_list, coverage_array, 50)
        )

    assert "frag_a" in frag_cov
    assert "frag_b" in frag_cov
    assert np.isclose(frag_cov["frag_a"], np.mean(np.arange(10, dtype=np.float64)))
    assert np.isclose(frag_cov["frag_b"], np.mean(np.arange(10, 20, dtype=np.float64)))
    assert frag_cov_std["frag_a"] >= 0.0
    assert frag_cov_std["frag_b"] >= 0.0
