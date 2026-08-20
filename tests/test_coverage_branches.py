import numpy as np
import pandas as pd
from loguru import logger
from unittest.mock import patch

from binferno.features.COMEBin_coverage import (
    _calculate_fragment_stats_vectorized,
    _process_contig_coverage_worker,
    _normalise_coverage_dicts,
    calculate_bam_cov,
    calculate_fragment_coverage,
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


def test_comebin_calculate_fragment_stats_vectorized_small_branch():
    coverage = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    coords = [(0, 2), (2, 5)]

    means, stds = _calculate_fragment_stats_vectorized(coverage, coords)

    assert means.shape == (2,)
    assert stds.shape == (2,)
    assert np.isclose(means[0], 1.5)
    assert np.isclose(means[1], 4.0)
    assert np.all(stds >= 0.0)


def test_comebin_calculate_fragment_stats_vectorized_large_branch():
    coverage = np.arange(100, dtype=np.float64)
    coords = [(i, i + 5) for i in range(0, 50, 5)] + [(50, 60), (60, 70)]

    means, stds = _calculate_fragment_stats_vectorized(coverage, coords)

    assert means.shape == (12,)
    assert stds.shape == (12,)
    assert np.all(np.isfinite(means))
    assert np.all(np.isfinite(stds))
    assert np.all(stds >= 0.0)


def test_comebin_calculate_fragment_stats_vectorized_fallback_on_bad_indices():
    coverage = np.arange(20, dtype=np.float64)
    coords = [(0, 5)] * 10
    coords[3] = (1000, 1005)

    means, stds = _calculate_fragment_stats_vectorized(coverage, coords)

    assert means.shape == (10,)
    assert stds.shape == (10,)
    assert np.all(np.isfinite(means))
    assert np.all(np.isfinite(stds))
    assert means[3] == 0.0
    assert stds[3] == 0.0


def test_comebin_process_contig_worker_none_coverage_returns_zeros():
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
        ("contig_1", contig_data_list, None, None, False)
    )

    assert cov == {"frag_a": 0.0, "frag_b": 0.0}
    assert cov_std == {"frag_a": 0.0, "frag_b": 0.0}
    assert extra == []


def test_comebin_process_contig_worker_handles_bad_fragment_positions():
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
        ("contig_1", contig_data_list, total_coverage, 20, False)
    )

    assert cov["missing_info"] == 0.0
    assert cov_std["missing_info"] == 0.0
    assert cov["neg_start"] == 0.0
    assert cov["beyond_end"] == 0.0
    assert cov["empty_range"] == 0.0
    assert cov["ok"] >= 0.0
    assert cov_std["ok"] >= 0.0


def test_comebin_process_contig_worker_inner_exception_fallback():
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

    def _raise(*args, **kwargs):
        raise RuntimeError("injected error to exercise fallback path")

    with patch(
        "binferno.features.COMEBin_coverage._calculate_fragment_stats_vectorized",
        _raise,
    ):
        frag_cov, frag_cov_std, _ = _process_contig_coverage_worker(
            ("contig_1", contig_data_list, coverage_array, 50, False)
        )

    assert "frag_a" in frag_cov
    assert "frag_b" in frag_cov
    assert np.isclose(frag_cov["frag_a"], np.mean(np.arange(10, dtype=np.float64)))
    assert np.isclose(frag_cov["frag_b"], np.mean(np.arange(10, 20, dtype=np.float64)))
    assert frag_cov_std["frag_a"] >= 0.0
    assert frag_cov_std["frag_b"] >= 0.0


def test_comebin_calculate_fragment_coverage_corrupt_bam_returns_empty(tmp_path):
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


def test_comebin_calculate_bam_cov_no_bams_returns_empty_df(tmp_path):
    df = calculate_bam_cov(
        bam_files=str(tmp_path),
        fragments_dict=_tiny_fragment_dict(),
        threads=1,
    )

    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_comebin_calculate_bam_cov_empty_bam_returns_zero_df(tmp_path):
    bam_dir = tmp_path / "bam"
    bam_dir.mkdir()
    (bam_dir / "sample1.bam").write_text("")

    df = calculate_bam_cov(
        bam_files=str(bam_dir),
        fragments_dict=_tiny_fragment_dict(),
        threads=1,
    )

    assert df.shape == (2, 2)
    assert list(df.columns) == ["sample1_coverage", "sample1_coverage_std"]
    assert np.all(df.values == 0.0)


def test_comebin_calculate_bam_cov_mode_comebin_column_order(tmp_path):
    bam_dir = tmp_path / "bam"
    bam_dir.mkdir()
    (bam_dir / "sample1.bam").write_text("")

    df = calculate_bam_cov(
        bam_files=str(bam_dir),
        fragments_dict=_tiny_fragment_dict(),
        threads=1,
        mode="comebin",
    )

    assert list(df.columns) == ["sample1_coverage_var", "sample1_coverage"]


def test_comebin_normalise_coverage_dicts_zero_total_mapped():
    cov = {"frag_1": 3.0, "frag_2": 4.0}
    std = {"frag_1": 1.0, "frag_2": 2.0}

    out_cov, out_std = _normalise_coverage_dicts(
        coverage=cov,
        coverage_std=std,
        all_fragment_headers=["frag_1", "frag_2"],
        total_mapped=0,
        rpkm_normalize=True,
    )

    assert out_cov == {"frag_1": 0.0, "frag_2": 0.0}
    assert out_std == {"frag_1": 0.0, "frag_2": 0.0}


def test_comebin_normalise_coverage_dicts_skips_rpkm_when_disabled():
    cov = {"frag_1": 3.0, "frag_2": 4.0}
    std = {"frag_1": 1.0, "frag_2": 2.0}

    out_cov, out_std = _normalise_coverage_dicts(
        coverage=cov,
        coverage_std=std,
        all_fragment_headers=["frag_1", "frag_2"],
        total_mapped=10,
        rpkm_normalize=False,
    )

    assert out_cov == cov
    assert out_std == std