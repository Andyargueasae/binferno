from pathlib import Path
import pandas as pd
import pytest
from loguru import logger

from binferno.postprocessing import BinfernoPostprocessor
from binferno.clustering.base import ClusterResult


def _write_tiny_contigs(path: Path) -> None:
    path.write_text(
        ">contig_1\n" + "ATGC" * 400 + "\n" ">contig_2\n" + "GCAT" * 400 + "\n"
    )


def test_postprocessor_validates_required_args(tmp_path):
    processor = BinfernoPostprocessor(checkm_database=tmp_path, minbin_size=1000)
    cluster_result = ClusterResult(n_genome=1, clusters={})

    with pytest.raises(ValueError):
        processor(
            contigs=None,
            cluster_result=cluster_result,
            output_path=tmp_path,
            num_threads=1,
            logger=logger,
            plastid=False,
        )

    contigs = tmp_path / "contigs.fna"
    _write_tiny_contigs(contigs)

    with pytest.raises(ValueError):
        processor(
            contigs=contigs,
            cluster_result=cluster_result,
            output_path=None,
            num_threads=1,
            logger=logger,
            plastid=False,
        )


def test_postprocessor_calls_expected_pipeline_non_plastid(tmp_path, monkeypatch):
    contigs = tmp_path / "contigs.fna"
    _write_tiny_contigs(contigs)

    output_dir = tmp_path / "out"
    cluster_result = ClusterResult(
        n_genome=2,
        clusters={
            "weight_seed_kmeans_k_2_result.tsv": pd.DataFrame(
                {
                    "contig_id": ["contig_1", "contig_2"],
                    "cluster_id": ["group0", "group1"],
                }
            )
        },
    )

    calls = {"profile_setup": 0, "run": 0, "run_pltd": 0, "estimate": 0, "filter": 0}

    class DummyProfiler:
        def __init__(self, cpus, logger):
            self.cpus = cpus
            self.logger = logger

        def profile_setup(self, database):
            calls["profile_setup"] += 1

        def run(self, bin_dirs, out_dir):
            calls["run"] += 1

        def run_pltd(self, bin_dirs, out_dir):
            calls["run_pltd"] += 1

    def fake_estimate_bins_quality_nobins(
        bac_mg_table, ar_mg_table, res_path, cluster_result
    ):
        calls["estimate"] += 1
        return "weight_seed_kmeans_k_2_result.tsv"

    def fake_filter_small_bins(
        log, contigs_path, best_method_path, output_path, minbinsize
    ):
        calls["filter"] += 1

    monkeypatch.setattr("binferno.postprocessing.GenomeProfiler", DummyProfiler)
    monkeypatch.setattr(
        "binferno.postprocessing.estimate_bins_quality_nobins",
        fake_estimate_bins_quality_nobins,
    )
    monkeypatch.setattr(
        "binferno.postprocessing.filter_small_bins",
        fake_filter_small_bins,
    )

    processor = BinfernoPostprocessor(checkm_database=tmp_path, minbin_size=1000)
    processor(
        contigs=contigs,
        cluster_result=cluster_result,
        output_path=output_dir,
        num_threads=1,
        logger=logger,
        plastid=False,
    )

    assert calls["profile_setup"] == 1
    assert calls["run"] == 1
    assert calls["run_pltd"] == 0
    assert calls["estimate"] == 1
    assert calls["filter"] == 1
    assert (output_dir / "cluster_res" / "unitem_profile").exists()


def test_postprocessor_calls_expected_pipeline_plastid(tmp_path, monkeypatch):
    contigs = tmp_path / "contigs.fna"
    _write_tiny_contigs(contigs)

    output_dir = tmp_path / "out"
    cluster_result = ClusterResult(
        n_genome=1,
        clusters={"weight_seed_kmeans_k_1_result.tsv": pd.DataFrame()},
    )

    calls = {"run": 0, "run_pltd": 0, "estimate_pltd": 0}

    class DummyProfiler:
        def __init__(self, cpus, logger):
            self.cpus = cpus
            self.logger = logger

        def profile_setup(self, database):
            pass

        def run(self, bin_dirs, out_dir):
            calls["run"] += 1

        def run_pltd(self, bin_dirs, out_dir):
            calls["run_pltd"] += 1

    def fake_estimate_bins_quality_nobins_pltd(pltd_mg_table, res_path, cluster_result):
        calls["estimate_pltd"] += 1
        return "weight_seed_kmeans_k_1_result.tsv"

    def fake_filter_small_bins(
        log, contigs_path, best_method_path, output_path, minbinsize
    ):
        pass

    monkeypatch.setattr("binferno.postprocessing.GenomeProfiler", DummyProfiler)
    monkeypatch.setattr(
        "binferno.postprocessing.estimate_bins_quality_nobins_pltd",
        fake_estimate_bins_quality_nobins_pltd,
    )
    monkeypatch.setattr(
        "binferno.postprocessing.filter_small_bins",
        fake_filter_small_bins,
    )

    processor = BinfernoPostprocessor(checkm_database=tmp_path, minbin_size=1000)
    processor(
        contigs=contigs,
        cluster_result=cluster_result,
        output_path=output_dir,
        num_threads=1,
        logger=logger,
        plastid=True,
    )

    assert calls["run"] == 0
    assert calls["run_pltd"] == 1
    assert calls["estimate_pltd"] == 1
