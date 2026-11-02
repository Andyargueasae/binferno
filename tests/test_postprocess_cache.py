"""Regression coverage for profiling reuse without invoking external CheckM."""

from pathlib import Path
import subprocess

import pytest
from loguru import logger

from binferno.postprocessing.profile_cache import database_signature
from binferno.postprocessing.writer_utils import GenomeProfiler, compute_bin_quality_cache


@pytest.fixture
def profile_job(tmp_path, monkeypatch):
    bins = tmp_path / "bins"
    bins.mkdir()
    (bins / "one.fa").write_text(">contig_1\nATGC\n")
    marker = tmp_path / "markers.ms"
    marker.write_text("markers")
    database = tmp_path / "database"
    database.mkdir()
    (database / "checkm.hmm").write_text("model")
    monkeypatch.setattr("binferno.postprocessing.writer_utils.check_on_path", lambda _: None)
    monkeypatch.setattr("binferno.postprocessing.writer_utils.shutil.which", lambda _: None)
    profiler = GenomeProfiler(cpus=1, logger=logger)
    profiler._database_signature = database_signature(database)
    calls = []

    def run(cmd, step):
        calls.append(cmd)
        if cmd[1] == "analyze":
            storage = Path(cmd[-1]) / "storage"
            storage.mkdir()
            for name in ("bin_stats.analyze.tsv", "checkm_hmm_info.pkl.gz"):
                (storage / name).write_text("analysis")
        else:
            Path(cmd[cmd.index("-f") + 1]).write_text("header\nresult\n")

    monkeypatch.setattr(profiler, "_run_cmd", run)
    args = (str(bins), "fa", str(marker), str(tmp_path / "profile"))
    return profiler, args, calls, run, database


def test_unchanged_profile_reuses_all_steps(profile_job):
    profiler, args, calls, _, _ = profile_job
    first = profiler._run_checkm_job(*args)
    assert len(calls) == 3
    assert profiler._run_checkm_job(*args) == first
    assert len(calls) == 3
    profiler._run_checkm_job(*args, skip_existing=False)
    assert len(calls) == 6


@pytest.mark.parametrize("changed", ["bin", "marker", "database", "tools", "added_bin", "removed_bin"])
def test_profile_invalidates_changed_inputs(profile_job, changed):
    profiler, args, calls, _, database = profile_job
    extra = Path(args[0]) / "extra.fa"
    if changed == "removed_bin":
        extra.write_text(">contig_2\nATGC\n")
    profiler._run_checkm_job(*args)
    if changed == "bin":
        # Same length and filename: file size alone must not allow stale reuse.
        (Path(args[0]) / "one.fa").write_text(">contig_1\nAAAA\n")
    elif changed == "marker":
        Path(args[2]).write_text("changed")
    elif changed == "database":
        (database / "checkm.hmm").write_text("changed")
        profiler._database_signature = database_signature(database)
    elif changed == "tools":
        profiler._tools_signature["checkm_version"] = "another version"
    elif changed == "added_bin":
        extra.write_text(">contig_2\nATGC\n")
    else:
        extra.unlink()
    profiler._run_checkm_job(*args)
    assert len(calls) == 6


def test_failed_qa_resumes_analysis_but_replaces_partial_table(profile_job, monkeypatch):
    profiler, args, calls, run, _ = profile_job

    def fail(cmd, step):
        run(cmd, step)
        if cmd[1] == "qa":
            raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(profiler, "_run_cmd", fail)
    with pytest.raises(subprocess.CalledProcessError):
        profiler._run_checkm_job(*args)
    monkeypatch.setattr(profiler, "_run_cmd", run)
    profiler._run_checkm_job(*args)
    assert [cmd[1] for cmd in calls] == ["analyze", "qa", "qa", "qa"]


def test_failed_analysis_restarts_instead_of_reusing_directory(profile_job, monkeypatch):
    profiler, args, calls, run, _ = profile_job

    def fail(cmd, step):
        run(cmd, step)
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(profiler, "_run_cmd", fail)
    with pytest.raises(subprocess.CalledProcessError):
        profiler._run_checkm_job(*args)
    monkeypatch.setattr(profiler, "_run_cmd", run)
    profiler._run_checkm_job(*args)
    assert [cmd[1] for cmd in calls] == ["analyze", "analyze", "qa", "qa"]


def test_missing_report_only_reruns_that_qa_step(profile_job):
    profiler, args, calls, _, _ = profile_job
    marker_table, _ = profiler._run_checkm_job(*args)
    Path(marker_table).unlink()
    profiler._run_checkm_job(*args)
    assert len(calls) == 4
    assert calls[-1][calls[-1].index("-o") + 1] == "5"


def test_legacy_or_corrupt_cache_is_rebuilt(profile_job):
    profiler, args, calls, _, _ = profile_job
    profiler._run_checkm_job(*args)
    (Path(args[3]) / "profile_state.json").write_text("{broken")
    profiler._run_checkm_job(*args)
    assert len(calls) == 6


def test_identical_bins_share_scores_across_candidate_clusterings():
    class Markers:
        def __init__(self):
            self.calls = []

        def bin_quality(self, contigs):
            self.calls.append(frozenset(contigs))
            return "Bacteria", 50 * len(contigs), 0.0

    markers = Markers()
    bins = {
        "first": {"a": {"c1", "c2"}, "b": {"c3"}},
        "second": {"renamed": {"c2", "c1"}, "different": {"c2", "c3"}},
    }
    scores = compute_bin_quality_cache(bins, sorted(bins), markers)
    assert len(markers.calls) == 3
    assert scores == {
        "first": {"a": (100, 0.0), "b": (50, 0.0)},
        "second": {"renamed": (100, 0.0), "different": (100, 0.0)},
    }


def test_marker_only_mode_skips_optional_qa_and_can_add_it_later(profile_job):
    profiler, args, calls, _, _ = profile_job
    profiler._run_checkm_job(*args, quality_reports=False)
    assert [cmd[1] for cmd in calls] == ["analyze", "qa"]
    assert calls[-1][calls[-1].index("-o") + 1] == "5"
    profiler._run_checkm_job(*args, quality_reports=False)
    assert len(calls) == 2
    profiler._run_checkm_job(*args, quality_reports=True)
    assert len(calls) == 3
    assert calls[-1][calls[-1].index("-o") + 1] == "2"


@pytest.mark.parametrize("plastid", [False, True])
def test_marker_only_profiling_does_not_parse_or_write_quality_reports(
    profile_job, monkeypatch, plastid
):
    profiler, args, calls, _, _ = profile_job
    for name in ("CHECKM_BAC_MS", "CHECKM_AR_MS", "CHECKM_PLTD_MS"):
        monkeypatch.setattr("binferno.postprocessing.writer_utils." + name, args[2])

    def unexpected(*args, **kwargs):
        pytest.fail("Marker-only profiling must not consume quality reports")

    monkeypatch.setattr(profiler, "_report_genome_quality", unexpected)
    monkeypatch.setattr(profiler, "_parse_quality_table", unexpected)
    run = profiler.run_pltd if plastid else profiler.run
    run({"method": (args[0], "fa")}, args[3], quality_reports=False)
    expected = 2 if plastid else 4
    assert len(calls) == expected
    run({"method": (args[0], "fa")}, args[3], quality_reports=False)
    assert len(calls) == expected
