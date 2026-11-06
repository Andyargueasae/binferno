from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from loguru import logger

import binferno.clustering.leiden as module
from binferno.clustering.leiden import LeidenClustering, _available_memory_bytes


@pytest.mark.parametrize("v2", [True, False])
@pytest.mark.parametrize("mount_root", ["/", "/job"])
def test_memory_headroom_includes_ancestor_limit(tmp_path, monkeypatch, v2, mount_root):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "meminfo").write_text("MemAvailable: 999999 kB\n")
    mount = tmp_path / "cgroup"
    group = mount / ("job/step" if mount_root == "/" else "step")
    group.mkdir(parents=True)
    (proc / "self/cgroup").write_text(
        "0::/job/step\n" if v2 else "5:memory:/job/step\n"
    )
    fs = "cgroup2 cgroup rw" if v2 else "cgroup cgroup rw,memory"
    (proc / "self/mountinfo").write_text(f"1 0 0:1 {mount_root} {mount} rw - {fs}\n")
    limit, usage = (
        ("memory.max", "memory.current")
        if v2
        else ("memory.limit_in_bytes", "memory.usage_in_bytes")
    )
    (group / limit).write_text("max" if v2 else str(2**63 - 4096))
    (group / usage).write_text("10")
    (group.parent / limit).write_text("1000")
    (group.parent / usage).write_text("600")
    assert _available_memory_bytes(proc) == 400
    (group / limit).write_text("50")
    (group / usage).write_text("70")
    assert _available_memory_bytes(proc) == 0


def test_slurm_without_job_accounting_does_not_use_node_ram(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    (tmp_path / "meminfo").write_text("MemAvailable: 999999 kB\n")
    assert _available_memory_bytes(tmp_path) is None


def test_worker_selection_scales_with_graph_and_respects_caps(monkeypatch):
    monkeypatch.setattr(module, "_available_memory_bytes", lambda: 100 * 2**30)
    model = LeidenClustering()
    small = model._select_workers(303451, 15, 12, 60, logger)
    large = model._select_workers(303451, 100, 12, 60, logger)
    assert small == 12
    assert 1 <= large < small
    assert (
        LeidenClustering(max_workers=2)._select_workers(303451, 15, 12, 60, logger) == 2
    )
    assert model._select_workers(303451, 15, 12, 1, logger) == 1
    assert model._select_workers(303451, 15, 1, 60, logger) == 1
    assert (
        LeidenClustering(worker_memory_gb=30)._select_workers(10, 15, 12, 60, logger)
        == 2
    )


def test_low_and_unknown_memory(monkeypatch):
    model = LeidenClustering()
    monkeypatch.setattr(module, "_available_memory_bytes", lambda: 100 * 2**20)
    with pytest.raises(MemoryError, match="Insufficient RAM"):
        model._select_workers(303451, 100, 12, 60, logger)
    monkeypatch.setattr(module, "_available_memory_bytes", lambda: None)
    assert model._select_workers(303451, 100, 12, 60, logger) == 2
    assert model._select_workers(303451, 100, 1, 60, logger) == 1


@pytest.mark.parametrize("cpm", [False, True])
def test_hnsw_and_leiden_have_separate_cpu_budgets(tmp_path, monkeypatch, cpm):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "6")
    monkeypatch.setattr(module.os, "sched_getaffinity", lambda _: set(range(4)))
    monkeypatch.setattr(module, "_available_memory_bytes", lambda: 100 * 2**30)
    monkeypatch.setattr(module, "gen_seed", lambda *args, **kw: 1)
    monkeypatch.setattr(module, "calc_n50", lambda _: 2000)
    monkeypatch.setattr(module, "_read_seed_count", lambda _: 1)
    monkeypatch.setattr(module, "gen_seed_idx", lambda *args, **kw: [0])
    monkeypatch.setattr(module, "seed_kmeans_full", lambda **kw: {})
    monkeypatch.setattr(
        module,
        "_filter_embeddings_by_length",
        lambda **kw: (
            np.zeros((120, 4), dtype=np.float32),
            np.array([f"c{i}" for i in range(120)]),
            np.full(120, 2000),
        ),
    )
    events = []
    submitted = []
    active = False

    def fit(self, **kw):
        assert not active
        events.append(("build", kw["num_threads"]))

        def query(features, k, num_threads):
            assert not active
            events.append(("query", num_threads))
            return np.zeros((120, k), dtype=np.uint64), np.zeros(
                (120, k), dtype=np.float32
            )

        return SimpleNamespace(knn_query=query)

    class Pool:
        def __init__(self, processes):
            events.append(("pool", processes))

        def __enter__(self):
            nonlocal active
            active = True
            return self

        def __exit__(self, *args):
            nonlocal active
            active = False

        def apply_async(self, fn, kwds):
            assert active
            submitted.append(kwds)
            for output_file, resolution in kwds["runs"]:
                output_file.write_text("c0\tgroup0\n")
            return SimpleNamespace(get=lambda: None)

    monkeypatch.setattr(LeidenClustering, "fit_hnsw_index", fit)
    monkeypatch.setattr(module.mp, "Pool", Pool)
    out = tmp_path / "out"
    out.mkdir()
    # A previous sweep's extra candidates must not enter the reduced sweep.
    unrelated = (
        out / "Leiden_bandwidth_0.3_maxedges_15_respara_30_partgraph_ratio_80.tsv"
    )
    unrelated.write_text("c0\told\n")
    model = LeidenClustering(max_workers=2, CPM=cpm)
    result = model(
        contigs=tmp_path / "contigs.fa",
        output_path=out,
        embeddings=pd.DataFrame([[0.0]]),
        threads=8,
        logger=logger,
    )
    assert events == [
        ("build", 4),
        ("query", 4),
        ("pool", 2),
        ("build", 4),
        ("query", 4),
        ("pool", 2 if cpm else 1),
    ]
    assert len(result.clusters) == (120 if cpm else 7)
    assert len(submitted) == (24 if cpm else 5)
    assert unrelated.name not in result.clusters
    expected = {
        f"Leiden_bandwidth_{b}_maxedges_{k}_respara_{r}_partgraph_ratio_{p}.tsv"
        for b, k, r, p in model._sweep_configurations()
    }
    assert set(result.clusters) == expected

    # Fully cached outputs are returned without submitting graph jobs.
    submitted.clear()
    reloaded = model(
        contigs=tmp_path / "contigs.fa",
        output_path=out,
        embeddings=pd.DataFrame([[0.0]]),
        threads=8,
        logger=logger,
    )
    assert not submitted
    assert set(reloaded.clusters) == expected

    # Resume only one missing resolution, preserving its cached graph sibling.
    missing = out / (
        "Leiden_bandwidth_0.05_maxedges_100_respara_"
        f"{'0.005' if cpm else '1'}_partgraph_ratio_50.tsv"
    )
    missing.unlink()
    model(
        contigs=tmp_path / "contigs.fa",
        output_path=out,
        embeddings=pd.DataFrame([[0.0]]),
        threads=8,
        logger=logger,
    )
    assert len(submitted) == 1
    assert [p for p, _ in submitted[0]["runs"]] == [missing]
