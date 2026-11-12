"""Graph reuse must preserve independent, seeded Leiden optimisations."""

import numpy as np
import pandas as pd
import pytest
import leidenalg
from igraph import Graph
from loguru import logger

import binferno.clustering.leiden as module
from binferno.clustering.leiden import LeidenClustering


def test_shortlist_runs_in_process_pool(tmp_path, monkeypatch):
    """Exercise actual HNSW, memmaps, grouped process jobs and TSV collection."""
    contigs = tmp_path / "contigs.fa"
    names = [f"c{i}" for i in range(120)]
    contigs.write_text("".join(f">{name}\n{'ACGT' * 300}\n" for name in names))
    output = tmp_path / "out"
    output.mkdir()
    (output / f"{contigs.name}.bacar_marker.2quarter_lencutoff_1001.seed").write_text(
        "c0\n"
    )
    monkeypatch.setattr(module, "gen_seed", lambda *args, **kwargs: 1)
    monkeypatch.setattr(module, "seed_kmeans_full", lambda **kwargs: {})
    monkeypatch.setattr(module, "_available_memory_bytes", lambda: 100 * 2**30)
    features = np.random.default_rng(17).normal(size=(120, 8)).astype(np.float32)
    result = LeidenClustering(max_workers=2, sweep_config="mini")(
        contigs=contigs,
        output_path=output,
        embeddings=pd.DataFrame(features, index=names),
        threads=2,
        logger=logger,
    )
    assert result.n_genome == 1
    assert len(result.clusters) == 7
    for assignments in result.clusters.values():
        assert assignments["contig_id"].tolist() == names
        assert assignments["cluster_id"].notna().all()


@pytest.mark.parametrize("cpm", [False, True])
@pytest.mark.parametrize("mutual", [False, True])
@pytest.mark.parametrize("taxonomy", [False, True])
def test_shared_graph_matches_independent_partitions(
    tmp_path, monkeypatch, cpm, mutual, taxonomy
):
    rng = np.random.default_rng(12)
    features = rng.normal(size=(30, 4))
    distances = ((features[:, None] - features[None, :]) ** 2).sum(axis=-1)
    neighbors = np.argsort(distances, axis=1)[:, :9]
    distances = np.take_along_axis(distances, neighbors, axis=1)
    names = [f"c{i}" for i in range(30)]
    sizes = rng.integers(1, 4, size=30).astype(float)
    initial = list(range(30))
    fixed = np.zeros(30, dtype=bool)
    fixed[[0, 7]] = True
    ids = np.ones((30, 7), dtype=np.int32)
    ids[15:, 5] = 2
    masks = np.ones_like(ids, dtype=np.uint8)
    model = LeidenClustering(
        CPM=cpm, mutual=mutual, taxonomy_edge_factor=0.35 if taxonomy else 1.0
    )
    resolutions = [0.01, 0.1, 0.5] if cpm else [0.5, 1, 5]

    # Independent reference follows the pre-refactor partition construction.
    sources, targets, weights = model._prepare_leiden_edges(
        neighbors,
        distances,
        bandwidth=0.1,
        lmode="l2",
        partgraph_ratio=80,
        mutual=mutual,
    )
    if taxonomy:
        weights, _ = module.penalize_taxonomy_conflicts(
            sources,
            targets,
            weights,
            ids,
            masks,
            conflict_rank="genus",
            conflict_factor=0.35,
        )
    expected = []
    for resolution in resolutions:
        graph = Graph(n=30, edges=list(zip(sources.tolist(), targets.tolist())))
        partition_type = (
            leidenalg.CPMVertexPartition if cpm else leidenalg.RBERVertexPartition
        )
        kwargs = {} if cpm else {"node_sizes": list(sizes)}
        partition = partition_type(
            graph,
            weights=weights.tolist(),
            initial_membership=initial,
            resolution_parameter=resolution,
            **kwargs,
        )
        optimiser = leidenalg.Optimiser()
        optimiser.set_rng_seed(model.seed)
        optimiser.optimise_partition(
            partition, is_membership_fixed=fixed, n_iterations=-1
        )
        labels = {
            node: f"group{ci}"
            for ci, members in enumerate(partition)
            for node in members
        }
        expected.append(
            "".join(f"{name}\t{labels[i]}\n" for i, name in enumerate(names))
        )

    paths = {}
    for name, array in [
        ("neighbor", neighbors),
        ("distance", distances),
        ("length_weight", sizes),
        ("is_membership_fixed", fixed),
    ]:
        paths[f"{name}_path"] = str(tmp_path / f"{name}.npy")
        np.save(paths[f"{name}_path"], array)
    if taxonomy:
        paths["ids_path"] = tmp_path / "ids.npy"
        paths["masks_path"] = tmp_path / "masks.npy"
        np.save(paths["ids_path"], ids)
        np.save(paths["masks_path"], masks)

    graphs = []
    penalties = []
    original_penalty = module.penalize_taxonomy_conflicts

    def construct_graph(*args, **kwargs):
        graph = Graph(*args, **kwargs)
        graphs.append(graph)
        return graph

    def penalize(*args, **kwargs):
        penalties.append(1)
        return original_penalty(*args, **kwargs)

    monkeypatch.setattr(module, "Graph", construct_graph)
    monkeypatch.setattr(module, "penalize_taxonomy_conflicts", penalize)
    outputs = [(tmp_path / f"res_{r}.tsv", r) for r in resolutions]
    model.run_leiden_resolutions_memmap(
        runs=outputs,
        namelist=names,
        bandwidth=0.1,
        lmode="l2",
        initial_list=initial,
        partgraph_ratio=80,
        **paths,
    )
    assert len(graphs) == 1
    assert len(penalties) == int(taxonomy)
    assert initial == list(range(30))
    assert [p.read_text() for p, _ in outputs] == expected
