import numpy as np
from .base import BaseClustering
from typing import Sequence, Optional, List, Union, Any
from igraph import Graph
import leidenalg
import time
import hnswlib


class LeidenClustering(BaseClustering):
    def __call__(
        self,
        output_file: str,
        namelist: Sequence[str],
        ann_neighbor_indices: np.ndarray,
        ann_distances: np.ndarray,
        length_weight: Sequence[float],
        bandwidth: float = 0.1,
        lmode: str = "l2",
        initial_list: Optional[List[Union[int, None]]] = None,
        is_membership_fixed: Optional[Sequence[bool]] = None,
        resolution_parameter: float = 1.0,
        partgraph_ratio: int = 50,
    ) -> None:
        """
        Run Leiden on a sparsified ANN graph and write TSV assignments.
        """
        sources, targets, weights = self._prepare_leiden_edges(
            ann_neighbor_indices=ann_neighbor_indices,
            ann_distances=ann_distances,
            bandwidth=bandwidth,
            lmode=lmode,
            partgraph_ratio=partgraph_ratio,
        )

        vcount = len(namelist)
        graph = Graph(
            n=vcount,
            edges=list(zip(sources.tolist(), targets.tolist())),
            directed=False,
        )

        partition = leidenalg.RBERVertexPartition(
            graph,
            weights=weights.tolist(),
            initial_membership=initial_list,
            resolution_parameter=resolution_parameter,
            node_sizes=list(length_weight),
        )

        optimiser = leidenalg.Optimiser()
        optimiser.optimise_partition(
            partition,
            is_membership_fixed=is_membership_fixed,
            n_iterations=-1,
        )

        assignments = {}
        for cluster_id, members in enumerate(partition):
            label = f"group{cluster_id}"
            for node_idx in members:
                assignments[namelist[node_idx]] = label

        with open(output_file, "w") as handle:
            for contig_name in namelist:
                handle.write(f"{contig_name}\t{assignments[contig_name]}\n")

        return

    def _prepare_leiden_edges(
        self,
        ann_neighbor_indices: np.ndarray,
        ann_distances: np.ndarray,
        bandwidth: float,
        lmode: str,
        partgraph_ratio: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build a sparsified undirected edge list and transformed weights from ANN output.
        """
        if bandwidth <= 0:
            raise ValueError("bandwidth must be > 0")

        neighbor_targets = ann_neighbor_indices[:, 1:]
        neighbor_distances = ann_distances[:, 1:]

        k = neighbor_targets.shape[1]
        sources = np.repeat(np.arange(neighbor_targets.shape[0]), k)
        targets = neighbor_targets.reshape(-1)
        distances = neighbor_distances.reshape(-1)

        dist_cutoff = np.percentile(distances, partgraph_ratio)
        keep = distances <= dist_cutoff

        sources = sources[keep]
        targets = targets[keep]
        distances = distances[keep]

        non_self = sources != targets
        sources = sources[non_self]
        targets = targets[non_self]
        distances = distances[non_self]

        upper = sources < targets
        sources = sources[upper]
        targets = targets[upper]
        distances = distances[upper]

        if lmode == "l1":
            weights = np.exp(-np.sqrt(distances) / bandwidth)
        elif lmode == "l2":
            weights = np.exp(-distances / bandwidth)
        else:
            raise ValueError(f"Unsupported lmode: {lmode}")

        return (
            sources.astype(np.int32),
            targets.astype(np.int32),
            weights.astype(np.float64),
        )

    def fit_hnsw_index(
        self,
        features: np.ndarray,
        num_threads: int,
        ef: int = 100,
        M: int = 16,
        space: str = "l2",
        save_index_file: Optional[str] = None,
        logger: Optional[Any] = None,
    ) -> hnswlib.Index:
        """
        Build an HNSW index for the embedding matrix.
        """
        start = time.time()

        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError("features must be a non-empty 2D array")

        n_samples, dim = features.shape
        labels = np.arange(n_samples, dtype=np.int32)

        index = hnswlib.Index(space=space, dim=dim)
        index.init_index(max_elements=n_samples, ef_construction=ef, M=M)
        index.add_items(features, labels, num_threads=num_threads)
        index.set_ef(ef)

        if save_index_file:
            index.save_index(save_index_file)

        logger.info(f"HNSW build time:\t{time.time() - start:.3f}s")
        return index
