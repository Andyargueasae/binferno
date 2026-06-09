from pathlib import Path
import os
import itertools

import numpy as np
import pandas as pd
from typing import Sequence, Optional, List, Union, Any
from igraph import Graph
from sklearn.preprocessing import normalize
import leidenalg
import time
import hnswlib
import loguru
import multiprocessing as mp

from ..utils import gen_seed, calc_n50
from .base import BaseClustering, ClusterResult
from .cluster_utils import (
    _filter_embeddings_by_length,
    seed_kmeans_full,
    gen_seed_idx,
    _read_seed_count,
    apply_dimension_reduction
)


class LeidenClustering(BaseClustering):
    seed:int = 42

    def __call__(
        self,
        contigs: Path,
        output_path: Path,
        embeddings: pd.DataFrame,
        logger: loguru._logger.Logger,
        threads: int=1,
        plastid: bool = False,
    ) -> ClusterResult:
        np.random.seed(self.seed)

        result = ClusterResult(
            kmeans_clusters=[],
            leiden_cluster_assignments={},
            n_genome=0,
        ) # instantiate an empty ClusterResult to store results and return at the end.

        if plastid:
            n_genome = gen_seed(
                logger,
                contig_file=contigs,
                output_dir=output_path,
                threads=threads,
                contig_length_threshold=1001,
                marker_name="pltd_marker",
                quarter="2quarter",
            )
            seed_file = output_path / f"{contigs.name}.pltd_marker.2quarter_lencutoff_1001.seed"
        else:
            n_genome = gen_seed(
                logger,
                contig_file=contigs,
                output_dir=output_path,
                threads=threads,
                contig_length_threshold=1001,
                marker_name="bacar_marker",
                quarter="2quarter",
            )
            seed_file = output_path / f"{contigs.name}.bacar_marker.2quarter_lencutoff_1001.seed"

        result.n_genome = n_genome

        if embeddings is None or embeddings.empty:
            raise ValueError("embeddings must be a non-empty DataFrame")
        if seed_file is None or contigs is None or output_path is None:
            raise ValueError("seed_file, contigs, and output_path are required")

        logger.info("Start clustering.")

        output_path.mkdir(parents=True, exist_ok=True)

        emb_mat, namelist, length_weight = _filter_embeddings_by_length(
            emb_df=embeddings,
            contig_file=contigs,
            contig_len=self.contig_len,
        )

        if len(namelist) == 0:
            raise ValueError("No contigs remain after length filtering")

        logger.info(f"N50:\t{calc_n50(contigs)}")
        logger.info(f"Contigs retained after length filter: {len(namelist)}")

        norm_embeddings = normalize(emb_mat).astype(np.float32, copy=False)

        # check if to apply dimension reduction.
        if self.dimension_reduction:
            norm_embeddings = apply_dimension_reduction(norm_embeddings)

        seed_num = _read_seed_count(seed_file)
        mode = f"weight_seed_kmeans"

        bin_nums = seed_num
        bin_nums = sorted(set(bin_nums))

        logger.info(f"Bin_numbers:\t{bin_nums}")
        
        kmeans_output = seed_kmeans_full(
            logger=logger,
            contig_file=contigs,
            namelist=namelist,
            out_path=str(output_path) + os.sep,
            X_mat=norm_embeddings,
            bin_number=bin_nums,
            prefix=mode,
            length_weight=length_weight,
            seed_file=seed_file,
        )
        result.kmeans_clusters = kmeans_output

        num_workers = max(1, int(threads or 1))

        parameter_list = [0.5, 1, 10]
        bandwidth_list = [0.05, 0.1, 0.2, 0.3]
        partgraph_ratio_list = [50, 80, 100]
        max_edges_list = [15, 100]

        seed_indices = gen_seed_idx(seed_file, contig_id_list=namelist)
        initial_list = list(np.arange(len(namelist), dtype=int))
        seed_set = set(seed_indices)
        is_membership_fixed = np.array(
            [i in seed_set for i in initial_list],
            dtype=bool,
        )

        memmap_dir = output_path / "leiden_memmap"
        memmap_dir.mkdir(parents=True, exist_ok=True)

        length_weight_path = memmap_dir / "length_weight.npy"
        fixed_path = memmap_dir / "is_membership_fixed.npy"

        np.save(length_weight_path, np.asarray(length_weight, dtype=np.float64))
        np.save(fixed_path, np.asarray(is_membership_fixed, dtype=bool))

        jobs = []
        # after the last two runs, update clustering. Also, debug the "both" mode, and add-corgi mode.
        with mp.Pool(processes=num_workers) as pool:
            for max_edges in max_edges_list:
                index = self.fit_hnsw_index(
                    logger=logger,
                    features=norm_embeddings,
                    num_threads=num_workers,
                    ef=max_edges * 10,
                    M=16,
                    space="l2",
                )

                ann_neighbor_indices, ann_distances = index.knn_query(
                    norm_embeddings,
                    k=max_edges + 1,
                    num_threads=num_workers,
                )

                neighbor_path = memmap_dir / f"neighbors_maxedges_{max_edges}.npy"
                distance_path = memmap_dir / f"distances_maxedges_{max_edges}.npy"

                np.save(neighbor_path, ann_neighbor_indices)
                np.save(distance_path, ann_distances)

                del ann_neighbor_indices, ann_distances

                for partgraph_ratio, bandwidth, resolution in itertools.product(
                    partgraph_ratio_list,
                    bandwidth_list,
                    parameter_list,
                ):
                    output_file = (
                        output_path
                        / f"Leiden_bandwidth_{bandwidth}_maxedges_{max_edges}"
                        f"_respara_{resolution}_partgraph_ratio_{partgraph_ratio}.tsv"
                    )

                    if output_file.exists():
                        continue

                    job = pool.apply_async(
                        self.run_leiden_memmap,
                        kwds={
                            "output_file": output_file,
                            "namelist": namelist,
                            "neighbor_path": str(neighbor_path),
                            "distance_path": str(distance_path),
                            "length_weight_path": str(length_weight_path),
                            "bandwidth": bandwidth,
                            "lmode": "l2",
                            "result_wrapper": result,
                            "initial_list": initial_list,
                            "is_membership_fixed_path": str(fixed_path),
                            "resolution_parameter": resolution,
                            "partgraph_ratio": partgraph_ratio,
                        },
                    )
                    jobs.append(job)

            for job in jobs:
                job.get()
        return result
    
    def run_leiden(
        self,
        output_file: Path,
        namelist: Sequence[str],
        ann_neighbor_indices: np.ndarray,
        ann_distances: np.ndarray,
        length_weight: Sequence[float],
        result_wrapper: ClusterResult,
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

        df = pd.DataFrame.from_dict(assignments, orient="index", columns=["cluster"])
        df.reset_index(inplace=True)
        df.rename(columns={"index": "contig_id", "cluster": "cluster_id"}, inplace=True)
        result_wrapper.leiden_cluster_assignments[output_file.name] = df

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

    def run_leiden_memmap(
        self,
        output_file: Path,
        namelist: Sequence[str],
        neighbor_path: str,
        distance_path: str,
        length_weight_path: str,
        bandwidth: float,
        lmode: str,
        initial_list,
        is_membership_fixed_path: str,
        resolution_parameter: float,
        partgraph_ratio: int,
    ) -> None:
        ann_neighbor_indices = np.load(neighbor_path, mmap_mode="r")
        ann_distances = np.load(distance_path, mmap_mode="r")
        length_weight = np.load(length_weight_path, mmap_mode="r")
        is_membership_fixed = np.load(is_membership_fixed_path, mmap_mode="r")

        self.run_leiden(
            output_file=output_file,
            namelist=namelist,
            ann_neighbor_indices=ann_neighbor_indices,
            ann_distances=ann_distances,
            length_weight=length_weight,
            bandwidth=bandwidth,
            lmode=lmode,
            initial_list=initial_list,
            is_membership_fixed=is_membership_fixed,
            resolution_parameter=resolution_parameter,
            partgraph_ratio=partgraph_ratio,
        )