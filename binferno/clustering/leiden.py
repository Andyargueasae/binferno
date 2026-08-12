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
from pydantic import Field

from ..utils import gen_seed, calc_n50
from .base import BaseClustering, ClusterResult
from .cluster_utils import (
    _filter_embeddings_by_length,
    seed_kmeans_full,
    gen_seed_idx,
    _read_seed_count,
    apply_dimension_reduction,
)

class LeidenClustering(BaseClustering):
    seed: int = 42
    mutual: bool = Field(False, description="Whether to use mutual kNN filtering in Leiden clustering.")
    cluster_min_length: int = Field(1000, description="Minimum contig length for phase-1 clustering.")
    CPM: bool = Field(False, description="Whether to use CPM (Constant Potts Model) for Leiden clustering instead of RBER (Reichardt-Bornholdt).")

    def __call__(
        self,
        contigs: Path,
        output_path: Path,
        embeddings: pd.DataFrame,
        logger: loguru._logger.Logger,
        threads: int = 1,
        plastid: bool = False,
        longread: bool = False,
    ) -> ClusterResult:
        np.random.seed(self.seed)

        result = ClusterResult(
            clusters={},
            n_genome=0,
        )  # instantiate an empty ClusterResult to store results and return at the end.

        if plastid:
            n_genome = gen_seed(
                logger,
                contig_file=contigs,
                output_dir=output_path,
                threads=threads,
                contig_length_threshold=self.contig_len + 1,
                marker_name="pltd_marker",
                quarter="2quarter",
            )
            seed_file = (
                output_path
                / f"{contigs.name}.pltd_marker.2quarter_lencutoff_{self.contig_len + 1}.seed"
            )
        else:
            n_genome = gen_seed(
                logger,
                contig_file=contigs,
                output_dir=output_path,
                threads=threads,
                contig_length_threshold=self.contig_len + 1,
                marker_name="bacar_marker",
                quarter="2quarter",
            )
            seed_file = (
                output_path
                / f"{contigs.name}.bacar_marker.2quarter_lencutoff_{self.contig_len + 1}.seed"
            )

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
            contig_len=self.cluster_min_length,
        )

        if len(namelist) == 0:
            raise ValueError("No contigs remain after length filtering")

        logger.info(f"N50:\t{calc_n50(contigs)}")
        logger.info(f"Contigs retained after length filter: {len(namelist)}")

        norm_embeddings = normalize(emb_mat).astype(np.float32, copy=False)

        # check if to apply dimension reduction.
        if self.dimension_reduction:
            norm_embeddings = apply_dimension_reduction(norm_embeddings)

        # Two-phase clustering will incorporate shorter contigs below 2500 bp. We can first try with mutual KNN, and then decide what to do. 
        # seed_thresh = 2500  # or self.seed_len, a new param
        # length_arr = np.asarray(length_weight)          # note: length_weight is per-contig length here
        # long_idx = np.flatnonzero(length_arr >= seed_thresh)
        # short_idx = np.flatnonzero(length_arr < seed_thresh)

        # emb_long = norm_embeddings[long_idx]
        # namelist_long = [namelist[i] for i in long_idx]

        seed_num = _read_seed_count(seed_file)
        mode = f"weight_seed_kmeans"

        bin_nums = seed_num
        logger.info(f"Bin_numbers:\t{bin_nums}")

        kmeans_out = seed_kmeans_full(
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
        for key, val in kmeans_out.items():
            result.clusters[key] = val

        num_workers = max(1, int(threads or 1))

        if not self.CPM:
            parameter_list = [0.5, 1, 5, 10, 30, 50, 70, 90, 100]
        else:
            parameter_list = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
        bandwidth_list = [0.05, 0.1, 0.2, 0.3] # we may have to replace this by actual-data driven sweeping.
        partgraph_ratio_list = [50, 80, 100]
        max_edges_list = [15, 100] # [15 may not always be optimal]

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
                        logger.warning(
                            f"Output file {output_file} already exists. Skipping Leiden run and save to ClusterResult."
                        )
                        df = pd.read_csv(
                            output_file,
                            sep="\t",
                            header=None,
                            names=["contig_id", "cluster_id"],
                        )
                        result.clusters[output_file.name] = df
                        continue

                    cluster_res = pool.apply_async(
                        self.run_leiden_memmap,
                        kwds={
                            "output_file": output_file,
                            "namelist": namelist,
                            "neighbor_path": str(neighbor_path),
                            "distance_path": str(distance_path),
                            "length_weight_path": str(length_weight_path),
                            "bandwidth": bandwidth,
                            "lmode": "l2",
                            "initial_list": initial_list,
                            "is_membership_fixed_path": str(fixed_path),
                            "resolution_parameter": resolution,
                            "partgraph_ratio": partgraph_ratio,
                        },
                    )
                    jobs.append(cluster_res)

            for cluster_res in jobs:
                cluster_res.get()

        # save the leiden clustering results to ClusterResult.
        for file in output_path.glob("Leiden_*.tsv"):
            df = pd.read_csv(
                file, sep="\t", header=None, names=["contig_id", "cluster_id"]
            )
            result.clusters[file.name] = df

        return result

    def run_leiden(
        self,
        output_file: Path,
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
            mutual=self.mutual,
        )

        vcount = len(namelist)
        graph = Graph(
            n=vcount,
            edges=list(zip(sources.tolist(), targets.tolist())),
            directed=False,
        )

        if not self.CPM:
            partition = leidenalg.RBERVertexPartition(
                graph,
                weights=weights.tolist(),
                initial_membership=initial_list,
                resolution_parameter=resolution_parameter,
                node_sizes=list(length_weight),
            )
        else:
            partition = leidenalg.CPMVertexPartition(
                graph,
                weights=weights.tolist(),
                initial_membership=initial_list,
                resolution_parameter=resolution_parameter,
                # node_sizes=list(length_weight),
            ) 

        optimiser = leidenalg.Optimiser()
        optimiser.optimise_partition(
            partition,
            is_membership_fixed=is_membership_fixed, # OK, this one should also be looked for. Run once on and once off. 
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

    def _prepare_leiden_edges(
        self,
        ann_neighbor_indices: np.ndarray,
        ann_distances: np.ndarray,
        bandwidth: float,
        lmode: str,
        partgraph_ratio: float,
        mutual: bool = True,          # NEW, mutual-KNN filter.
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if bandwidth <= 0:
            raise ValueError("bandwidth must be > 0")

        bandwidth_weight = [0.5, 1.0, 2.0]

        neighbor_targets = ann_neighbor_indices[:, 1:]
        neighbor_distances = ann_distances[:, 1:]

        k = neighbor_targets.shape[1]
        sources = np.repeat(np.arange(neighbor_targets.shape[0]), k)
        targets = neighbor_targets.reshape(-1)
        distances = neighbor_distances.reshape(-1)

        # global distance prune (existing behaviour)
        dist_cutoff = np.percentile(distances, partgraph_ratio)
        keep = distances <= dist_cutoff
        sources, targets, distances = sources[keep], targets[keep], distances[keep]

        non_self = sources != targets
        sources, targets, distances = sources[non_self], targets[non_self], distances[non_self]

        # ---- NEW: mutual-kNN filter, before the upper-triangle collapse ----
        if mutual:
            directed = set(zip(sources.tolist(), targets.tolist()))
            is_mutual = np.fromiter(
                ((t, s) in directed for s, t in zip(sources.tolist(), targets.tolist())),
                dtype=bool, count=len(sources),
            ) # this line of code checks if the reverse edge (t, s) exists in the directed set of edges (s, t). If it does, then the edge is mutual and is kept; otherwise, it is filtered out.
            sources, targets, distances = sources[is_mutual], targets[is_mutual], distances[is_mutual] # update the sources, targets, and distances arrays to only include mutual edges

        # collapse to undirected upper triangle (existing, will reduce the number of edges by half)
        upper = sources < targets
        sources, targets, distances = sources[upper], targets[upper], distances[upper]

        # the similarity weighting scheme.
        if lmode == "l1":
            weights = np.exp(-np.sqrt(distances) / bandwidth)
        elif lmode == "l2":
            weights = np.exp(-distances / bandwidth)
        else:
            raise ValueError(f"Unsupported lmode: {lmode}")

        return sources.astype(np.int32), targets.astype(np.int32), weights.astype(np.float64)

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
        initial_list: Optional[List[Union[int, None]]],
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

    def recruit_short(self, index_long, emb_short, short_names, long_assignments,
                  namelist_long, k=15, agreement=0.6, num_threads=1):
        """Assign each short contig to the majority community among its long-contig
        neighbours, if agreement clears the floor; else leave unbinned."""
        nbr_idx, _ = index_long.knn_query(emb_short, k=k, num_threads=num_threads)
        out = {}
        for i, name in enumerate(short_names):
            labels = [long_assignments[namelist_long[j]] for j in nbr_idx[i]]
            vals, counts = np.unique(labels, return_counts=True)
            top = counts.argmax()
            if counts[top] / counts.sum() >= agreement:
                out[name] = vals[top]
            # else: unbinned, deliberately omitted
        return out
