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
)
from .taxonomy_cluster import penalize_taxonomy_conflicts, prepare_taxonomy_arrays
from ..features.taxonomy import TAXONOMIC_RANKS


def _available_memory_bytes(proc_root: Path = Path("/proc")) -> Optional[int]:
    """Remaining host/job RAM, including limits on ancestor cgroups (v1/v2)."""
    headrooms = []
    job_limit_found = False
    try:
        for line in (proc_root / "meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                headrooms.append(int(line.split()[1]) * 1024)
        memberships = []
        for line in (proc_root / "self/cgroup").read_text().splitlines():
            _, controllers, member = line.split(":", 2)
            if not controllers or "memory" in controllers.split(","):
                memberships.append((not controllers, Path(member)))
        for line in (proc_root / "self/mountinfo").read_text().splitlines():
            before, after = line.split(" - ", 1)
            fields, fs = before.split(), after.split()
            for v2, member in memberships:
                if fs[0] != ("cgroup2" if v2 else "cgroup"):
                    continue
                if not v2 and "memory" not in fs[2].split(","):
                    continue
                mount = Path(fields[4].replace(r"\040", " "))
                root = Path(fields[3].replace(r"\040", " "))
                try:
                    relative = member.relative_to(root)
                except ValueError:
                    # Cgroup namespaces can report membership relative to the mount.
                    relative = member.relative_to("/")
                if ".." in relative.parts:
                    continue
                group = mount / relative
                while True:
                    limit_name = "memory.max" if v2 else "memory.limit_in_bytes"
                    usage_name = "memory.current" if v2 else "memory.usage_in_bytes"
                    try:
                        limit = int((group / limit_name).read_text().strip())
                        usage = int((group / usage_name).read_text().strip())
                        # v1 represents unlimited RAM with a near-int64-max value.
                        if 0 <= limit < (1 << 60):
                            headrooms.append(max(0, limit - usage))
                            job_limit_found = True
                    except (OSError, ValueError):
                        pass
                    if group == mount:
                        break
                    group = group.parent
    except (OSError, ValueError, IndexError):
        pass
    if os.environ.get("SLURM_JOB_ID") and not job_limit_found:
        # Node RAM alone does not establish a Slurm job's remaining allocation.
        return None
    return min(headrooms) if headrooms else None


class LeidenClustering(BaseClustering):
    seed: int = 42
    mutual: bool = Field(
        False, description="Whether to use mutual kNN filtering in Leiden clustering."
    )
    contig_len: int = Field(
        1000, description="Minimum contig length for seed generation."
    )
    cluster_min_length: int = Field(
        1000, description="Minimum contig length for phase-1 clustering."
    )
    CPM: bool = Field(
        False,
        description="Whether to use CPM (Constant Potts Model) for Leiden clustering instead of RBER (Reichardt-Bornholdt) [DEPRECATED].",
    )
    taxonomy_edge_factor: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description="Genus conflict edge multiplier; 1.0 disables taxonomy refinement.",
    )

    max_workers: int = Field(
        default=0,
        ge=0,
        description="Maximum concurrent Leiden runs; 0 selects from CPU and RAM budgets.",
    )
    worker_memory_gb: float = Field(
        default=0.0,
        ge=0.0,
        description="Estimated peak additional GiB per Leiden worker; 0 estimates from graph size.",
    )
    memory_reserve_fraction: float = Field(
        default=0.25,
        ge=0.0,
        lt=1.0,
        description="Fraction of remaining RAM reserved when selecting Leiden workers.",
    )

    def _select_workers(
        self,
        n_contigs: int,
        max_edges: int,
        cpu_budget: int,
        pending_jobs: int,
        logger: Any,
    ) -> int:
        """Choose concurrency before each graph sweep; estimates are not an OOM guarantee."""
        # Heuristic for array copies, Python edge tuples, igraph and optimiser state.
        # Mutual filtering also builds a Python set of the directed edges.
        bytes_per_edge = 1024 if self.mutual else 768
        estimate = (
            max(1, int(self.worker_memory_gb * 2**30))
            if self.worker_memory_gb
            else 512 * 2**20 + n_contigs * (256 + max_edges * bytes_per_edge)
        )
        available = _available_memory_bytes()
        cap = min(cpu_budget, pending_jobs, self.max_workers or cpu_budget)
        if available is None:
            workers = min(cap, 2)
            logger.warning(
                "Job RAM headroom unavailable; limiting Leiden to {} workers.", workers
            )
        else:
            usable = int(available * (1.0 - self.memory_reserve_fraction))
            workers = min(cap, usable // estimate)
            if workers < 1:
                raise MemoryError(
                    f"Insufficient RAM for one estimated Leiden worker: "
                    f"{usable / 2**30:.2f} GiB usable, {estimate / 2**30:.2f} GiB needed. "
                    "Request more job RAM or reduce graph size; worker_memory_gb "
                    "can override the estimate if measured usage is available."
                )
        logger.info(
            "Leiden max_edges={}: {} workers / {} CPUs; estimated {:.2f} GiB per worker; "
            "RAM headroom {} GiB (reserve {:.0%}).",
            max_edges,
            workers,
            cpu_budget,
            estimate / 2**30,
            f"{available / 2**30:.2f}" if available is not None else "unknown",
            self.memory_reserve_fraction,
        )
        return workers

    def __call__(
        self,
        contigs: Path,
        output_path: Path,
        embeddings: pd.DataFrame,
        logger: loguru._logger.Logger,
        threads: int = 1,
        plastid: bool = False,
        longread: bool = False,
        taxonomy: pd.DataFrame | None = None,
    ) -> ClusterResult:
        use_taxonomy = self.taxonomy_edge_factor < 1.0
        if use_taxonomy and taxonomy is None:
            raise ValueError(
                "Taxonomy edge refinement requires a taxonomy table. "
                "Set taxonomy_edge_factor=1.0 to use ordinary Leiden clustering."
            )
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

        norm_embeddings = emb_mat

        # check if to apply dimension reduction.

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

        cpu_budget = max(1, int(threads or 1))
        if hasattr(os, "sched_getaffinity"):
            cpu_budget = min(cpu_budget, len(os.sched_getaffinity(0)))
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        if slurm_cpus and slurm_cpus.isdigit():
            cpu_budget = min(cpu_budget, max(1, int(slurm_cpus)))
        logger.info("HNSW construction and query CPU budget: {}", cpu_budget)

        configurations = self._sweep_configurations()
        max_edges_list = sorted({config[1] for config in configurations})

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

        ids_path = masks_path = None
        if use_taxonomy:
            tax_ids, tax_masks = prepare_taxonomy_arrays(
                taxonomy, namelist, TAXONOMIC_RANKS
            )
            ids_path = memmap_dir / "taxonomy_ids.npy"
            masks_path = memmap_dir / "taxonomy_masks.npy"
            np.save(ids_path, tax_ids)
            np.save(masks_path, tax_masks)
            logger.info(
                "Taxonomy edge refinement enabled: genus conflict factor {}",
                self.taxonomy_edge_factor,
            )

        output_prefix = (
            f"Leiden_taxedge_genus_factor_{self.taxonomy_edge_factor:g}_"
            if use_taxonomy
            else "Leiden_"
        )

        output_files = []
        for max_edges in max_edges_list:
            index = self.fit_hnsw_index(
                logger=logger,
                features=norm_embeddings,
                num_threads=cpu_budget,
                ef=max_edges * 10,
                M=16,
                space="l2",
            )

            ann_neighbor_indices, ann_distances = index.knn_query(
                norm_embeddings,
                k=max_edges + 1,
                num_threads=cpu_budget,
            )

            neighbor_path = memmap_dir / f"neighbors_maxedges_{max_edges}.npy"
            distance_path = memmap_dir / f"distances_maxedges_{max_edges}.npy"

            np.save(neighbor_path, ann_neighbor_indices)
            np.save(distance_path, ann_distances)

            del ann_neighbor_indices, ann_distances, index

            pending = {}

            for bandwidth, neighbors, resolution, partgraph_ratio in configurations:
                if neighbors != max_edges:
                    continue
                output_file = (
                    output_path
                    / f"{output_prefix}bandwidth_{bandwidth}_maxedges_{max_edges}"
                    f"_respara_{resolution}_partgraph_ratio_{partgraph_ratio}.tsv"
                )
                output_files.append(output_file)

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

                # Resolutions share the weighted graph, but start from fresh
                # partitions and optimiser state inside the worker.
                group = pending.setdefault(
                    (bandwidth, partgraph_ratio),
                    {
                        "runs": [],
                        "namelist": namelist,
                        "neighbor_path": str(neighbor_path),
                        "distance_path": str(distance_path),
                        "length_weight_path": str(length_weight_path),
                        "bandwidth": bandwidth,
                        "lmode": "l2",
                        "initial_list": initial_list,
                        "is_membership_fixed_path": str(fixed_path),
                        "partgraph_ratio": partgraph_ratio,
                        "ids_path": ids_path,
                        "masks_path": masks_path,
                    },
                )
                group["runs"].append((output_file, resolution))

            if not pending:
                continue
            workers = self._select_workers(
                len(namelist), max_edges, cpu_budget, len(pending), logger
            )
            # Do not overlap multithreaded HNSW with memory-heavy Leiden runs.
            with mp.Pool(processes=workers) as pool:
                jobs = [
                    pool.apply_async(self.run_leiden_resolutions_memmap, kwds=kwds)
                    for kwds in pending.values()
                ]
                for cluster_res in jobs:
                    cluster_res.get()

        # Only return this sweep's candidates, including when an earlier full
        # sweep left additional TSV files in the same output directory.
        for file in output_files:
            df = pd.read_csv(
                file, sep="\t", header=None, names=["contig_id", "cluster_id"]
            )
            result.clusters[file.name] = df

        return result

    def _sweep_configurations(self) -> list[tuple[float, int, float, int]]:
        """Return (bandwidth, neighbours, resolution, pruning percentile) tuples."""
        if self.CPM:
            # The eight RBER quality tables do not justify reducing CPM's grid.
            return [
                (bandwidth, neighbors, resolution, percentile)
                for neighbors, percentile, bandwidth, resolution in itertools.product(
                    [15, 100],
                    [50, 80, 100],
                    [0.05, 0.1, 0.2, 0.3],
                    [0.001, 0.005, 0.01, 0.05, 0.1],
                )
            ]
        # Seven explicit configurations retain the best (num_905, sum_cont5)
        # pair in each of the eight benchmark tables; do not expand as a grid.
        return [
            (0.1, 15, 1, 100),
            (0.1, 15, 5, 100),
            (0.1, 15, 1, 80),
            (0.2, 15, 1, 100),
            (0.05, 15, 0.5, 100),
            (0.05, 100, 0.5, 50),
            (0.05, 100, 1, 50),
        ]

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
        taxonomy_ids: Optional[np.ndarray] = None,
        taxonomy_mask: Optional[np.ndarray] = None,
    ) -> None:
        """
        Run Leiden on a sparsified ANN graph and write TSV assignments.
        """
        graph, weights = self._build_leiden_graph(
            namelist,
            ann_neighbor_indices,
            ann_distances,
            bandwidth,
            lmode,
            partgraph_ratio,
            taxonomy_ids,
            taxonomy_mask,
        )
        self._optimise_leiden_graph(
            graph,
            weights,
            output_file,
            namelist,
            length_weight,
            initial_list,
            is_membership_fixed,
            resolution_parameter,
        )

    def _build_leiden_graph(
        self,
        namelist: Sequence[str],
        ann_neighbor_indices: np.ndarray,
        ann_distances: np.ndarray,
        bandwidth: float,
        lmode: str,
        partgraph_ratio: int,
        taxonomy_ids: Optional[np.ndarray] = None,
        taxonomy_mask: Optional[np.ndarray] = None,
    ) -> tuple[Graph, list[float]]:
        """Prepare one weighted graph for all resolutions with these settings."""
        sources, targets, weights = self._prepare_leiden_edges(
            ann_neighbor_indices=ann_neighbor_indices,
            ann_distances=ann_distances,
            bandwidth=bandwidth,
            lmode=lmode,
            partgraph_ratio=partgraph_ratio,
            mutual=self.mutual,
        )

        if taxonomy_ids is not None and self.taxonomy_edge_factor < 1.0:
            if taxonomy_mask is None:
                raise ValueError("Taxonomy masks are required with taxonomy IDs.")
            weights, _ = penalize_taxonomy_conflicts(
                sources,
                targets,
                weights,
                taxonomy_ids,
                taxonomy_mask,
                conflict_rank="genus",
                conflict_factor=self.taxonomy_edge_factor,
            )

        vcount = len(namelist)
        graph = Graph(
            n=vcount,
            edges=list(zip(sources.tolist(), targets.tolist())),
            directed=False,
        )
        return graph, weights.tolist()

    def _optimise_leiden_graph(
        self,
        graph: Graph,
        weights: Sequence[float],
        output_file: Path,
        namelist: Sequence[str],
        length_weight: Sequence[float],
        initial_list: Optional[List[Union[int, None]]],
        is_membership_fixed: Optional[Sequence[bool]],
        resolution_parameter: float,
    ) -> None:
        """Optimise a fresh partition; never warm-start from another resolution."""
        if not self.CPM:
            partition = leidenalg.RBERVertexPartition(
                graph,
                weights=weights,
                initial_membership=initial_list,
                resolution_parameter=resolution_parameter,
                node_sizes=list(length_weight),
            )
        else:
            partition = leidenalg.CPMVertexPartition(
                graph,
                weights=weights,
                initial_membership=initial_list,
                resolution_parameter=resolution_parameter,
                # node_sizes=list(length_weight),
            )

        optimiser = leidenalg.Optimiser()
        optimiser.set_rng_seed(self.seed)
        optimiser.optimise_partition(
            partition,
            is_membership_fixed=is_membership_fixed,  # OK, this one should also be looked for. Run once on and once off.
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
        mutual: bool = True,  # NEW, mutual-KNN filter.
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
        dist_cutoff = np.percentile(distances, partgraph_ratio) # what is this? partgraph_ratio is a percentile threshold for pruning edges based on their distances. It calculates the distance value below which a certain percentage (partgraph_ratio) of the distances fall. Edges with distances greater than this cutoff are filtered out, effectively reducing the number of edges in the graph and focusing on closer neighbors for clustering.
        keep = distances <= dist_cutoff
        sources, targets, distances = sources[keep], targets[keep], distances[keep]

        non_self = sources != targets
        sources, targets, distances = (
            sources[non_self],
            targets[non_self],
            distances[non_self],
        )

        # ---- NEW: mutual-kNN filter, before the upper-triangle collapse ----
        if mutual:
            directed = set(zip(sources.tolist(), targets.tolist()))
            is_mutual = np.fromiter(
                (
                    (t, s) in directed
                    for s, t in zip(sources.tolist(), targets.tolist())
                ),
                dtype=bool,
                count=len(sources),
            )  # this line of code checks if the reverse edge (t, s) exists in the directed set of edges (s, t). If it does, then the edge is mutual and is kept; otherwise, it is filtered out.
            sources, targets, distances = (
                sources[is_mutual],
                targets[is_mutual],
                distances[is_mutual],
            )  # update the sources, targets, and distances arrays to only include mutual edges

        # collapse to undirected upper triangle (existing, will reduce the number of edges by half)
        upper = sources < targets
        sources, targets, distances = sources[upper], targets[upper], distances[upper]

        # the similarity weighting scheme. bandwidth is a hyperparameter that controls the scale of the similarity function. lmode determines the type of distance metric used to compute the weights. The weights are computed using an exponential decay function based on the distances and bandwidth.
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
        # temporarily add num_threads=1 for the two rep checks. After running the two-rep check, move back.
        # num_threads = 1
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
        ids_path: Optional[Path] = None,
        masks_path: Optional[Path] = None,
    ) -> None:
        self.run_leiden_resolutions_memmap(
            runs=[(output_file, resolution_parameter)],
            namelist=namelist,
            neighbor_path=neighbor_path,
            distance_path=distance_path,
            length_weight_path=length_weight_path,
            bandwidth=bandwidth,
            lmode=lmode,
            initial_list=initial_list,
            is_membership_fixed_path=is_membership_fixed_path,
            partgraph_ratio=partgraph_ratio,
            ids_path=ids_path,
            masks_path=masks_path,
        )

    def run_leiden_resolutions_memmap(
        self,
        runs: Sequence[tuple[Path, float]],
        namelist: Sequence[str],
        neighbor_path: str,
        distance_path: str,
        length_weight_path: str,
        bandwidth: float,
        lmode: str,
        initial_list: Optional[List[Union[int, None]]],
        is_membership_fixed_path: str,
        partgraph_ratio: int,
        ids_path: Optional[Path] = None,
        masks_path: Optional[Path] = None,
    ) -> None:
        """Load ANN arrays and build a graph once per group of resolutions."""
        ann_neighbor_indices = np.load(neighbor_path, mmap_mode="r")
        ann_distances = np.load(distance_path, mmap_mode="r")
        length_weight = np.load(length_weight_path, mmap_mode="r")
        is_membership_fixed = np.load(is_membership_fixed_path, mmap_mode="r")

        taxonomy_ids = (
            np.load(ids_path, mmap_mode="r") if ids_path is not None else None
        )
        taxonomy_mask = (
            np.load(masks_path, mmap_mode="r") if masks_path is not None else None
        )

        graph, weights = self._build_leiden_graph(
            namelist=namelist,
            ann_neighbor_indices=ann_neighbor_indices,
            ann_distances=ann_distances,
            bandwidth=bandwidth,
            lmode=lmode,
            partgraph_ratio=partgraph_ratio,
            taxonomy_ids=taxonomy_ids,
            taxonomy_mask=taxonomy_mask,
        )
        for output_file, resolution in runs:
            self._optimise_leiden_graph(
                graph=graph,
                weights=weights,
                output_file=output_file,
                namelist=namelist,
                length_weight=length_weight,
                initial_list=initial_list,
                is_membership_fixed=is_membership_fixed,
                resolution_parameter=resolution,
            )

    def recruit_short(
        self,
        index_long,
        emb_short,
        short_names,
        long_assignments,
        namelist_long,
        k=15,
        agreement=0.6,
        num_threads=1,
    ):
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
