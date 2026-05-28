import itertools
import os
from pathlib import Path
from typing import Sequence
from sklearn.preprocessing import normalize
from pydantic import Field
import numpy as np
import pandas as pd
from .base import BaseClustering
from .leiden import LeidenClustering
import multiprocessing as mp


class BinfernoClustering(BaseClustering):

    leiden: LeidenClustering = Field(
        default_factory=LeidenClustering,
        description="Leiden clustering module. Only used if use_leiden is True.",
    )
    dimension_reduction: bool = Field(
        default=False,
        description="Whether to perform dimension reduction before clustering.",
    )

    class Config:
        arbitrary_types_allowed = True

    def __call__(
        self,
        embeddings: pd.DataFrame,
        seed_file: str = "",
        contig_file: str = "",
        output_path: Path = None,
        contig_len: int = 1000,
        cluster_num: int = None,
        threads: int = 1,
        **kwargs,
    ) -> None:
        np.random.seed(42)
        logger = kwargs.get("logger", None)
        if embeddings is None or embeddings.empty:
            raise ValueError("embeddings must be a non-empty DataFrame")
        if seed_file is None or contig_file is None or output_path is None:
            raise ValueError("seed_file, contig_file, and output_path are required")

        logger.info("Start clustering.")

        output_path.mkdir(parents=True, exist_ok=True)

        emb_mat, namelist, length_weight = self._filter_embeddings_by_length(
            emb_df=embeddings,
            contig_file=contig_file,
            contig_len=contig_len,
        )

        if len(namelist) == 0:
            raise ValueError("No contigs remain after length filtering")

        logger.info(f"N50:\t{self.calculate_n50(length_weight.tolist())}")
        logger.info(f"Contigs retained after length filter: {len(namelist)}")

        norm_embeddings = normalize(emb_mat).astype(np.float32, copy=False)

        # check if to apply dimension reduction.
        if self.dimension_reduction:
            norm_embeddings = self.apply_dimension_reduction(norm_embeddings)

        seed_num = self._read_seed_count(seed_file)
        mode = f"weight_seed_kmeans"

        bin_nums = [seed_num]
        if cluster_num is not None:
            bin_nums.append(cluster_num)
        bin_nums = sorted(set(bin_nums))

        logger.info(f"Bin_numbers:\t{bin_nums}")

        logger.info(f"Bin_numbers:\t{bin_nums}")
        for k in bin_nums:
            self.seed_kmeans_full(
                logger=logger,
                contig_file=contig_file,
                namelist=namelist,
                out_path=str(output_path) + os.sep,
                X_mat=norm_embeddings,
                bin_number=k,
                prefix=mode,
                length_weight=length_weight,
                seed_file=seed_file,
            )

        num_workers = max(1, int(threads or 1))

        parameter_list = [0.5, 1, 10]
        bandwidth_list = [0.05, 0.1, 0.2, 0.3]
        partgraph_ratio_list = [50, 80, 100]
        max_edges_list = [15, 100]

        seed_indices = self.gen_seed_idx(seed_file, contig_id_list=namelist)
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
                index = self.leiden.fit_hnsw_index(
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
                            "output_file": str(output_file),
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
                    jobs.append(job)

            for job in jobs:
                job.get()
        return

    def run_leiden_memmap(
        self,
        output_file: str,
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

        self.leiden(
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
