from __future__ import annotations

from glob import glob
from pathlib import Path
import shutil

import loguru
from pydantic import Field
import pandas as pd

from .base import BasePostprocessor
from ..clustering.base import ClusterResult
from .writer_utils import (
    estimate_bins_quality_nobins,
    estimate_bins_quality_nobins_pltd,
    filter_small_bins,
    GenomeProfiler,
)

class BinfernoPostprocessor(BasePostprocessor):
    checkm_database: str = Field(default="", description="Path to CheckM database")
    minbin_size: int = Field(
        default=200_000,
        description="Minimum bin size for filtering contigs before clustering.",
    )

    class Config:
        arbitrary_types_allowed = True

    def __call__(
        self,
        contigs: Path,
        cluster_result: ClusterResult,
        output_path: Path,
        num_threads: int = 5,
        logger: loguru._logger.Logger = None,
        plastid: bool = False,
    ) -> None:
        
        checkm_database = self.checkm_database

        if output_path is None:
            raise ValueError("output_path is required")
        if contigs is None:
            raise ValueError("contigs is required")

        output_path = Path(output_path)
        cluster_res_dir = output_path / "cluster_res"
        unitem_profile_dir = cluster_res_dir / "unitem_profile"

        if unitem_profile_dir.exists():
            shutil.rmtree(unitem_profile_dir)

        n_genome = cluster_result.n_genome

        logger.info(f"Seed_num:\t{n_genome}")
        logger.info(f"Run unitem profile:\t{n_genome}")

        res_name = f"weight_seed_kmeans_k_{n_genome}_result.tsv"

        # Dict[str, Tuple[str, str]]: res_name -> (bin_path, suffix)
        bin_dirs = {res_name: (str(cluster_res_dir / f"{res_name}_bins"), "fa")}

        assert not unitem_profile_dir.exists()
        unitem_profile_dir.mkdir(parents=True, exist_ok=False)
        
        profile = GenomeProfiler(cpus=num_threads, logger=logger)
        profile.profile_setup(checkm_database)

        if plastid:
            profile.run_pltd(bin_dirs, str(unitem_profile_dir))
        else:
            profile.run(bin_dirs, str(unitem_profile_dir)) # make sure that kmeans bins should be there. 

        bac_mg_table = (
            unitem_profile_dir
            / "binning_methods"
            / res_name
            / "checkm_bac"
            / "marker_gene_table.tsv"
        )
        ar_mg_table = (
            unitem_profile_dir
            / "binning_methods"
            / res_name
            / "checkm_ar"
            / "marker_gene_table.tsv"
        )
        pltd_mg_table = (
            unitem_profile_dir
            / "binning_methods"
            / res_name
            / "checkm_pltd"
            / "marker_gene_table.tsv"
        )

        if plastid:
            best_method = estimate_bins_quality_nobins_pltd(
                pltd_mg_table=str(pltd_mg_table),
                res_path=cluster_res_dir,
                cluster_result=cluster_result,
            )
        else:
            best_method = estimate_bins_quality_nobins(
                bac_mg_table=str(bac_mg_table),
                ar_mg_table=str(ar_mg_table),
                res_path=cluster_res_dir,
                cluster_result=cluster_result,
            )

        logger.info(f"Final result:\t{cluster_res_dir / best_method}")

        filter_small_bins(
            logger,
            str(contigs),
            str(cluster_res_dir / best_method),
            str(output_path),
            minbinsize=self.minbin_size,
        )
        return