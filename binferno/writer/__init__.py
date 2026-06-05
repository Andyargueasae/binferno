from __future__ import annotations

from pathlib import Path
import shutil

import loguru
from pydantic import Field
from biolib.common import make_sure_path_exists
import pandas as pd

from .base import BaseWriter
from .writer_utils import (
    estimate_bins_quality_nobins,
    estimate_bins_quality_nobins_pltd,
    filter_small_bins,
    Profile,
)

class BinfernoWriter(BaseWriter):
    checkm_database: str = Field(default="", description="Path to CheckM database")
    minbin_size: int = Field(
        default=200000,
        description="Minimum bin size for filtering contigs before clustering.",
    )

    class Config:
        arbitrary_types_allowed = True

    def __call__(
        self,
        contigs: Path,
        output_path: Path,
        seed_num: int = 0,
        ignore_kmeans_res: bool = False,
        num_threads: int = 5,
        logger: loguru._logger.Logger = None,
        res_name: str = None,
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

        logger.info(f"Seed_num:\t{seed_num}")
        logger.info(f"Run unitem profile:\t{seed_num}")

        if res_name is None:
            res_name = f"weight_seed_kmeans_k_{seed_num}_result.tsv"

        bin_dirs = {res_name: (str(cluster_res_dir / f"{res_name}_bins"), "fa")}

        if not unitem_profile_dir.exists():
            make_sure_path_exists(str(unitem_profile_dir))
            profile = Profile(cpus=num_threads, logger=logger)
            profile.profile_setup(checkm_database)

            if plastid:
                profile.run_pltd(bin_dirs, str(unitem_profile_dir))
            else:
                profile.run(bin_dirs, str(unitem_profile_dir))
                # if the job fails, remove the unitem_profile_dir to allow re-running the profile step/
                if not unitem_profile_dir.exists():
                    shutil.rmtree(unitem_profile_dir)

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
                res_path=str(cluster_res_dir),
                ignore_kmeans_res=ignore_kmeans_res,
            )
        else:
            best_method = estimate_bins_quality_nobins(
                bac_mg_table=str(bac_mg_table),
                ar_mg_table=str(ar_mg_table),
                res_path=str(cluster_res_dir),
                ignore_kmeans_res=ignore_kmeans_res,
            )

        logger.info(f"Final result:\t{cluster_res_dir / best_method}")

        filter_small_bins(
            logger,
            contigs,
            str(cluster_res_dir / best_method),
            str(output_path),
            minbinsize=self.minbin_size,
        )
        return