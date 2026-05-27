from __future__ import annotations
from pathlib import Path

from .base import BaseResults
from pydantic import Field
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
from biolib.common import make_sure_path_exists
import pandas as pd
from .filter_small_bins import filter_small_bins
from .scripts.unitem_markers import Markers, PltdMarkers

BinDict = Dict[str, Dict[str, set]]
ContigAssignmentDict = Dict[str, Dict[str, str]]
QualityCache = Dict[str, Dict[str, Tuple[float, float]]]


def get_binstats(bin_contig_names, markers):
    """
    Return (completeness, contamination) for a bin.
    """
    _, comp, cont = markers.bin_quality(bin_contig_names)
    return comp, cont


def list_result_tsvs(res_path: str, ignore_kmeans_res: bool = False) -> List[str]:
    """
    Collect clustering result TSV filenames from a result directory.
    """
    selected = []
    for filename in os.listdir(res_path):
        if not filename.endswith(".tsv"):
            continue
        if ignore_kmeans_res and filename.startswith("weight"):
            continue
        selected.append(filename)

    selected.sort()
    return selected


def read_bins_nosequences(
    bin_dirs: Dict[str, str],
) -> Tuple[BinDict, ContigAssignmentDict]:
    """
    Read bin assignments from TSV result files.

    Each TSV is expected to contain:
        contig_id <tab> cluster_id
    or equivalent first-two-column assignment layout.

    Returns
    -------
    bins:
        bins[method_id][cluster_id] -> set(contig_ids)
    contigs_in_bins:
        contigs_in_bins[contig_id][method_id] -> cluster_id
    """
    bins: BinDict = defaultdict(lambda: defaultdict(set))
    contigs_in_bins: ContigAssignmentDict = defaultdict(dict)

    for method_id, bin_file in bin_dirs.items():
        df = pd.read_csv(
            bin_file,
            sep="\t",
            header=None,
            usecols=[0, 1],
            names=["contig_id", "cluster_id"],
        )

        for contig_id, cluster_id in zip(df["contig_id"], df["cluster_id"]):
            cluster_key = str(cluster_id)
            bins[method_id][cluster_key].add(contig_id)
            contigs_in_bins[contig_id][method_id] = cluster_key

    return bins, contigs_in_bins


def compute_bin_quality_cache(
    bins: BinDict,
    methods_sorted: List[str],
    markers,
) -> QualityCache:
    """
    Compute and cache (completeness, contamination) for every bin of every method.
    """
    quality_cache: QualityCache = defaultdict(dict)

    for method_id in methods_sorted:
        for bin_id, contig_names in bins[method_id].items():
            comp, cont = get_binstats(contig_names, markers)
            quality_cache[method_id][bin_id] = (comp, cont)

    return quality_cache


def summarize_method_quality(
    bins: BinDict,
    methods_sorted: List[str],
    quality_cache: QualityCache,
):
    """
    Build per-method summary counts and choose the best method.

    Scoring criteria preserved from the original code.
    """
    bin_quality_dict = defaultdict(dict)
    method_scores = []
    method_scores_cont5 = []

    for method_id in methods_sorted:
        num_5010 = 0
        num_7010 = 0
        num_9010 = 0
        num_505 = 0
        num_705 = 0
        num_905 = 0

        for bin_id in bins[method_id]:
            comp, cont = quality_cache[method_id][bin_id]

            if comp > 50 and cont < 10:
                num_5010 += 1
            if comp > 70 and cont < 10:
                num_7010 += 1
            if comp > 90 and cont < 10:
                num_9010 += 1
            if comp > 50 and cont < 5:
                num_505 += 1
            if comp > 70 and cont < 5:
                num_705 += 1
            if comp > 90 and cont < 5:
                num_905 += 1

        total_score = num_5010 + num_7010 + num_9010 + num_505 + num_705 + num_905
        total_score_cont5 = num_505 + num_705 + num_905

        bin_quality_dict[method_id]["num_5010"] = num_5010
        bin_quality_dict[method_id]["num_7010"] = num_7010
        bin_quality_dict[method_id]["num_9010"] = num_9010
        bin_quality_dict[method_id]["num_505"] = num_505
        bin_quality_dict[method_id]["num_705"] = num_705
        bin_quality_dict[method_id]["num_905"] = num_905
        bin_quality_dict[method_id]["sum"] = total_score
        bin_quality_dict[method_id]["sum_cont5"] = total_score_cont5

        method_scores.append(total_score)
        method_scores_cont5.append(total_score_cont5)

    max_score = max(method_scores)
    tied_methods = []
    tied_cont5_scores = []

    for i, method_id in enumerate(methods_sorted):
        if method_scores[i] == max_score:
            tied_methods.append(method_id)
            tied_cont5_scores.append(method_scores_cont5[i])

    if len(tied_methods) == 1:
        best_method = tied_methods[0]
    else:
        best_method = tied_methods[tied_cont5_scores.index(max(tied_cont5_scores))]

    return bin_quality_dict, best_method


def savecontigs_with_high_bin_quality(
    bins: BinDict,
    best_method: str,
    quality_cache: QualityCache,
    outpath: str,
) -> None:
    """
    Save contigs from bins passing:
    - completeness > 50 and contamination < 10
    - completeness > 50 and contamination < 5
    """
    out_dir = Path(outpath)
    file_5010 = out_dir / f"{best_method}5010_res.txt"
    file_5005 = out_dir / f"{best_method}5005_res.txt"

    bin_count_5010 = 0
    bin_count_5005 = 0

    with open(file_5010, "w") as handle_5010, open(file_5005, "w") as handle_5005:
        for bin_id, contig_names in bins[best_method].items():
            comp, cont = quality_cache[best_method][bin_id]

            if comp > 50 and cont < 10:
                for contig_id in contig_names:
                    handle_5010.write(f"{contig_id}\t{bin_count_5010}\n")
                bin_count_5010 += 1

            if comp > 50 and cont < 5:
                for contig_id in contig_names:
                    handle_5005.write(f"{contig_id}\t{bin_count_5005}\n")
                bin_count_5005 += 1


def write_estimated_bin_quality(bin_quality_dict, output_file):
    """
    Write method-level quality summary table, sorted by num_905 and sum_cont5.
    """
    with open(output_file, "w") as fout:
        fout.write(
            "Binning_method\tnum_5010\tnum_7010\tnum_9010\t"
            "num_505\tnum_705\tnum_905\tsum\tsum_cont5\n"
        )
        sorted_methods = sorted(
            bin_quality_dict.items(),
            key=lambda x: (x[1]["num_905"], x[1]["sum_cont5"]),
            reverse=True,
        )
        for method_id, stats in sorted_methods:
            fout.write(
                f"{method_id}\t"
                f"{stats['num_5010']}\t"
                f"{stats['num_7010']}\t"
                f"{stats['num_9010']}\t"
                f"{stats['num_505']}\t"
                f"{stats['num_705']}\t"
                f"{stats['num_905']}\t"
                f"{stats['sum']}\t"
                f"{stats['sum_cont5']}\n"
            )


def estimate_bins_quality_nobins(
    bac_mg_table,
    ar_mg_table,
    res_path,
    ignore_kmeans_res: bool = False,
):
    """
    Estimate bin quality using bacterial/archaeal marker information.
    """
    markers = Markers()

    # Keep these calls only if required for internal marker setup / side effects.
    markers.marker_gene_tables(bac_mg_table, ar_mg_table)

    return _estimate_bins_quality_common(
        markers=markers,
        res_path=res_path,
        ignore_kmeans_res=ignore_kmeans_res,
        save_high_quality_contigs=True,
    )


def estimate_bins_quality_nobins_pltd(
    pltd_mg_table,
    res_path,
    ignore_kmeans_res: bool = False,
):
    """
    Estimate bin quality using plastid marker information.
    """
    markers = PltdMarkers()

    # Keep this call only if required for internal marker setup / side effects.
    markers.marker_gene_tables_pltd(pltd_mg_table)

    return _estimate_bins_quality_common(
        markers=markers,
        res_path=res_path,
        ignore_kmeans_res=ignore_kmeans_res,
        save_high_quality_contigs=False,
    )


def _estimate_bins_quality_common(
    markers,
    res_path: str,
    ignore_kmeans_res: bool = False,
    save_high_quality_contigs: bool = False,
):
    """
    Shared internal implementation for bin-quality estimation.
    """
    result_files = list_result_tsvs(res_path, ignore_kmeans_res=ignore_kmeans_res)

    bin_dirs = {filename: os.path.join(res_path, filename) for filename in result_files}

    bins, contigs = read_bins_nosequences(bin_dirs)
    methods_sorted = sorted(bins.keys())

    if not methods_sorted:
        raise ValueError(f"No clustering TSV files found in {res_path}")

    quality_cache = compute_bin_quality_cache(
        bins=bins,
        methods_sorted=methods_sorted,
        markers=markers,
    )

    bin_quality_dict, best_method = summarize_method_quality(
        bins=bins,
        methods_sorted=methods_sorted,
        quality_cache=quality_cache,
    )

    if save_high_quality_contigs:
        savecontigs_with_high_bin_quality(
            bins=bins,
            best_method=best_method,
            quality_cache=quality_cache,
            outpath=res_path,
        )

    output_file = os.path.join(res_path, "estimate_res.txt")
    write_estimated_bin_quality(bin_quality_dict, output_file)

    return best_method


class BinfernoResults(BaseResults):
    checkm_database: str = Field(default="", description="Path to CheckM database")

    class Config:
        arbitrary_types_allowed = True

    def __call__(
        self,
        seed_num: int = 0,
        ignore_kmeans_res: bool = False,
        output_path: Path = None,
        num_threads: int = 5,
        logger: Any = None,
        res_name: str = None,
        plastid: bool = False,
        contig_file: str = None,
        minbinsize: int = 200000,
    ) -> None:
        checkm_database = self.checkm_database

        if output_path is None:
            raise ValueError("output_path is required")
        if contig_file is None:
            raise ValueError("contig_file is required")

        output_path = Path(output_path)
        cluster_res_dir = output_path / "cluster_res"
        unitem_profile_dir = cluster_res_dir / "unitem_profile"

        if unitem_profile_dir.exists():
            os.rmdir(unitem_profile_dir)

        logger.info(f"Seed_num:\t{seed_num}")
        logger.info(f"Run unitem profile:\t{seed_num}")

        if res_name is None:
            res_name = f"weight_seed_kmeans_k_{seed_num}_result.tsv"

        bin_dirs = {res_name: (str(cluster_res_dir / f"{res_name}_bins"), "fa")}

        if not unitem_profile_dir.exists():
            make_sure_path_exists(str(unitem_profile_dir))
            profile = self.Profile(cpus=num_threads)
            profile.profile_setup(checkm_database)

            if plastid:
                profile.run_pltd(bin_dirs, str(unitem_profile_dir))
            else:
                profile.run(bin_dirs, str(unitem_profile_dir))
                # if the job fails, remove the unitem_profile_dir to allow re-running the profile step/

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
            contig_file,
            str(cluster_res_dir / best_method),
            str(output_path),
            minbinsize=minbinsize,
        )
        return
