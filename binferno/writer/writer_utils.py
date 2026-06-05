from __future__ import annotations

import sys
from pathlib import Path
import os
import gzip
import shutil
from collections import defaultdict
import subprocess
import csv
import loguru

import pandas as pd
from typing import Dict, List, Tuple, Iterator, TextIO, Optional
from biolib.common import make_sure_path_exists
from biolib.external.execute import check_on_path

from .scripts.unitem_markers import Markers, PltdMarkers
from .scripts.unitem_defaults import *

BinDict = Dict[str, Dict[str, set]]
ContigAssignmentDict = Dict[str, Dict[str, str]]
QualityCache = Dict[str, Dict[str, Tuple[float, float]]]

class Profile:
    """
    Profile genome bins across binning methods using CheckM.
    """

    def __init__(self, cpus: int, logger: loguru._logger.Logger):
        self.logger = logger
        check_on_path("checkm")
        self.cpus = cpus

    def profile_setup(self, database):
        """
        Set the active CheckM database root.
        """
        if not os.path.exists(database):
            self.logger.error("CheckM database %s does not exist.", database)
            sys.exit(1)

        cmd = ["checkm", "data", "setRoot", str(database)]
        self.logger.info(
            "Setting up CheckM database with command: %s",
            " ".join(cmd),
        )
        subprocess.run(cmd, check=True)

    def _run_cmd(self, cmd: List[str], step_name: str) -> None:
        """
        Run a subprocess command with logging.
        """
        self.logger.info("[%s] %s", step_name, " ".join(cmd))
        subprocess.run(cmd, check=True)

    def _run_checkm_job(
        self,
        bin_dir: str,
        bin_ext: str,
        marker_set_file: str,
        output_dir: str,
        skip_existing: bool = True,
    ) -> Tuple[str, str]:
        """
        Run CheckM analyze + qa outputs for a single marker-set job.

        Returns
        -------
        marker_gene_table, genome_quality_table
        """
        make_sure_path_exists(output_dir)

        marker_gene_table = os.path.join(output_dir, MARKER_GENE_TABLE)
        genome_quality_table = os.path.join(output_dir, GENOME_QUALITY_TABLE)

        analyze_done = (
            os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0
        )
        marker_done = os.path.exists(marker_gene_table)
        quality_done = os.path.exists(genome_quality_table)

        if not (skip_existing and analyze_done and marker_done and quality_done):
            if not (skip_existing and analyze_done):
                analyze_cmd = [
                    "checkm",
                    "analyze",
                    "-t",
                    str(self.cpus),
                    "-x",
                    str(bin_ext),
                    str(marker_set_file),
                    str(bin_dir),
                    str(output_dir),
                ]
                self._run_cmd(analyze_cmd, "checkm analyze")

            if not (skip_existing and marker_done):
                marker_cmd = [
                    "checkm",
                    "qa",
                    "-t",
                    str(self.cpus),
                    "-o",
                    "5",
                    "--tab_table",
                    "-f",
                    str(marker_gene_table),
                    str(marker_set_file),
                    str(output_dir),
                ]
                self._run_cmd(marker_cmd, "checkm qa marker_gene_table")

            if not (skip_existing and quality_done):
                quality_cmd = [
                    "checkm",
                    "qa",
                    "-t",
                    str(self.cpus),
                    "-o",
                    "2",
                    "--tab_table",
                    "-f",
                    str(genome_quality_table),
                    str(marker_set_file),
                    str(output_dir),
                ]
                self._run_cmd(quality_cmd, "checkm qa genome_quality_table")

        return marker_gene_table, genome_quality_table

    def _parse_quality_table(
        self,
        quality_table: str,
        domain_label: Optional[str] = None,
    ) -> Dict[str, Tuple]:
        """
        Parse a CheckM genome quality table.

        If domain_label is provided, return:
            gid -> (domain_label, comp, cont)
        Otherwise return:
            gid -> (comp, cont)
        """
        parsed = {}

        with open(quality_table, newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            next(reader, None)  # skip header

            for row in reader:
                if not row:
                    continue

                gid = row[0]
                comp = float(row[5])
                cont = float(row[6])

                if domain_label is None:
                    parsed[gid] = (comp, cont)
                else:
                    parsed[gid] = (domain_label, comp, cont)

        return parsed

    def _genome_quality(self, bac_quality_table: str, ar_quality_table: str):
        """
        Combine Bac and Ar CheckM quality tables, choosing the better domain call.
        """
        bac = self._parse_quality_table(bac_quality_table, domain_label=None)
        ar = self._parse_quality_table(ar_quality_table, domain_label=None)

        genome_quality = {}
        all_gids = set(bac).union(ar)

        for gid in all_gids:
            bac_comp, bac_cont = bac.get(gid, (0.0, float("inf")))
            ar_comp, ar_cont = ar.get(gid, (0.0, float("inf")))

            if bac_comp + bac_cont > ar_comp + ar_cont:
                genome_quality[gid] = ("Bacteria", bac_comp, bac_cont)
            else:
                genome_quality[gid] = ("Archaea", ar_comp, ar_cont)

        return genome_quality

    def _genome_quality_pltd(self, pltd_quality_table: str):
        """
        Parse plastid CheckM quality table.
        """
        return self._parse_quality_table(pltd_quality_table, domain_label="plastid")

    def _report_genome_quality(
        self, genome_quality: Dict[str, Dict[str, Tuple]], output_dir: str
    ):
        """
        Write per-method and global bin quality summaries.
        """
        output_dir = str(output_dir)

        # Per-method genome tables
        for method_id, method_quality in genome_quality.items():
            table = os.path.join(output_dir, f"{method_id}_quality.tsv")
            with open(table, "w") as fout:
                fout.write(
                    "Genome ID\tMarker Set Domain\tCompleteness (%)\tContamination (%)\tQuality\n"
                )
                for gid, (domain, comp, cont) in method_quality.items():
                    quality = comp - 5 * cont
                    fout.write(
                        f"{gid}\t{domain}\t{comp:.2f}\t{cont:.2f}\t{quality:.2f}\n"
                    )

        # Global summary
        summary = {}
        for method_id, method_quality in genome_quality.items():
            comp_cont_count = defaultdict(lambda: defaultdict(int))
            quality_count = defaultdict(int)
            total_comp = 0.0
            total_cont = 0.0
            total_q = 0.0

            for _, (_, comp, cont) in method_quality.items():
                quality = comp - 5 * cont

                for test_comp in (90, 80, 70):
                    for test_cont in (5, 10):
                        if comp >= test_comp and cont <= test_cont:
                            comp_cont_count[test_comp][test_cont] += 1

                for test_q in (90, 70, 50):
                    if quality >= test_q:
                        quality_count[test_q] += 1

                total_comp += comp
                total_cont += cont
                total_q += quality

            summary[method_id] = (
                comp_cont_count,
                quality_count,
                total_comp,
                total_cont,
                total_q,
            )

        summary_file = os.path.join(output_dir, "bin_quality_summary.tsv")
        with open(summary_file, "w") as fout:
            method_ids = sorted(summary)
            fout.write("\t" + "\t".join(method_ids) + "\n")

            for comp in (90, 80, 70):
                for cont in (5, 10):
                    fout.write(f"Comp. >= {comp}, cont. <= {cont}")
                    for method_id in method_ids:
                        fout.write(f"\t{summary[method_id][0][comp][cont]}")
                    fout.write("\n")

            for q in (90, 70, 50):
                fout.write(f"Quality >= {q}")
                for method_id in method_ids:
                    fout.write(f"\t{summary[method_id][1][q]}")
                fout.write("\n")

            fout.write("Total compl.")
            for method_id in method_ids:
                fout.write(f"\t{summary[method_id][2]:.1f}")
            fout.write("\n")

            fout.write("Total cont.")
            for method_id in method_ids:
                fout.write(f"\t{summary[method_id][3]:.1f}")
            fout.write("\n")

            fout.write("Total quality")
            for method_id in method_ids:
                fout.write(f"\t{summary[method_id][4]:.1f}")
            fout.write("\n")

    def _profile_methods(
        self,
        bin_dirs,
        output_dir: str,
        marker_jobs: List[Tuple[str, str]],
        plastid_mode: bool = False,
        skip_existing: bool = True,
    ):
        """
        Shared implementation for bacterial/archaeal/plastid profiling.

        marker_jobs:
            list of (subdir_name, marker_set_file)
        """
        self.logger.info("Profiling genomes in %d directories.", len(bin_dirs))

        genome_quality = defaultdict(dict)

        for idx, (method_id, (bin_dir, bin_ext)) in enumerate(
            bin_dirs.items(), start=1
        ):
            self.logger.info(
                "Profiling %s (%d of %d).",
                method_id,
                idx,
                len(bin_dirs),
            )

            quality_tables = {}

            for subdir_name, marker_set_file in marker_jobs:
                cur_output_dir = os.path.join(
                    output_dir,
                    BINNING_METHOD_DIR,
                    method_id,
                    subdir_name,
                )

                _, genome_quality_table = self._run_checkm_job(
                    bin_dir=bin_dir,
                    bin_ext=bin_ext,
                    marker_set_file=marker_set_file,
                    output_dir=cur_output_dir,
                    skip_existing=skip_existing,
                )
                quality_tables[subdir_name] = genome_quality_table

            if plastid_mode:
                pltd_quality_table = quality_tables[CHECKM_PLTD_DIR]
                if not os.path.exists(pltd_quality_table):
                    self.logger.error(
                        "Missing plastid quality table for %s.", method_id
                    )
                    sys.exit(1)

                genome_quality[method_id] = self._genome_quality_pltd(
                    pltd_quality_table
                )
            else:
                bac_quality_table = quality_tables[CHECKM_BAC_DIR]
                ar_quality_table = quality_tables[CHECKM_AR_DIR]

                if not os.path.exists(bac_quality_table) or not os.path.exists(
                    ar_quality_table
                ):
                    self.logger.error(
                        "Missing Bac/Ar quality table for %s.", method_id
                    )
                    self.logger.error(
                        "Please verify there were bins in the bin directory specified for this method."
                    )
                    sys.exit(1)

                genome_quality[method_id] = self._genome_quality(
                    bac_quality_table,
                    ar_quality_table,
                )

        self._report_genome_quality(genome_quality, output_dir)

    def run(self, bin_dirs, output_dir, skip_existing: bool = True):
        """
        Profile bins with bacterial and archaeal marker sets.
        """
        marker_jobs = [
            (CHECKM_BAC_DIR, CHECKM_BAC_MS),
            (CHECKM_AR_DIR, CHECKM_AR_MS),
        ]
        self._profile_methods(
            bin_dirs=bin_dirs,
            output_dir=output_dir,
            marker_jobs=marker_jobs,
            plastid_mode=False,
            skip_existing=skip_existing,
        )

    def run_pltd(self, bin_dirs, output_dir, skip_existing: bool = True):
        """
        Profile bins with plastid marker set.
        """
        marker_jobs = [
            (CHECKM_PLTD_DIR, CHECKM_PLTD_MS),
        ]
        self._profile_methods(
            bin_dirs=bin_dirs,
            output_dir=output_dir,
            marker_jobs=marker_jobs,
            plastid_mode=True,
            skip_existing=skip_existing,
        )


def get_binstats(bin_contig_names, markers):
    """
    Return (completeness, contamination) for a bin.
    """
    _, comp, cont = markers.bin_quality(bin_contig_names)
    return comp, cont

def list_result_tsvs(
    res_path: str, ignore_kmeans_res: bool = False
) -> List[str]:
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
    result_files = list_result_tsvs(
        res_path, ignore_kmeans_res=ignore_kmeans_res
    )

    bin_dirs = {
        filename: os.path.join(res_path, filename) for filename in result_files
    }

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

def _open_text_maybe_gzip(filepath: str) -> TextIO:
    """
    Open a plain-text or gzipped text file.
    """
    if filepath.endswith(".gz"):
        return gzip.open(filepath, "rt", encoding="utf-8")
    return open(filepath, "r", encoding="utf-8")

def _iter_fasta(filepath: str) -> Iterator[Tuple[str, str]]:
    """
    Yield (header_without_>, sequence) pairs from a FASTA file.
    Header is truncated at the first space to match the original behavior.
    Supports both plain-text and gzipped FASTA.
    """
    with _open_text_maybe_gzip(filepath) as handle:
        current_header = None
        seq_chunks = []

        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line:
                continue

            if line.startswith(">"):
                if current_header is not None:
                    yield current_header, "".join(seq_chunks)

                header = line[1:]
                current_header = header.split(" ", 1)[0]
                seq_chunks = []
            else:
                seq_chunks.append(line)

        if current_header is not None:
            yield current_header, "".join(seq_chunks)

def _read_result_map(
    resultfile: str
) -> Tuple[Dict[str, list], Dict[str, str]]:
    """
    Read contig-to-cluster assignments.

    Returns
    -------
    cluster_to_contigs : dict
        cluster_name -> list of contig names
    contig_to_cluster : dict
        contig_name -> cluster_name
    """
    cluster_to_contigs = defaultdict(list)
    contig_to_cluster = {}

    with open(resultfile, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue

            contig_name, cluster_name = stripped.split("\t")
            cluster_to_contigs[cluster_name].append(contig_name)
            contig_to_cluster[contig_name] = cluster_name

    return dict(cluster_to_contigs), contig_to_cluster

def gen_bins(fastafile: str, resultfile: str, outputdir: str) -> None:
    """
    Generate one FASTA file per cluster from a contig-to-cluster result TSV.
    Uses streaming FASTA parsing to avoid loading the entire assembly into memory.
    """
    print(f"Processing file:\t{fastafile}")
    print(f"Reading Map:\t{resultfile}")

    cluster_to_contigs, contig_to_cluster = _read_result_map(resultfile)

    print(f"Writing bins:\t{outputdir}")
    os.makedirs(outputdir, exist_ok=True)

    cluster_to_binfile = {}
    for bin_idx, cluster_name in enumerate(cluster_to_contigs):
        cluster_to_binfile[cluster_name] = os.path.join(outputdir, f"{bin_idx}.fa")

    open_handles = {}
    try:
        for header, sequence in _iter_fasta(fastafile):
            cluster_name = contig_to_cluster.get(header)
            if cluster_name is None:
                continue

            if cluster_name not in open_handles:
                open_handles[cluster_name] = open(
                    cluster_to_binfile[cluster_name],
                    "w",
                    encoding="utf-8",
                )

            handle = open_handles[cluster_name]
            handle.write(f">{header}\n{sequence}\n")
    finally:
        for handle in open_handles.values():
            handle.close()

def filter_small_bins(
    logger,
    fastafile: str,
    resultfile: str,
    output_path: str,
    minbinsize: int = 200000,
) -> None:
    """
    Filter bins whose total contig length is below minbinsize, write filtered TSV,
    generate FASTA bins, and copy final TSV to BTBin_res.tsv.
    """
    filtered_tsv = f"{resultfile}.filtersmallbins_{minbinsize}.tsv"

    logger.info(f"Processing file:\t{fastafile}")
    logger.info(f"Reading Map:\t{resultfile}")

    cluster_to_contigs, contig_to_cluster = _read_result_map(resultfile)

    # First pass over FASTA: compute contig lengths and accumulate per-bin sizes.
    bin_sizes = defaultdict(int)
    contig_lengths = {}

    for header, sequence in _iter_fasta(fastafile):
        seq_len = len(sequence)
        contig_lengths[header] = seq_len

        cluster_name = contig_to_cluster.get(header)
        if cluster_name is not None:
            bin_sizes[cluster_name] += seq_len

    kept_clusters = {
        cluster_name
        for cluster_name, total_size in bin_sizes.items()
        if total_size >= minbinsize
    }

    with open(filtered_tsv, "w", encoding="utf-8") as handle:
        for cluster_name, contig_names in cluster_to_contigs.items():
            if cluster_name not in kept_clusters:
                continue
            for contig_name in contig_names:
                handle.write(f"{contig_name}\t{cluster_name}\n")

    gen_bins(fastafile, filtered_tsv, os.path.join(output_path, "res_bins"))
    shutil.copy2(filtered_tsv, os.path.join(output_path, "binferno_res.tsv"))
