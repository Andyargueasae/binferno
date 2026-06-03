from __future__ import annotations
from pathlib import Path

import loguru
from .base import BaseWriter
from pydantic import Field
import os
from pathlib import Path
from typing import Dict, List, Tuple, Iterator, TextIO
from biolib.common import make_sure_path_exists
import pandas as pd
from .scripts.unitem_markers import Markers, PltdMarkers
import gzip
import shutil
from collections import defaultdict


BinDict = Dict[str, Dict[str, set]]
ContigAssignmentDict = Dict[str, Dict[str, str]]
QualityCache = Dict[str, Dict[str, Tuple[float, float]]]

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
        seed_num: int = 0,
        ignore_kmeans_res: bool = False,
        output_path: Path = None,
        num_threads: int = 5,
        logger: loguru.logger = None, # type: ignore
        res_name: str = None,
        plastid: bool = False,
        contigs: str = None,
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
            profile = self.Profile(cpus=num_threads)
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
            best_method = self.estimate_bins_quality_nobins_pltd(
                pltd_mg_table=str(pltd_mg_table),
                res_path=str(cluster_res_dir),
                ignore_kmeans_res=ignore_kmeans_res,
            )
        else:
            best_method = self.estimate_bins_quality_nobins(
                bac_mg_table=str(bac_mg_table),
                ar_mg_table=str(ar_mg_table),
                res_path=str(cluster_res_dir),
                ignore_kmeans_res=ignore_kmeans_res,
            )

        logger.info(f"Final result:\t{cluster_res_dir / best_method}")

        self.filter_small_bins(
            logger,
            contigs,
            str(cluster_res_dir / best_method),
            str(output_path),
            minbinsize=self.minbin_size,
        )
        return

    def get_binstats(self, bin_contig_names, markers):
        """
        Return (completeness, contamination) for a bin.
        """
        _, comp, cont = markers.bin_quality(bin_contig_names)
        return comp, cont


    def list_result_tsvs(self, res_path: str, ignore_kmeans_res: bool = False) -> List[str]:
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
        self,
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
        self,
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
                comp, cont = self.get_binstats(contig_names, markers)
                quality_cache[method_id][bin_id] = (comp, cont)

        return quality_cache


    def summarize_method_quality(
        self,
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
        self,
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


    def write_estimated_bin_quality(self, bin_quality_dict, output_file):
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
        self,
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

        return self._estimate_bins_quality_common(
            markers=markers,
            res_path=res_path,
            ignore_kmeans_res=ignore_kmeans_res,
            save_high_quality_contigs=True,
        )


    def estimate_bins_quality_nobins_pltd(
        self,
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

        return self._estimate_bins_quality_common(
            markers=markers,
            res_path=res_path,
            ignore_kmeans_res=ignore_kmeans_res,
            save_high_quality_contigs=False,
        )


    def _estimate_bins_quality_common(
        self,
        markers,
        res_path: str,
        ignore_kmeans_res: bool = False,
        save_high_quality_contigs: bool = False,
    ):
        """
        Shared internal implementation for bin-quality estimation.
        """
        result_files = self.list_result_tsvs(res_path, ignore_kmeans_res=ignore_kmeans_res)

        bin_dirs = {filename: os.path.join(res_path, filename) for filename in result_files}

        bins, contigs = self.read_bins_nosequences(bin_dirs)
        methods_sorted = sorted(bins.keys())

        if not methods_sorted:
            raise ValueError(f"No clustering TSV files found in {res_path}")

        quality_cache = self.compute_bin_quality_cache(
            bins=bins,
            methods_sorted=methods_sorted,
            markers=markers,
        )

        bin_quality_dict, best_method = self.summarize_method_quality(
            bins=bins,
            methods_sorted=methods_sorted,
            quality_cache=quality_cache,
        )

        if save_high_quality_contigs:
            self.savecontigs_with_high_bin_quality(
                bins=bins,
                best_method=best_method,
                quality_cache=quality_cache,
                outpath=res_path,
            )

        output_file = os.path.join(res_path, "estimate_res.txt")
        self.write_estimated_bin_quality(bin_quality_dict, output_file)

        return best_method

    def _open_text_maybe_gzip(self, filepath: str) -> TextIO:
        """
        Open a plain-text or gzipped text file.
        """
        if filepath.endswith(".gz"):
            return gzip.open(filepath, "rt", encoding="utf-8")
        return open(filepath, "r", encoding="utf-8")


    def _iter_fasta(self, filepath: str) -> Iterator[Tuple[str, str]]:
        """
        Yield (header_without_>, sequence) pairs from a FASTA file.
        Header is truncated at the first space to match the original behavior.
        Supports both plain-text and gzipped FASTA.
        """
        with self._open_text_maybe_gzip(filepath) as handle:
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


    def _read_result_map(self, resultfile: str) -> Tuple[Dict[str, list], Dict[str, str]]:
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


    def gen_bins(self, fastafile: str, resultfile: str, outputdir: str) -> None:
        """
        Generate one FASTA file per cluster from a contig-to-cluster result TSV.
        Uses streaming FASTA parsing to avoid loading the entire assembly into memory.
        """
        print(f"Processing file:\t{fastafile}")
        print(f"Reading Map:\t{resultfile}")

        cluster_to_contigs, contig_to_cluster = self._read_result_map(resultfile)

        print(f"Writing bins:\t{outputdir}")
        os.makedirs(outputdir, exist_ok=True)

        cluster_to_binfile = {}
        for bin_idx, cluster_name in enumerate(cluster_to_contigs):
            cluster_to_binfile[cluster_name] = os.path.join(outputdir, f"{bin_idx}.fa")

        open_handles = {}
        try:
            for header, sequence in self._iter_fasta(fastafile):
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
        self,
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

        cluster_to_contigs, contig_to_cluster = self._read_result_map(resultfile)

        # First pass over FASTA: compute contig lengths and accumulate per-bin sizes.
        bin_sizes = defaultdict(int)
        contig_lengths = {}

        for header, sequence in self._iter_fasta(fastafile):
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

        self.gen_bins(fastafile, filtered_tsv, os.path.join(output_path, "res_bins"))
        shutil.copy2(filtered_tsv, os.path.join(output_path, "binferno_res.tsv"))
