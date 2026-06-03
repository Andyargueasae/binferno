import os
import sys
from pydantic import BaseModel
import subprocess
from .scripts.unitem_defaults import *
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
import csv
import logging
from biolib.common import make_sure_path_exists
from biolib.external.execute import check_on_path


class BaseWriter(BaseModel):
    def __call__(
        self,
        contigs: str,
        cluster_res_dir: str,
        output_path: str,
        plastid: bool = False,
        minbin_size: int = 200000,
        ) -> None:
        raise NotImplementedError("BaseWriter is an abstract class. Please implement the __call__ method.")
    
    class Profile:
        """
        Profile genome bins across binning methods using CheckM.
        """

        def __init__(self, cpus: int):
            self.logger = logging.getLogger("timestamp")
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
