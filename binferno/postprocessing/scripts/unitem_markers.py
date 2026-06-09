from __future__ import absolute_import, division, print_function

import os
import logging
from collections import defaultdict
from multiprocessing import Pool
from typing import Dict, Iterable, List, Tuple

from checkm.markerSets import MarkerSetParser

from .unitem_defaults import *

# ---------------------------------------------------------------------
# Worker globals for multiprocessing
# ---------------------------------------------------------------------

_WORKER_MARKER_OBJ = None


def _init_marker_worker(marker_obj):
    """
    Initialize worker-local marker evaluator.
    """
    global _WORKER_MARKER_OBJ
    _WORKER_MARKER_OBJ = marker_obj


def _score_bin_worker(contig_ids):
    """
    Worker entry point for bin scoring.
    """
    return _WORKER_MARKER_OBJ.bin_quality(contig_ids)


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------


def _read_marker_gene_table(marker_gene_table: str) -> Dict[str, List[str]]:
    """
    Read a CheckM marker gene table into:
        marker_id -> [scaffold_id, scaffold_id, ...]
    """
    markers = defaultdict(list)

    if not os.path.exists(marker_gene_table):
        return markers

    with open(marker_gene_table) as handle:
        next(handle, None)  # skip header

        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue

            marker_id = fields[1]
            gene_id = fields[2]

            if "&&" in gene_id:
                gene_id = gene_id.split("&&", 1)[0]

            last_underscore = gene_id.rfind("_")
            if last_underscore == -1:
                continue

            scaffold_id = gene_id[:last_underscore]
            markers[marker_id].append(scaffold_id)

    return markers


def _invert_marker_table(
    marker_to_scaffolds: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """
    Convert:
        marker_id -> [scaffold_id, ...]
    into:
        scaffold_id -> [marker_id, ...]
    """
    scaffold_to_markers = defaultdict(list)

    for marker_id, scaffold_ids in marker_to_scaffolds.items():
        for scaffold_id in scaffold_ids:
            scaffold_to_markers[scaffold_id].append(marker_id)

    return scaffold_to_markers


# ---------------------------------------------------------------------
# Bacteria / Archaea markers
# ---------------------------------------------------------------------


class Markers:
    """
    Read marker information and estimate bin quality using bacterial and
    archaeal marker sets.
    """

    def __init__(self):
        self.logger = logging.getLogger("timestamp")

        parser = MarkerSetParser()

        bac_marker_sets = parser.parseTaxonomicMarkerSetFile(CHECKM_BAC_MS)
        self.bac_ms = bac_marker_sets.mostSpecificMarkerSet()

        ar_marker_sets = parser.parseTaxonomicMarkerSetFile(CHECKM_AR_MS)
        self.ar_ms = ar_marker_sets.mostSpecificMarkerSet()

        self.bac_markers_on_contigs: Dict[str, List[str]] = defaultdict(list)
        self.ar_markers_on_contigs: Dict[str, List[str]] = defaultdict(list)

    def read_table(self, marker_gene_table: str) -> Dict[str, List[str]]:
        """
        Read CheckM marker table:
            marker_id -> [scaffold_ids]
        """
        return _read_marker_gene_table(marker_gene_table)

    def marker_gene_tables(
        self,
        bac_mg_table: str,
        ar_mg_table: str,
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Load bacterial and archaeal marker gene tables and build
        contig -> marker mappings.
        """
        bac_gene_table = self.read_table(bac_mg_table)
        ar_gene_table = self.read_table(ar_mg_table)

        self.bac_markers_on_contigs = _invert_marker_table(bac_gene_table)
        self.ar_markers_on_contigs = _invert_marker_table(ar_gene_table)

        return bac_gene_table, ar_gene_table

    def create_gene_table(self, contig_ids: Iterable[str]):
        """
        Create bacterial and archaeal gene tables for a set of contigs.
        """
        bac_table = defaultdict(list)
        ar_table = defaultdict(list)

        for cid in contig_ids:
            for marker_id in self.bac_markers_on_contigs.get(cid, ()):
                bac_table[marker_id].append(cid)

            for marker_id in self.ar_markers_on_contigs.get(cid, ()):
                ar_table[marker_id].append(cid)

        return bac_table, ar_table

    def evaluate_bac(self, gene_table, individual_markers: bool = False):
        """
        Evaluate completeness / contamination using bacterial marker set.
        """
        comp, cont = self.bac_ms.genomeCheck(gene_table, individual_markers)
        return comp, cont

    def evaluate_ar(self, gene_table, individual_markers: bool = False):
        """
        Evaluate completeness / contamination using archaeal marker set.
        """
        comp, cont = self.ar_ms.genomeCheck(gene_table, individual_markers)
        return comp, cont

    def evaluate(self, bac_gene_table, ar_gene_table):
        """
        Evaluate completeness / contamination using the better of
        bacterial and archaeal marker sets.
        """
        bac_comp_i, bac_cont_i = self.evaluate_bac(bac_gene_table, True)
        ar_comp_i, ar_cont_i = self.evaluate_ar(ar_gene_table, True)

        if bac_comp_i + bac_cont_i > ar_comp_i + ar_cont_i:
            bac_comp, bac_cont = self.evaluate_bac(bac_gene_table, False)
            return "Bacteria", bac_comp, bac_cont

        ar_comp, ar_cont = self.evaluate_ar(ar_gene_table, False)
        return "Archaea", ar_comp, ar_cont

    def bin_quality(self, contig_ids: Iterable[str]):
        """
        Estimate bin quality for one bin.
        """
        bac_gene_table, ar_gene_table = self.create_gene_table(contig_ids)
        return self.evaluate(bac_gene_table, ar_gene_table)

    def score_bins_parallel(
        self,
        bins: Dict[str, Iterable[str]],
        processes: int = 1,
    ) -> Dict[str, Tuple[str, float, float]]:
        """
        Score many bins in parallel.

        Parameters
        ----------
        bins : dict
            bin_id -> iterable of contig ids
        processes : int
            Number of worker processes

        Returns
        -------
        dict
            bin_id -> (domain, completeness, contamination)
        """
        bin_ids = list(bins.keys())
        bin_contigs = [list(bins[bin_id]) for bin_id in bin_ids]

        if processes <= 1 or len(bin_ids) <= 1:
            results = [self.bin_quality(contigs) for contigs in bin_contigs]
        else:
            with Pool(
                processes=processes,
                initializer=_init_marker_worker,
                initargs=(self,),
            ) as pool:
                results = pool.map(_score_bin_worker, bin_contigs)

        return dict(zip(bin_ids, results))


# ---------------------------------------------------------------------
# Plastid markers
# ---------------------------------------------------------------------


class PltdMarkers:
    """
    Read plastid marker information and estimate plastid bin quality.
    """

    def __init__(self):
        self.logger = logging.getLogger("timestamp")

        parser = MarkerSetParser()
        pltd_marker_sets = parser.parseTaxonomicMarkerSetFile(CHECKM_PLTD_MS)
        self.pltd_ms = pltd_marker_sets.mostSpecificMarkerSet()

        self.pltd_markers_on_contigs: Dict[str, List[str]] = defaultdict(list)

    def read_table(self, marker_gene_table: str) -> Dict[str, List[str]]:
        """
        Read CheckM plastid marker table:
            marker_id -> [scaffold_ids]
        """
        return _read_marker_gene_table(marker_gene_table)

    def marker_gene_tables_pltd(
        self,
        pltd_mg_table: str,
    ) -> Dict[str, List[str]]:
        """
        Load plastid marker gene table and build contig -> marker mapping.
        """
        pltd_gene_table = self.read_table(pltd_mg_table)
        self.pltd_markers_on_contigs = _invert_marker_table(pltd_gene_table)
        return pltd_gene_table

    def create_pltd_gene_tables(self, contig_ids: Iterable[str]):
        """
        Create plastid gene table for a set of contigs.
        """
        pltd_table = defaultdict(list)

        for cid in contig_ids:
            for marker_id in self.pltd_markers_on_contigs.get(cid, ()):
                pltd_table[marker_id].append(cid)

        return pltd_table

    def evaluate_pltd(self, gene_table, individual_markers: bool = False):
        """
        Evaluate completeness / contamination using plastid marker set.
        """
        comp, cont = self.pltd_ms.genomeCheck(gene_table, individual_markers)
        return comp, cont

    def evaluate(self, pltd_gene_table):
        """
        Evaluate plastid bin quality.
        """
        pltd_comp, pltd_cont = self.evaluate_pltd(pltd_gene_table, True)
        return "Algae", pltd_comp, pltd_cont

    def bin_quality(self, contig_ids: Iterable[str]):
        """
        Estimate plastid bin quality for one bin.
        """
        pltd_gene_table = self.create_pltd_gene_tables(contig_ids)
        return self.evaluate(pltd_gene_table)

    def score_bins_parallel(
        self,
        bins: Dict[str, Iterable[str]],
        processes: int = 1,
    ) -> Dict[str, Tuple[str, float, float]]:
        """
        Score many plastid bins in parallel.

        Parameters
        ----------
        bins : dict
            bin_id -> iterable of contig ids
        processes : int
            Number of worker processes

        Returns
        -------
        dict
            bin_id -> (domain, completeness, contamination)
        """
        bin_ids = list(bins.keys())
        bin_contigs = [list(bins[bin_id]) for bin_id in bin_ids]

        if processes <= 1 or len(bin_ids) <= 1:
            results = [self.bin_quality(contigs) for contigs in bin_contigs]
        else:
            with Pool(
                processes=processes,
                initializer=_init_marker_worker,
                initargs=(self,),
            ) as pool:
                results = pool.map(_score_bin_worker, bin_contigs)

        return dict(zip(bin_ids, results))
