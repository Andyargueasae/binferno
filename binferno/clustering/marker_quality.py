"""Marker-gene parsing and bin quality estimation for binferno.

This module replaces the ad-hoc `parse_contig_to_markers` /
`count_markers_saturated` / `completeness_contamination` trio in dbscan.py.

Design goals
------------
1. **Never fail silently.** Every stage that can produce an empty or
   meaningless marker table raises or emits a loud report. The old code
   defaulted every contig to ``[]`` markers, so a total ORF-id/contig-id
   join failure was indistinguishable from "this bin has no SCGs".
2. **Format-agnostic HMMER parsing.** Auto-detects ``--tblout`` vs
   ``--domtblout`` (the query-name column differs: index 2 vs index 3)
   and collapses multi-domain rows.
3. **A fixed denominator.** Completeness is computed against the marker
   vocabulary declared in the ``.hmm`` file(s), not against "whatever
   HMMs happened to hit this assembly". Supports per-domain marker sets
   so a bacterium is not penalised for lacking archaeal markers.
4. **Vectorised.** Bin counting is a CSR row-slice + column sum, so
   scoring thousands of candidate clusterings across a parameter sweep
   is cheap. No early-exit sentinel, hence no ``None``.

CLI
---
    python -m marker_quality diagnose \
        --hmmout run/contigs.fa.bacar_marker.hmmout \
        --contigs contigs.fa \
        [--hmm-file bacteria=bacar_marker.hmm] \
        [--min-contig-len 1000]
"""

from __future__ import annotations

import gzip
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import csr_matrix

__all__ = [
    "HmmHit",
    "MarkerTable",
    "BinQuality",
    "sniff_hmmout_format",
    "parse_hmmout",
    "resolve_orf_to_contig",
    "load_hmm_names",
    "build_marker_table",
    "diagnose_marker_table",
]


# --------------------------------------------------------------------------
# low-level IO
# --------------------------------------------------------------------------
def _open_text(path):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


@dataclass(frozen=True)
class HmmHit:
    orf_id: str
    hmm_name: str
    score: float
    evalue: float


def sniff_hmmout_format(path) -> str:
    """Return 'tblout' or 'domtblout'.

    HMMER writes a two-line ``#`` header. ``--domtblout`` has a ``tlen``
    column at index 2 and the query name at index 3; ``--tblout`` has the
    query name at index 2. We first try the header, then fall back to
    checking whether field 2 of the first data row is an integer.
    """
    header_lines: List[str] = []
    first_data: Optional[List[str]] = None
    with _open_text(path) as fh:
        for line in fh:
            if line.startswith("#"):
                if len(header_lines) < 4:
                    header_lines.append(line)
                continue
            if line.strip():
                first_data = line.split()
                break

    header_blob = " ".join(header_lines).lower()
    if "tlen" in header_blob and "qlen" in header_blob:
        return "domtblout"
    if "full sequence" in header_blob and "tlen" not in header_blob:
        return "tblout"

    if first_data is not None and len(first_data) >= 4:
        # domtblout col 2 is target length -> pure integer.
        if first_data[2].isdigit() and not first_data[3].isdigit():
            return "domtblout"
        return "tblout"

    raise ValueError(
        f"Could not determine HMMER output format for {path}: no data rows found. "
        "The file is empty -- check that gen_seed's hmmsearch actually ran."
    )


def parse_hmmout(
    path,
    fmt: str = "auto",
    max_evalue: Optional[float] = 1e-5,
    min_score: Optional[float] = None,
    best_hit_per_orf: bool = True,
) -> List[HmmHit]:
    """Parse a HMMER tblout/domtblout into deduplicated per-ORF hits.

    ``best_hit_per_orf`` keeps only the highest-scoring HMM per ORF. This
    matters: without it, one ORF hitting five related HMMs is counted as
    five markers, which inflates contamination and (in the old code)
    tripped the saturation bail-out that returned ``None``.

    Multi-domain rows in domtblout are collapsed automatically because we
    key on ``(orf_id, hmm_name)`` and keep the best full-sequence score.
    """
    if fmt == "auto":
        fmt = sniff_hmmout_format(path)
    if fmt == "domtblout":
        qcol, scol, ecol = 3, 7, 6  # query name, full-seq score, full-seq E-value
    elif fmt == "tblout":
        qcol, scol, ecol = 2, 5, 4
    else:
        raise ValueError(f"fmt must be 'tblout', 'domtblout' or 'auto', got {fmt!r}")

    best: Dict[Tuple[str, str], HmmHit] = {}
    n_rows = 0
    with _open_text(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            cols = line.split()
            if len(cols) <= max(qcol, scol, ecol):
                continue
            n_rows += 1
            orf, hmm = cols[0], cols[qcol]
            try:
                score = float(cols[scol])
                evalue = float(cols[ecol])
            except ValueError:
                continue
            if max_evalue is not None and evalue > max_evalue:
                continue
            if min_score is not None and score < min_score:
                continue
            key = (orf, hmm)
            prev = best.get(key)
            if prev is None or score > prev.score:
                best[key] = HmmHit(orf, hmm, score, evalue)

    hits = list(best.values())
    if best_hit_per_orf:
        per_orf: Dict[str, HmmHit] = {}
        for h in hits:
            prev = per_orf.get(h.orf_id)
            if prev is None or h.score > prev.score:
                per_orf[h.orf_id] = h
        hits = list(per_orf.values())

    if not hits:
        raise ValueError(
            f"{path} ({fmt}) yielded 0 usable hits from {n_rows} rows. "
            f"Loosen max_evalue (currently {max_evalue}) or check the file."
        )
    return hits


# --------------------------------------------------------------------------
# ORF id -> contig id
# --------------------------------------------------------------------------
_ORF_STRIPPERS: Dict[str, re.Pattern] = {
    # FragGeneScan / MetaGeneMark: <contig>_<start>_<end>_<strand>
    "fraggenescan": re.compile(r"_\d+_\d+_[+-]$"),
    # Prodigal / pyrodigal: <contig>_<ordinal>
    "prodigal": re.compile(r"_\d+$"),
    # Prodigal with strand suffix used by some wrappers
    "prodigal_strand": re.compile(r"_\d+_[+-]$"),
}


@dataclass
class OrfJoinReport:
    style: str
    matched: int
    total: int

    @property
    def rate(self) -> float:
        return self.matched / max(self.total, 1)


def resolve_orf_to_contig(
    orf_ids: Sequence[str],
    known_contigs: Iterable[str],
    min_match_rate: float = 0.5,
) -> Tuple[Dict[str, str], OrfJoinReport, List[OrfJoinReport]]:
    """Pick the ORF-id-stripping rule that best matches the real contig names.

    This is the single most important safety check in the module. If the
    gene caller changed (e.g. FragGeneScan -> pyrodigal), a hard-coded
    regex silently maps every ORF onto a contig name that does not exist,
    every bin ends up with zero markers, and completeness is 0 everywhere.
    """
    known = set(known_contigs)
    if not known:
        raise ValueError("known_contigs is empty")

    uniq_orfs = list(dict.fromkeys(orf_ids))
    candidates: List[Tuple[str, Dict[str, str], OrfJoinReport]] = []

    def _try(style: str, fn):
        mapping = {}
        matched = 0
        for o in uniq_orfs:
            c = fn(o)
            mapping[o] = c
            if c in known:
                matched += 1
        candidates.append(
            (style, mapping, OrfJoinReport(style, matched, len(uniq_orfs)))
        )

    _try("identity", lambda o: o)
    for style, rx in _ORF_STRIPPERS.items():
        _try(style, lambda o, rx=rx: rx.sub("", o))
    # whitespace-split ids (some tools append " # start # end")
    _try("first_token", lambda o: o.split()[0])

    candidates.sort(key=lambda t: t[2].matched, reverse=True)
    style, mapping, report = candidates[0]
    all_reports = [c[2] for c in candidates]

    if report.rate < min_match_rate:
        detail = ", ".join(f"{r.style}={r.rate:.1%}" for r in all_reports)
        example_orf = uniq_orfs[0]
        example_contig = next(iter(known))
        raise ValueError(
            "Could not map ORF ids onto contig ids. Best match rate "
            f"{report.rate:.1%} (< {min_match_rate:.0%}). Tried: {detail}.\n"
            f"  example ORF id    : {example_orf!r}\n"
            f"  example contig id : {example_contig!r}\n"
            "Fix: add the right stripper to _ORF_STRIPPERS, or check that the "
            "hmmout was generated from THIS assembly."
        )
    return mapping, report, all_reports


def load_hmm_names(hmm_path) -> List[str]:
    """Extract the NAME (falling back to ACC) of every model in a .hmm file."""
    names: List[str] = []
    pending_name = None
    with _open_text(hmm_path) as fh:
        for line in fh:
            if line.startswith("NAME"):
                pending_name = line.split(None, 1)[1].strip()
            elif line.startswith("//"):
                if pending_name:
                    names.append(pending_name)
                pending_name = None
    if not names:
        raise ValueError(f"No NAME records found in {hmm_path}")
    return names


def read_fasta_lengths(path) -> Dict[str, int]:
    lengths: Dict[str, int] = {}
    name = None
    n = 0
    with _open_text(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    lengths[name] = n
                name = line[1:].split()[0]
                n = 0
            else:
                n += len(line.strip())
    if name is not None:
        lengths[name] = n
    return lengths


# --------------------------------------------------------------------------
# quality
# --------------------------------------------------------------------------
@dataclass
class BinQuality:
    completeness: float
    contamination: float
    n_unique: int
    n_markers: int
    marker_set: str
    n_contigs: int
    bp: int

    def is_hq(self, min_comp: float = 0.9, max_cont: float = 0.05) -> bool:
        return self.completeness >= min_comp and self.contamination <= max_cont

    def is_mq(self, min_comp: float = 0.5, max_cont: float = 0.10) -> bool:
        return self.completeness >= min_comp and self.contamination <= max_cont


class MarkerTable:
    """contig x marker count matrix plus domain-aware quality estimation."""

    def __init__(
        self,
        namelist: Sequence[str],
        counts: csr_matrix,
        marker_names: Sequence[str],
        marker_sets: Mapping[str, np.ndarray],
        contig_lengths: Optional[np.ndarray] = None,
    ):
        self.namelist = list(namelist)
        self.counts = counts.tocsr()
        self.counts.sort_indices()
        self.marker_names = list(marker_names)
        self.marker_sets = {
            k: np.asarray(v, dtype=np.int64) for k, v in marker_sets.items()
        }
        if contig_lengths is None:
            contig_lengths = np.zeros(len(self.namelist), dtype=np.int64)
        self.contig_lengths = np.asarray(contig_lengths, dtype=np.int64)
        self._name_to_idx = {n: i for i, n in enumerate(self.namelist)}

    # -- basic stats ------------------------------------------------------
    @property
    def n_contigs(self) -> int:
        return self.counts.shape[0]

    @property
    def n_markers(self) -> int:
        return self.counts.shape[1]

    def contigs_with_markers(self) -> int:
        return int((np.diff(self.counts.indptr) > 0).sum())

    def bin_counts(self, idx: Sequence[int]) -> np.ndarray:
        idx = np.fromiter(idx, dtype=np.int64)
        if idx.size == 0:
            return np.zeros(self.n_markers, dtype=np.int64)
        return np.asarray(self.counts[idx].sum(axis=0)).ravel()

    def quality(self, idx: Sequence[int]) -> BinQuality:
        """Completeness / contamination, CheckM-style, best domain wins.

        completeness  = |markers present in set| / |set|
        contamination = sum(copies - 1 over present markers) / |set|

        Always returns a BinQuality. It never returns None, so a caller
        can distinguish "clean but empty" from "hopelessly contaminated"
        instead of silently dropping the bin from the score.
        """
        idx = np.fromiter(idx, dtype=np.int64)
        c = self.bin_counts(idx)
        best: Optional[BinQuality] = None
        bp = int(self.contig_lengths[idx].sum()) if idx.size else 0
        for sname, ids in self.marker_sets.items():
            sub = c[ids]
            n_unique = int((sub > 0).sum())
            n_total = int(sub.sum())
            denom = float(len(ids))
            comp = n_unique / denom
            cont = (n_total - n_unique) / denom
            q = BinQuality(comp, cont, n_unique, n_total, sname, int(idx.size), bp)
            if best is None or (q.completeness, -q.contamination) > (
                best.completeness,
                -best.contamination,
            ):
                best = q
        assert best is not None
        return best

    def quality_many(self, bins: Sequence[Sequence[int]]) -> List[BinQuality]:
        return [self.quality(b) for b in bins]

    def max_achievable_completeness(self) -> Dict[str, float]:
        """Upper bound on completeness if ALL markers in the assembly were
        placed in one bin. If this is < your min_completeness threshold,
        your sweep can never score above zero -- which is exactly the
        'everything is None/0' failure mode."""
        total = np.asarray(self.counts.sum(axis=0)).ravel()
        out = {}
        for sname, ids in self.marker_sets.items():
            out[sname] = float((total[ids] > 0).sum()) / float(len(ids))
        return out


def build_marker_table(
    hmmout_path,
    namelist: Sequence[str],
    contig_lengths: Optional[Sequence[int]] = None,
    hmm_files: Optional[Mapping[str, str]] = None,
    max_evalue: Optional[float] = 1e-5,
    min_score: Optional[float] = None,
    best_hit_per_orf: bool = True,
    min_join_rate: float = 0.5,
    logger=None,
) -> Tuple[MarkerTable, dict]:
    """Build a MarkerTable from a HMMER output file.

    ``hmm_files`` maps a marker-set name to the .hmm file that defines it,
    e.g. ``{"bacteria": ".../bacar_marker.hmm"}`` or, better, separate
    bacterial and archaeal files. When omitted, the vocabulary is taken
    from the observed hits, which makes completeness non-comparable across
    runs -- a warning is emitted.
    """
    info: dict = {}
    fmt = sniff_hmmout_format(hmmout_path)
    info["hmmout_format"] = fmt

    hits = parse_hmmout(
        hmmout_path,
        fmt=fmt,
        max_evalue=max_evalue,
        min_score=min_score,
        best_hit_per_orf=best_hit_per_orf,
    )
    info["n_hits"] = len(hits)

    orf_ids = [h.orf_id for h in hits]
    mapping, report, all_reports = resolve_orf_to_contig(
        orf_ids, namelist, min_match_rate=min_join_rate
    )
    info["orf_id_style"] = report.style
    info["orf_join_rate"] = report.rate
    info["orf_join_all"] = {r.style: r.rate for r in all_reports}

    # marker vocabulary
    if hmm_files:
        set_names: Dict[str, List[str]] = {
            k: load_hmm_names(v) for k, v in hmm_files.items()
        }
        vocab_order = list(dict.fromkeys(n for ns in set_names.values() for n in ns))
    else:
        if logger:
            logger.warning(
                "No .hmm file supplied: completeness denominator is taken from "
                "observed hits and is NOT comparable between runs. Pass "
                "hmm_files={'bacteria': '<path>.hmm'} for a fixed denominator."
            )
        vocab_order = sorted({h.hmm_name for h in hits})
        set_names = {"observed": list(vocab_order)}

    vocab = {name: i for i, name in enumerate(vocab_order)}
    marker_sets = {
        k: np.array([vocab[n] for n in ns if n in vocab], dtype=np.int64)
        for k, ns in set_names.items()
    }
    marker_sets = {k: v for k, v in marker_sets.items() if v.size > 0}
    if not marker_sets:
        raise ValueError(
            "No marker set survived: none of the HMM names in the .hmm file(s) "
            "match the query names in the hmmout. Are they from the same run?"
        )
    info["n_marker_sets"] = {k: int(v.size) for k, v in marker_sets.items()}

    name_to_idx = {n: i for i, n in enumerate(namelist)}
    rows, cols = [], []
    n_unmapped_contig = 0
    n_unknown_hmm = 0
    for h in hits:
        contig = mapping[h.orf_id]
        ri = name_to_idx.get(contig)
        if ri is None:
            n_unmapped_contig += 1
            continue
        ci = vocab.get(h.hmm_name)
        if ci is None:
            n_unknown_hmm += 1
            continue
        rows.append(ri)
        cols.append(ci)

    info["n_hits_dropped_contig_not_in_namelist"] = n_unmapped_contig
    info["n_hits_dropped_hmm_not_in_vocab"] = n_unknown_hmm
    info["n_hits_used"] = len(rows)

    if not rows:
        raise ValueError(
            "Marker table is empty after joining. "
            f"{n_unmapped_contig} hits had contigs outside namelist "
            f"(usually a length-filter mismatch) and {n_unknown_hmm} hits had "
            "HMM names outside the vocabulary."
        )

    counts = csr_matrix(
        (np.ones(len(rows), dtype=np.int32), (rows, cols)),
        shape=(len(namelist), len(vocab)),
    )
    counts.sum_duplicates()

    if contig_lengths is None:
        contig_lengths = np.zeros(len(namelist), dtype=np.int64)

    table = MarkerTable(namelist, counts, vocab_order, marker_sets, contig_lengths)
    info["contigs_with_markers"] = table.contigs_with_markers()
    info["max_achievable_completeness"] = table.max_achievable_completeness()
    return table, info


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------
def diagnose_marker_table(table: MarkerTable, info: dict, out=sys.stdout) -> None:
    p = lambda s: print(s, file=out)
    p("=" * 68)
    p("MARKER TABLE DIAGNOSTIC")
    p("=" * 68)
    p(f"hmmout format detected      : {info['hmmout_format']}")
    p(f"hits after dedup/filtering  : {info['n_hits']}")
    p(
        f"ORF->contig rule chosen     : {info['orf_id_style']} "
        f"({info['orf_join_rate']:.1%} of ORFs matched a contig)"
    )
    for style, rate in sorted(info["orf_join_all"].items(), key=lambda t: -t[1]):
        p(f"    {style:<18} {rate:6.1%}")
    p(f"hits dropped (contig n/a)   : {info['n_hits_dropped_contig_not_in_namelist']}")
    p(f"hits dropped (hmm n/a)      : {info['n_hits_dropped_hmm_not_in_vocab']}")
    p(f"hits used                   : {info['n_hits_used']}")
    p(f"contigs                     : {table.n_contigs}")
    p(
        f"contigs carrying >=1 marker : {info['contigs_with_markers']} "
        f"({info['contigs_with_markers'] / max(table.n_contigs,1):.1%})"
    )
    p(f"marker sets                 : {info['n_marker_sets']}")
    p("")
    p("Ceiling check -- completeness if the WHOLE assembly were one bin:")
    for sname, v in info["max_achievable_completeness"].items():
        flag = "" if v >= 0.9 else "   <-- LOW, thresholds above this are unreachable"
        p(f"    {sname:<18} {v:6.1%}{flag}")
    p("")

    per_contig = np.diff(table.counts.indptr)
    if per_contig.max() > 0:
        p("Per-contig unique marker counts (top 5):")
        top = np.argsort(per_contig)[::-1][:5]
        for i in top:
            q = table.quality([int(i)])
            p(
                f"    {table.namelist[i][:48]:<48} "
                f"markers={per_contig[i]:4d} comp={q.completeness:6.1%} "
                f"cont={q.contamination:6.1%}"
            )
    p("")
    whole = table.quality(range(table.n_contigs))
    p(
        f"Whole-assembly bin: comp={whole.completeness:.1%} "
        f"cont={whole.contamination:.1%} set={whole.marker_set}"
    )
    p(
        "(a huge contamination here is expected and healthy -- it means the "
        "markers are being counted; near-zero means the join is broken)"
    )
    p("=" * 68)


def _cli(argv: Sequence[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="marker_quality")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("diagnose")
    d.add_argument("--hmmout", required=True)
    d.add_argument("--contigs", required=True, help="assembly FASTA")
    d.add_argument(
        "--hmm-file",
        action="append",
        default=[],
        help="name=path/to/markers.hmm (repeatable)",
    )
    d.add_argument("--min-contig-len", type=int, default=1000)
    d.add_argument("--max-evalue", type=float, default=1e-5)
    d.add_argument("--no-best-hit-per-orf", action="store_true")
    args = ap.parse_args(argv)

    lengths = read_fasta_lengths(args.contigs)
    namelist = [n for n, L in lengths.items() if L >= args.min_contig_len]
    clen = np.array([lengths[n] for n in namelist], dtype=np.int64)
    hmm_files = dict(kv.split("=", 1) for kv in args.hmm_file) or None

    table, info = build_marker_table(
        args.hmmout,
        namelist,
        contig_lengths=clen,
        hmm_files=hmm_files,
        max_evalue=args.max_evalue,
        best_hit_per_orf=not args.no_best_hit_per_orf,
    )
    diagnose_marker_table(table, info)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv[1:]))
