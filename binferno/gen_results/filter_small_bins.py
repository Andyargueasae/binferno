import gzip
import os
import shutil
from collections import defaultdict
from typing import Dict, Iterator, Tuple, TextIO


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


def _read_result_map(resultfile: str) -> Tuple[Dict[str, list], Dict[str, str]]:
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
    shutil.copy2(filtered_tsv, os.path.join(output_path, "BTBin_res.tsv"))
