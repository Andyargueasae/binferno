import gzip
from Bio import SeqIO
import mimetypes
import os
import subprocess
import pyfastx
from pathlib import Path
import sys


def get_length(fastx_file):
    file_type = mimetypes.guess_type(fastx_file)[1]
    if file_type == "gzip":
        f = gzip.open(fastx_file, "rt")
    elif not file_type:
        f = open(fastx_file, "rt")
    else:
        raise RuntimeError("Unknown type of file: '{}".format(fastx_file))
    length = {}
    if os.path.getsize(fastx_file) == 0:
        return length
    file_format = None
    line = f.readline()
    if line.startswith("@"):
        file_format = "fastq"
    elif line.startswith(">"):
        file_format = "fasta"
    f.seek(0)
    if not file_format:
        raise RuntimeError("Invalid sequence file: '{}".format(fastx_file))
    for seq_record in SeqIO.parse(f, file_format):
        length[seq_record.id] = len(seq_record.seq)

    f.close()
    return length


def save_result(result, filepath, namelist):
    filedir, filename = os.path.split(filepath)
    if not filename:
        filename = "result.tsv"
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    f = open(filepath, "w")
    for contigIdx in range(len(result)):
        f.write(namelist[contigIdx] + "\t" + str(result[contigIdx].item(0)) + "\n")
    f.close()


def calc_n50(contigs: Path, min_length: int = 1000) -> int:
    seq_record = pyfastx.Fastx(contigs)
    lengths = sorted(
        [len(record) for _, record in seq_record if len(record) >= min_length],
        reverse=True,
    )
    # only care about the lengths over 1000bp.
    total_length = sum(lengths)
    cumulative_length = 0
    for length in lengths:
        cumulative_length += length
        if cumulative_length >= total_length / 2:
            return length
    return 0


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def gen_seed(
    logger,
    contig_file: str,
    output_dir: str,
    threads: int,
    contig_length_threshold: int,
    marker_name: str = "marker",
    quarter: str = "3quarter",
):
    """
    Generate seed sequences from contigs using FragGeneScan, HMMsearch, and custom markers.

    :param contig_file: Path to the input contig file.
    :param threads: The number of threads to use for processing.
    :param contig_length_threshold: The contig length threshold.
    :param marker_name: The marker name (default: "marker").
    :param quarter: The quarter identifier (default: "3quarter").
    :return: The number of candidate seeds generated.
    """
    fragScanURL = "run_FragGeneScan.pl"

    hmmExeURL = "hmmsearch"
    markerExeURL = os.path.join(
        os.path.dirname(__file__), "auxiliary", "test_getmarker_" + quarter + ".pl"
    )
    markerURL = os.path.join(
        os.path.dirname(__file__), "auxiliary", marker_name + ".hmm"
    )

    fragResult_prefix = os.path.join(
        output_dir, os.path.basename(contig_file) + ".frag"
    )
    hmmResultURL = os.path.join(
        output_dir, os.path.basename(contig_file) + "." + marker_name + ".hmmout"
    )
    seedURL = os.path.join(
        output_dir,
        os.path.basename(contig_file)
        + "."
        + marker_name
        + "."
        + quarter
        + "_lencutoff_"
        + str(contig_length_threshold)
        + ".seed",
    )
    marker_gene_faa = f"{fragResult_prefix}.faa"
    if not (os.path.exists(marker_gene_faa)):
        fragCmd = (
            fragScanURL
            + " -genome="
            + contig_file
            + " -out="
            + fragResult_prefix
            + " -complete=0 -train=complete -thread="
            + str(threads)
            + " 1>"
            + fragResult_prefix
            + ".out 2>"
            + fragResult_prefix
            + ".err"
        )
        logger.info("exec cmd: " + fragCmd)
        subprocess.run(fragCmd, shell=True, check=True)

    if os.path.exists(marker_gene_faa):
        if not (os.path.exists(hmmResultURL)):
            hmmCmd = (
                hmmExeURL
                + " --domtblout "
                + hmmResultURL
                + " -E 1e-10 --cpu "
                + str(threads)
                + " "
                + markerURL
                + " "
                + marker_gene_faa
                + " 1>"
                + hmmResultURL
                + ".out 2>"
                + hmmResultURL
                + ".err"
            )
            if marker_name == "pltd_marker":
                hmmCmd = (
                    hmmExeURL
                    + " --domtblout "
                    + hmmResultURL
                    + " -E 1e-10 --cpu "
                    + str(threads)
                    + " "
                    + markerURL
                    + " "
                    + marker_gene_faa
                    + " 1>"
                    + hmmResultURL
                    + ".out 2>"
                    + hmmResultURL
                    + ".err"
                )
            logger.info("exec cmd: " + hmmCmd)
            subprocess.run(hmmCmd, shell=True, check=True)

        if os.path.exists(hmmResultURL):
            if not (os.path.exists(seedURL)):
                markerCmd = (
                    markerExeURL
                    + " "
                    + hmmResultURL
                    + " "
                    + contig_file
                    + " "
                    + str(contig_length_threshold)
                    + " "
                    + seedURL
                )
                logger.info("exec cmd: " + markerCmd)
                subprocess.run(markerCmd, shell=True, check=True)

            if os.path.exists(seedURL):
                candK = file_len(seedURL)
            else:
                logger.info("markerCmd failed! Not exist: " + markerCmd)
                candK = 0
        else:
            logger.info("Hmmsearch failed! Not exist: " + hmmResultURL)
            os.remove(hmmResultURL)
            sys.exit()
    else:
        logger.info("FragGeneScan failed! Not exist: " + marker_gene_faa)
        os.remove(marker_gene_faa)
        sys.exit()
    return candK
