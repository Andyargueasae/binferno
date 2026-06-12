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
    contig_file: Path,
    output_dir: Path,
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
    # rewrite to Path.
    # markerExeURL = Path(__file__).parent / "auxiliary" / f"test_getmarker_{quarter}.pl"
    markerExeURL = Path(__file__).parent / "auxiliary" / f"getmarker_{quarter}.py"
    markerURL = Path(__file__).parent / "auxiliary" / f"{marker_name}.hmm"

    fragResult_prefix = output_dir / (contig_file.name + ".frag")
    hmmResultURL = output_dir / (contig_file.name + "." + marker_name + ".hmmout")
    seedURL = output_dir / (
        contig_file.name
        + "."
        + marker_name
        + "."
        + quarter
        + "_lencutoff_"
        + str(contig_length_threshold)
        + ".seed"
    )
    marker_gene_faa = output_dir / (contig_file.name + ".frag.faa")

    if not (os.path.exists(marker_gene_faa)):
        fragCmd = (
            fragScanURL
            + " -genome="
            + str(contig_file)
            + " -out="
            + str(fragResult_prefix)
            + " -complete=0 -train=complete -thread="
            + str(threads)
            + " 1>"
            + str(fragResult_prefix)
            + ".out 2>"
            + str(fragResult_prefix)
            + ".err"
        )
        logger.info("exec cmd: " + fragCmd)
        subprocess.run(fragCmd, shell=True, check=True)

    if os.path.exists(marker_gene_faa):
        if not (os.path.exists(hmmResultURL)):
            hmmCmd = (
                str(hmmExeURL)
                + " --domtblout "
                + str(hmmResultURL)
                + " -E 1e-10 --cpu "
                + str(threads)
                + " "
                + str(markerURL)
                + " "
                + str(marker_gene_faa)
                + " 1>"
                + str(hmmResultURL)
                + ".out 2>"
                + str(hmmResultURL)
                + ".err"
            )
            if marker_name == "pltd_marker":
                hmmCmd = (
                    str(hmmExeURL)
                    + " --domtblout "
                    + str(hmmResultURL)
                    + " -E 1e-10 --cpu "
                    + str(threads)
                    + " "
                    + str(markerURL)
                    + " "
                    + str(marker_gene_faa)
                    + " 1>"
                    + str(hmmResultURL)
                    + ".out 2>"
                    + str(hmmResultURL)
                    + ".err"
                )
            logger.info("exec cmd: " + hmmCmd)
            subprocess.run(hmmCmd, shell=True, check=True)

        if os.path.exists(hmmResultURL):
            if not (os.path.exists(seedURL)):
                markerCmd = (
                    str(markerExeURL)
                    + " "
                    + str(hmmResultURL)
                    + " "
                    + str(contig_file)
                    + " "
                    + str(contig_length_threshold)
                    + " "
                    + str(seedURL)
                )
                logger.info("exec cmd: " + markerCmd)
                subprocess.run(markerCmd, shell=True, check=True)

            if os.path.exists(seedURL):
                candK = file_len(seedURL)
            else:
                logger.info("markerCmd failed! Not exist: " + str(seedURL))
                candK = 0
        else:
            logger.info("Hmmsearch failed! Not exist: " + str(hmmResultURL))
            os.remove(hmmResultURL)
            sys.exit()
    else:
        logger.info("FragGeneScan failed! Not exist: " + str(marker_gene_faa))
        os.remove(marker_gene_faa)
        sys.exit()
    return candK
