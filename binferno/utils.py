import gzip
from pathlib import Path
import sys
import mimetypes
import os
import subprocess

from Bio import SeqIO
import pyfastx
import pyhmmer
import pyrodigal


def run_hmmsearch_pyhmmer(
    hmm_path: Path,
    faa_path: Path,
    domtblout_path: Path,
    threads: int,
    e_value: float = 1e-10,
):
    # Load sequences and digitize against the HMM alphabet
    with pyhmmer.easel.SequenceFile(str(faa_path), digital=True) as seq_file:
        sequences = seq_file.read_block()

    with pyhmmer.plan7.HMMFile(str(hmm_path)) as hmm_file:
        hmms = list(hmm_file)

    # E=1e-10 corresponds to the full-sequence reporting threshold
    all_hits = list(pyhmmer.hmmsearch(hmms, sequences, cpus=threads, E=e_value))

    # Write domain table in the same format hmmsearch --domtblout produces
    with open(domtblout_path, "wb") as f:
        for hits in all_hits:
            hits.write(f, format="domains", header=True)


def run_pyrodigal(
    contig_file: Path,
    faa_path: Path,
    threads: int,
):
    # meta=True is the equivalent of FragGeneScan's metagenome mode:
    # no per-sample training, uses prebuilt metagenomic models.
    orf_finder = pyrodigal.GeneFinder(meta=True)

    records = list(SeqIO.parse(str(contig_file), "fasta"))

    def process(record):
        genes = orf_finder.find_genes(bytes(record.seq))
        return record.id, genes

    results = []
    if threads > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=threads) as pool:
            results = list(pool.map(process, records))
    else:
        results = [process(r) for r in records]

    with open(faa_path, "w") as f:
        for contig_id, genes in results:
            genes.write_translations(f, sequence_id=contig_id)


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
    fragResult_prefix = output_dir / (Path(contig_file).name + ".frag")
    # hmmer: try rewriting this cmd-based run to pyhmmer api.
    hmmExeURL = "hmmsearch"
    markerExeURL = Path(__file__).parent / "auxiliary" / f"getmarker_{quarter}.py"
    markerURL = Path(__file__).parent / "auxiliary" / f"{marker_name}.hmm"
    hmmResultURL = output_dir / (Path(contig_file).name + "." + marker_name + ".hmmout")
    seedURL = output_dir / (
        Path(contig_file).name
        + "."
        + marker_name
        + "."
        + quarter
        + "_lencutoff_"
        + str(contig_length_threshold)
        + ".seed"
    )
    marker_gene_faa = output_dir / (Path(contig_file).name + ".frag.faa")

    train_name = "simclr" if Path(output_dir.parent / "simclr").exists() else "barlowtwins"
    scg_faa = output_dir.parent / train_name / "training" / "scg" / f"{contig_file}.scg.faa"
    if scg_faa.exists():
        # copy the scg.faa to marker_gene_faa
        import shutil
        shutil.copyfile(scg_faa, marker_gene_faa)

    if not (os.path.exists(marker_gene_faa)):
        run_pyrodigal(
            contig_file=contig_file,
            faa_path=marker_gene_faa,
            threads=threads,
        )

        logger.info("FragGeneScan/Pyrodigal Finished. Output: " + str(marker_gene_faa))

    if os.path.exists(marker_gene_faa):
        if not (os.path.exists(hmmResultURL)):
            run_hmmsearch_pyhmmer(
                hmm_path=markerURL,
                faa_path=marker_gene_faa,
                domtblout_path=hmmResultURL,
                threads=threads,
                e_value=1e-10,
            )

            logger.info("Hmmer/Pyhmmer search finished. Output: " + str(hmmResultURL))

        if os.path.exists(hmmResultURL):
            if not (os.path.exists(seedURL)):
                markerCmd = (
                    "python "
                    + str(markerExeURL)
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

                logger.info("gethmmmarker finished. Output: " + str(seedURL))

            if os.path.exists(seedURL):
                candK = file_len(seedURL)
            else:
                logger.info("gethmmmarker failed! Not exist: " + str(seedURL))
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
