import glob
from pathlib import Path
import tyro

from binferno.training import BinfernoTrainer
from binferno.clustering import BinfernoClustering
from binferno.gen_results import BinfernoResults
from binferno.features import BinfernoFeatureBuilder
from loguru import logger
from .utils import gen_seed, calc_n50


def binferno(
    contigs: Path,
    bam: Path,
    output_dir: Path,
    training: BinfernoTrainer,
    clustering: BinfernoClustering,
    results: BinfernoResults,
    features: BinfernoFeatureBuilder,
    threads: int = 4,
    only_training: bool = False,  # True: only run training, False: full process.
    plastid: bool = False,
):

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.add(f"{output_dir}/binferno.log", rotation="1 MB", retention="10 days")


    embeddings = training(
        contigs=contigs,
        bam=bam,
        output_dir=output_dir,
        temperature=temperature,
        logger=logger,
        threads=threads,
        features=features,
    )

    if only_training:
        logger.info(
            "Only training stage is set, temperarily skipping clustering and results generations."
        )
        return

    logger.info("Starting clustering stage.")
    minbin_size = 200000
    if plastid:
        seed_num = gen_seed(
            logger,
            contig_file=str(fasta),
            threads=threads,
            contig_length_threshold=1001,
            marker_name="pltd_marker",
            quarter="2quarter",
        )
        seed_file = f"{str(fasta)}.pltd_marker.2quarter_lencutoff_1001.seed"
        minbin_size = 30000
    else:
        seed_num = gen_seed(
            logger,
            contig_file=str(fasta),
            threads=threads,
            contig_length_threshold=1001,
            marker_name="bacar_marker",
            quarter="2quarter",
        )
        seed_file = f"{str(fasta)}.bacar_marker.2quarter_lencutoff_1001.seed"

    cluster_output_dir = output_dir / "cluster_res"
    if not glob.glob(str(cluster_output_dir / "Leiden*.tsv")):
        clustering(
            contig_file=str(fasta),
            embeddings=embeddings,
            seed_file=seed_file,
            output_path=cluster_output_dir,
            logger=logger,
            threads=threads,
        )
    else:
        logger.info("Clustering results already exist, skipping clustering step.")

    # Run Output results generation
    results_output_dir = output_dir / "res_bins"
    results_output_dir.mkdir(parents=True, exist_ok=True)

    results(
        contig_file=str(fasta),
        seed_num=seed_num,
        num_threads=threads,
        output_path=output_dir,
        plastid=plastid,
        minbinsize=minbin_size,
        logger=logger,
    )

    logger.info("Binferno pipeline completed successfully.")
    return

def main():
    tyro.cli(binferno)


if __name__ == "__main__":
    main()
