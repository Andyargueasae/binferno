from pathlib import Path
import shutil

import tyro
from typing_extensions import Annotated
from loguru import logger

from binferno.training import BinfernoTrainer
from binferno.clustering import BinfernoClustering
from binferno.postprocessing import BinfernoPostprocessor
from binferno.features import BinfernoFeatureBuilder


def binferno(
    # Input contigs in fasta format.
    contigs: Path,
    # Input bam file for the contigs, used for coverage estimation.
    bam: Path,
    # Output directory to store all results, including intermediate results and final bins.
    output_dir: Path,
    
    features: BinfernoFeatureBuilder,
    training: BinfernoTrainer,
    clustering: BinfernoClustering,
    postprocessor: BinfernoPostprocessor,
    threads: Annotated[
        int,
        tyro.conf.arg(
            name="threads",
            metavar="THREADS", 
            help="Number of threads to use for parallel processing."),
    ] = 4,
    train_only: Annotated[
        bool,
        tyro.conf.arg(
            name="train_only",
            metavar="TRAIN_ONLY",
            help="True: only run training, False: full process.",
        ),
    ] = False,

) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.add(f"{output_dir}/binferno.log", rotation="1 MB", retention="10 days")

    embeddings = training(
        contigs=contigs,
        bam=bam,
        output_dir=output_dir,
        features=features,
        threads=threads,
        logger=logger,
    )

    if train_only:
        logger.info(
            "Only training stage is set, temperarily skipping clustering and results generations."
        )
        return

    logger.info("Starting clustering and writing MAGs.")
    cluster_output_dir = output_dir / "cluster_res"
    if cluster_output_dir.exists():
        logger.warning(
            f"Cluster output directory {cluster_output_dir} already exists. It will be overwritten."
        )
        shutil.rmtree(cluster_output_dir)
    cluster_output_dir.mkdir(parents=True, exist_ok=True)

    cluster_result = clustering(
        contigs=contigs,
        embeddings=embeddings,
        output_path=cluster_output_dir,
        threads=threads,
        logger=logger,
    )

    with open(cluster_output_dir / "cluster_results.pkl", "wb") as f:
        import pickle
        pickle.dump(cluster_result, f)

    # Run Output results generation
    logger.info("Start writing results.")
    results_output_dir = output_dir / "res_bins"
    results_output_dir.mkdir(parents=True, exist_ok=True)

    postprocessor(
        contigs=contigs,
        cluster_result=cluster_result,
        num_threads=threads,
        output_path=output_dir,
        plastid=clustering.plastid,
        logger=logger,
    )

    logger.info("Binferno pipeline completed successfully.")
    return


def main():
    tyro.cli(binferno)


if __name__ == "__main__":
    main()
