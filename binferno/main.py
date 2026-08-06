from pathlib import Path

import tyro
from loguru import logger
from typing_extensions import Annotated

from binferno.training import BinfernoTrainer
from binferno.clustering import BinfernoClustering
from binferno.postprocessing import BinfernoPostprocessor
from binferno.features import BinfernoFeatureBuilder


def binferno(
    contigs: Annotated[
        Path,
        tyro.conf.arg(
            name="contigs",
            metavar="CONTIGS",
            help="Path to the input contigs in fasta format.",
        ),
    ],
    bam_dir: Annotated[
        Path,
        tyro.conf.arg(
            name="bam_dir",
            metavar="BAM_DIR",
            help="Path to the input directory of BAM files, must be paired with contigs. Used for calculating coverage mean and std.",
        ),
    ],
    output_dir: Annotated[
        Path,
        tyro.conf.arg(
            name="output_dir",
            metavar="OUTPUT_DIR",
            help="Path to the output directory to store all results, including intermediate results and final bins.",
        ),
    ],
    features: BinfernoFeatureBuilder,
    training: BinfernoTrainer,
    clustering: BinfernoClustering,
    postprocessor: BinfernoPostprocessor,
    threads: Annotated[
        int,
        tyro.conf.arg(
            name="threads",
            metavar="THREADS",
            help="Number of threads to use for parallel processing.",
        ),
    ] = 4,
    longread: Annotated[
            bool,
            tyro.conf.arg(
                name="longread",
                metavar="LONGREAD",
                help="True: input contigs are from long-read assembly, False: input contigs are from short-read assembly.",
            ),
        ]=False,
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
        bam=bam_dir,
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
        logger.warning(f"Cluster output directory {cluster_output_dir} already exists.")

    cluster_output_dir.mkdir(parents=True, exist_ok=True)

    cluster_result = clustering(
        contigs=contigs,
        embeddings=embeddings,
        output_path=cluster_output_dir,
        threads=threads,
        logger=logger,
        longread=longread,
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
