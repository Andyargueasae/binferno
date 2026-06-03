from pathlib import Path
import shutil
import tyro
from binferno.training import BinfernoTrainer
from binferno.clustering import BinfernoClustering
from binferno.gen_results import BinfernoWriter
from binferno.features import BinfernoFeatureBuilder
from loguru import logger


def binferno(
    contigs: Path,
    bam: Path,
    output_dir: Path,
    training: BinfernoTrainer,
    clustering: BinfernoClustering,
    results: BinfernoWriter,
    features: BinfernoFeatureBuilder,
    threads: int = 4,
    only_training: bool = False,  # True: only run training, False: full process.
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.add(f"{output_dir}/binferno.log", rotation="1 MB", retention="10 days")

    embeddings = training(
        contigs=contigs,
        bam=bam,
        output_dir=output_dir,
        logger=logger,
        threads=threads,
        features=features,
    )

    if only_training:
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

    
    n_genome = clustering(
            contigs=str(contigs),
            embeddings=embeddings,
            output_path=cluster_output_dir,
            logger=logger,
            threads=threads,
        )
    
    # Run Output results generation
    logger.info("Start writing results.")
    results_output_dir = output_dir / "res_bins"
    results_output_dir.mkdir(parents=True, exist_ok=True)

    results(
        contigs=str(contigs),
        seed_num=n_genome,
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
