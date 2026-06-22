import tempfile
from pathlib import Path

from loguru import logger
import pyfastx

# import two clustering modules to test them both in the same way
from binferno.clustering.leiden import LeidenClustering
from binferno.clustering.dbscan import DbscanClustering
from binferno.clustering.base import ClusterResult
from binferno.utils import gen_seed
from .conftest import rng, toy_embeddings, sim_cluster_fasta, CONTIG

"""
Mainly tests leiden path from embedding matrix to ClusterResult.
"""


def test_leiden_clustering() -> None:
    embeddings = toy_embeddings(contig=CONTIG)

    clustering = LeidenClustering()
    logger.bind(test="leiden_clustering")

    cluster_result = clustering(
        logger=logger,
        embeddings=embeddings,
        contigs=CONTIG,
        output_path=Path(tempfile.mkdtemp()),
    )
    # now, we should test the cluster_result object, to see if there are anything.
    assert (
        cluster_result.n_genome
    ), "Leiden clustering should find at least one genome in the tiny contigs."
    assert (
        cluster_result.clusters
    ), "Leiden clustering should assign clusters to the contigs."
    return


def test_dbscan_clustering() -> None:
    embeddings = toy_embeddings(contig=CONTIG)
    clustering = DbscanClustering()
    logger.bind(test="dbscan_clustering")

    cluster_result = clustering(
        logger=logger,
        embeddings=embeddings,
        contigs=CONTIG,
        output_path=Path(tempfile.mkdtemp()),
    )
    # now, we should test the cluster_result object, to see if there are anything.
    assert (
        cluster_result.n_genome
    ), "DBSCAN clustering should find at least one genome in the tiny contigs."
    assert (
        cluster_result.clusters
    ), "DBSCAN clustering should assign clusters to the contigs."
    return


def test_utils_marker_gene_finder():
    n_genome = gen_seed(
        logger=logger,
        contig_file=CONTIG,
        output_dir=Path(tempfile.mkdtemp()),
        threads=1,
        contig_length_threshold=1000,
        marker_name="bacar_marker",
        quarter="2quarter",
    )

    assert (
        n_genome
    ), "Marker gene finder should find at least one genome in the tiny contigs."
    return
