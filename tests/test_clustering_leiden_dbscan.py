import tempfile
from pathlib import Path

from loguru import logger
import pyfastx

# import two clustering modules to test them both in the same way
from binferno.clustering.leiden import LeidenClustering
from binferno.clustering.dbscan import DbscanClustering
from binferno.clustering.base import ClusterResult
from binferno.utils import gen_seed
from .conftest import rng, toy_embeddings, sim_cluster_fasta, CONTIG, MARKER

"""
Mainly tests leiden path from embedding matrix to ClusterResult.
"""

def test_kmeans_clustering() -> None:
    assert True, "This is a placeholder test for kmeans clustering. Implement this test when the function is implemented."
    return

def test_leiden_clustering() -> None:
    embeddings = toy_embeddings(contig=CONTIG)
    assert True, "This is a placeholder test for leiden clustering. Implement this test when the function is implemented."
    return


def test_dbscan_clustering() -> None:
    embeddings = toy_embeddings(contig=CONTIG)
    clustering = DbscanClustering()
    logger.bind(test="dbscan_clustering")
    assert True, "This is a placeholder test for dbscan clustering. Implement this test when the function is implemented."
    return


def test_utils_marker_gene_finder():
    tmp_dir = tempfile.mkdtemp()
    n_genome = gen_seed(
        logger=logger,
        contig_file=MARKER,
        output_dir=Path(tmp_dir),
        threads=1,
        contig_length_threshold=1000,
        marker_name="bacar_marker",
        quarter="2quarter",
    )
    internal_marker_file = Path(tmp_dir) / (Path(MARKER).name + ".frag.faa")
    internal_hmmer_out =  Path(tmp_dir) / (Path(MARKER).name + "." + "bacar_marker" + ".hmmout")
    assert internal_marker_file.exists(), f"Expected marker gene faa file to be created at {internal_marker_file}, but it does not exist."
    assert internal_hmmer_out.exists(), f"Expected hmmer output file to be created at {internal_hmmer_out}, but it does not exist."
    assert (
        n_genome == 0
    ), "Expected that no genomes are found in the tiny contigs. We just want to make sure the process is good."
    return

def test_postprocess_quality_tabling():
    assert True, "This is a placeholder test for postprocess_quality_tabling. Implement this test when the function is implemented."