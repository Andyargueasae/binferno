import tempfile
import sys
import logging
from pathlib import Path
import os

import numpy as np
try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from .conftest import tiny_fasta, CONTIG, BAM

from binferno.features.kmer import KmerFeatureBuilder
from binferno.features.coverage import CoverageFeatureBuilder
from binferno.training.simclr import SimCLRTrainer
from binferno.training.barlowtwins import BarlowTwinsTrainer

'''
Mainly tests for feature building, including kmer composition features, coverage mean and std, and Corgi features.
Will assess whether values are of correct shape, non-negative, and non-zero where applicable.
'''

def dummy_kmer_args():
    return {
        "k": 4,
        "pseudocount": 1,
    }

def gen_simclr_fragments():
    tempdir=tempfile.TemporaryDirectory()
    contigs=tiny_fasta(tempdir)
    output_dir=tempdir
    
    num_view=6
    trainer = SimCLRTrainer()
    fragment_dict, all_fragments, namelist = trainer.gen_views(
        contigs=contigs, 
        output_dir=output_dir,
        view_num=num_view,
    )

    assert isinstance(fragment_dict, dict)
    assert isinstance(all_fragments, list)
    assert isinstance(namelist, list)
    assert len(all_fragments) == len(namelist)*num_view, f"Expected {len(namelist)*num_view} fragments, but got {len(all_fragments)}."

    # assert if each contig is reprersented num_view times.
    for contig in namelist:
        count = sum([1 for name in namelist if name.startswith(contig)])
        assert count == num_view, f"Contig {contig} should have {num_view} fragments, but got {count}."
    return 

def gen_barlowtwins_fragments():
    # Then the test would be barlow twins.
    
    tempdir=tempfile.TemporaryDirectory()
    contigs=tiny_fasta(tempdir)
    output_dir=tempdir
    num_view=8
    
    trainer = BarlowTwinsTrainer()
    fragment_dict, all_fragments, namelist = trainer.gen_views(
        contigs=contigs,
        output_dir=output_dir,
        view_num=num_view,
    )

    assert isinstance(fragment_dict, dict)
    assert isinstance(all_fragments, list)
    assert isinstance(namelist, list)

    assert len(namelist) <= len(all_fragments) <= len(namelist)*num_view, f"Expected at most {len(namelist)*num_view} fragments, but got {len(all_fragments)}."

def test_kmer_features(dummy_all_fragments):
    kmer_args = dummy_kmer_args()
    _, dummy_all_frags, _ = dummy_all_fragments
    builder = KmerFeatureBuilder(k=kmer_args.get("k"), pseudocount=kmer_args.get("pseudocount"))
    features = builder(dummy_all_frags).values
    assert features.shape == (len(dummy_all_frags), 136), f"Expected feature shape ({len(dummy_all_frags)}, 136), but got {features.shape}."
    assert np.all(np.linalg.norm(features, axis=1) > 0)
    return

def test_coverage_features(bam_dir, dummy_all_fragments):    
    builder = CoverageFeatureBuilder()
    bams = bam_dir
    dummy_frags_dict, all_frags, _ = dummy_all_fragments

    logger.info(f"Generated {len(dummy_frags_dict)} fragments for testing.")
    cov_df = builder(fragment_dict=dummy_frags_dict, bam_path=bams, threads=1, logger=logger)
    assert cov_df.shape[0] == len(all_frags), f"Expected {len(all_frags)} rows, but got {cov_df.shape[0]}."
    assert np.all(cov_df.values >= 0), "Coverage values should be non-negative."
    # current bam is empty, so coverage should be all zeros.
    assert np.all(cov_df.values == 0), "Coverage values should be zero for an empty BAM file."

    # now try to get a pair of contigs and bams with actual non-zero coverage. 
    contigs=CONTIG

    trainer = SimCLRTrainer()
    fragments_dict, all_frags, _ = trainer.gen_views(
        contigs=contigs,
        output_dir=Path(tempfile.mkdtemp()),
        view_num=6,
    ) # get all fragments.

    assert os.path.exists(BAM), f"BAM file {BAM} does not exist."
    cov_df = builder(fragment_dict=fragments_dict, bam_path=BAM, threads=1, logger=logger)
    assert cov_df.shape[0] == len(all_frags), f"Expected {len(all_frags)} rows, but got {cov_df.shape[0]}."
    assert cov_df.shape[1] == 2, f"Expected 2 columns (mean and std) for one sample, but got {cov_df.shape[1]}."
    assert np.all(cov_df.values >= 0), "Coverage values should be non-negative."

