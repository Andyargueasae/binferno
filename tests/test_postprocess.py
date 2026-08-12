import tempfile
from pathlib import Path

import pytest
import numpy as np

from binferno.postprocessing import BinfernoPostprocessor

"""
From the ClusterResult to the output bin writing.
"""


def test_genome_profiler():
    assert (
        True
    ), "This test is a placeholder for testing the postprocessing step of Binferno. Implement this test to check that the postprocessor correctly takes the ClusterResult and writes the output bins as expected."

def test_write_bins():
    assert (
        True
    ), "This test is a placeholder for testing the write_bins function of the BinfernoPostprocessor. Implement this test to verify that the function correctly writes the output bins to disk in the expected format and structure."

def test_qc_select_best_method():
    assert (
        True
    ), "This test is a placeholder for testing the qc_select_best_method function of the BinfernoPostprocessor. Implement this test to ensure that the function correctly evaluates the quality of bins produced by different methods and selects the best one based on predefined criteria."