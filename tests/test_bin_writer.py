import tempfile 
from pathlib import Path

import pytest
import numpy as np

from binferno.postprocessing import BinfernoPostprocessor

@pytest.fixture
def binferno_postprocessor():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        postprocessor = BinfernoPostprocessor(checkm_database="path/to/checkm_db", minbin_size=200_000)
        yield postprocessor

def test_binferno_postprocessor(binferno_postprocessor):
    # Create dummy inputs
    contigs = Path("path/to/contigs.fasta")
    cluster_result = None  # You would need to create a mock ClusterResult object
    output_path = Path("path/to/output")
    
    # Run the postprocessor
    try:
        binferno_postprocessor(
            contigs=contigs,
            cluster_result=cluster_result,
            output_path=output_path,
            num_threads=5,
            logger=None,
            plastid=False
        )
    except Exception as e:
        pytest.fail(f"BinfernoPostprocessor raised an exception: {e}")