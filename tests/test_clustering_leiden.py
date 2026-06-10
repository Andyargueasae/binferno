import tempfile
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

from binferno.clustering.leiden import LeidenClustering

'''
Mainly tests leiden path from embedding matrix to ClusterResult.
'''

@pytest.fixture
def leiden_clustering():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        clustering = LeidenClustering(output_dir=output_dir, resolution=1.0)
        yield clustering

def test_leiden_clustering(leiden_clustering):
    # Create a sample distance matrix
    distance_matrix = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    
    # Run the clustering
    leiden_clustering.fit(distance_matrix)
    
    # Check that the output file exists
    output_file = leiden_clustering.output_dir / "leiden_clusters.csv"
    assert output_file.exists(), "Output file does not exist."
    
    # Check that the output file has the correct format
    df = pd.read_csv(output_file)
    assert "cluster" in df.columns, "Output file does not contain 'cluster' column."
    assert len(df) == distance_matrix.shape[0], "Output file does not have the correct number of rows."