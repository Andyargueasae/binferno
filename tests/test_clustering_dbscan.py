import tempfile
from pathlib import Path

import pytest
import pandas as pd
import numpy as np

from binferno.clustering import DbscanClustering

@pytest.fixture
def dbscan_clustering():
    with tempfile.TemporaryDirectory() as tmpdir:
        clustering = DbscanClustering(
            data_dir=Path(tmpdir) / "data",
            output_dir=Path(tmpdir) / "output",
            eps=0.5,
            min_samples=5,
        )
        yield clustering

def test_dbscan_clustering(dbscan_clustering):
    # Create dummy data
    data_dir = dbscan_clustering.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    for i in range(10):
        df = pd.DataFrame({
            "feature1": np.random.rand(100),
            "feature2": np.random.rand(100),
        })
        df.to_csv(data_dir / f"sample_{i}.csv", index=False)

    # Run clustering
    dbscan_clustering.cluster()

    # Check if output files are created
    output_dir = dbscan_clustering.output_dir
    assert (output_dir / "clusters.csv").exists(), "Clusters file not found"
    assert (output_dir / "clustering_log.txt").exists(), "Clustering log not found"