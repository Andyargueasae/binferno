import tempfile
from pathlib import Path

import pytest
import pandas as pd
import numpy as np

from binferno.training import SimCLRTrainer


'''
Mainly tests the SimCLR path: from embeddings to dataloading to training and encoding. 
Create dummy data to serve testing purposes.  
'''


@pytest.fixture
def simclr_trainer():
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = SimCLRTrainer(
            data_dir=Path(tmpdir) / "data",
            output_dir=Path(tmpdir) / "output",
            num_epochs=1,
            batch_size=4,
            learning_rate=1e-3,
            num_workers=0,
        )
        yield trainer

def test_simclr_training(simclr_trainer):
    # Create dummy data
    data_dir = simclr_trainer.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    for i in range(10):
        df = pd.DataFrame({
            "feature1": np.random.rand(100),
            "feature2": np.random.rand(100),
            "label": np.random.randint(0, 2, size=100),
        })
        df.to_csv(data_dir / f"sample_{i}.csv", index=False)

    # Run training
    simclr_trainer.train()

    # Check if output files are created
    output_dir = simclr_trainer.output_dir
    assert (output_dir / "model.pth").exists(), "Model file not found"
    assert (output_dir / "training_log.txt").exists(), "Training log not found"