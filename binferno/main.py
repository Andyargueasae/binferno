from pathlib import Path
import tyro

from .features import BinfernoFeatures
from .training import BinfernoTraining
from .clustering import BinfernoClustering

def binferno(
    fasta:Path,
    bam:Path,
    features:BinfernoFeatures,
    training:BinfernoTraining,
    clustering:BinfernoClustering,
):
    # Run Training
    results = training(fasta, bam, features)

    # Run clustering
    clusters = clustering(results)

    # TODO Output results
    

def main():
    tyro.cli(binferno)
