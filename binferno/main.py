from pathlib import Path
import typer

from . import features, training


@typer.command()
def main(
    fasta:Path,
    bam:Path,
    kmer:bool = typer.Option(True, help="Whether to include a kmer features"),
    corgi:bool = typer.Option(True, help="Whether to include a corgi features"),
    coverage:bool = typer.Option(True, help="Whether to include a coverage features"),
):
    # Build Features
    feature_module = features.build_feature_modules(kmer, corgi, coverage)
    features = feature_module("test.fasta")
    
    # Run Training
    training_module = training.build_training_modules()