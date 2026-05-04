from pathlib import Path
import typer

from . import features, training, clustering

app = typer.Typer()

@app.command()
def main(
    fasta:Path,
    bam:Path,
    kmer:bool = typer.Option(True, help="Whether to include a kmer features"),
    corgi:bool = typer.Option(True, help="Whether to include a corgi features"),
    coverage:bool = typer.Option(True, help="Whether to include a coverage features"),
    simclr:bool = typer.Option(True, help="Whether to include a SimCLR training module"),
    barlowtwins:bool = typer.Option(False, help="Whether to include a BarlowTwins training module"),
):
    # Build Features
    feature_module = features.build_feature_modules(kmer, corgi, coverage)
    features = feature_module(fasta, bam)
    
    # Run Training
    training_module = training.build_training_modules(simclr, barlowtwins)
    results = training_module(features)

    # Run clustering
    clustering_module = clustering.build_clustering_module()
    clusters = clustering_module(results)

    # TODO Output results
    