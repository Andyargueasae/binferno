from pathlib import Path

from pydantic import Field
import networkx as nx
import pandas as pd
import numpy as np

from .base import BaseFeatureBuilder

class TaxonomyFeatureBuilder(BaseFeatureBuilder):
    """
    A feature builder that constructs a taxonomy graph from contig sequences and their taxonomic classifications.
    
    Insights come from:
     - TaxVamb and taxometer: the extraordinary work to use taxonomy as a feature for binning, and the taxonomic classifications they provide for contigs.
    """

    def __call__(self, contigs: pd.DataFrame, taxonomy: Path, output_dir: Path = None, **kwargs):
        """
        Constructs a taxonomy graph and returns it as a DataFrame.

        Parameters:
        - contigs: A DataFrame containing contig information, including taxonomic classifications.
        - taxonomy: A path to a CSV file containing taxonomic classifications for each contig.
        - output_dir: Optional path to save the resulting graph.

        Returns:
        - A DataFrame representing the taxonomy graph.
        """
        # Load the taxonomy CSV file
        taxonomy_df = pd.read_csv(taxonomy)

        # Merge contigs with taxonomy information
        merged_df = pd.merge(contigs, taxonomy_df, on='contig_id', how='left')

        # Create a graph from the merged DataFrame
        G = nx.from_pandas_edgelist(merged_df, source='parent_taxon', target='child_taxon')

        # Convert the graph to a DataFrame
        graph_df = pd.DataFrame(G.edges(), columns=['parent_taxon', 'child_taxon'])

        # Optionally save the graph to the output directory
        if output_dir is not None:
            graph_df.to_csv(output_dir / 'taxonomy_graph.csv', index=False)

        return graph_df