from pathlib import Path

from pydantic import Field
import networkx as nx
import pandas as pd

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
        raise NotImplementedError("TaxonomyFeatureBuilder is not implemented yet. This is a placeholder for future development.")
    
    def _construct_taxonomy_graph(self, taxonomy_df: pd.DataFrame) -> nx.Graph:
        """
        Constructs a taxonomy graph from a DataFrame containing taxonomic classifications.

        Parameters:
        - taxonomy_df: A DataFrame with columns 'parent_taxon' and 'child_taxon' representing taxonomic relationships.

        Returns:
        - A NetworkX graph representing the taxonomy.
        """
        G = nx.from_pandas_edgelist(taxonomy_df, source='parent_taxon', target='child_taxon')
        return G