from pathlib import Path

from pydantic import Field
import networkx as nx
import pandas as pd
import loguru
from typing import List

from .base import BaseFeatureBuilder

TAXONOMY_HEADER = ["contigs", "predictions"]


class TaxonomyFeatureBuilder(BaseFeatureBuilder):
    """
    A feature builder that constructs a taxonomy graph from contig sequences and their taxonomic classifications.

    Insights come from: (June 22 2026)
     - TaxVamb and taxometer: the extraordinary work to use taxonomy as a feature for binning, and the taxonomic classifications they provide for contigs.
    """

    taxonomy: Path = Field(
        Path(),
        description="Path to a TSV file containing taxonomic classifications for each contig. Columns: contig_id, taxonomy_string.",
    )

    def __call__(self, contigs: Path, logger: loguru._logger.Logger) -> pd.DataFrame:
        """
        Constructs a taxonomy graph and returns it as a DataFrame.

        Parameters:
        - contigs: A Path object pointing to the contig file.
        - logger: A loguru logger for logging information.

        Returns:
        - A DataFrame representing the taxonomy graph.
        """
        taxonomy_df = pd.read_csv(self.taxonomy, sep="\t", header=0)
        # filter the contigs by contigs list.
        contig_ids = set()
        with open(contigs, "r") as f:
            for line in f:
                if line.startswith(">"):
                    contig_id = line[1:].strip().split()[0]
                    contig_ids.add(contig_id)
        taxonomy_df = taxonomy_df[taxonomy_df["contigs"].isin(contig_ids)]
        tax_nodes, tax_node_indices, parental_array = self._construct_taxonomy_graph(
            logger, taxonomy_df
        )

        # assign the node index per contig for the finest taxonomy level.
        contig_to_tax_node = {}
        for _, row in taxonomy_df.iterrows():
            contig_id, taxonomy_str = row["contigs"], row["predictions"]
            if not taxonomy_str or taxonomy_str == "unclassified":
                contig_to_tax_node[contig_id] = tax_node_indices[
                    "root"
                ]  # assign to root if unclassified, so they are not used in training. This is a design choice; you may choose to handle unclassified contigs differently.
            taxonomy_levels = taxonomy_str.split(";")
            finest_taxon = taxonomy_levels[-1]
            tax_node_idx = tax_node_indices[finest_taxon]
            contig_to_tax_node[contig_id] = tax_node_idx

        # turn that into a df.
        contig_taxonomy_df = pd.DataFrame(
            {
                "contig_id": list(contig_to_tax_node.keys()),
                "tax_node_idx": list(
                    contig_to_tax_node.values()
                ),  # it already maps conitgs to the finest tax_node index.
            }
        )
        return tax_nodes, tax_node_indices, parental_array, contig_taxonomy_df

    def _construct_taxonomy_graph(
        self, logger: loguru._logger.Logger, taxonomy_df: pd.DataFrame
    ) -> nx.Graph:
        """
        Constructs a taxonomy graph from a DataFrame containing taxonomic classifications.

        Parameters:
        - taxonomy_df: A DataFrame with columns 'parent_taxon' and 'child_taxon' representing taxonomic relationships.

        Returns:
        - A NetworkX graph representing the taxonomy.
        """
        logger.info("Constructing taxonomy graph from taxonomy DataFrame.")
        G = nx.DiGraph()
        root = "root"
        G.add_node(root)
        for _, row in taxonomy_df.iterrows():
            contig_id, taxonomy_str = row["contigs"], row["predictions"]
            if not taxonomy_str:
                continue
            taxonomy_levels = taxonomy_str.split(
                ";"
            )  # assume in "domain;phylum;class;order;family;genus;species" format, by having --tax-lineage 1 in mmseqs2.
            G.add_edge(root, taxonomy_levels[0])
            for i in range(1, len(taxonomy_levels)):
                parent_taxon = taxonomy_levels[i - 1]
                child_taxon = taxonomy_levels[i]
                G.add_node(child_taxon)
                G.add_edge(parent_taxon, child_taxon)

        edges = nx.bfs_edges(G, source=root)
        logger.info(
            f"Constructed taxonomy graph with {G.number_of_nodes()} nodes \
            and {G.number_of_edges()} edges."
        )

        nodes = [root] + [child for _, child in edges]
        node_indices = {node: idx for idx, node in enumerate(nodes)}

        table_parent: list[int] = [
            -1 if n == root else node_indices[next(iterator(G.predecessors(n)))]
            for n in nodes
        ]
        # Verify ancestor-first ordering: every parent precedes its child
        assert all(p < i for i, p in enumerate(table_parent) if p != -1)

        return nodes, node_indices, table_parent


def iterator(itr):
    for item in itr:
        yield item
