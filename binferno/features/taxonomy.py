from pathlib import Path

from pydantic import Field, PrivateAttr
import pandas as pd
import loguru
from typing import List, Dict, Tuple

from .base import BaseFeatureBuilder

TAXONOMY_HEADER = ["contigs", "predictions"]

TAXONOMIC_RANKS = [
    "domain",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "species",
]


class TaxonomyFeatureBuilder(BaseFeatureBuilder):
    """
    Convert hierarchical taxonomic predictions into rank-specific
    categorical IDs.

    Output
    ------
    taxonomy_df:
        index = contig_id

        columns:
            tax_domain
            tax_phylum
            tax_class
            tax_order
            tax_family
            tax_genus
            tax_species

            mask_domain
            mask_phylum
            mask_class
            mask_order
            mask_family
            mask_genus
            mask_species

    vocabularies:
        dict mapping rank -> {taxon_name: integer_id}

        Integer 0 is always reserved for <UNK>.
    """

    taxonomy: Path = Field(
        Path(),
        description="Path to a TSV file containing taxonomic classifications for each contig. Columns: contig_id, taxonomy_string.",
    )
    _ranks: List[str] = PrivateAttr(default_factory=lambda: TAXONOMIC_RANKS.copy())
    _vocabularies: Dict[str, Dict[str, int]] | None = PrivateAttr(default=None)

    def ranks(self) -> List[str]:
        return self._ranks.copy()

    def __call__(
        self, contigs: Path, logger: loguru._logger.Logger
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
        """Build rank-specific features for FASTA contigs represented in the TSV."""
        taxonomy_df = pd.read_csv(self.taxonomy, sep="\t", header=0)
        missing_columns = set(TAXONOMY_HEADER) - set(taxonomy_df.columns)
        if missing_columns:
            raise ValueError(
                f"Taxonomy TSV is missing required columns: {sorted(missing_columns)}"
            )
        if taxonomy_df["contigs"].duplicated().any():
            duplicates = taxonomy_df.loc[
                taxonomy_df["contigs"].duplicated(), "contigs"
            ].tolist()
            raise ValueError(
                f"Taxonomy TSV contains duplicate contig IDs: {duplicates[:5]}"
            )

        with Path(contigs).open() as fasta_handle:
            fasta_ids = {
                line[1:].split()[0] for line in fasta_handle if line.startswith(">")
            }
        in_fasta = taxonomy_df["contigs"].isin(fasta_ids)
        ignored = int((~in_fasta).sum())
        if ignored:
            logger.warning(
                f"Ignoring {ignored} taxonomy rows whose contig IDs are absent "
                "from the input FASTA."
            )
        taxonomy_df = taxonomy_df.loc[in_fasta].copy()
        if taxonomy_df.empty:
            raise ValueError("No taxonomy rows match the input FASTA contig IDs.")
        return self.fit_transform(taxonomy_df)

    @staticmethod
    def _clean_taxon(taxon: str) -> str | None:
        if taxon is None:
            return None

        taxon = str(taxon).strip()

        if taxon == "":
            return None

        if taxon.lower() in {
            "unclassified",
            "unknown",
            "na",
            "nan",
            "none",
            "root",
        }:
            return None

        return taxon

    def _parse_lineage(self, lineage: str) -> Dict[str, str | None]:
        """
        Convert:
            d__Bacteria;p__Proteobacteria;...

        or:
            Bacteria;Proteobacteria;...

        into a rank dictionary.
        """

        result = {rank: None for rank in self._ranks}

        if lineage is None or pd.isna(lineage):
            return result

        parts = [self._clean_taxon(x) for x in str(lineage).split(";")]

        parts = parts[: len(self._ranks)]

        for rank, taxon in zip(self._ranks, parts):
            result[rank] = taxon

        return result

    def fit(self, taxonomy_table: pd.DataFrame):
        """
        Construct rank-specific vocabularies.

        taxonomy_table requires:
            contigs
            predictions
        """

        parsed = [
            self._parse_lineage(lineage) for lineage in taxonomy_table["predictions"]
        ]

        vocabularies = {}

        for rank in self._ranks:
            taxa = sorted(
                {lineage[rank] for lineage in parsed if lineage[rank] is not None}
            )

            # 0 is reserved for missing / unknown
            vocab = {"<UNK>": 0}

            for idx, taxon in enumerate(taxa, start=1):
                vocab[taxon] = idx

            vocabularies[rank] = vocab

        self._vocabularies = vocabularies

        return self

    def transform(
        self,
        taxonomy_table: pd.DataFrame,
    ) -> pd.DataFrame:

        if self._vocabularies is None:
            raise RuntimeError(
                "TaxonomyFeatureBuilder.fit() must be called before transform()."
            )

        records = []

        for _, row in taxonomy_table.iterrows():

            contig_id = row["contigs"]
            lineage = self._parse_lineage(row["predictions"])

            record = {"contig_id": contig_id}

            for rank in self._ranks:

                taxon = lineage[rank]

                if taxon is None:
                    tax_id = 0
                    mask = 0.0

                else:
                    tax_id = self._vocabularies[rank].get(
                        taxon,
                        0,
                    )

                    # Unknown taxon at transform time
                    if tax_id == 0:
                        mask = 0.0
                    else:
                        mask = 1.0

                record[f"tax_{rank}"] = tax_id
                record[f"mask_{rank}"] = mask

            records.append(record)

        taxonomy_df = pd.DataFrame(records)

        taxonomy_df = taxonomy_df.set_index("contig_id")
        ordered_columns = [
            *[f"tax_{rank}" for rank in self._ranks],
            *[f"mask_{rank}" for rank in self._ranks],
        ]
        return taxonomy_df[ordered_columns]

    def fit_transform(
        self,
        taxonomy_table: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:

        self.fit(taxonomy_table)

        taxonomy_df = self.transform(taxonomy_table)

        return taxonomy_df, self._vocabularies
