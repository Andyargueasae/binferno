from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle

import pandas as pd
from pydantic import Field
import loguru
from sklearn.preprocessing import normalize

from .base import BaseFeatureBuilder
from .corgi import CorgiFeatureBuilder
from .coverage import CoverageFeatureBuilder
from .kmer import KmerFeatureBuilder, generate_feature_mapping
from .dnabert_s import DnaBertSFeatureBuilder
from .taxonomy import TaxonomyFeatureBuilder


class BinfernoFeatureBuilder(BaseFeatureBuilder):

    use_corgi: bool = Field(
        default=False,
        description="Whether to use Corgi to generate features for training.",
    )
    corgi: CorgiFeatureBuilder
    use_kmer: bool = Field(
        default=True, description="Whether to use k-mer features for training."
    )
    kmer: KmerFeatureBuilder
    use_coverage: bool = Field(
        default=True, description="Whether to use coverage features for training."
    )
    coverage: CoverageFeatureBuilder
    use_dnabert_s: bool = Field(
        default=False, description="Whether to use DNABERT-S features for training."
    )
    dnabert_s: DnaBertSFeatureBuilder
    use_taxonomy: bool = Field(
        default=False, description="Whether to use taxonomy features for training."
    )
    taxonomy: TaxonomyFeatureBuilder  # Placeholder for future taxonomy feature builder
    corgi_l2_normalize: bool = Field(
        default=True,
        description="Whether to L2 normalize Corgi features before concatenation.",
    )
    save_features: bool = Field(
        default=True,
        description="Whether to save the generated features as tsv files in the output directory.",
    )

    def _get_kmer_dim(self) -> int:
        if not self.use_kmer:
            return 0
        _, kmer_dim = generate_feature_mapping(self.kmer.kmer_len)
        return kmer_dim

    def __call__(
        self,
        FragMent_Dict: dict,
        All_Fragments: list,
        contigs: Path = None,
        bam_path: Path = None,
        threads: int = 4,
        logger: loguru.logger = None,  # type: ignore
        output_dir: Path = None,
        contig_length: int = 1000,  # used to calibrate corgi when run raw data.
    ) -> Tuple[List[pd.DataFrame | Optional[pd.DataFrame | Dict | None]], int]:

        if (
            (not self.use_kmer)
            and (not self.use_corgi)
            and (not self.use_coverage)
            and (not self.use_dnabert_s)
            and (not self.use_taxonomy)
        ):
            raise ValueError(
                "At least one feature type must be selected for training. \
                Please set at least one of use_kmer, use_corgi, use_coverage, \
                use_dnabert_s, or use_taxonomy to True."
            )
        feature_path = output_dir / "features.tsv"

        if feature_path.exists():
            logger.info("Loading existing features...")
            resulting_features = pd.read_csv(
                feature_path, sep="\t", index_col=0, header=0
            )
            # Always return None for second output (unified matrix approach)
            return [resulting_features, None], self._get_kmer_dim()

        canonical_features = pd.DataFrame()

        # STEP 1: Load COVERAGE features FIRST
        coverage_df = None
        if self.use_coverage:
            logger.info("Generating coverage features...")
            coverage_data = output_dir / "coverage_features.tsv"
            if coverage_data.exists():
                logger.info("Loading existing coverage features...")
                coverage_df = pd.read_csv(
                    coverage_data, sep="\t", index_col=0, header=0
                )
            else:
                coverage_df = self.coverage.__call__(
                    FragMent_Dict,
                    bam_path,
                    threads,
                    logger,
                )
                (
                    coverage_df.to_csv(coverage_data, sep="\t", index=True, header=True)
                    if self.save_features
                    else None
                )
            canonical_features = pd.concat(
                [canonical_features, coverage_df], axis=1
            )  # Start with: [coverage]

        # STEP 2: Load CORGI/TAXONOMY features and INTEGRATE into main matrix
        corgi_df = None
        dnabert_s_df = None

        if self.use_taxonomy:
            logger.info("Generating taxonomy features...")
            tax_nodes, tax_node_indices, parental_array, contig_taxonomy_df = (
                self.taxonomy.__call__(
                    contigs=contigs,
                    logger=logger,
                )
            )
            contig_taxonomy_df = contig_taxonomy_df.set_index("contig_id")
            # Integrate taxonomy features into main matrix
            canonical_features = pd.concat(
                [canonical_features, contig_taxonomy_df], axis=1
            )  # Now: [coverage, taxonomy]
            logger.info(
                f"Taxonomy of the data contains {len(tax_node_indices)} unique taxa."
            )
        elif self.use_corgi:
            logger.info("Generating Corgi features...")
            corgi_data = output_dir / "corgi_features.tsv"
            if corgi_data.exists():
                logger.info("Loading existing Corgi features...")
                corgi_df = pd.read_csv(corgi_data, sep="\t", index_col=0, header=0)
            else:
                corgi_df = self.corgi(
                    output_dir, contigs=contigs, min_length=contig_length
                )
                (
                    corgi_df.to_csv(corgi_data, sep="\t", index=True, header=True)
                    if self.save_features
                    else None
                )

            # Apply L2 normalization BEFORE concatenation (matching old behavior)
            if self.corgi_l2_normalize:
                corgi_df = pd.DataFrame(
                    normalize(corgi_df), index=corgi_df.index, columns=corgi_df.columns
                )

            canonical_features = pd.concat(
                [canonical_features, corgi_df], axis=1
            )  # Now: [coverage, corgi]
        elif self.use_dnabert_s:
            logger.info("Generating DNABERT-S features...")
            dnabert_s_data = output_dir / "dnabert_s_features.tsv"
            if dnabert_s_data.exists():
                logger.info("Loading existing DNABERT-S features...")
                dnabert_s_df = pd.read_csv(
                    dnabert_s_data, sep="\t", index_col=0, header=0
                )
            else:
                dnabert_s_df = self.dnabert_s(
                    fragments=All_Fragments,
                    logger=logger,
                )
                (
                    dnabert_s_df.to_csv(
                        dnabert_s_data, sep="\t", index=True, header=True
                    )
                    if self.save_features
                    else None
                )

            dnabert_s_df = pd.DataFrame(
                normalize(dnabert_s_df),
                index=dnabert_s_df.index,
                columns=dnabert_s_df.columns,
            )
            canonical_features = pd.concat(
                [canonical_features, dnabert_s_df], axis=1
            )  # Now: [coverage, dnabert_s]

        # STEP 3: Load KMER features LAST
        kmer_df = None
        if self.use_kmer:
            kmer_data = output_dir / "kmer_features.tsv"
            if kmer_data.exists():
                logger.info("Loading existing k-mer features...")
                kmer_df = pd.read_csv(kmer_data, sep="\t", index_col=0, header=0)
            else:
                logger.info("Generating k-mer features...")
                kmer_df = self.kmer(All_Fragments)
                (
                    kmer_df.to_csv(kmer_data, sep="\t", index=True, header=True)
                    if self.save_features
                    else None
                )

            kmer_df = pd.DataFrame(
                normalize(kmer_df), index=kmer_df.index, columns=kmer_df.columns
            )
            canonical_features = pd.concat(
                [canonical_features, kmer_df], axis=1
            )  # Final: [coverage, corgi/dnabert_s/taxonomy, kmer]

        # STEP 4: Save and return UNIFIED feature matrix
        canonical_features.to_csv(
            output_dir / "features.tsv", index=True, header=True, sep="\t"
        )

        kmer_dim = self._get_kmer_dim()
        coverage_dim = coverage_df.shape[1] if coverage_df is not None else 0
        corgi_dim = corgi_df.shape[1] if corgi_df is not None else 0
        dnabert_s_dim = dnabert_s_df.shape[1] if dnabert_s_df is not None else 0

        logger.info(
            f"Generated unified features with dimensions - \
            Coverage: {coverage_dim}, Corgi: {corgi_dim}, DNABERT-S: {dnabert_s_dim}, Kmer: {kmer_dim}. \
            Total dimensions: {canonical_features.shape[1]}"
        )

        # Return UNIFIED feature matrix and None (no separate taxonomy features)
        # This matches old binferno behavior
        return [canonical_features, None], kmer_dim
