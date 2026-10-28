import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle

import pandas as pd
import numpy as np
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
    corgi_reduction: bool = Field(
        default=False,
        description="Whether to apply dimensionality reduction to Corgi features before concatenation, mutually exclusive with corgi_l2_normalize.",
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
        taxonomy_path = output_dir / "taxonomy_features.tsv"
        fragment_ids = [header for header, _ in All_Fragments]

        if feature_path.exists():
            logger.info("Loading existing features...")
            resulting_features = pd.read_csv(
                feature_path, sep="\t", index_col=0, header=0
            )
            cached_ids = resulting_features.index.tolist()
            if (
                resulting_features.index.is_unique
                and len(cached_ids) == len(fragment_ids)
                and set(cached_ids) == set(fragment_ids)
            ):
                taxonomy_features = None
                if self.use_taxonomy:
                    if taxonomy_path.exists():
                        taxonomy_features = pd.read_csv(
                            taxonomy_path, sep="\t", index_col=0, header=0
                        )
                        if (
                            taxonomy_features.index.is_unique
                            and taxonomy_features.index.tolist() == cached_ids
                        ):
                            return [
                                resulting_features,
                                taxonomy_features,
                            ], self._get_kmer_dim()
                    logger.warning(
                        "Ignoring cached canonical features because its taxonomy "
                        "cache is missing or has different fragment IDs."
                    )
                else:
                    return [resulting_features, None], self._get_kmer_dim()
            logger.warning(
                "Ignoring cached features because their contig/view IDs do not "
                "match the current augmented fragments."
            )

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
        contig_taxonomy_df = None

        if self.use_taxonomy:
            logger.info("Generating taxonomy features...")
            contig_taxonomy_df, taxonomy_vocabularies = self.taxonomy(
                contigs=contigs,
                logger=logger,
            )
            if not contig_taxonomy_df.index.is_unique:
                raise ValueError("Taxonomy TSV contains duplicate contig IDs.")
            if fragment_ids:
                # replace everything after a full stop to void.
                base_contig_ids = [
                    re.sub(r"\.(?:h[12]\.)?(?:original|\d+)$", "", fragment_id)
                    for fragment_id in fragment_ids
                ]
                missing_taxonomy = sorted(
                    set(base_contig_ids) - set(contig_taxonomy_df.index)
                )
                if missing_taxonomy:
                    preview = ", ".join(missing_taxonomy[:5])
                    raise ValueError(
                        f"Taxonomy is missing {len(missing_taxonomy)} contig IDs "
                        f"required by augmented fragments (examples: {preview})."
                    )
                contig_taxonomy_df = contig_taxonomy_df.reindex(base_contig_ids).copy()
                contig_taxonomy_df.index = pd.Index(
                    fragment_ids, name=contig_taxonomy_df.index.name
                )
            # Taxon IDs are categorical; keep them out of the continuous
            # coverage/composition matrix and encode them in a dedicated model.
            if not np.isfinite(contig_taxonomy_df.to_numpy(dtype=float)).all():
                raise ValueError("Taxonomy features contain NaN or infinite values.")
            if self.save_features:
                contig_taxonomy_df.to_csv(
                    taxonomy_path, index=True, header=True, sep="\t"
                )
            unique_taxa = sum(
                len(vocabulary) - 1 for vocabulary in taxonomy_vocabularies.values()
            )
            logger.info(
                f"Taxonomy of the data contains {unique_taxa} rank-specific taxa."
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
            if self.corgi_reduction:
                corgi_df = pd.DataFrame(
                    self.dim_reduce(corgi_df.values, n_components=50),
                    index=corgi_df.index,
                )
                corgi_df.columns = [str(col) for col in corgi_df.columns]

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

            # so here there might be an information loss if we normalize kmer again,
            # because kmer itself is already normalized inside the kmer feature computation.
            # kmer_df = pd.DataFrame(
            #     normalize(kmer_df), index=kmer_df.index, columns=kmer_df.columns
            # )

            canonical_features = pd.concat(
                [canonical_features, kmer_df], axis=1
            )  # Final: [coverage, corgi/dnabert_s/taxonomy, kmer]

        # STEP 4: Enforce exact alignment before saving. Pandas concatenation
        # otherwise turns missing feature rows into silent NaNs.
        if fragment_ids:
            expected_index = pd.Index(fragment_ids)
            if (
                not canonical_features.index.is_unique
                or not canonical_features.index.equals(expected_index)
            ):
                missing = expected_index.difference(canonical_features.index).tolist()
                extra = canonical_features.index.difference(expected_index).tolist()
                raise ValueError(
                    "Feature IDs do not exactly match augmented fragment IDs "
                    f"(missing={missing[:5]}, extra={extra[:5]})."
                )
        if not np.isfinite(canonical_features.to_numpy(dtype=float)).all():
            raise ValueError("Canonical features contain NaN or infinite values.")

        canonical_features.to_csv(
            output_dir / "features.tsv", index=True, header=True, sep="\t"
        )

        kmer_dim = self._get_kmer_dim()
        coverage_dim = coverage_df.shape[1] if coverage_df is not None else 0
        corgi_dim = corgi_df.shape[1] if corgi_df is not None else 0
        dnabert_s_dim = dnabert_s_df.shape[1] if dnabert_s_df is not None else 0
        taxonomy_dim = (
            contig_taxonomy_df.shape[1] if contig_taxonomy_df is not None else 0
        )

        logger.info(
            f"Generated unified features with dimensions - \
            Coverage: {coverage_dim}, Corgi: {corgi_dim}, DNABERT-S: {dnabert_s_dim}, \
            Taxonomy (separate): {taxonomy_dim}, Kmer: {kmer_dim}. \
            Canonical dimensions: {canonical_features.shape[1]}"
        )

        return [canonical_features, contig_taxonomy_df], kmer_dim

    def dim_reduce(self, embeddings: np.ndarray, n_components: int = 50) -> np.ndarray:
        """
        Reduce the dimensionality of the embeddings using PCA.
        """
        from sklearn.decomposition import PCA

        pca = PCA(n_components=n_components)
        reduced_embeddings = pca.fit_transform(embeddings)
        return reduced_embeddings
