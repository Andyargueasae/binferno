from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle

import pandas as pd
from pydantic import Field
import loguru

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
        max_norm_stacked: bool = False,  # whether to normalize the coverage features before feeding into the model. we found that normalizing the coverage features before feeding into the model can lead to better performance, so we set std_first to True for simclr training.
    ) -> Tuple[List[pd.DataFrame|Optional[pd.DataFrame|Dict|None]], int]:

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
        taxonomy_path = output_dir / "taxonomy_features.pkl"

        if feature_path.exists():
            logger.info("Loading existing features...")
            resulting_features = pd.read_csv(
                feature_path, sep="\t", index_col=0, header=0
            )
            if taxonomy_path.exists():
                logger.info("Loading existing taxonomy features...")
                with open(taxonomy_path, "rb") as f:
                    taxonomy_features = pickle.load(f)
            else:
                taxonomy_features = None
            return [resulting_features, taxonomy_features], self._get_kmer_dim()

        canonical_features = pd.DataFrame()
        taxonomy_features: Optional[pd.DataFrame|Dict|None] = None
        
        if self.use_kmer:
            kmer_data = output_dir / "kmer_features.tsv"
            if kmer_data.exists():
                logger.info("Loading existing k-mer features...")
                kmer_df = pd.read_csv(kmer_data, sep="\t", index_col=0, header=0)
            else:
                logger.info("Generating k-mer features...")
                kmer_df = self.kmer(All_Fragments)

            canonical_features = pd.concat([canonical_features, kmer_df], axis=1)

        # the taxonomy-related feature
        if self.use_taxonomy:
            logger.info("Generating taxonomy features...")
            tax_nodes, tax_node_indices, parental_array, contig_taxonomy_df = (
                self.taxonomy(
                    contigs=contigs,
                    taxonomy=contigs,  # assuming the taxonomy information is provided in the contigs file for now, can be changed to a separate file in the future.
                    output_dir=output_dir,
                    logger=logger,
                )
            )
            contig_taxonomy_df = contig_taxonomy_df.set_index("contig_id")
            taxonomy_hloss_features = {
                "tax_node": tax_nodes,
                "tax_node_index": tax_node_indices,
                "parental_array": parental_array,
                "contig_taxonomy_df": contig_taxonomy_df,
            }
            taxonomy_features = taxonomy_hloss_features # Dict type.

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
            taxonomy_features = corgi_df # pd.DataFrame type.

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
                )
            taxonomy_features = dnabert_s_df # pd.DataFrame type.
            
        # Finally, coverages are loaded.
        if self.use_coverage:
            logger.info("Generating coverage features...")
            coverage_data = output_dir / "coverage_features.tsv"
            if coverage_data.exists():
                logger.info("Loading existing coverage features...")
                coverage_df = pd.read_csv(
                    coverage_data, sep="\t", index_col=0, header=0
                )
            else:
                coverage_df = self.coverage(
                    FragMent_Dict,
                    bam_path,
                    threads,
                    logger,
                    max_norm_stacked=max_norm_stacked,
                )
                if coverage_df.shape[0] != canonical_features.shape[0]:
                    raise ValueError(
                        f"Coverage features have different number of rows ({coverage_df.shape[0]}) compared \
                        to other features ({canonical_features.shape[0]}). Please check the coverage feature \
                        generation."
                    )
            canonical_features = pd.concat([canonical_features, coverage_df], axis=1)

        if taxonomy_features:
            with open(output_dir / "taxonomy_features.pkl", "wb") as f:
                pickle.dump(taxonomy_features, f)
        

        kmer_dim = self._get_kmer_dim()
        corgi_dim = corgi_df.shape[1] if self.use_corgi else 0
        taxonomy_dim = taxonomy_features.shape[1] if taxonomy_features is not None and isinstance(taxonomy_features, pd.DataFrame) else 0
        coverage_dim = coverage_df.shape[1] if self.use_coverage else 0
        dnabert_s_dim = dnabert_s_df.shape[1] if self.use_dnabert_s else 0

        logger.info(
            f"Generated features with dimensions - \
            Canonical features: k-mer: {kmer_dim}, Coverage: {coverage_dim},  \
            The third modality: Corgi: {corgi_dim} | Taxonomy: {taxonomy_dim} | DNABERT-S: {dnabert_s_dim}."
        )

        del (
            FragMent_Dict,
            All_Fragments,
        )
        del (
            kmer_df,
            coverage_df,
        )

        if self.use_corgi:
            del corgi_df
        if self.use_dnabert_s:
            del dnabert_s_df

        # Final feature looks like:
        # [kmer, coverage]
        # If taxonomy features are used: a separate data structure will store them. 
        
        canonical_features.to_csv(
            output_dir / "features.tsv", index=True, header=True, sep="\t"
        )

        return [canonical_features, taxonomy_features], kmer_dim
