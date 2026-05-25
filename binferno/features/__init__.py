import pandas as pd
from pydantic import Field
from .base import BaseFeatures
from .corgi import CorgiFeatures  # the only one without required multisample mode.
from .coverage import CoverageFeatures
from .kmer import KmerFeatures


class BinfernoFeatures(BaseFeatures):
    use_corgi: bool = Field(
        default=False,
        description="Whether to use Corgi to generate features for training.",
    )
    corgi: CorgiFeatures
    use_kmer: bool = Field(
        default=True, description="Whether to use k-mer features for training."
    )
    kmer: KmerFeatures
    use_coverage: bool = Field(
        default=True, description="Whether to use coverage features for training."
    )
    coverage: CoverageFeatures
    save_features: bool = Field(
        default=True,
        description="Whether to save the generated features as tsv files in the output directory.",
    )

    def __call__(
        self,
        FragMent_Dict,
        All_Fragments,
        contigs=None,
        bam_path=None,
        threads=4,
        logger=None,
        output_dir=None,
    ) -> pd.DataFrame:

        if not self.use_kmer and not self.use_corgi and not self.use_coverage:
            raise ValueError(
                "At least one feature type must be selected for training. Please set at least one of use_kmer, use_corgi or use_coverage to True."
            )

        resulting_features = pd.DataFrame()
        if self.use_kmer:
            logger.info("Generating k-mer features...")
            kmer_df = self.kmer(All_Fragments)
            kmer_df.to_csv(
                output_dir / "kmer_features.tsv", index=True, header=True, sep="\t"
            )
            resulting_features = pd.concat(
                [resulting_features, kmer_df], axis=1
            )  # the first to add is k-mer features.
        if self.use_corgi:
            logger.info("Generating Corgi features...")
            corgi_df = self.corgi(output_dir, contigs=contigs)
            corgi_df.to_csv(
                output_dir / "corgi_features.tsv", index=True, header=True, sep="\t"
            )
            resulting_features = pd.concat(
                [resulting_features, corgi_df], axis=1
            )  # if applicable, the second is corgi features.
        if self.use_coverage:
            logger.info("Generating coverage features...")
            coverage_df = self.coverage(FragMent_Dict, bam_path, threads, logger)
            if coverage_df.shape[0] != resulting_features.shape[0]:
                raise ValueError(
                    f"Coverage features have different number of rows ({coverage_df.shape[0]}) compared to other features ({resulting_features.shape[0]}). Please check the coverage feature generation."
                )
            resulting_features = pd.concat([resulting_features, coverage_df], axis=1)
            coverage_df.to_csv(
                output_dir / "coverage_features.tsv", index=True, header=True, sep="\t"
            )  # if applicable, the last is coverage features.

        del (
            FragMent_Dict,
            All_Fragments,
        )  # free up memory after generating the features.
        del (
            kmer_df,
            corgi_df,
            coverage_df,
        )  # free up memory after generating the features.

        namelist = list(FragMent_Dict.keys())
        # [kmer, corgi, coverage] features are already sorted by method in the sort_features_by_method function, 
        # so we can directly save the resulting features without further sorting.

        resulting_features.to_csv(
            output_dir / "features.tsv", index=True, header=True, sep="\t"
        )
        return resulting_features

# class BinfernoMultiFeatures(BinfernoFeatures):
#     kmer: KmerMultiFeatures
#     coverage: CoverageMultiFeatures
#     corgi: CorgiFeatures

#     def __call__():
#         return