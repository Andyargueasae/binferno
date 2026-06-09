from pathlib import Path
from pydantic import Field
import pandas as pd
from .base import BaseClustering
from .leiden import LeidenClustering
from .dbscan import DbscanClustering
import loguru


class BinfernoClustering(BaseClustering):
    use_leiden: bool = Field(
        default=True,
        description="Whether to use Leiden clustering. If False, KMeans will be used.",
    )
    leiden: LeidenClustering

    use_DBSCAN: bool = Field(
        default=False,
        description="Whether to use DBSCAN clustering. If False, KMeans will be used.",
    )
    dbscan: DbscanClustering

    dimension_reduction: bool = Field(
        default=False,
        description="Whether to perform dimension reduction before clustering."
        "Used only if training.concat is set to True and the dimension of embeddings > 128.",
    )
    plastid: bool = Field(
        default=False,
        description="Whether to use plastid-specific seed generation and minimum bin size.",
    )

    class Config:
        arbitrary_types_allowed = True

    def __call__(
        self,
        embeddings: pd.DataFrame,
        contigs: Path,
        output_path: Path = None,
        contig_len: int = 1000,
        logger: loguru.logger = None,  # type: ignore
        threads: int = 1,
    ) -> None:
        if self.use_leiden:
            n_genome = self.leiden(
                contigs=contigs,
                output_path=output_path,
                embeddings=embeddings,
                logger=logger,
                threads=threads,
                contig_len=contig_len,
                plastid=self.plastid,
                dimension_reduction=self.dimension_reduction,
            )
        elif self.use_DBSCAN:
            n_genome = self.dbscan(
                contigs=contigs,
                output_path=output_path,
                embeddings=embeddings,
                logger=logger,
                threads=threads,
                contig_len=contig_len,
                plastid=self.plastid,
                dimension_reduction=self.dimension_reduction,
            )
        return n_genome

