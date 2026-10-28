from pathlib import Path
from pydantic import Field
import pandas as pd

from .base import BaseClustering, ClusterResult, ClusterMethod
from .leiden import LeidenClustering
from .dbscan import DbscanClustering
import loguru


class BinfernoClustering(BaseClustering):
    cluster_method: ClusterMethod = Field(
        default=ClusterMethod.leiden,
        description="The clustering method to use. DBSCAN is DEPRECATED and will be removed in future versions. Use Leiden instead.",
    )
    leiden: LeidenClustering
    dbscan: DbscanClustering
    plastid: bool = Field(
        default=False,
        description="Whether to use plastid-specific seed generation and minimum bin size.",
    )
    longread: bool = Field(
        False,
        description="Whether the input data is from long-read sequencing. This will impact the downstream binning choices of parameters and thresholds.",
    )

    class Config:
        arbitrary_types_allowed = True

    def __call__(
        self,
        embeddings: pd.DataFrame,
        contigs: Path,
        output_path: Path = None,
        logger: loguru._logger.Logger = None,  # type: ignore
        threads: int = 1,
        taxonomy: pd.DataFrame | None = None,
    ) -> ClusterResult:

        match self.cluster_method:
            case ClusterMethod.leiden:
                result = self.leiden(
                    contigs=contigs,
                    output_path=output_path,
                    embeddings=embeddings,
                    logger=logger,
                    threads=threads,
                    plastid=self.plastid,
                    longread=self.longread,
                    taxonomy=taxonomy,
                )
            case ClusterMethod.dbscan:
                result = self.dbscan(
                    contigs=contigs,
                    output_path=output_path,
                    embeddings=embeddings,
                    logger=logger,
                    threads=threads,
                    plastid=self.plastid,
                    longread=self.longread,
                )
        return result
