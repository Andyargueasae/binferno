from enum import Enum
from pathlib import Path
from pydantic import Field
import pandas as pd

from .base import BaseClustering, ClusterResult
from .leiden import LeidenClustering
from .dbscan import DbscanClustering
import loguru


class ClusterMethod(Enum):
    leiden = "leiden"
    dbscan = "dbscan"


class BinfernoClustering(BaseClustering):
    cluster_method: ClusterMethod = Field(
        default=ClusterMethod.leiden,
        description="The clustering method to use.",
    )
    leiden: LeidenClustering
    dbscan: DbscanClustering

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
        logger: loguru._logger.Logger = None,  # type: ignore
        threads: int = 1,
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
                )
            case ClusterMethod.dbscan:
                result = self.dbscan(
                    contigs=contigs,
                    output_path=output_path,
                    embeddings=embeddings,
                    logger=logger,
                    threads=threads,
                    plastid=self.plastid,
                )
        return result
