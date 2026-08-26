from enum import Enum
from pathlib import Path
from pydantic import Field
import pandas as pd

from .base import BaseClustering, ClusterResult
from .leiden import LeidenClustering
from .dbscan import DbscanClustering
from .cluster_utils import apply_dimension_reduction
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
    dimension_reduction: bool = Field(False, description="Whether to apply dimension reduction to embeddings before clustering.")
    pca: bool = Field(False, description="Whether to apply PCA for dimension reduction.")
    umap: bool = Field(False, description="Whether to apply UMAP for dimension reduction.")
    default_dim: int = Field(50, description="Default dimension for dimension reduction.")
    longread: bool = Field(False, description="Whether the input data is from long-read sequencing. This will impact the downstream binning choices of parameters and thresholds.")

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

        if self.dimension_reduction:
            embeddings = apply_dimension_reduction(
                embeddings,
                pca=self.pca,
                umap=self.umap,
                res_dim=self.default_dim,
            )
        
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
