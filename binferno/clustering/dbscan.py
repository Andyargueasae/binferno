from pathlib import Path

from pydantic import Field
import pandas as pd 

from .base import BaseClustering


class DbscanClustering(BaseClustering):
    def __call__(
        self,
        contigs: Path,
        output_path: Path,
        embeddings: pd.DataFrame,
        logger: loguru._logger.Logger,
        threads: int=1,
        contig_len: int = 1000,
    ) -> None:

        raise NotImplementedError("DBSCAN clustering is not yet implemented.")