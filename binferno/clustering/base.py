from pathlib import Path
from typing import Dict
from pydantic import BaseModel, Field
import loguru
import pandas as pd
from dataclasses import dataclass

# make this a data object to store clustering results, including cluster assignments and other relevant information.


# Pydantic class replace.
@dataclass
class ClusterResult:
    n_genome: int
    clusters: Dict[
        str, pd.DataFrame
    ]  # a generic dictionary to store all the results for a key and the corresponding dataframes.


class BaseClustering(BaseModel):
    contig_len: int = (
        1000  # minimum contig length to consider for clustering, default 1000 bp.
    )
    dimension_reduction: bool = Field(
        default=False,
        description="Whether to perform dimension reduction before clustering."
        "Used only if training.concat is set to True and the dimension of embeddings > 128.",
    )

    def __call__(
        self,
        contigs: Path,
        output_path: Path,
        embeddings: pd.DataFrame,
        logger: loguru._logger.Logger,
        threads: int = 1,
        plastid: bool = False,
    ) -> ClusterResult:
        raise NotImplementedError("Subclasses must implement __call__ method.")
