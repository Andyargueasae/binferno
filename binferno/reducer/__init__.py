from pathlib import Path

import loguru
from pydantic import Field
import numpy as np
import pandas as pd

from .base import BaseReducer
from .pca import PcaReducer
from .umap import UmapReducer


class BinfernoReducer(BaseReducer):
    class Config:
        arbitrary_types_allowed = True

    pca: PcaReducer
    umap: UmapReducer
    use_pca: bool = Field(
        False, description="Whether to apply PCA for dimension reduction."
    )
    use_umap: bool = Field(
        False, description="Whether to apply UMAP for dimension reduction."
    )
    use_pipeline: bool = Field(
        False,
        description="Whether to apply a pipeline of PCA and UMAP for dimension reduction.",
    )
    default_dim: int = Field(
        32, description="Default dimension for dimension reduction."
    )

    def __call__(
        self, data: pd.DataFrame, logger: loguru._logger.Logger = None, threads: int = 1
    ):
        x_ind = data.index
        x_columns = data.columns
        data = self.l2_normalize(data).astype(
            np.float32, copy=False
        )  # normalize the data before dimension reduction. Now it is already a numpy.ndarray.
        if self.use_pipeline:
            logger.info(
                "Applying PCA followed by UMAP for dimension reduction: a pipeline used in Seurat (https://satijalab.org/seurat/)."
            )
            data = self.pca(
                data, n_components=50
            )  # first reduce to 50 dimensions using PCA
            data = self.umap(data, n_components=self.default_dim)
            df = pd.DataFrame(
                data,
                columns=[f"i{idx}" for idx in range(data.shape[1])],
                index=x_ind,
                dtype=np.float32,
            )
        elif self.use_pca:
            logger.info("Directly applying PCA for dimension reduction.")
            data = self.pca(data, n_components=self.default_dim)
            df = pd.DataFrame(
                data,
                columns=[f"i{idx}" for idx in range(data.shape[1])],
                index=x_ind,
                dtype=np.float32,
            )
        elif self.use_umap:
            logger.info("Directly applying UMAP for dimension reduction.")
            data = self.umap(data, n_components=self.default_dim)
            df = pd.DataFrame(
                data,
                columns=[f"i{idx}" for idx in range(data.shape[1])],
                index=x_ind,
                dtype=np.float32,
            )
        else:
            logger.info(
                "No dimension reduction method selected. Returning original data."
            )
            df = pd.DataFrame(data, index=x_ind, columns=x_columns)

        logger.info(f"Dimension reduction completed. Final dimension: {data.shape[1]}")

        return df

