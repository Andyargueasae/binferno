from pathlib import Path

import loguru
from pydantic import Field

from .base import BaseReducer
from .pca import PcaReducer
from .umap import UmapReducer

class BinfernoReducer(BaseReducer):
    class Config:
        arbitrary_types_allowed = True

    pca: PcaReducer
    umap: UmapReducer
    use_pca: bool = Field(False, description="Whether to apply PCA for dimension reduction.")
    use_umap: bool = Field(False, description="Whether to apply UMAP for dimension reduction.")
    use_pipeline: bool = Field(False, description="Whether to apply a pipeline of PCA and UMAP for dimension reduction.")
    default_dim: int = Field(32, description="Default dimension for dimension reduction.")

    def __call__(self, data, logger: loguru._logger.Logger = None, threads: int = 1):
        data = self.l2_normalize(data).astype(np.float32, copy=False)  # normalize the data before dimension reduction.
        if self.use_pipeline:
            logger.info("Applying PCA followed by UMAP for dimension reduction: a pipeline used in Seurat (https://satijalab.org/seurat/).")
            data = self.pca(data, n_components=self.default_dim)
            data = self.umap(data, n_components=self.default_dim)
        elif self.use_pca:
            logger.info("Directly applying PCA for dimension reduction.")
            data = self.pca(data, n_components=self.default_dim)
        elif self.use_umap:
            logger.info("Directly applying UMAP for dimension reduction.")
            data = self.umap(data, n_components=self.default_dim)
        else:
            logger.info("No dimension reduction method selected. Returning original data.")

        logger.info(f"Dimension reduction completed. Final dimension: {data.shape[1]}")
        return data
