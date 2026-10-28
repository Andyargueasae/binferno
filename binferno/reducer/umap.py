from pathlib import Path

import loguru
import pandas as pd
from pydantic import Field

from .base import BaseReducer


class UmapReducer(BaseReducer):
    class Config:
        arbitrary_types_allowed = True

    n_neighbors: int = Field(
        default=30, description="The number of neighbors parameter for UMAP."
    )
    min_dist: float = Field(
        default=0.1, description="The minimum distance parameter for UMAP."
    )

    def __call__(
        self,
        data: pd.DataFrame,
        n_components: int = 15,
    ):
        try:
            import umap
        except ModuleNotFoundError as exc:
            if exc.name != "umap":
                raise
            raise ImportError(
                "UMAP reduction requires umap-learn. Install umap-learn before "
                "enabling use_umap or use_pipeline."
            ) from exc

        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric="cosine",
            random_state=42,
            n_jobs=-1,
        )
        reduced_data = reducer.fit_transform(data)
        return reduced_data

