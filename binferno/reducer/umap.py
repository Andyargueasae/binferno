from pathlib import Path

import loguru
import umap
from pydantic import Field

class UmapReducer:
    class Config:
        arbitrary_types_allowed = True

    def __call__(self, 
        data: pd.DataFrame, 
        n_components: int = 15, 
        n_neighbors: int = 30, 
        min_dist: float = 0.1
    ):

        reducer = umap.UMAP(
            n_components=n_components, 
            n_neighbors=n_neighbors, 
            min_dist=min_dist,
            metric="cosine",
            random_state=42,
            n_jobs=-1,
        )
        reduced_data = reducer.fit_transform(data)
        return reduced_data