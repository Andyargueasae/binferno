from pathlib import Path

import loguru
from pydantic import Field
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
from .base import BaseReducer


class PcaReducer(BaseReducer):
    class Config:
        arbitrary_types_allowed = True

    def __call__(self, data, n_components=50):

        PCA_dim_reducer = KernelPCA(
            n_components=n_components, kernel="linear", random_state=42
        )
        if data.shape[0] > 50000:
            PCA_dim_reducer = PCA(
                n_components=n_components, random_state=42, svd_solver="randomized"
            )

        reduced_data = PCA_dim_reducer.fit_transform(data)
        return reduced_data
