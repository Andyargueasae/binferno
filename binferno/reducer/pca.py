from pathlib import Path

import loguru
from pydantic import Field
from sklearn.decomposition import PCA

class PcaReducer:
    class Config:
        arbitrary_types_allowed = True

    def __call__(self, data, n_components=50):

        pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(data)
        return reduced_data