from pathlib import Path

import loguru
from pydantic import Field
from sklearn.manifold import TSNE

class TsneReducer:
    class Config:
        arbitrary_types_allowed = True

    def __call__(self, data, n_components=15, perplexity=30.0, learning_rate=200.0, n_iter=1000):

        tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter)
        reduced_data = tsne.fit_transform(data)
        return reduced_data