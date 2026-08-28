from pathlib import Path

import loguru
from pydantic import Field
from sklearn.manifold import TSNE
from .base import BaseReducer


class TsneReducer(BaseReducer):
    class Config:
        arbitrary_types_allowed = True

    perplexity: float = Field(
        default=30.0, description="The perplexity parameter for t-SNE."
    )
    learning_rate: float = Field(
        default=200.0, description="The learning rate parameter for t-SNE."
    )
    n_iter: int = Field(default=1000, description="The number of iterations for t-SNE.")

    def __call__(self, data, n_components=15):
        tsne = TSNE(
            n_components=n_components,
            perplexity=self.perplexity,
            learning_rate=self.learning_rate,
            n_iter=self.n_iter,
        )
        reduced_data = tsne.fit_transform(data)
        return reduced_data
