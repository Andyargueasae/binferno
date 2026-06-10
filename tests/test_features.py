import pytest

import numpy as np
import pandas as pd

from binferno.features.kmer import KmerFeatureBuilder
from binferno.features.coverage import CoverageFeatureBuilder
from binferno.features.corgi import CorgiFeatureBuilder

'''
Mainly tests for feature building, including kmer composition features, coverage mean and std, and Corgi features.
Will assess whether values are of correct shape, non-negative, and non-zero where applicable.
'''

def test_kmer_features(toy_embeddings):
    builder = KmerFeatureBuilder(k=4, embedding_dim=64)
    features = builder.build(toy_embeddings.index)
    assert features.shape == (12, 64)
    assert np.all(np.linalg.norm(features, axis=1) > 0)

def test_coverage_features(toy_embeddings):
    builder = CoverageFeatureBuilder()
    features = builder.build(toy_embeddings.index)
    assert features.shape == (12, 1)
    assert np.all(features >= 0)

def test_corgi_features(toy_embeddings):
    builder = CorgiFeatureBuilder()
    features = builder.build(toy_embeddings.index)
    assert features.shape == (12, 128)  # assuming Corgi produces 128-dim features
    assert np.all(np.linalg.norm(features, axis=1) > 0)

