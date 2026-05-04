from .corgi import CorgiFeatures
from .coverage import CoverageFeatures
from .kmer import KmerFeatures
from ..multi import MultiModule



def build_feature_modules(kmer:bool, corgi:bool, coverage:bool):
    features = []
    if kmer:
        features.append(KmerFeatures())
    if corgi:
        features.append(CorgiFeatures())
    if coverage:
        features.append(CoverageFeatures())

    return MultiModule(*features)