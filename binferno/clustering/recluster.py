# SCG-guided reclustering/refining of current clusters by the SCG algorithm:
# 1. SCG genes are expected to occur once in the same cluster, so if they are found in multiple clusters, they are used to merge those clusters.
# 2. Two clusters are merged if their SCGs after merged can make a complementary set of SCGs (i.e. the merged cluster has all SCGs in the SCG list).
# 3. If a cluster has no SCGs, not possible here. 
from pathlib import Path
import os
from unittest.mock import Base

from loguru import logger
import numpy as np
from pydantic import BaseModel, Field

from .base import BaseClustering, ClusterResult

# TODO: implement a current cluster check by reclustering based on the list above.
class Recluster(BaseClustering):
    def __call__(self):
        pass

