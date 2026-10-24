"""Canonical tensors with separate integer protein lookup metadata.

The named tuple is collated recursively by PyTorch, including worker processes.
Row/column slicing preserves lookup keys; columns always refer to features only.
"""
from collections import namedtuple
import numpy as np
import torch


class ProteinBatch(namedtuple("ProteinBatchBase", "features contig_ids view_ids")):
    __slots__ = ()

    def __len__(self):
        return len(self.features)

    @property
    def shape(self):
        return self.features.shape

    def size(self, dim=None):
        return self.features.size() if dim is None else self.features.size(dim)

    def __getitem__(self, key):
        rows = key[0] if isinstance(key, tuple) else key
        return ProteinBatch(self.features[key], self.contig_ids[rows], self.view_ids[rows])

    def to(self, device, **kwargs):
        return ProteinBatch(*(value.to(device, **kwargs) for value in self))

    def pin_memory(self):
        return ProteinBatch(*(value.pin_memory() for value in self))


def feature_tensor(data):
    """Convert a feature dataframe, aligning metadata by exact row header."""
    features = torch.from_numpy(np.asarray(data.values, dtype=np.float32))
    metadata = data.attrs.get("esmc_lookup")
    if metadata is None:
        return features
    ids = torch.tensor([metadata[header] for header in data.index], dtype=torch.long)
    return ProteinBatch(features, ids[:, 0], ids[:, 1])


def cat_features(batches, dim=0):
    if isinstance(batches[0], ProteinBatch):
        if dim != 0:
            raise ValueError("Protein batches can only be concatenated by row")
        return ProteinBatch(*(torch.cat(values, dim=0) for values in zip(*batches)))
    return torch.cat(batches, dim=dim)


def unpack_features(x, contig_ids=None, view_ids=None):
    if isinstance(x, ProteinBatch):
        return x.features, x.contig_ids, x.view_ids
    return x, contig_ids, view_ids
