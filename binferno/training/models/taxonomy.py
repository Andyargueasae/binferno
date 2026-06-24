import itertools
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np

class TaxonomyHead(nn.Module):
    def __init__(self, in_dim, table_parent: list, tax_nodes: list = None,
                 node_indices: dict = None, hidden=512, dropout=0.2):
        super().__init__()
        self.tree = Hierarchy(np.asarray(table_parent))
        self.n_leaf = int(self.tree.leaf_mask().sum())
        self.n_nodes = self.tree.num_nodes()
        self.tax_nodes = tax_nodes            # optional: index -> name, for readout
        self.node_indices = node_indices      # optional: name  -> index
 
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.n_leaf),   # logits over leaves
        )
        self.loss_fn = FlatSoftmaxNLL(self.tree)
        self.sum_leaf = SumLeafDescendants(self.tree, strict=False)
 
    def forward(self, fused):
        return self.net(fused)                # [B, n_leaf] logits
 
    def loss(self, logits, target_idx, mask):
        # mask: bool [B], True for contigs that carry a taxonomy label
        if mask.sum() == 0:
            return logits.new_zeros(())
        return self.loss_fn(logits[mask], target_idx[mask])
 
    @torch.no_grad()
    def predict(self, logits):
        # [B, n_nodes] probability mass per node (leaf softmax summed up the tree)
        return self.sum_leaf(F.softmax(logits, dim=-1), dim=-1)

    
class Hierarchy:
    def __init__(self, parents: np.ndarray):
        assert np.all(parents[1:] < np.arange(1, len(parents)))  # topo-order guarantee
        self._parents = parents

    def edges(self):
        return list(zip(self._parents[1:], itertools.count(1)))

    def num_children(self):
        n = len(self._parents)
        unique, counts = np.unique(self._parents[1:], return_counts=True)
        r = np.zeros(n, dtype=int); r[unique] = counts; return r

    def leaf_mask(self):
        return self.num_children() == 0 # leaf nodes have no children

    def ancestor_mask(self, strict=False):
        n = len(self._parents)
        is_desc = np.zeros((n, n), dtype=bool)
        if not strict: 
            is_desc[0, 0] = 1
        for i, j in self.edges():          # i is parent of j, i < j
            is_desc[j, :] = is_desc[i, :]
            is_desc[j, j] = 1
        return is_desc.T

class Sum(nn.Module):
    def __init__(self, tree, transpose, subset=None, strict=False):
        super().__init__()
        m = tree.ancestor_mask(strict=strict)
        if subset is not None: 
            m = m[:, subset]
        if transpose: 
            m = m.T
        self.matrix = torch.from_numpy(m).type(torch.get_default_dtype())
    def _apply(self, fn):
        super()._apply(fn)
        self.matrix = fn(self.matrix)
        return self
    def forward(self, values, dim=-1):
        return torch.tensordot(values, self.matrix, dims=1)

SumDescendants = partial(Sum, transpose=True)
def SumLeafDescendants(tree, **kw):
    return SumDescendants(tree, subset=tree.leaf_mask(), **kw)

class FlatSoftmaxNLL(nn.Module):
    """Tree-aware NLL. Network predicts a flat softmax over LEAVES; targets are
    full-tree NODE indices (internal nodes allowed). Loss rewards probability
    mass anywhere in the target node's leaf subtree."""

    def __init__(self, parents: np.ndarray, reduction: str = "mean"):
        super().__init__()
        assert reduction in ("mean", "none")
        hierarchy = Hierarchy(parents)
        is_anc = hierarchy.ancestor_mask(strict=False)          # [n_nodes, n_nodes]
        leaf_mask = hierarchy.leaf_mask()                        # [n_nodes]
        leaf_masks = is_anc[:, leaf_mask]                      # [n_nodes, n_leaf]
        self.register_buffer("leaf_masks", torch.from_numpy(leaf_masks))
        self.reduction = reduction

    def forward(self, scores: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
        # scores: [B, n_leaf] logits.  target_idx: [B] node indices in the full tree.
        logp_leaf = F.log_softmax(scores, dim=-1)              # [B, n_leaf]
        allowed = self.leaf_masks[target_idx.long()]           # [B, n_leaf] bool
        neg_inf = torch.tensor(float("-inf"), device=scores.device)
        logp_allowed = torch.where(allowed, logp_leaf, neg_inf)
        logp_label = torch.logsumexp(logp_allowed, dim=-1)     # log P(subtree) [B]
        loss = -logp_label
        return loss.mean() if self.reduction == "mean" else loss
