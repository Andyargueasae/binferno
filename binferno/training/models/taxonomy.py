import itertools
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TaxonomyHead(nn.Module):
    def __init__(self, in_dim, table_parent: list, tax_nodes: list = None,
                 node_indices: dict = None, hidden=512, dropout=0.2):
        super().__init__()
        self.tree = Hierarchy(np.asarray(table_parent))
        self.n_leaf = int(self.tree.leaf_mask().sum())
        self.n_nodes = len(table_parent)  # number of nodes in the tree, including internal nodes and leaves
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
        self.loss_fn = FlatSoftmaxNLL(np.asarray(table_parent))
        self.sum_leaf = SumLeafDescendants(Hierarchy(np.asarray(table_parent)), strict=False)
 
    def forward(self, 
        fused: torch.Tensor,  # [B, in_dim]
    ) -> torch.Tensor:
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

class TaxonomyFusion(nn.Module):
    """
    Wraps a TaxonomyHead and fuses its dense prediction back into the trunk
    representation to form the clustering latent.
 
        fused ──┬──────────────────────────────► (kept)
                ├─► tax_head ─► logits ─► loss (labeled subset only)
                └─► softmax/predict ─► detach ─► tax_proj ─► tax_emb
        cluster_latent = fuse( fused , tax_emb )         [gated by warmup]
 
    Cluster on `forward(...)['latent']`.
    """
    def __init__(self, in_dim, table_parent, tax_nodes=None, node_indices=None,
                 hidden=512, dropout=0.2,
                 tax_emb_dim=64, latent_dim: int=32,
                 warmup_steps=2000, detach_prediction=True,
                 use_node_probs=True):
        super().__init__()
        self.head = TaxonomyHead(in_dim, table_parent, tax_nodes, node_indices,
                                 hidden=hidden, dropout=dropout)
        self.detach_prediction = detach_prediction
        self.warmup_steps = warmup_steps
        self.use_node_probs = use_node_probs          # True: [B,n_nodes] probs; False: [B,n_leaf]
        self.register_buffer("_step", torch.zeros((), dtype=torch.long))
 
        pred_dim = self.head.n_nodes if use_node_probs else self.head.n_leaf
        # Compress the (large, sparse) prediction vector to a small dense embedding.
        self.tax_proj = nn.Sequential(
            nn.Linear(pred_dim, tax_emb_dim),
            nn.LayerNorm(tax_emb_dim),
            nn.LeakyReLU(),
        )
        latent_dim = latent_dim or in_dim
        # Fuse trunk + taxonomy embedding -> clustering latent.
        self.fuse = nn.Sequential(
            nn.Linear(in_dim + tax_emb_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
        )
        self.latent_dim = latent_dim
 
    def _gate(self) -> float:
        # 0 during warmup, ramped to 1 over an equal-length window after it.
        s = int(self._step.item())
        if s < self.warmup_steps:
            return 0.0
        ramp = min(1.0, (s - self.warmup_steps) / max(1, self.warmup_steps))
        return ramp
 
    def forward(self, fused, advance_step: bool = True):
        logits = self.head(fused)                      # [B, n_leaf]
 
        if self.use_node_probs:
            pred = self.head.sum_leaf(F.softmax(logits, dim=-1), dim=-1)  # [B, n_nodes]
        else:
            pred = F.softmax(logits, dim=-1)                              # [B, n_leaf]
 
        if self.detach_prediction:
            pred = pred.detach()
 
        tax_emb = self.tax_proj(pred)                  # [B, tax_emb_dim]
        gate = self._gate()
        tax_emb = gate * tax_emb                       # closed during warmup
 
        latent = self.fuse(torch.cat([fused, tax_emb], dim=-1))  # [B, latent_dim]
 
        if self.training and advance_step:
            self._step += 1
 
        return {"latent": latent, "logits": logits, "gate": gate}
 
    def loss(self, logits: torch.Tensor, target_idx: torch.Tensor, mask: torch.Tensor, weights=None):
        # delegate to the head's masked hierarchical loss (optionally length-weighted)
        return self.head.loss(logits, target_idx, mask, weights=weights)

# target index: the node index in integer.
# mask: boolean mask for the contigs that have taxonomy labels.
# logits: the output of the taxonomy head, which is a tensor of shape [B, n_leaf].
# weights: optional tensor of shape [B] for weighting the loss per sample by contig length.