import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np

class HierarchicalTaxonomyEncoder(nn.Module):
    # June 24, 2026 TODO.
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        output_dim: int = 128,
        dropout: float = 0.1,
        seed: int = 42,
        taxonomy: bool = False,
    ):
        super(HierarchicalTaxonomyEncoder, self).__init__()
        self.taxonomy = taxonomy
        self.taxonomy_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.taxonomy_encoder(x)

class TaxonomyHead(nn.Module):
    def __init__(self, in_dim, table_parent, hidden=512, dropout=0.2):
        super().__init__()
        self.tree = self.Hierarchy(np.asarray(table_parent))
        self.n_leaf = int(self.tree.leaf_mask().sum())

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),  # LN, per your convention
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.n_leaf),  # logits over leaves
        )
        self.loss_fn = self.FlatSoftmaxNLL(self.tree)
        self.sum_leaf = self.SumLeafDescendants(self.tree, strict=False)

    def forward(self, fused):  # fused: [B, in_dim] encoder output
        return self.net(fused)  # [B, n_leaf]

    def loss(self, logits, target_idx, mask):  # mask: contigs that HAVE a label
        if mask.sum() == 0:
            return logits.new_zeros(())
        return self.loss_fn(logits[mask], target_idx[mask])

    @torch.no_grad()
    def predict(self, logits):
        return self.sum_leaf(F.softmax(logits, -1), dim=-1)  # per-node probs

    def Hierarchy(self, table_parent):
        # Build a tree from the parent-child relationships in table_parent
        G = nx.DiGraph()
        for child, parent in enumerate(table_parent):
            if parent != -1:  # Assuming -1 indicates no parent (root)
                G.add_edge(parent, child)
        return G

    def FlatSoftmaxNLL(self, tree):
        # This function should implement the negative log-likelihood loss for the flat softmax over the leaf nodes,
        # taking into account the hierarchy defined by the tree.
        raise NotImplementedError(
            "This function should be implemented to compute the flat softmax NLL loss."
        )
    
    def LabelHloss(self, tree):
        raise NotImplementedError(
            "This function should be implemented to compute the hierarchical loss based on the tree structure."
        )

    def SumLeafDescendants(self, tree, strict=False):
        # This function should implement the summation of probabilities from leaf nodes to their ancestors in the hierarchy defined by the tree.
        raise NotImplementedError(
            "This function should be implemented to sum leaf descendants' probabilities to their ancestors."
        )


def _build_tax_emb_encoder(
    tax_emb_dim: int, 
    hidden_dim: int = 512, 
    output_dim: int = 128, 
    dropout: float = 0.1
):
    n_layers = 3
    modules = []
    input_dim = tax_emb_dim
    for _ in range(n_layers):
        modules.append(nn.Linear(input_dim, hidden_dim))
        modules.append(nn.BatchNorm1d(hidden_dim))
        modules.append(nn.LeakyReLU())
        modules.append(nn.Dropout(dropout))
        input_dim = hidden_dim
    modules.append(nn.Linear(hidden_dim, output_dim))
    modules.append(nn.BatchNorm1d(output_dim))
    modules.append(nn.LeakyReLU())
    return nn.Sequential(*modules)