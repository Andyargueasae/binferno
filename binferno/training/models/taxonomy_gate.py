"""Bounded taxonomy contribution to a composition and coverage embedding."""

import numpy as np
import torch
import torch.nn.functional as F


def bounded_taxonomy_residual(base, taxonomy, tax_features, alpha):
    """Add a unit length taxonomy direction, weighted by valid lineage depth.

    Taxonomy features contain rank IDs followed by their presence masks, ordered
    from broad to specific. A missing ancestor invalidates deeper ranks.
    """
    if base.shape != taxonomy.shape:
        raise ValueError("Base and taxonomy embeddings must have the same shape.")
    if tax_features.ndim != 2 or tax_features.shape[1] % 2:
        raise ValueError("Taxonomy features must contain rank IDs and masks.")
    n_ranks = tax_features.shape[1] // 2
    masks = tax_features[:, n_ranks:].to(dtype=base.dtype).clamp(0, 1)
    confidence = masks.cumprod(dim=1).mean(dim=1, keepdim=True)
    base = F.normalize(base, dim=-1)
    taxonomy = F.normalize(taxonomy, dim=-1)
    return F.normalize(base + alpha * confidence * taxonomy, dim=-1)


def validate_rank_taxonomy_table(table, tax_columns, mask_columns):
    """Validate cached taxonomy once on CPU and return embedding vocab sizes."""
    if table.empty:
        raise ValueError("Taxonomy table must contain at least one row.")
    ids = table[tax_columns].to_numpy()
    masks = table[mask_columns].to_numpy()
    if not np.isfinite(ids).all() or (ids < 0).any() or (ids != np.floor(ids)).any():
        raise ValueError("Taxonomy IDs must be finite, nonnegative integers.")
    if not np.isfinite(masks).all() or not np.isin(masks, (0, 1)).all():
        raise ValueError("Taxonomy masks must contain only 0 or 1.")
    return [int(ids[:, rank].max()) + 1 for rank in range(ids.shape[1])]
