from typing import Sequence

import numpy as np
import pandas as pd

RANKS = (
    "domain", "phylum", "class", "order", "family", "genus", "species"
)


def prepare_taxonomy_arrays(
    taxonomy: pd.DataFrame,
    namelist: Sequence[str],
    ranks: Sequence[str] = RANKS,
) -> tuple[np.ndarray, np.ndarray]:
    """Align taxonomy with Leiden vertices; absent contigs remain unknown."""
    if not taxonomy.index.is_unique:
        raise ValueError("Taxonomy contig IDs must be unique.")
    id_columns = [f"tax_{rank}" for rank in ranks]
    mask_columns = [f"mask_{rank}" for rank in ranks]
    missing = set(id_columns + mask_columns) - set(taxonomy.columns)
    if missing:
        raise ValueError(f"Taxonomy columns are missing: {sorted(missing)}")

    ordered = taxonomy.reindex(namelist)[id_columns + mask_columns].fillna(0)
    raw_ids = ordered[id_columns].to_numpy(dtype=np.float64)
    raw_masks = ordered[mask_columns].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(raw_ids).all()
        or (raw_ids < 0).any()
        or (raw_ids != np.floor(raw_ids)).any()
    ):
        raise ValueError("Taxonomy IDs must be finite nonnegative integers.")
    if not np.isin(raw_masks, (0, 1)).all():
        raise ValueError("Taxonomy masks must contain only 0 or 1.")
    if ((raw_masks == 1) & (raw_ids == 0)).any():
        raise ValueError("A present taxonomy rank must have a nonzero ID.")
    if raw_ids.size and raw_ids.max() > np.iinfo(np.int32).max:
        raise ValueError("Taxonomy IDs exceed the int32 range.")
    return raw_ids.astype(np.int32), raw_masks.astype(np.uint8)


def penalize_taxonomy_conflicts(
    sources: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    tax_ids: np.ndarray,
    tax_masks: np.ndarray,
    *,
    conflict_rank: str = "genus",
    conflict_factor: float = 0.35,
) -> tuple[np.ndarray, int]:
    """Weaken edges with a supported conflict at one rank.

    Array rows must follow the same contig order as Leiden's `namelist`.
    Missing taxonomy leaves an edge unchanged.
    """
    rank = RANKS.index(conflict_rank)
    if not 0.0 < conflict_factor <= 1.0:
        raise ValueError("conflict_factor must be in (0, 1]")

    # Require a complete, matching lineage above the disputed rank.
    parents_agree = np.all(
        (tax_masks[sources, :rank] > 0)
        & (tax_masks[targets, :rank] > 0)
        & (tax_ids[sources, :rank] == tax_ids[targets, :rank]),
        axis=1,
    )
    both_known = (
        (tax_masks[sources, rank] > 0)
        & (tax_masks[targets, rank] > 0)
    )
    disagree = tax_ids[sources, rank] != tax_ids[targets, rank]
    conflict = parents_agree & both_known & disagree

    adjusted = weights.copy()
    adjusted[conflict] *= conflict_factor
    return adjusted, int(conflict.sum())
