import numpy as np
import pandas as pd
import pytest

from binferno.clustering.leiden import LeidenClustering
from binferno.clustering.taxonomy_cluster import (
    penalize_taxonomy_conflicts,
    prepare_taxonomy_arrays,
)


def test_taxonomy_edges_only_weaken_supported_conflicts():
    # Rows 0/1 disagree at genus with matching known parents. Row 2 has
    # no genus assignment, so its edge must retain the embedding weight.
    ids = np.array([
        [1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 2, 0],
        [1, 1, 1, 1, 1, 0, 0],
    ])
    masks = (ids > 0).astype(np.uint8)
    weights, conflicts = penalize_taxonomy_conflicts(
        np.array([0, 0]), np.array([1, 2]),
        np.array([0.8, 0.6]), ids, masks, conflict_factor=0.35,
    )
    assert conflicts == 1
    np.testing.assert_allclose(weights, [0.28, 0.6])


def test_taxonomy_rows_align_with_filtered_contigs_and_allow_missing():
    ranks = ("domain", "genus")
    table = pd.DataFrame(
        [[1, 3, 1, 1], [1, 4, 1, 1]],
        index=["b", "a"],
        columns=["tax_domain", "tax_genus", "mask_domain", "mask_genus"],
        dtype=float,
    )
    ids, masks = prepare_taxonomy_arrays(table, ["a", "missing", "b"], ranks)
    np.testing.assert_array_equal(ids, [[1, 4], [0, 0], [1, 3]])
    np.testing.assert_array_equal(masks[1], [0, 0])

    table.loc["a", "mask_genus"] = 0.5
    with pytest.raises(ValueError, match="masks"):
        prepare_taxonomy_arrays(table, ["a", "b"], ranks)


def test_leiden_runs_without_taxonomy(tmp_path):
    leiden = LeidenClustering()
    assert leiden.taxonomy_edge_factor == 1.0
    neighbors = np.array([[0, 1, 2], [1, 0, 2], [2, 0, 1]])
    distances = np.array([[0, 0.1, 0.2], [0, 0.1, 0.2], [0, 0.2, 0.2]])
    output = tmp_path / "baseline.tsv"
    leiden.run_leiden(
        output_file=output,
        namelist=["a", "b", "c"],
        ann_neighbor_indices=neighbors,
        ann_distances=distances,
        length_weight=[1.0, 1.0, 1.0],
        initial_list=[0, 1, 2],
        is_membership_fixed=np.zeros(3, dtype=bool),
        partgraph_ratio=100,
    )
    assert len(output.read_text().splitlines()) == 3
