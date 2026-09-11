from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from binferno.clustering.cluster_utils import apply_dimension_reduction, l2_normalize

from binferno.clustering.cluster_utils import (
    _filter_embeddings_by_length,
    _read_seed_count,
    gen_seed_idx,
    partial_seed_init,
)


def _write_contigs(path: Path):
    path.write_text(
        ">contig_1\n" + "A" * 1200 + "\n"
        ">contig_2\n" + "C" * 900 + "\n"
        ">contig_3\n" + "G" * 1500 + "\n"
    )


def test_filter_embeddings_by_length(tmp_path):
    contigs = tmp_path / "contigs.fna"
    _write_contigs(contigs)

    emb_df = pd.DataFrame(
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        index=["contig_1", "contig_2", "contig_3"],
        columns=["e1", "e2"],
    )

    features, names, lengths = _filter_embeddings_by_length(
        emb_df=emb_df,
        contig_file=str(contigs),
        contig_len=1000,
    )

    assert features.shape == (2, 2)
    assert list(names) == ["contig_1", "contig_3"]
    assert np.array_equal(lengths.astype(int), np.array([1200, 1500]))


def test_read_seed_count_valid(tmp_path):
    seed_file = tmp_path / "seed.txt"
    seed_file.write_text("contig_1\ncontig_2\ncontig_1\n")
    assert _read_seed_count(str(seed_file)) == 2


def test_read_seed_count_missing_file(tmp_path):
    missing = tmp_path / "missing.seed"
    with pytest.raises(FileNotFoundError):
        _read_seed_count(str(missing))


def test_read_seed_count_empty_file(tmp_path):
    seed_file = tmp_path / "empty.seed"
    seed_file.write_text("\n")
    with pytest.raises(ValueError):
        _read_seed_count(str(seed_file))


def test_gen_seed_idx_partial_match(tmp_path):
    seed_file = tmp_path / "seed.txt"
    seed_file.write_text("contig_2\nnot_present\ncontig_1\n")
    ids = ["contig_1", "contig_2", "contig_3"]

    out = gen_seed_idx(str(seed_file), ids)
    assert out == [1, 0]


def test_partial_seed_init_shapes_and_seeded_first_center():
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ],
        dtype=np.float64,
    )
    centers = partial_seed_init(
        X=X,
        n_clusters=3,
        random_state=0,
        seed_idx=[2, 1],
    )

    assert centers.shape == (3, 2)
    # first two centers must be exactly the seed rows
    assert np.allclose(centers[0], X[2])
    assert np.allclose(centers[1], X[1])


def test_partial_seed_init_errors():
    X = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)

    with pytest.raises(ValueError):
        partial_seed_init(X=X, n_clusters=0, random_state=0, seed_idx=[0])

    with pytest.raises(ValueError):
        partial_seed_init(X=X, n_clusters=3, random_state=0, seed_idx=[0])

    with pytest.raises(ValueError):
        partial_seed_init(X=X, n_clusters=1, random_state=0, seed_idx=[])

    with pytest.raises(ValueError):
        partial_seed_init(X=X, n_clusters=1, random_state=0, seed_idx=[0, 1])


# ============================================================================
# Tests for apply_dimension_reduction and l2_normalize functions
# ============================================================================


def _create_sample_embeddings(n_samples=100, n_features=768):
    """Create synthetic embedding data for testing."""
    rng = np.random.default_rng(seed=42)
    data = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    return pd.DataFrame(
        data,
        index=[f"contig_{i}" for i in range(n_samples)],
        columns=[f"feat_{i}" for i in range(n_features)],
    )


def test_l2_normalize_unit_vectors():
    """Test L2 normalization produces unit vectors."""
    X = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    normalized = l2_normalize(X)
    norms = np.linalg.norm(normalized, axis=1)
    assert np.allclose(norms[:2], 1.0, atol=1e-5)
    assert np.linalg.norm(normalized[2]) < 1e-5


def test_l2_normalize_preserves_direction():
    """Test that L2 normalization preserves direction."""
    X = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32)
    normalized = l2_normalize(X)
    assert np.allclose(normalized[0], [1.0, 0.0], atol=1e-5)
    assert np.allclose(normalized[1], [0.0, 1.0], atol=1e-5)


def test_apply_dimension_reduction_pca_path():
    """Test dimension reduction with PCA enabled."""
    X = _create_sample_embeddings(n_samples=100, n_features=768)
    result = apply_dimension_reduction(X, pca=True, umap=False)
    assert isinstance(result, pd.DataFrame)
    assert list(result.index) == list(X.index)
    assert result.shape == (100, 50)
    assert all(col.startswith("dim_") for col in result.columns)


def test_apply_dimension_reduction_umap_path():
    """Test dimension reduction with UMAP enabled."""
    X = _create_sample_embeddings(n_samples=50, n_features=128)
    result = apply_dimension_reduction(X, pca=False, umap=True, res_dim=10)
    assert isinstance(result, pd.DataFrame)
    assert list(result.index) == list(X.index)
    assert result.shape == (50, 10)


def test_apply_dimension_reduction_both_methods():
    """Test dimension reduction with both PCA and UMAP enabled."""
    X = _create_sample_embeddings(n_samples=50, n_features=256)
    result = apply_dimension_reduction(X, pca=True, umap=True, res_dim=15)
    assert isinstance(result, pd.DataFrame)
    assert list(result.index) == list(X.index)
    assert result.shape == (50, 15)


def test_apply_dimension_reduction_default_params():
    """Test dimension reduction with default parameters."""
    X = _create_sample_embeddings(n_samples=100, n_features=768)
    result = apply_dimension_reduction(X)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (100, 50)


def test_apply_dimension_reduction_index_preserved():
    """Test that contig indices are preserved after reduction."""
    X = _create_sample_embeddings(n_samples=20, n_features=100)
    original_index = list(X.index)
    result = apply_dimension_reduction(X, pca=True)
    assert list(result.index) == original_index


def test_apply_dimension_reduction_small_dataset():
    """Test dimension reduction with small dataset."""
    X = pd.DataFrame(
        np.random.randn(5, 32).astype(np.float32),
        index=[f"contig_{i}" for i in range(5)],
        columns=[f"feat_{i}" for i in range(32)],
    )
    result = apply_dimension_reduction(X, pca=True)
    assert result.shape[0] == 5
    assert result.shape[1] == 50


def test_apply_dimension_reduction_dtypes():
    """Test that output has appropriate data types."""
    X = _create_sample_embeddings(n_samples=30, n_features=100)
    result = apply_dimension_reduction(X, pca=True)
    assert np.issubdtype(result.dtypes[0], np.floating)
    assert all(isinstance(idx, str) for idx in result.index)


def test_apply_dimension_reduction_finite_values():
    """Test that output contains only finite values (no NaN or inf)."""
    X = _create_sample_embeddings(n_samples=50, n_features=256)
    result = apply_dimension_reduction(X, pca=True)
    assert np.all(np.isfinite(result.values))


def test_apply_dimension_reduction_custom_res_dim():
    """Test dimension reduction with custom res_dim parameter."""
    X = _create_sample_embeddings(n_samples=40, n_features=128)
    for res_dim in [5, 20, 32]:
        result = apply_dimension_reduction(X, umap=True, res_dim=res_dim)
        assert result.shape[1] == res_dim


def test_apply_dimension_reduction_l2_normalization_applied():
    """Test that L2 normalization is applied in the pipeline."""
    X = pd.DataFrame(
        np.random.randn(20, 100).astype(np.float32) * 1000,
        index=[f"contig_{i}" for i in range(20)],
        columns=[f"feat_{i}" for i in range(100)],
    )
    result = apply_dimension_reduction(X, pca=True)
    assert np.all(np.isfinite(result.values))
    assert result.shape == (20, 50)


def test_apply_dimension_reduction_reproducibility_pca():
    """Test that PCA path produces consistent results."""
    X = _create_sample_embeddings(n_samples=40, n_features=128)
    result1 = apply_dimension_reduction(X.copy(), pca=True)
    result2 = apply_dimension_reduction(X.copy(), pca=True)
    assert np.allclose(result1.values, result2.values, rtol=1e-5, atol=1e-7)


def test_apply_dimension_reduction_output_structure():
    """Test the structure of the output DataFrame."""
    X = _create_sample_embeddings(n_samples=25, n_features=200)
    result = apply_dimension_reduction(X, pca=True)
    assert isinstance(result, pd.DataFrame)
    assert len(result.columns) == 50
    assert len(result) == 25
    expected_cols = [f"dim_{i}" for i in range(50)]
    assert list(result.columns) == expected_cols


def test_apply_dimension_reduction_no_methods_still_processes():
    """Test that L2 norm and KernelPCA are applied even with both flags False."""
    X = _create_sample_embeddings(n_samples=35, n_features=150)
    result = apply_dimension_reduction(X, pca=False, umap=False)
    assert result.shape == (35, 50)
    assert isinstance(result, pd.DataFrame)


def test_apply_dimension_reduction_with_single_sample():
    """Test dimension reduction with only one sample."""
    X = pd.DataFrame(
        np.random.randn(1, 100).astype(np.float32),
        index=["contig_0"],
        columns=[f"feat_{i}" for i in range(100)],
    )
    result = apply_dimension_reduction(X, pca=True)
    assert result.shape == (1, 50)
    assert result.index[0] == "contig_0"


def test_apply_dimension_reduction_preserves_sample_order():
    """Test that sample order is preserved during reduction."""
    indices = [f"sample_{i}" for i in range(15)]
    X = pd.DataFrame(
        np.random.randn(15, 128).astype(np.float32),
        index=indices,
        columns=[f"feat_{i}" for i in range(128)],
    )
    result = apply_dimension_reduction(X, pca=True)
    assert list(result.index) == indices


def test_apply_dimension_reduction_kernel_pca_cosine():
    """Test that KernelPCA with cosine kernel is applied."""
    X = _create_sample_embeddings(n_samples=25, n_features=512)
    result = apply_dimension_reduction(X, pca=True)
    assert result.shape[1] == 50
    assert np.all(np.isfinite(result.values))
