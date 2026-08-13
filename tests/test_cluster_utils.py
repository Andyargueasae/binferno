from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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