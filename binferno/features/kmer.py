from typing import Tuple, List
from itertools import product
from pathlib import Path
import itertools
import os
from .base import BaseFeatureBuilder
import pandas as pd
import numpy as np
from pydantic import Field

KMER_LEN = 4
KERNEL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "kernel.npz"
)  # give the actual path of the kernel.npz.


def validate_input_array(array: np.ndarray) -> np.ndarray:
    "Returns array similar to input array but C-contiguous and with own data. Derived from VAMB."
    if not array.flags["C_CONTIGUOUS"]:
        array = np.ascontiguousarray(array)
    if not array.flags["OWNDATA"]:
        array = array.copy()

    assert array.flags["C_CONTIGUOUS"] and array.flags["OWNDATA"]
    return array


def generate_feature_mapping(kmer_len):
    BASE_COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G"}
    kmer_hash = {}
    counter = 0

    # all kmers, len=256.
    kmers = ["".join(kmer) for kmer in itertools.product("ATGC", repeat=kmer_len)]

    for kmer in kmers:
        if kmer not in kmer_hash:
            kmer_hash[kmer] = counter
            rev_compl = kmer.translate(str.maketrans(BASE_COMPLEMENT))[::-1]
            kmer_hash[rev_compl] = counter
            counter += 1
    return kmer_hash, counter


def _calculate_kmer_composition(
    sequences_to_process: List[Tuple[str, str]],
    kmer_len: int = 4,
    pseudocount: float = 1,
) -> pd.DataFrame:
    """
    Compute normalized canonical k-mer frequencies for each input sequence.

    Args:
        sequences_to_process: List of (header, sequence) tuples.
        kmer_len: Length of k-mers to evaluate.
        pseudocount: Small constant added to every feature before normalization.

    Returns:
        DataFrame indexed by sequence header, with normalized feature values.
    """
    feature_map, n_features = generate_feature_mapping(kmer_len)

    headers = []
    rows = []

    for header, sequence in sequences_to_process:
        seq = sequence.upper()
        n_kmers = len(seq) - kmer_len + 1

        if n_kmers > 0:
            encoded = np.fromiter(
                (
                    idx
                    for i in range(n_kmers)
                    for idx in [feature_map.get(seq[i : i + kmer_len])]
                    if idx is not None
                ),
                dtype=np.int64,
            )
        else:
            encoded = np.empty(0, dtype=np.int64)

        if encoded.size:
            counts = np.bincount(encoded, minlength=n_features).astype(
                np.float64, copy=False
            )
        else:
            counts = np.zeros(n_features, dtype=np.float64)

        headers.append(header.replace(">", ""))
        rows.append(counts)

    if not rows:
        return pd.DataFrame()

    # Stack rows into a matrix and add pseudocount before normalization.
    matrix = np.vstack(rows)
    print("Use pseudocount of ", pseudocount, " for k-mer features.")
    matrix += pseudocount

    # normalize each row to sum to 1, handling zero-sum cases by setting them to zero.
    totals = matrix.sum(axis=1, keepdims=True)
    valid = totals[:, 0] > 1e-9

    matrix[valid] /= totals[
        valid
    ]  # only normalize rows with non-zero totals. so that for each contig the sum of k-mer features is 1.0.
    matrix[~valid] = 0.0

    return pd.DataFrame(
        matrix,
        index=headers,
        columns=[str(i) for i in range(n_features)],
    )


def generate_kmer_projection(contigs: dict) -> Tuple[np.ndarray, List[str]]:

    # load kernel first.
    npz = np.load(KERNEL)
    array = validate_input_array(npz["arr_0"])
    npz.close()
    assert array.shape == (256, 103)
    kmer_dict = {
        "".join(kmer): idx for idx, kmer in enumerate(product("ACGT", repeat=KMER_LEN))
    }
    num_contigs = sum([1 for i, _ in enumerate(contigs)])
    projection_matrix = np.zeros(
        (num_contigs, 103), dtype=np.float32
    )  # We knew that the resulting projection will be 103-dimensional, from the 2009 paper.
    identifiers = []

    raw_mat = np.zeros((num_contigs, 4**KMER_LEN), dtype=np.float32)
    for i, (name, seq) in enumerate(contigs):
        identifiers.append(name)
        kmer_counts = np.zeros(4**KMER_LEN, dtype=np.float32)

        for j in range(len(seq) - KMER_LEN + 1):
            kmer = seq[j : j + KMER_LEN]
            if all(base in "ACGT" for base in kmer):
                kmer_counts[kmer_dict[kmer]] += 1
        raw_mat[i, :] = (
            kmer_counts  # store the 256-size kmer count for the iterating sequence.
        )

    s = raw_mat.sum(axis=1).reshape(-1, 1)
    s[s == 0] = 1.0
    raw_mat *= 1 / s  # normalized by summed frequencies.
    raw_mat -= 1 / 256  # add pseudocount 1/256.
    projection_matrix = np.dot(raw_mat, array)  # TL = P.
    assert (
        projection_matrix.shape[1] == 103
    ), "Projection matrix should have 103 features as per the kernel dimensions."
    return projection_matrix, identifiers


class KmerFeatureBuilder(BaseFeatureBuilder):
    pseudocount: int = Field(
        1,
        description="Pseudocount to add to k-mer counts before normalization. Default is 1.",
    )
    # TODO: think of using larger k.
    kmer_len: int = Field(
        KMER_LEN,
        description=f"Length of k-mers to compute. Default is {KMER_LEN}.",
    )

    def __call__(self, fragments: list) -> pd.DataFrame:
        # generate hashable 4-mer feature mapping, prepare for counting tnfs.
        kmer_df = _calculate_kmer_composition(
            fragments, kmer_len=self.kmer_len, pseudocount=self.pseudocount
        )
        return kmer_df


# ======= Multisample kmer features. =======
