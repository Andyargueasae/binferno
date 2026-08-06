import itertools

from typing import Tuple, List
import pandas as pd
import numpy as np
from pydantic import Field

from .base import BaseFeatureBuilder

KMER_LEN = 4

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