import pandas as pd
import numpy as np
import pysam
from tqdm import tqdm
from typing import Dict, List, Set, Tuple, Union

FragmentDict = Dict[str, Dict[str, Union[str, List[str]]]]
CoverageDict = Dict[str, float]

def _normalise_coverage_dicts(
    coverage: dict,
    coverage_std: dict,
    all_fragment_headers: list,
    total_mapped: int,
    rpkm_normalize: bool = True,
) -> tuple[dict, dict]:
    """
    Supports:
    - none: raw mean/std
    - rpm/rpkm: divide mean/std by total_mapped / 1e6
    - max: add 1e-5, then divide mean/std separately by their own max
    """
    if total_mapped <= 0:
        zero = {fh: 0.0 for fh in all_fragment_headers}
        return zero, zero

    if not rpkm_normalize:
        return coverage, coverage_std

    else:
        factor = total_mapped / 1_000_000
        if factor <= 0:
            zero = {fh: 0.0 for fh in all_fragment_headers}
            return zero, zero

        return (
            {k: v / factor for k, v in coverage.items()},
            {k: v / factor for k, v in coverage_std.items()},
        )

def _map_fasta_to_bam_refs(
    fragments_dict: FragmentDict,
    bam_references: Set[str],
    disable_progress: bool = False,
) -> Tuple[Dict[str, str], List[str]]:
    bam_ref_map = {}
    unmapped_fasta_headers = []

    iterator = (
        fragments_dict.keys()
        if disable_progress
        else tqdm(fragments_dict.keys(), desc="Mapping headers")
    )

    for original_header in iterator:
        fasta_key = original_header  # key id.

        # Try direct match
        if fasta_key in bam_references:
            bam_ref_map[original_header] = (
                fasta_key  # key as orig header and fasta_key as value.
            )
            continue

        # Try matching after splitting on space
        base_name_space = fasta_key.split(" ")[0]
        if base_name_space in bam_references:
            bam_ref_map[original_header] = base_name_space
            continue

        # Try matching after splitting on last dot
        base_name_dot = fasta_key.rsplit(".", 1)[0] if "." in fasta_key else None
        if base_name_dot and base_name_dot in bam_references:
            bam_ref_map[original_header] = base_name_dot
            continue

        bam_ref_map[original_header] = None
        unmapped_fasta_headers.append(original_header)

    return bam_ref_map, unmapped_fasta_headers
