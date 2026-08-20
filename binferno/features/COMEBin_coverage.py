# features/coverage.py
import os
from multiprocessing import Pool
from typing import Dict, List, Set, Tuple, Union, Any, Literal
import glob
import pandas as pd
from .base import BaseFeatureBuilder
import numpy as np
import pysam
from tqdm import tqdm
from pydantic import Field

from .coverage_utils import (
    _normalise_coverage_dicts,
    _map_fasta_to_bam_refs,
    FragmentDict,
    CoverageDict,
)

def _calculate_fragment_stats_vectorized(
    coverage_array,
    fragment_coords,
    return_variance: bool = False,
):
    n_fragments = len(fragment_coords)
    means = np.zeros(n_fragments, dtype=np.float64)
    dispersions = np.zeros(n_fragments, dtype=np.float64)  # std or var

    if n_fragments < 10:
        for i, (start, end) in enumerate(fragment_coords):
            fragment_coverage = coverage_array[start:end]
            if len(fragment_coverage) > 0:
                means[i] = np.mean(fragment_coverage)
                if len(fragment_coverage) > 1:
                    dispersions[i] = (
                        np.var(fragment_coverage)
                        if return_variance
                        else np.std(fragment_coverage)
                    )
                else:
                    dispersions[i] = 0.0
            else:
                means[i] = 0.0
                dispersions[i] = 0.0
        return means, dispersions

    try:
        all_indices = []
        fragment_starts = []
        fragment_lengths = []

        for start, end in fragment_coords:
            length = end - start
            if length > 0:
                all_indices.extend(range(start, end))
                fragment_starts.append(len(all_indices) - length)
                fragment_lengths.append(length)
            else:
                fragment_starts.append(0)
                fragment_lengths.append(0)

        if all_indices:
            all_fragment_data = coverage_array[all_indices]

            fragment_lengths_arr = np.array(fragment_lengths)
            cumulative_indices_arr = np.cumsum([0] + fragment_lengths[:-1])

            valid_mask = fragment_lengths_arr > 0

            if np.any(valid_mask):
                valid_indices = cumulative_indices_arr[valid_mask]
                valid_lengths = fragment_lengths_arr[valid_mask]

                sums = np.add.reduceat(all_fragment_data, valid_indices)
                means_valid = sums / valid_lengths

                sums_sq = np.add.reduceat(all_fragment_data**2, valid_indices)
                vars_valid = (sums_sq / valid_lengths) - (means_valid**2)

                vars_valid = np.maximum(vars_valid, 0)

                means[valid_mask] = means_valid
                if return_variance:
                    dispersions[valid_mask] = vars_valid
                else:
                    dispersions[valid_mask] = np.sqrt(vars_valid)

        return means, dispersions

    except (IndexError, ValueError, ZeroDivisionError) as e:
        print(f"Vectorized processing failed, using fallback: {e}")
        for i, (start, end) in enumerate(fragment_coords):
            fragment_coverage = coverage_array[start:end]
            if len(fragment_coverage) > 0:
                means[i] = np.mean(fragment_coverage)
                if len(fragment_coverage) > 1:
                    dispersions[i] = (
                        np.var(fragment_coverage)
                        if return_variance
                        else np.std(fragment_coverage)
                    )
                else:
                    dispersions[i] = 0.0
            else:
                means[i] = 0.0
                dispersions[i] = 0.0
        return means, dispersions


def _process_contig_coverage_worker(args):
    (
        bam_contig_name,
        contig_data_list,
        total_coverage_per_base,
        bam_contig_length,
        return_variance,
    ) = args

    fragment_coverage = {}
    fragment_coverage_dispersion = {}

    try:
        if total_coverage_per_base is None or bam_contig_length is None:
            for original_header, data in contig_data_list:
                for fragment_header in data["fragments"]:
                    fragment_coverage[fragment_header] = 0.0
                    fragment_coverage_dispersion[fragment_header] = 0.0
            return fragment_coverage, fragment_coverage_dispersion, []

        fragment_coords = []
        fragment_headers = []

        for original_header, data in contig_data_list:
            fragment_info = data.get("fragment_info", {})

            for fragment_header in data["fragments"]:
                if fragment_header not in fragment_info:
                    print(
                        f"Missing fragment info for {fragment_header}, setting zero coverage"
                    )
                    fragment_coverage[fragment_header] = 0.0
                    fragment_coverage_dispersion[fragment_header] = 0.0
                    continue

                start_pos = fragment_info[fragment_header]["start_pos"]
                frag_len = fragment_info[fragment_header]["length"]
                end_pos = start_pos + frag_len

                if start_pos < 0:
                    print(
                        f"Fragment {fragment_header} has negative start position: {start_pos}"
                    )
                    fragment_coverage[fragment_header] = 0.0
                    fragment_coverage_dispersion[fragment_header] = 0.0
                    continue

                if start_pos >= bam_contig_length:
                    print(
                        f"Fragment {fragment_header} starts beyond contig end: {start_pos} >= {bam_contig_length}"
                    )
                    fragment_coverage[fragment_header] = 0.0
                    fragment_coverage_dispersion[fragment_header] = 0.0
                    continue

                effective_start = max(0, start_pos)
                effective_end = min(end_pos, bam_contig_length)

                if effective_start >= effective_end:
                    fragment_coverage[fragment_header] = 0.0
                    fragment_coverage_dispersion[fragment_header] = 0.0
                    continue

                fragment_coords.append((effective_start, effective_end))
                fragment_headers.append(fragment_header)

        if fragment_coords:
            try:
                means, dispersions = _calculate_fragment_stats_vectorized(
                    total_coverage_per_base,
                    fragment_coords,
                    return_variance=return_variance,
                )

                for i, fragment_header in enumerate(fragment_headers):
                    fragment_coverage[fragment_header] = float(means[i])
                    fragment_coverage_dispersion[fragment_header] = float(
                        dispersions[i]
                    )

            except Exception as e:
                print(f"Error in vectorized coverage calculation: {e}")
                for i, (start, end) in enumerate(fragment_coords):
                    fragment_header = fragment_headers[i]
                    try:
                        fragment_coverage_array = total_coverage_per_base[start:end]
                        if fragment_coverage_array.size > 0:
                            fragment_coverage[fragment_header] = float(
                                np.mean(fragment_coverage_array)
                            )
                            fragment_coverage_dispersion[fragment_header] = float(
                                np.var(fragment_coverage_array)
                                if return_variance
                                else np.std(fragment_coverage_array)
                            )
                        else:
                            fragment_coverage[fragment_header] = 0.0
                            fragment_coverage_dispersion[fragment_header] = 0.0
                    except Exception as e2:
                        print(
                            f"Error calculating coverage for fragment {fragment_header}: {e2}"
                        )
                        fragment_coverage[fragment_header] = 0.0
                        fragment_coverage_dispersion[fragment_header] = 0.0

    except Exception as e:
        print(f"Error processing contig {bam_contig_name}: {e}")
        for original_header, data in contig_data_list:
            for fragment_header in data["fragments"]:
                fragment_coverage[fragment_header] = 0.0
                fragment_coverage_dispersion[fragment_header] = 0.0

    return fragment_coverage, fragment_coverage_dispersion, []


def calculate_fragment_coverage(
    bam_file: str,
    fragments_dict: FragmentDict,
    cores: int = 16,
    coverage_batch_size: int = 100000,
    disable_progress: bool = False,
    return_variance: bool = False,
):
    fragment_coverage: CoverageDict = {}
    fragment_coverage_std: CoverageDict = {}

    try:
        with pysam.AlignmentFile(bam_file, "rb") as bamf:
            bam_references = set(bamf.references)
            bam_lengths = dict(zip(bamf.references, bamf.lengths))
            if not bam_references:
                print("Alignment file contains no reference sequences.")
                return {}, {}

            bam_ref_map, unmapped_headers = _map_fasta_to_bam_refs(
                fragments_dict, bam_references, disable_progress=disable_progress
            )

            if unmapped_headers:
                print(
                    f"{len(unmapped_headers)} FASTA headers could not be matched to BAM references."
                )

            contig_fragments = {}
            for original_header, data in fragments_dict.items():
                bam_contig_name = bam_ref_map.get(original_header)
                if bam_contig_name is not None:
                    if bam_contig_name not in contig_fragments:
                        contig_fragments[bam_contig_name] = []
                    contig_fragments[bam_contig_name].append((original_header, data))

            total_bases = sum(
                bam_lengths.get(contig, 0) for contig in contig_fragments.keys()
            )

            use_batching = total_bases > 250000000
            batch_size = coverage_batch_size if use_batching else len(contig_fragments)

            contig_list = list(contig_fragments.items())
            num_batches = (len(contig_list) + batch_size - 1) // batch_size

            if use_batching:
                print(
                    f"Processing {len(contig_fragments)} contigs ({total_bases:,} total bases) in {num_batches} batches (batch_size={batch_size}) using {cores} cores..."
                )
                print(
                    f"Estimated peak memory for coverage: ~{(batch_size * (total_bases / len(contig_fragments)) * 4 / 1e9):.2f} GB per batch"
                )
            else:
                print(
                    f"Processing {len(contig_fragments)} contigs ({total_bases:,} total bases) using {cores} cores..."
                )

            results = []

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(contig_list))
                batch_contigs = contig_list[start_idx:end_idx]

                print(
                    f"Processing batch {batch_idx + 1}/{num_batches} ({len(batch_contigs)} contigs)..."
                )

                coverage_data = {}
                batch_iterator = (
                    batch_contigs
                    if disable_progress
                    else tqdm(
                        batch_contigs,
                        desc=f"Loading coverage (batch {batch_idx + 1}/{num_batches})",
                    )
                )
                for bam_contig_name, _ in batch_iterator:
                    bam_contig_length = bam_lengths.get(bam_contig_name)
                    if bam_contig_length is None:
                        coverage_data[bam_contig_name] = (None, None)
                        continue

                    try:
                        coverage_arrays = bamf.count_coverage(
                            contig=bam_contig_name,
                            start=0,
                            stop=bam_contig_length,
                            quality_threshold=0,
                        )
                        total_coverage_per_base = np.sum(coverage_arrays, axis=0)
                        coverage_data[bam_contig_name] = (
                            total_coverage_per_base,
                            bam_contig_length,
                        )
                    except Exception as e:
                        print(
                            f"Error loading coverage for contig {bam_contig_name}: {e}"
                        )
                        coverage_data[bam_contig_name] = (None, None)

                worker_args = [
                    (
                        bam_contig_name,
                        contig_data_list,
                        coverage_data[bam_contig_name][0],
                        coverage_data[bam_contig_name][1],
                        return_variance,
                    )
                    for bam_contig_name, contig_data_list in batch_contigs
                ]

                with Pool(processes=cores) as pool:
                    if disable_progress:
                        batch_results = list(
                            pool.imap(_process_contig_coverage_worker, worker_args)
                        )
                    else:
                        batch_results = list(
                            tqdm(
                                pool.imap(_process_contig_coverage_worker, worker_args),
                                total=len(worker_args),
                                desc=f"Processing fragments (batch {batch_idx + 1}/{num_batches})",
                            )
                        )

                results.extend(batch_results)
                del coverage_data
                print(f"Completed batch {batch_idx + 1}/{num_batches}")

            for res_cov, res_std, _ in results:
                fragment_coverage.update(res_cov)
                fragment_coverage_std.update(res_std)

            for original_header in unmapped_headers:
                if original_header in fragments_dict:
                    for fragment_header in fragments_dict[original_header]["fragments"]:
                        fragment_coverage[fragment_header] = 0.0
                        fragment_coverage_std[fragment_header] = 0.0

    except Exception as e:
        import traceback

        print(f"Error processing alignment file {bam_file}: {e}")
        print(f"Traceback: {traceback.format_exc()}")

        try:
            if bamf:
                total_mapped = bamf.mapped
                total_unmapped = bamf.unmapped
                print(
                    f"BAM file stats: {total_mapped} mapped reads, {total_unmapped} unmapped reads"
                )
                if total_mapped == 0:
                    print(
                        f"BAM file {bam_file} has NO mapped reads - cannot calculate coverage"
                    )
        except:
            print(f"Could not read BAM file stats for {bam_file}")

        return {}, {}

    all_fragment_headers = {
        fh for data in fragments_dict.values() for fh in data["fragments"]
    }
    missing_fragments = all_fragment_headers - set(fragment_coverage.keys())

    for missing_fh in missing_fragments:
        fragment_coverage[missing_fh] = 0.0
        fragment_coverage_std[missing_fh] = 0.0

    print(
        f"Coverage calculation complete. Total fragments: {len(all_fragment_headers)}"
    )

    del results
    return fragment_coverage, fragment_coverage_std


def calculate_bam_cov(
    bam_files: str,
    bam_batch: int = 100000,
    fragments_dict: dict = None,
    threads: int = 16,
    rpkm_normalize: bool = True,
    mode: Literal["default", "comebin"] = "default",
) -> pd.DataFrame:
    """
    mode="default":
        - mean + std output
        - existing normalization behavior

    mode="comebin":
        - mean + variance output (variance instead of std)
        - output column order is [all variance columns, all mean columns]
          to match COMEBin feature stacking style [vars, cov]
        - per-BAM normalization is skipped by default (raw stats), so stacked
          max-normalization can be applied later in COMEBin style.
    """
    if mode not in {"default", "comebin"}:
        raise ValueError(f"Unsupported mode: {mode}")

    bams = sorted(glob.glob(f"{bam_files}/*.bam"))
    if len(bams) == 0:
        return pd.DataFrame()

    all_fragment_headers = [
        fragment_header
        for data in fragments_dict.values()
        for fragment_header in data["fragments"]
    ]

    mean_series_list = []
    dispersion_series_list = []  # std in default mode, var in comebin mode

    disable_inner_progress = len(bams) > 1
    bam_iterator = (
        tqdm(bams, desc="Processing BAM files") if disable_inner_progress else bams
    )

    base_names = [os.path.splitext(os.path.basename(f))[0] for f in bams]
    sample_names_map = {}

    if len(set(base_names)) == len(base_names):
        for f in bams:
            sample_names_map[f] = os.path.splitext(os.path.basename(f))[0]
    else:
        print(
            "Duplicate alignment filenames detected. "
            "Using parent directory to disambiguate sample names."
        )
        for f in bams:
            base = os.path.splitext(os.path.basename(f))[0]
            parent = os.path.basename(os.path.dirname(f))
            sample_names_map[f] = f"{parent}_{base}"

    for i, bam_file in enumerate(bam_iterator):
        print(
            f"Processing alignment file {i + 1}/{len(bams)}: "
            f"{os.path.basename(bam_file)}"
        )

        disp_suffix = "coverage_var" if mode == "comebin" else "coverage_std"

        if os.path.getsize(bam_file) == 0:
            print(
                f"BAM file {bam_file} is empty. Assigning zero coverage for all fragments."
            )
            zero_cov = pd.DataFrame(
                index=all_fragment_headers,
                columns=[
                    f"{sample_names_map[bam_file]}_coverage",
                    f"{sample_names_map[bam_file]}_{disp_suffix}",
                ],
            ).fillna(0.0)

            mean_series_list.append(zero_cov.iloc[:, 0])
            dispersion_series_list.append(zero_cov.iloc[:, 1])
            continue

        if not os.path.exists(str(bam_file) + ".bai"):
            pysam.index(bam_file)

        with pysam.AlignmentFile(bam_file, "rb") as bam:
            total_mapped = bam.mapped

        coverage, coverage_dispersion = calculate_fragment_coverage(
            bam_file,
            fragments_dict,
            threads,
            bam_batch,
            disable_progress=disable_inner_progress,
            return_variance=(mode == "comebin"),
        )
    
        # mode == "comebin"
        coverage, coverage_dispersion = _normalise_coverage_dicts(
            coverage=coverage,
            coverage_std=coverage_dispersion,
            all_fragment_headers=all_fragment_headers,
            total_mapped=total_mapped,
            rpkm_normalize=False,
        )

        sample_name = sample_names_map[bam_file]

        mean_series = pd.Series(
            coverage,
            name=f"{sample_name}_coverage",
            dtype=float,
        )
        dispersion_series = pd.Series(
            coverage_dispersion,
            name=f"{sample_name}_{disp_suffix}",
            dtype=float,
        )

        dispersion_series = dispersion_series.apply(np.sqrt)

        mean_series_list.append(mean_series)
        dispersion_series_list.append(dispersion_series)

    if mode == "comebin":
        # Match COMEBin ordering style: [vars..., means...]
        all_coverage_series = dispersion_series_list + mean_series_list
    else:
        all_coverage_series = mean_series_list + dispersion_series_list

    coverage_features = pd.concat(all_coverage_series, axis=1)
    coverage_features = coverage_features.reindex(all_fragment_headers).fillna(0.0)

    del mean_series_list, dispersion_series_list
    print(
        f"Coverage calculation complete. Created "
        f"{len(coverage_features.columns)} coverage columns."
    )

    return coverage_features

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