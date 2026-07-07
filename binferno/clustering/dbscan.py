import os
from pathlib import Path
from collections import defaultdict
import re
from typing import Any, Tuple

import pandas as pd
import loguru
import numpy as np
from pydantic import Field
from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN, Birch
from sklearn.neighbors import kneighbors_graph

from .base import BaseClustering, ClusterResult
from ..utils import gen_seed
from .cluster_utils import (
    _filter_embeddings_by_length,
    seed_kmeans_full,
    _read_seed_count,
)


class DbscanClustering(BaseClustering): # add the choice of HDBSCAN in addition to DBSCAN.
    hdbscan: bool = Field(
        default=False,
        description="Whether to use HDBSCAN instead of DBSCAN for clustering.",
    )
    def __call__(
        self,
        contigs: Path,
        output_path: Path,
        embeddings: pd.DataFrame,
        logger: loguru._logger.Logger,
        threads: int = 1,
        plastid: bool = False,
    ) -> ClusterResult:

        result = ClusterResult(
            n_genome=0,
            clusters={},
        )

        if plastid:
            n_genome = gen_seed(
                logger,
                contig_file=contigs,
                output_dir=output_path,
                threads=threads,
                contig_length_threshold=1001,
                marker_name="pltd_marker",
                quarter="2quarter",
            )
            seed_file = os.path.join(
                output_path,
                f"{os.path.basename(str(contigs))}.pltd_marker.2quarter_lencutoff_1001.seed",
            )
        else:
            n_genome = gen_seed(
                logger,
                contig_file=contigs,
                output_dir=output_path,
                threads=threads,
                contig_length_threshold=1001,
                marker_name="bacar_marker",
                quarter="2quarter",
            )
            seed_file = os.path.join(
                output_path,
                f"{os.path.basename(str(contigs))}.bacar_marker.2quarter_lencutoff_1001.seed",
            )

        result.n_genome = n_genome

        if embeddings is None or embeddings.empty:
            raise ValueError("embeddings must be a non-empty DataFrame")
        if seed_file is None or contigs is None or output_path is None:
            raise ValueError("seed_file, contigs, and output_path are required")

        logger.info("Start clustering.")

        output_path.mkdir(parents=True, exist_ok=True)

        emb_mat, namelist, length_weight = _filter_embeddings_by_length(
            emb_df=embeddings,
            contig_file=contigs,
            contig_len=self.contig_len,
        )

        logger.info(f"Contigs retained after length filter: {len(namelist)}")

        norm_embeddings = normalize(emb_mat).astype(np.float32, copy=False)

        # your part to run these things.
        marker_name = "pltd_marker" if plastid else "bacar_marker"
        hmmout_path = _resolve_hmmout_path(contigs, output_path, marker_name)
        if not os.path.exists(hmmout_path):
            raise FileNotFoundError(
                f"Expected hmm output at {hmmout_path}; check gen_seed's "
                f"output naming and update _resolve_hmmout_path."
            )
        contig_to_markers, n_scgs = parse_contig_to_markers(hmmout_path)
        # contigs with zero marker hits simply won't be keys; the scorer
        # must treat a missing key as an empty list:
        contig_to_markers = {
            **{name: [] for name in namelist},
            **contig_to_markers,
        }

        logger.info(
            f"Parsed markers for {sum(1 for v in contig_to_markers.values() if v)} "
            f"contigs over {n_scgs} SCGs"
        )

        # KMeans extablish the initial run to be used in post processing later.
        seed_num = _read_seed_count(seed_file)
        mode = f"weight_seed_kmeans"

        bin_nums = seed_num
        logger.info(f"Bin_numbers:\t{bin_nums}")

        kmeans_out = seed_kmeans_full(
            logger=logger,
            contig_file=contigs,
            namelist=namelist,
            out_path=str(output_path) + os.sep,
            X_mat=norm_embeddings,
            bin_number=bin_nums,
            prefix=mode,
            length_weight=length_weight,
            seed_file=seed_file,
        )
        for key, val in kmeans_out.items():
            result.clusters[key] = val

        # --- DBSCAN sweep, marker-quality-only scoring, with noise rescue ---
        best_bins, best_score, eps_grid = dbscan_marker_sweep(
            latent=norm_embeddings,
            contig_lengths=length_weight.astype(np.int64),
            contig_to_markers=contig_to_markers,
            namelist=namelist,
            n_scgs=n_scgs,
            K_seed=n_genome,  # sanity check only, per your choice
            min_samples=1,  # long-read: see note
            metric="cosine",
            patience=3,
            scoring="count",  # Path A; switch to "smooth" to compare
            do_rescue=True,  # long-read singleton + BIRCH rescue
            logger=logger,
            hdbscan=self.hdbscan
        )
        logger.info(
            f"DBSCAN sweep: {len(best_bins)} bins, score={best_score:.3f}, "
            f"swept {len(eps_grid)} eps values; gen_seed K={n_genome}"
        )

        # turn best_bins into a dataframe of contig_name -> bin_label, where bin_label is an integer from 0 to len(best_bins)-1.
        best_bin_df = pd.DataFrame(
            [
                (namelist[ci], bin_label)
                for bin_label, bin_indices in enumerate(best_bins)
                for ci in bin_indices
            ],
            columns=["contig_id", "cluster_id"],
        )
        # save a copy of this file for later use in post-processing.
        best_bin_df.to_csv(output_path / "dbscan_clusters.tsv", index=False, sep="\t")
        result.clusters["dbscan_clusters.tsv"] = best_bin_df

        with open(output_path / "dbscan_eps_grid.txt", "w") as f:
            for eps in eps_grid:
                f.write(f"{eps}\n")

        with open(output_path / "cluster_result.pkl", "wb") as f:
            import pickle

            pickle.dump(result, f)

        logger.info(
            "Clustering completed. Saved results and epsilon grid to output directory."
        )
        return result


def completeness_contamination(counts: np.ndarray, n_scgs: int) -> Tuple[float, float]:
    n_total = counts.sum()
    n_unique = (counts > 0).sum()
    return n_unique / n_scgs, (n_total - n_unique) / n_scgs


def count_markers_saturated(
    contig_indices: list, contig_to_markers: dict, namelist: list, n_scgs: int
):
    """Count SCGs in a bin; bail early once hopelessly contaminated."""
    counts = np.zeros(n_scgs, dtype=np.int32)
    n_markers = n_unique = 0
    max_dup = n_scgs  # contamination == 1.0
    for ci in contig_indices:
        for gene_id in contig_to_markers[namelist[ci]]:
            existing = counts[gene_id]
            n_unique += existing == 0
            counts[gene_id] = existing + 1
            n_markers += 1
            if (n_markers - n_unique) > max_dup:
                return None
    return counts


def score_count(
    bins: list,
    contig_to_markers: dict,
    namelist: list,
    n_scgs: int,
    min_completeness: float = 0.75,
    max_contamination: float = 0.30,
):
    """PATH A: hard count of near-complete, low-contamination bins."""
    good = 0
    for b in bins:
        counts = count_markers_saturated(b, contig_to_markers, namelist, n_scgs)
        if counts is None:
            continue
        comp, cont = completeness_contamination(counts, n_scgs)
        if comp >= min_completeness and cont <= max_contamination:
            good += 1
    return float(good)


def score_smooth(
    bins: list,
    contig_to_markers: dict,
    namelist: list,
    n_scgs: int,
    lam: float = 1.0,
    min_completeness: float = 0.50,
):
    """PATH B: sum of (completeness - lam*contamination) over qualifying bins.

    Smoother than the hard count: no 0.75/0.30 cliff. A loose
    min_completeness gate keeps junk bins from contributing noise, but
    the reward scales continuously so the sweep won't over-split just to
    manufacture more bins that barely clear a threshold.
    """
    total = 0.0
    for b in bins:
        counts = count_markers_saturated(b, contig_to_markers, namelist, n_scgs)
        if counts is None:
            continue
        comp, cont = completeness_contamination(counts, n_scgs)
        if comp >= min_completeness:
            total += comp - lam * cont
    return total


def build_knn_graph(
    latent: np.ndarray, k: int = None, metric: str = "cosine", n_jobs: int = -1
):
    N = latent.shape[0]
    k = k or min(200, N - 1)
    return kneighbors_graph(
        latent, n_neighbors=k, mode="distance", metric=metric, n_jobs=n_jobs
    )


def derive_eps_grid(
    graph: Any,
    n_points: int,
    k_steps: range = range(1, 80),
    extra: Tuple = (0.12, 0.15, 0.2),
):
    data = graph.data
    eps = []
    for k in k_steps:
        idx = n_points * k - 1
        if idx >= len(data):
            break
        v = np.partition(data, idx)[idx]
        eps.append(round(0.7 * v, 4))
        eps.append(round(1.2 * v, 4))
    eps = sorted(set(eps) | set(extra))
    return [e for e in eps if e > 1e-3]


def rescue_noise(
    noise_indices,
    latent,
    contig_to_markers,
    namelist,
    n_scgs,
    min_completeness=0.50,
    max_contamination=0.30,
):
    """Recover genomes DBSCAN dumped as noise (label -1).

    Two-tier: (1) emit any single noise contig that alone passes the
    marker check — on long-read assemblies a near-complete genome can be
    one contig, and min_samples would have discarded it. (2) BIRCH the
    remaining noise (LorBin-style) and keep clusters that pass.
    """
    rescued = []
    remaining = []

    # tier 1: singleton rescue
    for ci in noise_indices:
        counts = count_markers_saturated([ci], contig_to_markers, namelist, n_scgs)
        if counts is not None:
            comp, cont = completeness_contamination(counts, n_scgs)
            if comp >= min_completeness and cont <= max_contamination:
                rescued.append({ci})
                continue
        remaining.append(ci)

    # tier 2: BIRCH over leftover noise
    if len(remaining) > 5:
        sub = latent[remaining]
        # threshold grid from sub-distances; cheap since noise set is small
        g = build_knn_graph(sub, k=min(50, len(remaining) - 1))
        data = g.data
        thresholds = sorted(
            {
                round(
                    np.partition(data, len(remaining) * k - 1)[len(remaining) * k - 1],
                    4,
                )
                for k in (5, 15, 30)
                if len(remaining) * k - 1 < len(data)
            }
        )
        for thr in thresholds:
            if thr < 1e-5:
                continue
            labels = Birch(threshold=thr, n_clusters=None).fit_predict(sub)
            bins = defaultdict(set)
            for local_i, lab in zip(remaining, labels):
                bins[lab].add(local_i)
            for b in bins.values():
                counts = count_markers_saturated(b, contig_to_markers, namelist, n_scgs)
                if counts is None:
                    continue
                comp, cont = completeness_contamination(counts, n_scgs)
                if comp >= min_completeness and cont <= max_contamination:
                    rescued.append(b)
    return rescued


def dbscan_marker_sweep(
    latent: pd.DataFrame,
    contig_lengths: list,
    contig_to_markers: dict,
    namelist: list,
    n_scgs: int,
    K_seed: int = None,
    min_samples: int = 5,
    metric: str = "cosine",
    patience: int = 3,
    scoring: str = "count",  # "count" (Path A) or "smooth" (Path B)
    lam: float = 1.0,
    do_rescue: bool = False,  # noise-rescue pass on the winning clustering
    logger: loguru._logger.Logger = None,
    hdbscan: bool = False,  # Whether to use HDBSCAN instead of DBSCAN
):
    graph = build_knn_graph(latent, metric=metric)
    eps_grid = derive_eps_grid(graph, latent.shape[0])

    if scoring == "count":
        score_fn = lambda bins: score_count(bins, contig_to_markers, namelist, n_scgs)
    elif scoring == "smooth":
        score_fn = lambda bins: score_smooth(
            bins, contig_to_markers, namelist, n_scgs, lam=lam
        )
    else:
        raise ValueError(f"unknown scoring: {scoring}")

    best_bins, best_eps, best_score = None, None, -np.inf
    n_worse = 0

    for eps in eps_grid:
        labels = DBSCAN(
            eps=eps, min_samples=min_samples, metric="precomputed", n_jobs=-1
        ).fit_predict(graph, sample_weight=contig_lengths)

        bins = defaultdict(set)
        for idx, lab in enumerate(labels):
            if lab != -1:
                bins[lab].add(idx)
        bins = list(bins.values())
        if not bins:
            continue

        score = score_fn(bins)
        if score > best_score:
            best_score, best_bins, best_eps = score, bins, eps
            n_worse = 0
        else:
            n_worse += 1
            if n_worse > patience:
                break

    if best_bins is None:
        return [], best_score, eps_grid

    # noise rescue on the winning eps only (re-run to recover its labels)
    if do_rescue:
        labels = DBSCAN(
            eps=best_eps, min_samples=min_samples, metric="precomputed", n_jobs=-1
        ).fit_predict(graph, sample_weight=contig_lengths)
        noise = [i for i, lab in enumerate(labels) if lab == -1]
        if noise:
            rescued = rescue_noise(
                noise, latent, contig_lengths, contig_to_markers, namelist, n_scgs
            )
            if logger:
                logger.info(
                    f"rescued {len(rescued)} bins from {len(noise)} noise contigs"
                )
            best_bins = best_bins + rescued

    if logger and K_seed is not None:
        logger.info(
            f"best eps={best_eps}, bins={len(best_bins)}, "
            f"score={best_score:.3f}, K_seed={K_seed}"
        )

    return best_bins, best_score, eps_grid


def parse_contig_to_markers(hmmout_path: str):
    """Parse HMMER tblout (or domtblout) produced during gen_seed into
    {contig_name: [marker_id, ...]} and the marker-id vocabulary.

    ASSUMPTIONS — verify on one real run:
      * file is hmmsearch --tblout format (whitespace-delimited, '#'-comments)
      * column 0 = target ORF id, column 2 (or 3) = query HMM name
      * ORF id is '<contig>_<start>_<end>_<strand>' (FragGeneScan).
        We strip the trailing 3 underscore-fields to recover the contig.
    Adjust the two marked lines if your gen_seed differs.
    """
    marker_vocab = {}  # marker name -> integer id
    contig_to_markers = {}

    orf_suffix = re.compile(r"_\d+_\d+_[+-]$")  # FragGeneScan tail

    with open(hmmout_path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            cols = line.split()
            orf_id = cols[0]  # <-- ADJUST if needed
            hmm_name = cols[2]  # <-- ADJUST (tblout col 3 = query name)
            contig = orf_suffix.sub("", orf_id)  # <-- ADJUST split if not FGS
            mid = marker_vocab.setdefault(hmm_name, len(marker_vocab))
            contig_to_markers.setdefault(contig, []).append(mid)

    n_scgs = len(marker_vocab)
    return contig_to_markers, n_scgs


def _resolve_hmmout_path(contigs: Path, output_path: Path, marker_name: str) -> str:
    """Mirror gen_seed's naming. VERIFY against an actual gen_seed run."""
    base = os.path.basename(str(contigs))
    # common COMEBin pattern; correct the suffix if yours differs
    candidate = os.path.join(str(output_path), f"{base}.{marker_name}.hmmout")
    return candidate


def write_bins(
    best_bins: list,
    namelist: list,
    contig_seqs: dict,
    out_dir: str,
    min_bin_size: int = 1,
):
    os.makedirs(out_dir, exist_ok=True)
    written = 0
    for i, bin_indices in enumerate(best_bins):
        names = [namelist[ci] for ci in bin_indices]
        if sum(len(contig_seqs[n]) for n in names) < min_bin_size:
            continue
        with open(os.path.join(out_dir, f"bin.{i}.fa"), "w") as fh:
            for n in names:
                fh.write(f">{n}\n{contig_seqs[n]}\n")
        written += 1
    return written
