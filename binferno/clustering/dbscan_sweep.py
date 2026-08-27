"""Density-based clustering sweep for binferno, scored on marker quality.

Standalone (no binferno imports) so it can be unit-tested and reused by the
Leiden module. The clustering module in dbscan.py is a thin wrapper.

Key differences from the first draft:

* Scoring is **lexicographic** ``(n_HQ, n_MQ, sum(comp - lam*cont))``. A flat
  count of HQ bins produces a plateau of exact ties across the whole eps
  grid; with strict ``>`` comparison the first eps wins and `patience`
  aborts the sweep after 4 evaluations. The continuous third term breaks
  ties in the right direction.
* Bins are never dropped from the score. The old saturation bail-out
  returned ``None`` for contaminated bins, so an eps producing one giant
  chimera scored identically to one producing nothing at all.
* The eps grid comes from genuine per-point k-th-neighbour distances, not
  a global order statistic over the flattened CSR data array.
* The kNN graph is symmetrised and carries an explicit diagonal, so a
  point counts toward its own core criterion (required for long-read
  single-contig genomes).
* ``min_samples`` can be expressed in **base pairs** rather than contig
  counts, which is what makes the module portable across assembly sizes.
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import DBSCAN, Birch
from sklearn.neighbors import kneighbors_graph
from typing import Any, Dict, List, Optional, Sequence, Tuple
from rich.progress import track

try:  # in-package
    from .marker_quality import BinQuality, MarkerTable
except ImportError:  # standalone / testing
    from marker_quality import BinQuality, MarkerTable

__all__ = [
    "SweepConfig",
    "SweepResult",
    "build_knn_graph",
    "derive_eps_grid",
    "score_bins",
    "density_sweep",
    "rescue_noise",
    "recruit_unbinned",
]


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
@dataclass
class SweepConfig:
    # quality thresholds
    hq_completeness: float = 0.90
    hq_contamination: float = 0.05
    mq_completeness: float = 0.50
    mq_contamination: float = 0.10
    lam: float = 1.0          # contamination weight in the tie-break term
    smooth_min_completeness: float = 0.30

    # graph
    k: Optional[int] = None            # None -> scaled from N
    k_min: int = 15
    k_max: int = 150
    max_graph_entries: int = 30_000_000  # memory guard: N*k
    metric: str = "cosine"

    # eps grid
    eps_k_values: Tuple[int, ...] = (1, 2, 3, 5, 8, 12, 20, 30)
    eps_quantiles: Tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90)
    max_eps_values: int = 48

    # core-point criterion
    core_bp: Optional[int] = 100_000   # bp within eps for a core point;
                                       # None -> classic contig-count semantics
    min_samples: int = 5               # used only when core_bp is None

    # output filtering
    min_bin_bp: int = 100_000
    min_bin_contigs: int = 1

    # sweep control
    patience: Optional[int] = None     # None -> evaluate the full grid
    min_evals_before_stop: int = 10

    # hdbscan
    hdbscan_min_cluster_sizes: Optional[Sequence[int]] = None
    hdbscan_min_samples: Sequence[Optional[int]] = (None, 5, 15)
    hdbscan_selection_methods: Sequence[str] = ("eom", "leaf")


@dataclass
class SweepResult:
    bins: List[List[int]]
    params: Any
    score: Tuple[float, float, float]
    audit: pd.DataFrame
    qualities: List[BinQuality]
    unbinned: List[int]


# --------------------------------------------------------------------------
# graph
# --------------------------------------------------------------------------
def choose_k(n: int, cfg: SweepConfig) -> int:
    if cfg.k is not None:
        return max(1, min(cfg.k, n - 1))
    k = int(round(5 * math.log2(max(n, 2))))
    k = int(np.clip(k, cfg.k_min, cfg.k_max))
    if n * k > cfg.max_graph_entries:
        k = max(cfg.k_min, cfg.max_graph_entries // max(n, 1))
    return max(1, min(k, n - 1))


def build_knn_graph(
    latent: np.ndarray,
    k: Optional[int] = None,
    metric: str = "cosine",
    n_jobs: int = -1,
    cfg: Optional[SweepConfig] = None,
    add_self_loop: bool = True,
) -> sparse.csr_matrix:
    """Symmetric sparse distance graph with an explicit diagonal.

    Three things the naive ``kneighbors_graph`` call got wrong for a
    precomputed DBSCAN:

    1. The kNN graph is directed. If b is among a's k nearest but a is not
       among b's, DBSCAN sees an asymmetric neighbourhood and core/border
       assignment becomes order-dependent. ``maximum(G, G.T)`` takes the
       union of edges while preserving the true distances.
    2. sklearn treats *stored* entries as the neighbourhood. A genuine
       zero distance (duplicate embeddings, very common for short contigs
       after L2 normalisation) is an implicit "not a neighbour". We floor
       distances at a tiny positive value.
    3. ``include_self=False`` means a point does not count toward its own
       ``min_samples``. With ``sample_weight`` in bp this is fatal: a 5 Mb
       long-read contig could never become a core point on its own.
    """
    cfg = cfg or SweepConfig()
    n = latent.shape[0]
    k = k or choose_k(n, cfg)
    g = kneighbors_graph(
        latent, n_neighbors=k, mode="distance", metric=metric, n_jobs=n_jobs
    ).tocsr()
    g.data = np.maximum(g.data.astype(np.float64), 1e-9)
    g = g.maximum(g.T).tocsr()
    if add_self_loop:
        g = g.tolil()
        g.setdiag(1e-12)
        g = g.tocsr()
    g.sort_indices()
    # sklearn re-sorts a precomputed sparse graph on *every* fit otherwise;
    # across a 48-value eps grid that is 48 redundant sorts of an
    # N*k-element buffer.
    try:
        from sklearn.neighbors import sort_graph_by_row_values
        g = sort_graph_by_row_values(g, copy=False, warn_when_not_sorted=False)
    except Exception:  # noqa: BLE001  (older sklearn)
        pass
    return g


def knn_distance_profile(latent: np.ndarray, k: int, metric: str, n_jobs: int = -1) -> np.ndarray:
    """(n, k) array of sorted per-point neighbour distances, self excluded."""
    g = kneighbors_graph(
        latent, n_neighbors=k, mode="distance", metric=metric, n_jobs=n_jobs
    ).tocsr()
    counts = np.diff(g.indptr)
    if not np.all(counts == k):
        # ragged (shouldn't happen with kneighbors_graph) -- fall back
        out = np.zeros((g.shape[0], k))
        for i in range(g.shape[0]):
            row = np.sort(g.data[g.indptr[i]:g.indptr[i + 1]])
            out[i, : row.size] = row
            if row.size < k and row.size:
                out[i, row.size:] = row[-1]
        return out
    d = g.data.reshape(g.shape[0], k)
    return np.sort(d, axis=1)


def derive_eps_grid(latent: np.ndarray, cfg: SweepConfig, n_jobs: int = -1) -> List[float]:
    """eps candidates from quantiles of the k-th nearest-neighbour distance.

    The original version took ``np.partition(graph.data, n*k-1)``, i.e. a
    global order statistic over the flattened CSR buffer. That is the
    (k/K)-quantile of *all* pairwise kNN distances pooled together, not
    the k-th neighbour distance of a point, so the grid did not adapt to
    a heterogeneous assembly (few very dense genomes + a long tail of
    singletons) at all.
    """
    n = latent.shape[0]
    kmax = min(max(cfg.eps_k_values), max(1, n - 1))
    prof = knn_distance_profile(latent, kmax, cfg.metric, n_jobs=n_jobs)
    eps = set()
    for k in cfg.eps_k_values:
        if k > kmax:
            continue
        dk = prof[:, k - 1]
        for q in cfg.eps_quantiles:
            eps.add(round(float(np.quantile(dk, q)), 6))
    eps = sorted(e for e in eps if e > 1e-6)
    if not eps:
        return []
    if len(eps) > cfg.max_eps_values:
        idx = np.linspace(0, len(eps) - 1, cfg.max_eps_values).round().astype(int)
        eps = [eps[i] for i in sorted(set(idx))]
    return eps


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def score_bins(
    bins: Sequence[Sequence[int]],
    table: MarkerTable,
    cfg: SweepConfig,
) -> Tuple[Tuple[float, float, float], List[BinQuality]]:
    """Lexicographic (n_HQ, n_MQ, smooth) score.

    Returns the score and the per-bin qualities so the caller can write an
    audit table without recomputing.
    """
    quals = table.quality_many(bins)
    n_hq = n_mq = 0
    smooth = 0.0
    for q in quals:
        if q.completeness >= cfg.hq_completeness and q.contamination <= cfg.hq_contamination:
            n_hq += 1
        if q.completeness >= cfg.mq_completeness and q.contamination <= cfg.mq_contamination:
            n_mq += 1
        if q.completeness >= cfg.smooth_min_completeness:
            smooth += q.completeness - cfg.lam * q.contamination
    return (float(n_hq), float(n_mq), float(smooth)), quals


# --------------------------------------------------------------------------
# labels -> bins
# --------------------------------------------------------------------------
def labels_to_bins(
    labels: np.ndarray,
    contig_lengths: np.ndarray,
    cfg: SweepConfig,
) -> Tuple[List[List[int]], List[int]]:
    groups: Dict[int, List[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        if lab != -1:
            groups[int(lab)].append(i)
    bins, dropped = [], []
    for lab in sorted(groups):
        members = sorted(groups[lab])
        bp = int(contig_lengths[members].sum())
        if bp < cfg.min_bin_bp or len(members) < cfg.min_bin_contigs:
            dropped.extend(members)
            continue
        bins.append(members)
    unbinned = sorted([i for i, lab in enumerate(labels) if lab == -1] + dropped)
    # deterministic ordering: biggest bins first
    bins.sort(key=lambda b: (-int(contig_lengths[b].sum()), b[0]))
    return bins, unbinned


# --------------------------------------------------------------------------
# sweep
# --------------------------------------------------------------------------
def _run_dbscan(graph, eps, cfg, lengths, n_jobs):
    if cfg.core_bp is not None:
        # sample_weight in bp: a point is core when the total assembled
        # sequence within eps (including itself) reaches core_bp. This is
        # the assembly-size-portable version of min_samples -- a single
        # 2 Mb long-read contig is core on its own, while a drift of 2 kb
        # fragments is not.
        return DBSCAN(
            eps=eps, min_samples=cfg.core_bp, metric="precomputed", n_jobs=n_jobs
        ).fit_predict(graph, sample_weight=lengths)
    return DBSCAN(
        eps=eps, min_samples=cfg.min_samples, metric="precomputed", n_jobs=n_jobs
    ).fit_predict(graph)


def _hdbscan_grid(n: int, cfg: SweepConfig, longread: bool) -> List[int]:
    if cfg.hdbscan_min_cluster_sizes is not None:
        return list(cfg.hdbscan_min_cluster_sizes)
    if longread:
        # a long-read genome can be 1-3 contigs; min_cluster_size=50 makes
        # every real MAG noise by construction.
        base = [2, 3, 5, 10, 25]
    else:
        base = [5, 10, 25, 50, 100]
    return [c for c in base if c <= max(2, n // 4)] or [2]


def density_sweep(
    latent: np.ndarray,
    contig_lengths: np.ndarray,
    table: MarkerTable,
    cfg: Optional[SweepConfig] = None,
    use_hdbscan: bool = False,
    longread: bool = False,
    do_rescue: bool = True,
    n_jobs: int = -1,
    logger=None,
) -> SweepResult:
    cfg = cfg or SweepConfig()
    latent = np.ascontiguousarray(np.asarray(latent, dtype=np.float32))
    contig_lengths = np.asarray(contig_lengths, dtype=np.float64)
    n = latent.shape[0]

    rows: List[dict] = []
    best_score = (-np.inf, -np.inf, -np.inf)
    best_bins: List[List[int]] = []
    best_params: Any = None
    best_quals: List[BinQuality] = []
    best_labels: Optional[np.ndarray] = None

    def consider(params, labels):
        nonlocal best_score, best_bins, best_params, best_quals, best_labels
        bins, unbinned = labels_to_bins(labels, contig_lengths, cfg)
        if not bins:
            rows.append(dict(params=str(params), n_bins=0, n_hq=0, n_mq=0,
                             smooth=0.0, binned_bp=0, unbinned_frac=1.0))
            return False
        score, quals = score_bins(bins, table, cfg)
        binned_bp = int(sum(q.bp for q in quals))
        rows.append(dict(
            params=str(params), n_bins=len(bins), n_hq=int(score[0]),
            n_mq=int(score[1]), smooth=round(score[2], 4),
            binned_bp=binned_bp,
            unbinned_frac=round(1 - binned_bp / max(contig_lengths.sum(), 1), 4),
        ))
        improved = score > best_score
        if improved:
            best_score, best_bins, best_params = score, bins, params
            best_quals, best_labels = quals, labels
        return improved

    if not use_hdbscan:
        graph = build_knn_graph(latent, metric=cfg.metric, n_jobs=n_jobs, cfg=cfg)
        eps_grid = derive_eps_grid(latent, cfg, n_jobs=n_jobs)
        # or can we directly use latent to run DBSCAN?
        if logger:
            logger.info(
                f"kNN graph: n={n}, k={choose_k(n, cfg)}, nnz={graph.nnz}; "
                f"sweeping {len(eps_grid)} eps values in "
                f"[{eps_grid[0]:.4g}, {eps_grid[-1]:.4g}]" if eps_grid else "empty eps grid"
            )
        n_worse = 0
        for i, eps in enumerate(eps_grid):
            labels = _run_dbscan(graph, eps, cfg, contig_lengths, n_jobs)
            if consider(("eps", eps), labels):
                n_worse = 0
            else:
                n_worse += 1
            if (cfg.patience is not None
                    and i + 1 >= cfg.min_evals_before_stop
                    and n_worse > cfg.patience):
                if logger:
                    logger.info(f"early stop at eps={eps} after {n_worse} non-improving steps")
                break
    else:
        HDB = _import_hdbscan()
        sizes = _hdbscan_grid(n, cfg, longread)

        def run_hdbscan_single(mcs, ms, sel):
            try:
                labels = HDB(
                    min_cluster_size=int(mcs),
                    min_samples=None if ms is None else int(ms),
                    metric="euclidean",
                    cluster_selection_method=sel,
                ).fit_predict(latent)
                return ("mcs", int(mcs), "ms", ms, "sel", sel), labels
            except Exception as e:
                if logger:
                    logger.warning(f"hdbscan({mcs},{ms},{sel}) failed: {e}")
                return None

        cpu_count = os.cpu_count() or 1
        if n_jobs < 0:
            max_workers = max(1, cpu_count + n_jobs)
        else:
            max_workers = n_jobs if n_jobs > 0 else cpu_count
        
        tasks = list(itertools.product(sizes, cfg.hdbscan_min_samples, cfg.hdbscan_selection_methods))

        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all parameter combinations to the thread pool
                future_to_params = {
                    executor.submit(run_hdbscan_single, mcs, ms, sel): (mcs, ms, sel)
                    for mcs, ms, sel in tasks
                }
                
                # Process results as soon as each thread finishes
                for future in track(as_completed(future_to_params), total=len(tasks), description="Sweeping HDBSCAN parameters in parallel..."):
                    result = future.result()
                    if result is not None:
                        params_tuple, labels = result
                        consider(params_tuple, labels)
        else:
            # Fallback to synchronous execution if 1 CPU / worker is specified
            for mcs, ms, sel in track(tasks, description="Sweeping HDBSCAN parameters sequentially..."):
                result = run_hdbscan_single(mcs, ms, sel)
                if result is not None:
                    params_tuple, labels = result
                    consider(params_tuple, labels)

        graph = None

    audit = pd.DataFrame(rows)

    if best_labels is None:
        if logger:
            logger.error(
                "No parameter setting produced any bin passing the size filter. "
                "Check the marker diagnostic and min_bin_bp."
            )
        return SweepResult([], None, best_score, audit, [], list(range(n)))

    _, unbinned = labels_to_bins(best_labels, contig_lengths, cfg)

    if do_rescue and unbinned:
        rescued = rescue_noise(unbinned, latent, contig_lengths, table, cfg)
        if rescued:
            used = {i for b in rescued for i in b}
            best_bins = best_bins + rescued
            unbinned = [i for i in unbinned if i not in used]
            best_quals = table.quality_many(best_bins)
            if logger:
                logger.info(f"rescued {len(rescued)} bins from {len(unbinned) + len(used)} noise contigs")

    best_bins.sort(key=lambda b: (-int(contig_lengths[b].sum()), b[0]))
    best_quals = table.quality_many(best_bins)
    return SweepResult(best_bins, best_params, best_score, audit, best_quals, unbinned)

def _import_hdbscan():
    try:
        from hdbscan import HDBSCAN  # type: ignore
        return HDBSCAN
    except Exception:
        pass
    try:
        from sklearn.cluster import HDBSCAN  # sklearn >= 1.3
        return HDBSCAN
    except Exception as e:  # noqa: BLE001
        raise ImportError(
            "HDBSCAN requested but neither the `hdbscan` package nor "
            "sklearn.cluster.HDBSCAN (sklearn>=1.3) is available."
        ) from e


# --------------------------------------------------------------------------
# noise handling
# --------------------------------------------------------------------------
def rescue_noise(
    noise_indices: Sequence[int],
    latent: np.ndarray,
    contig_lengths: np.ndarray,
    table: MarkerTable,
    cfg: SweepConfig,
) -> List[List[int]]:
    """Recover genomes DBSCAN dumped as noise, without double-assigning.

    The original version looped over BIRCH thresholds and appended every
    passing cluster from every threshold, so one contig could land in
    several output bins simultaneously. Here each threshold only claims
    contigs that are still unclaimed, and thresholds are tried
    coarse-to-fine.
    """
    rescued: List[List[int]] = []
    available = list(dict.fromkeys(int(i) for i in noise_indices))

    # tier 1: singleton rescue (long-read: a genome can be one contig)
    remaining = []
    for ci in available:
        q = table.quality([ci])
        if (q.completeness >= cfg.mq_completeness
                and q.contamination <= cfg.mq_contamination):
            rescued.append([ci])
        else:
            remaining.append(ci)

    # tier 2: BIRCH over what is left
    if len(remaining) > 5:
        sub = latent[remaining]
        k = min(30, len(remaining) - 1)
        prof = knn_distance_profile(sub, k, cfg.metric)
        thresholds = sorted(
            {round(float(np.quantile(prof[:, min(j, k) - 1], 0.5)), 6)
             for j in (5, 15, 30) if j <= k},
            reverse=True,
        )
        claimed: set = set()
        for thr in thresholds:
            if thr < 1e-6:
                continue
            active = [i for i in remaining if i not in claimed]
            if len(active) < 2:
                break
            labels = Birch(threshold=thr, n_clusters=None).fit_predict(latent[active])
            groups: Dict[int, List[int]] = defaultdict(list)
            for local, lab in zip(active, labels):
                groups[int(lab)].append(local)
            for members in groups.values():
                if int(contig_lengths[members].sum()) < cfg.min_bin_bp:
                    continue
                q = table.quality(members)
                if (q.completeness >= cfg.mq_completeness
                        and q.contamination <= cfg.mq_contamination):
                    rescued.append(sorted(members))
                    claimed.update(members)
    return rescued


def recruit_unbinned(
    bins: List[List[int]],
    unbinned: Sequence[int],
    latent: np.ndarray,
    table: MarkerTable,
    cfg: SweepConfig,
    max_contamination_increase: float = 0.02,
    distance_gate_quantile: float = 0.95,
    distance_gate_slack: float = 1.25,
) -> Tuple[List[List[int]], List[int]]:
    """Assign leftover contigs to the nearest bin centroid, subject to two
    independent gates. Mirrors the two-phase (long contigs first, then
    short-contig recruitment) scheme already used in the Leiden branch.

    The contamination gate alone is not sufficient: a contig carrying no
    SCG at all can never raise contamination, so pure junk would be
    hoovered into whichever bin happens to be nearest. The distance gate
    requires the candidate to sit within the bin's own observed radius
    (times a slack factor), which is what actually excludes it.
    """
    if not bins or len(unbinned) == 0:
        return bins, list(unbinned)
    cent = np.vstack([latent[b].mean(axis=0) for b in bins])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-12
    X = latent[list(unbinned)]
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    sims = Xn @ cent.T
    nearest = np.argmax(sims, axis=1)

    # per-bin radius: quantile of member cosine distance to the centroid
    radius = np.empty(len(bins))
    for j, b in enumerate(bins):
        Mb = latent[b]
        Mb = Mb / (np.linalg.norm(Mb, axis=1, keepdims=True) + 1e-12)
        d = 1.0 - Mb @ cent[j]
        radius[j] = float(np.quantile(d, distance_gate_quantile)) * distance_gate_slack

    base = {j: table.quality(bins[j]) for j in range(len(bins))}
    proposals: Dict[int, List[int]] = defaultdict(list)
    rejected_by_distance: List[int] = []
    for row, (ci, j) in enumerate(zip(unbinned, nearest)):
        if (1.0 - sims[row, j]) > radius[j]:
            rejected_by_distance.append(int(ci))
            continue
        proposals[int(j)].append(int(ci))

    out = [list(b) for b in bins]
    still: List[int] = list(rejected_by_distance)
    for j, cand in proposals.items():
        trial = out[j] + cand
        q = table.quality(trial)
        if q.contamination <= base[j].contamination + max_contamination_increase:
            out[j] = sorted(trial)
        else:
            still.extend(cand)
    return out, sorted(still)


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------
def bins_to_dataframe(
    bins: Sequence[Sequence[int]],
    namelist: Sequence[str],
    unbinned: Sequence[int] = (),
    include_unbinned: bool = False,
) -> pd.DataFrame:
    recs = [(namelist[ci], bid) for bid, members in enumerate(bins) for ci in members]
    if include_unbinned:
        recs += [(namelist[ci], -1) for ci in unbinned]
    df = pd.DataFrame(recs, columns=["contig_id", "cluster_id"])
    dup = df["contig_id"].duplicated().sum()
    if dup:
        raise AssertionError(f"{dup} contigs assigned to more than one bin")
    return df


def quality_dataframe(
    bins: Sequence[Sequence[int]],
    quals: Sequence[BinQuality],
    namelist: Sequence[str],
) -> pd.DataFrame:
    return pd.DataFrame([
        dict(cluster_id=i, n_contigs=q.n_contigs, bp=q.bp,
             completeness=round(q.completeness, 4),
             contamination=round(q.contamination, 4),
             n_unique_markers=q.n_unique, n_markers=q.n_markers,
             marker_set=q.marker_set,
             longest_contig=max((namelist[c] for c in b),
                                key=lambda s: len(s)) if b else "")
        for i, (b, q) in enumerate(zip(bins, quals))
    ])