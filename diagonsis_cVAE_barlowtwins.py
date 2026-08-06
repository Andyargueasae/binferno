"""
embedding_fusion.py
===================
Tools for deciding whether (and how) to fuse multiple contig embeddings
(e.g. Barlow-Twins, SimCLR, cVAE-salient) for binning.

Workflow
--------
1. DIAGNOSE: measure whether two embeddings carry complementary signal.
     - knn_agreement : Jaccard overlap of per-contig nearest neighbours.
     - linear_cka    : representational similarity of the two spaces.
   High agreement / high CKA  -> fusion won't help; pick the better single one.
   Low-to-moderate            -> complementary signal exists; fuse.

2. FUSE (two options):
     - early_fusion : per-block L2-normalize (+ optional PCA whiten), concatenate.
                      Simple, single vector, but blends mismatched geometries.
     - late_fusion  : build a kNN graph per embedding in its NATIVE metric,
                      union the graphs, run one community-detection pass.
                      Robust to geometry mismatch; preferred default.

Dependencies: numpy, scikit-learn. (python-igraph + leidenalg optional, for
the Leiden community step; a networkx/greedy fallback is provided.)
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


# --------------------------------------------------------------------------- #
# 1. Diagnostics
# --------------------------------------------------------------------------- #
def knn_agreement(emb_a: np.ndarray,
                  emb_b: np.ndarray,
                  k: int = 15,
                  metric_a: str = "euclidean",
                  metric_b: str = "cosine") -> dict:
    """Mean Jaccard overlap of each contig's k nearest neighbours across two
    embeddings, each queried in its OWN natural metric.

    Returns mean/median Jaccard and the per-contig array.
      ~1.0 -> the two spaces define the same neighbourhoods (redundant).
      ~0.0 -> they define different neighbourhoods (complementary).
    """
    nn_a = NearestNeighbors(n_neighbors=k + 1, metric=metric_a).fit(emb_a)
    nn_b = NearestNeighbors(n_neighbors=k + 1, metric=metric_b).fit(emb_b)
    idx_a = nn_a.kneighbors(emb_a, return_distance=False)[:, 1:]   # drop self
    idx_b = nn_b.kneighbors(emb_b, return_distance=False)[:, 1:]

    jacc = np.empty(len(emb_a))
    for i in range(len(emb_a)):
        sa, sb = set(idx_a[i]), set(idx_b[i])
        union = len(sa | sb)
        jacc[i] = len(sa & sb) / union if union else 0.0

    return {
        "mean_jaccard": float(jacc.mean()),
        "median_jaccard": float(np.median(jacc)),
        "per_contig": jacc,
    }


def linear_cka(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Linear Centered Kernel Alignment between two embeddings (in [0, 1]).
    1.0 = representationally identical up to rotation/scale; near 0 = unrelated.
    Computed in feature space for efficiency (works when n > d)."""
    A = emb_a - emb_a.mean(0, keepdims=True)
    B = emb_b - emb_b.mean(0, keepdims=True)
    # ||A^T B||_F^2 / (||A^T A||_F * ||B^T B||_F)
    hsic = np.linalg.norm(A.T @ B, "fro") ** 2
    norm_a = np.linalg.norm(A.T @ A, "fro")
    norm_b = np.linalg.norm(B.T @ B, "fro")
    return float(hsic / (norm_a * norm_b + 1e-12))


def diagnose(embeddings: dict[str, np.ndarray],
             k: int = 15,
             metrics: dict[str, str] | None = None) -> None:
    """Print pairwise agreement for a set of named embeddings.
    `metrics` maps embedding name -> 'euclidean' | 'cosine' (defaults: cosine)."""
    metrics = metrics or {}
    names = list(embeddings)
    print(f"{'pair':<28}{'kNN Jaccard':>14}{'linear CKA':>14}")
    print("-" * 56)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            ag = knn_agreement(
                embeddings[a], embeddings[b], k=k,
                metric_a=metrics.get(a, "cosine"),
                metric_b=metrics.get(b, "cosine"),
            )
            cka = linear_cka(embeddings[a], embeddings[b])
            print(f"{a + ' vs ' + b:<28}{ag['mean_jaccard']:>14.3f}{cka:>14.3f}")
    print("\nGuide: high (->1) = redundant, skip fusion;  low/moderate = fuse.")


# --------------------------------------------------------------------------- #
# 2a. Early fusion (per-block normalize + optional whiten, then concatenate)
# --------------------------------------------------------------------------- #
def early_fusion(embeddings: list[np.ndarray],
                 weights: list[float] | None = None,
                 whiten_dims: list[int | None] | None = None) -> np.ndarray:
    """Per-block L2-normalize (optionally PCA-whiten to `whiten_dims[i]` first),
    scale by `weights[i]`, then concatenate into one fused vector.

    Per-block normalization is the critical step: without it the block with the
    larger norm dominates every distance.
    """
    n = len(embeddings)
    weights = weights or [1.0] * n
    whiten_dims = whiten_dims or [None] * n
    blocks = []
    for emb, w, d in zip(embeddings, weights, whiten_dims):
        x = emb
        if d is not None:
            x = PCA(n_components=d, whiten=True).fit_transform(x)
        x = normalize(x, norm="l2", axis=1)   # each block on the unit sphere
        blocks.append(w * x)
    return np.concatenate(blocks, axis=1)


# --------------------------------------------------------------------------- #
# 2b. Late fusion (union of per-embedding kNN graphs -> one clustering)
# --------------------------------------------------------------------------- #
def _knn_graph(emb: np.ndarray, k: int, metric: str):
    """Return (row, col, weight) edges of a kNN similarity graph.
    Weight = 1 - distance (cosine) or a Gaussian kernel (euclidean)."""
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(emb)
    dist, idx = nn.kneighbors(emb)
    dist, idx = dist[:, 1:], idx[:, 1:]            # drop self
    rows = np.repeat(np.arange(len(emb)), k)
    cols = idx.reshape(-1)
    d = dist.reshape(-1)
    if metric == "cosine":
        w = 1.0 - d
    else:  # euclidean -> Gaussian kernel with median-distance bandwidth
        sigma = np.median(d) + 1e-9
        w = np.exp(-(d ** 2) / (2 * sigma ** 2))
    return rows, cols, np.clip(w, 0.0, None)


def late_fusion_graph(embeddings: list[np.ndarray],
                      metrics: list[str],
                      k: int = 15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build one kNN graph per embedding in its native metric and UNION them
    (summing edge weights). Returns symmetric (rows, cols, weights)."""
    n_nodes = len(embeddings[0])
    acc: dict[tuple[int, int], float] = {}
    for emb, metric in zip(embeddings, metrics):
        r, c, w = _knn_graph(emb, k, metric)
        for ri, ci, wi in zip(r, c, w):
            a, b = (int(ri), int(ci)) if ri < ci else (int(ci), int(ri))
            if a != b:
                acc[(a, b)] = acc.get((a, b), 0.0) + float(wi)
    edges = np.array(list(acc.keys()))
    wts = np.array(list(acc.values()))
    return edges[:, 0], edges[:, 1], wts


def cluster_graph(rows: np.ndarray, cols: np.ndarray, weights: np.ndarray,
                  n_nodes: int, resolution: float = 1.0,
                  seed: int = 0) -> np.ndarray:
    """Community-detect on the fused graph. Uses Leiden if available, else a
    greedy-modularity fallback via networkx."""
    try:
        import igraph as ig
        import leidenalg as la
        g = ig.Graph(n=n_nodes, edges=list(zip(rows.tolist(), cols.tolist())))
        g.es["weight"] = weights.tolist()
        part = la.find_partition(
            g, la.RBConfigurationVertexPartition,
            weights="weight", resolution_parameter=resolution, seed=seed,
        )
        labels = np.array(part.membership)
    except ImportError:
        import networkx as nx
        from networkx.algorithms.community import greedy_modularity_communities
        g = nx.Graph()
        g.add_nodes_from(range(n_nodes))
        for r, c, w in zip(rows, cols, weights):
            g.add_edge(int(r), int(c), weight=float(w))
        comms = greedy_modularity_communities(g, weight="weight")
        labels = np.empty(n_nodes, dtype=int)
        for cid, nodes in enumerate(comms):
            for node in nodes:
                labels[node] = cid
    return labels


def late_fusion_cluster(embeddings: list[np.ndarray],
                        metrics: list[str],
                        k: int = 15,
                        resolution: float = 1.0) -> np.ndarray:
    """End-to-end late fusion: graph union -> community detection -> bin labels."""
    rows, cols, wts = late_fusion_graph(embeddings, metrics, k=k)
    return cluster_graph(rows, cols, wts, n_nodes=len(embeddings[0]),
                         resolution=resolution)


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, true_k = 600, 6
    centers = rng.normal(scale=6.0, size=(true_k, 8))
    labels_true = rng.integers(0, true_k, size=n)

    # Two "views" of the same contigs: shared genome signal + view-specific noise.
    shared = centers[labels_true]
    bt = np.concatenate([shared, rng.normal(scale=2.0, size=(n, 8))], axis=1)   # BT-like
    simclr = np.concatenate([shared, rng.normal(scale=2.0, size=(n, 8))], axis=1)  # SimCLR-like
    bt += rng.normal(scale=1.0, size=bt.shape)
    simclr += rng.normal(scale=1.0, size=simclr.shape)

    embs = {"barlow_twins": bt, "simclr_cvae": simclr}
    metrics = {"barlow_twins": "euclidean", "simclr_cvae": "cosine"}

    print("=== Diagnostic ===")
    diagnose(embs, k=15, metrics=metrics)

    print("\n=== Early fusion ===")
    fused = early_fusion([bt, simclr], weights=[1.0, 1.0], whiten_dims=[8, 8])
    print("fused shape:", fused.shape)

    print("\n=== Late fusion ===")
    labels = late_fusion_cluster([bt, simclr],
                                 metrics=["euclidean", "cosine"],
                                 k=15, resolution=1.0)
    from sklearn.metrics import adjusted_rand_score
    print("recovered clusters:", len(np.unique(labels)),
          " ARI vs truth:", round(adjusted_rand_score(labels_true, labels), 3))