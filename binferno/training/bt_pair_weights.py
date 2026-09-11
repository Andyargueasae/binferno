"""
Helper for preparing the `w` argument to WeightedBTLoss.

`w` is a per-PAIR precision weight derived from the lengths of the two
augmented views in that pair -- not from the parent contig length.

Usage is three lines inside BarlowTwinsTrainer; see the patch notes at the
bottom.  Run this file directly for a self-test on synthetic headers.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Step 1: FragMent_Dict -> {fragment_header: length}
# ---------------------------------------------------------------------------


def build_fragment_length_map(fragment_dict: Dict) -> Dict[str, int]:
    """Flatten FragMent_Dict into an exact-match {fragment_header: length} map.

    Covers every header shape barlow_twins_augmentation can emit:
        {contig}.original          full sequence
        {contig}.{i}               crop from a contig <= 50 kb
        {contig}.h1.{i}            crop from the left half of a long contig
        {contig}.h2.{i}            crop from the right half

    Deliberately does NOT parse headers.  fragment_info is already keyed by the
    exact header string, so lookup is exact and contig names containing dots
    (SPAdes: NODE_1_length_5000_cov_12.345) are handled correctly.
    """
    lengths: Dict[str, int] = {}
    for contig_name, payload in fragment_dict.items():
        contig_len = len(payload.get("sequence", ""))
        if contig_len:
            # `.original` is appended by barlow_twins_augmentation but is not
            # always written into fragment_info, so seed it explicitly.
            lengths[f"{contig_name}.original"] = contig_len
        for frag_header, info in payload.get("fragment_info", {}).items():
            length = info.get("length")
            lengths[frag_header] = int(length) if length else contig_len
    return lengths


# ---------------------------------------------------------------------------
# Step 2: {header: length} + pair indices -> per-pair weights
# ---------------------------------------------------------------------------


def make_pair_weights(
    index: Sequence[str],
    idx1: np.ndarray,
    idx2: np.ndarray,
    length_map: Dict[str, int],
    tau: float = 0.5,
    clip_percentile: Optional[float] = 99.0,
    default_length: int = 1000,
    per_contig_group: Optional[np.ndarray] = None,
    strict: bool = True,
) -> np.ndarray:
    """Precision weight for each positive pair.

    Two views of lengths L1, L2 each estimate the same features with sampling
    variance ~ 1/L.  Barlow Twins compares them, so the noise in the comparison
    has variance ~ (1/L1 + 1/L2) and the precision of the comparison is

        w = L1 * L2 / (L1 + L2)

    which is half the harmonic mean.  This is the pair-level analogue of
    MetaCAT's x_i ~ N(mu, Sigma / w_i).

    Parameters
    ----------
    index
        data_df.index, i.e. the fragment header for every row.
    idx1, idx2
        The two columns of BuildDataset.training_pairs.
    tau
        Tempering exponent.  0 -> uniform (identical to the current loss),
        1 -> full linear precision.  Check N_eff before going above ~0.5.
    clip_percentile
        Cap the raw precision at this percentile before tempering.  Matters
        here specifically because `.original` views of long contigs are orders
        of magnitude longer than crops and would otherwise dominate.
    strict
        Raise if any header is missing from length_map.  Leave this on the
        first time you run it -- a silent fallback to default_length would look
        like the weighting simply not working.

    Returns
    -------
    np.ndarray, float32, shape (n_pairs,), mean 1.0
    """
    index = list(index)
    idx1 = np.asarray(idx1, dtype=np.int64)
    idx2 = np.asarray(idx2, dtype=np.int64)

    missing = [h for h in index if h not in length_map]
    if missing:
        msg = (
            f"{len(missing)} of {len(index)} fragment headers are absent from "
            f"the length map (first few: {missing[:3]}). The map is built from "
            "FragMent_Dict; if these differ, feature_df and FragMent_Dict have "
            "drifted out of sync."
        )
        if strict:
            raise KeyError(msg)

    lens = np.array(
        [float(length_map.get(h, default_length)) for h in index], dtype=np.float64
    )
    lens = np.maximum(lens, 1.0)

    L1, L2 = lens[idx1], lens[idx2]
    prec = (L1 * L2) / (L1 + L2)

    if clip_percentile is not None and clip_percentile < 100.0:
        prec = np.minimum(prec, np.percentile(prec, clip_percentile))

    if tau == 0.0:
        return np.ones_like(prec, dtype=np.float32)

    prec = prec ** float(tau)

    if per_contig_group is not None:
        # Measured beta(tau=0) = 1.17 on real data: pair multiplicity already
        # weights contigs superlinearly in length, past MetaCAT's L^1. Rescale
        # so each contig's pairs sum to their tau=0 total -- tau then only
        # redistributes WITHIN a contig (crop vs .original), which is the part
        # multiplicity cannot express.
        g = np.asarray(per_contig_group, dtype=np.int64)
        assert g.shape == prec.shape, "per_contig_group must be one id per pair"
        tot = np.bincount(g, weights=prec, minlength=g.max() + 1)
        cnt = np.bincount(g, minlength=g.max() + 1)
        prec = prec * (cnt[g] / np.maximum(tot[g], 1e-12))

    prec = prec / prec.mean()
    return prec.astype(np.float32)


def weight_diagnostics(w: np.ndarray, batch_size: int = 1024) -> Dict[str, float]:
    """Report N_eff for a given weight vector, at batch scale.

    N_eff is what tells you whether the weighting has quietly destroyed your
    ability to estimate a proj_head_dim x proj_head_dim cross-correlation.
    Call this BEFORE training, on the full pair weight vector, using a random
    batch-sized sample -- it is a property of the weight distribution, not of
    the model.
    """
    w = np.asarray(w, dtype=np.float64)
    rng = np.random.default_rng(0)
    n_eff_samples = []
    for _ in range(50):
        s = rng.choice(w, size=min(batch_size, len(w)), replace=False)
        s = s / s.sum()
        n_eff_samples.append(1.0 / (s**2).sum())
    n_eff = float(np.mean(n_eff_samples))
    return {
        "n_pairs": int(len(w)),
        "weight_min": float(w.min()),
        "weight_max": float(w.max()),
        "weight_p50": float(np.percentile(w, 50)),
        "weight_p99": float(np.percentile(w, 99)),
        "n_eff_per_batch": n_eff,
        "n_eff_ratio": n_eff / min(batch_size, len(w)),
    }


# ---------------------------------------------------------------------------
# Patch notes: where these three lines go
# ---------------------------------------------------------------------------
#
# (a) barlowtwins.py __call__, right after gen_views (~line 103).
#     fragment_dict is already returned; it is just never used again.
#
#         fragment_dict, all_fragments, namelist = self.gen_views(...)
#         self._frag_lengths = build_fragment_length_map(fragment_dict)
#
# (b) prepare_trainer (~line 498), pass it into BuildDataset:
#
#         dataset_builder = self.BuildDataset(
#             data_df=data_df,
#             fragment_lengths=getattr(self, "_frag_lengths", None),
#             weight_tau=self.weight_tau,      # new pydantic Field, default 0.0
#         )
#
# (c) BuildDataset.__init__ stores them; then in
#     construct_tensordatasets_from_pairs, after idx1/idx2 are formed
#     (~line 446), build the weights and append LAST to every return branch:
#
#         if self.fragment_lengths is not None and self.weight_tau > 0:
#             w = make_pair_weights(
#                 self.data_df.index, idx1, idx2,
#                 self.fragment_lengths, tau=self.weight_tau,
#             )
#             w = torch.from_numpy(w)
#         else:
#             w = torch.ones(len(idx1), dtype=torch.float32)
#
#     ... then `return TensorDataset(features1, features2, base_ids, w)` etc.
#
# (d) train_an_epoch dispatches on len(batch_data).  With w always appended,
#     the three arities become 4 / 6 / 7 instead of 3 / 5 / 6.  Simplest edit
#     that does not disturb the existing branches:
#
#         *batch_data, w = batch_data
#         w = w.to(device, non_blocking=True)
#
#     immediately after the loop header, leaving every `len(batch_data) == 3`
#     test intact.  Then pass w=w through calculate_loss to criterion.
#
# NOTE on the taxonomy branch: calculate_loss calls criterion a second time on
# (tax_enc1, tax_enc2).  Pass the same w -- those are two views of the same
# pair, so the same precision applies.
#
# NOTE on where contig length IS the right weight: at the embedding/clustering
# stage the final vector comes from the `.original` view, so there the parent
# contig length is the correct precision (for graph edge weighting or medoid
# selection). Different stage, different observation.


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     import itertools

#     rng = np.random.default_rng(0)

#     # Synthetic FragMent_Dict mimicking barlow_twins_augmentation, including a
#     # SPAdes-style name with a dot in it and a >50 kb contig that gets halved.
#     contigs = {
#         "k141_43927": 8_000,
#         "NODE_1_length_5000_cov_12.345": 5_000,
#         "k141_99999": 180_000,
#     }
#     frag_dict = {}
#     for name, clen in contigs.items():
#         info = {}
#         if clen > 50_000:
#             for half in ("h1", "h2"):
#                 for i in range(8):
#                     info[f"{name}.{half}.{i}"] = {
#                         "start_pos": 0,
#                         "length": int(rng.integers(1000, clen // 2)),
#                     }
#         else:
#             for i in range(8):
#                 info[f"{name}.{i}"] = {
#                     "start_pos": 0,
#                     "length": int(rng.integers(1000, clen)),
#                 }
#         frag_dict[name] = {
#             "sequence": "A" * clen,
#             "fragments": list(info),
#             "fragment_info": info,
#         }

#     lmap = build_fragment_length_map(frag_dict)

#     # feature_df index = every fragment plus the .original rows
#     index = []
#     for name in contigs:
#         index.append(f"{name}.original")
#         index.extend(frag_dict[name]["fragment_info"].keys())

#     assert all(h in lmap for h in index), "length map does not cover the index"
#     print(f"length map covers {len(index)}/{len(index)} headers")
#     print(f"  .original of k141_99999 -> {lmap['k141_99999.original']:,} bp")
#     print(f"  dotted contig name resolved -> "
#           f"{lmap['NODE_1_length_5000_cov_12.345.original']:,} bp")

#     # pairs within each contig, as BuildDataset builds them
#     groups = {}
#     for r, h in enumerate(index):
#         base = h.rsplit(".original", 1)[0] if h.endswith(".original") else None
#         if base is None:
#             for c in contigs:
#                 if h.startswith(c + "."):
#                     base = c
#                     break
#         groups.setdefault(base, []).append(r)
#     pairs = [p for g in groups.values() for p in itertools.combinations(g, 2)]
#     pairs = np.array(pairs)
#     idx1, idx2 = pairs[:, 0], pairs[:, 1]

#     print()
#     for tau in (0.0, 0.25, 0.5, 1.0):
#         w = make_pair_weights(index, idx1, idx2, lmap, tau=tau)
#         d = weight_diagnostics(w, batch_size=min(1024, len(w)))
#         print(
#             f"tau={tau:<5} max/min={d['weight_max']/d['weight_min']:8.1f}  "
#             f"N_eff={d['n_eff_per_batch']:6.1f} ({d['n_eff_ratio']:.1%})"
#         )

#     # strict mode should fire on a header the map has never seen
#     try:
#         make_pair_weights(index + ["ghost.7"], idx1, idx2, lmap, strict=True)
#     except KeyError:
#         print("\nstrict=True correctly rejected an unmapped header")
