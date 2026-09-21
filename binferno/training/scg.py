"""SCG-guided cannot-link objective shared by contrastive trainers."""

import itertools
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..clustering.marker_quality import parse_hmmout, resolve_orf_to_contig
from ..utils import run_hmmsearch_pyhmmer, run_pyrodigal


class SCGMarginLoss(nn.Module):
    """Soft cannot-link loss for contigs carrying the same SCG.

    This consumes the representation that will be clustered, so it is
    independent of the contrastive objective and projection head.
    """

    def __init__(self, margin: float = 0.4):
        super().__init__()
        if not -1.0 <= margin <= 1.0:
            raise ValueError("margin must be between -1 and 1")
        self.margin = float(margin)

    def forward_pairs(self, embedding1, embedding2, weights=None):
        similarity = F.cosine_similarity(embedding1, embedding2, dim=1)
        losses = F.relu(similarity - self.margin).square()
        if weights is None:
            return losses.mean()
        weights = weights.to(device=losses.device, dtype=losses.dtype)
        return (losses * weights).sum() / weights.sum().clamp_min(1e-12)

    def forward(self, embeddings, marker_ids, contig_ids, pair_weights=None):
        embeddings = F.normalize(embeddings, dim=1)
        similarities = embeddings @ embeddings.T
        pair_mask = marker_ids[:, None].eq(marker_ids[None, :])
        pair_mask &= contig_ids[:, None].ne(contig_ids[None, :])
        pair_mask &= torch.triu(
            torch.ones_like(pair_mask, dtype=torch.bool), diagonal=1
        )
        if not pair_mask.any():
            return embeddings.sum() * 0.0
        losses = F.relu(similarities[pair_mask] - self.margin).square()
        if pair_weights is None:
            return losses.mean()
        pair_weights = pair_weights.to(device=losses.device, dtype=losses.dtype)
        weights = (pair_weights[:, None] * pair_weights[None, :])[pair_mask]
        return (losses * weights).sum() / weights.sum().clamp_min(1e-12)


class SCGPairDataset(Dataset):
    """Memory-efficient view of same-marker pairs and optional side features."""

    def __init__(
        self,
        features,
        pairs,
        marker_ids,
        contig_ids,
        weights,
        auxiliary_features=None,
    ):
        self.features = torch.from_numpy(features.astype(np.float32, copy=False))
        self.pairs = torch.from_numpy(pairs.astype(np.int64, copy=False))
        self.marker_ids = torch.from_numpy(marker_ids.astype(np.int64, copy=False))
        self.contig_ids = torch.from_numpy(contig_ids.astype(np.int64, copy=False))
        self.weights = torch.from_numpy(weights.astype(np.float32, copy=False))
        self.auxiliary_features = (
            None
            if auxiliary_features is None
            else torch.from_numpy(auxiliary_features.astype(np.float32, copy=False))
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        left, right = self.pairs[index]
        batch = [self.features[left], self.features[right]]
        if self.auxiliary_features is not None:
            batch.extend(
                [self.auxiliary_features[left], self.auxiliary_features[right]]
            )
        batch.extend(
            [
                self.marker_ids[index],
                self.contig_ids[index, 0],
                self.contig_ids[index, 1],
                self.weights[index],
            ]
        )
        return tuple(batch)


def build_scg_groups(
    contigs: Path,
    known_contigs: Sequence[str],
    output_dir: Path,
    threads: int,
    marker_hmm: Path,
    max_evalue: float,
    min_hmm_coverage: float,
    logger,
) -> Dict[str, List[str]]:
    """Run a cached prokaryotic SCG search and return trusted marker groups."""
    marker_hmm = Path(marker_hmm)
    if not marker_hmm.is_file():
        raise FileNotFoundError(f"SCG marker HMM does not exist: {marker_hmm}")
    output_dir.mkdir(parents=True, exist_ok=True)
    faa_path = output_dir / f"{Path(contigs).name}.scg.faa"
    hmmout_path = output_dir / f"{Path(contigs).name}.scg.domtblout"
    if not faa_path.exists():
        run_pyrodigal(contig_file=contigs, faa_path=faa_path, threads=threads)
    if not hmmout_path.exists() or hmmout_path.stat().st_size == 0:
        run_hmmsearch_pyhmmer(
            hmm_path=marker_hmm,
            faa_path=faa_path,
            domtblout_path=hmmout_path,
            threads=threads,
            e_value=max_evalue,
        )
    full_length_hits = set()
    with hmmout_path.open() as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            columns = line.split()
            if len(columns) < 17:
                continue
            try:
                hmm_length = float(columns[5])
                hmm_from, hmm_to = int(columns[15]), int(columns[16])
            except ValueError:
                continue
            coverage = (hmm_to - hmm_from + 1) / max(hmm_length, 1.0)
            if coverage >= min_hmm_coverage:
                full_length_hits.add((columns[0], columns[3]))
    hits = parse_hmmout(hmmout_path, max_evalue=max_evalue, best_hit_per_orf=False)
    hits = [hit for hit in hits if (hit.orf_id, hit.hmm_name) in full_length_hits]
    best_by_orf = {}
    for hit in hits:
        previous = best_by_orf.get(hit.orf_id)
        if previous is None or hit.score > previous.score:
            best_by_orf[hit.orf_id] = hit
    hits = list(best_by_orf.values())
    if not hits:
        raise ValueError("SCG search produced no hits passing the HMM coverage threshold")
    orf_to_contig, join_report, _ = resolve_orf_to_contig(
        [hit.orf_id for hit in hits], known_contigs
    )
    known_contig_set = set(known_contigs)
    hit_counts = Counter(
        (orf_to_contig[hit.orf_id], hit.hmm_name)
        for hit in hits
        if orf_to_contig[hit.orf_id] in known_contig_set
    )
    groups = defaultdict(list)
    for (contig_id, marker_id), count in hit_counts.items():
        if count == 1:
            groups[marker_id].append(contig_id)
    groups = {
        marker: sorted(set(contig_ids))
        for marker, contig_ids in groups.items()
        if len(set(contig_ids)) >= 2
    }
    if not groups:
        raise ValueError(
            "SCG search produced no marker shared by at least two single-copy contigs"
        )
    logger.info(
        f"SCG search: {len(hits)} accepted hits, {len(groups)} usable marker "
        f"groups, ORF join {join_report.rate:.1%}."
    )
    return groups


def prepare_scg_loader(
    data_df: pd.DataFrame,
    marker_groups: Dict[str, List[str]],
    original_suffix: str,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
    logger=None,
    worker_init_fn=None,
    auxiliary_df: Optional[pd.DataFrame] = None,
) -> DataLoader:
    """Construct same-marker pairs from a trainer's original-view rows."""
    original_rows = [
        row for row, name in enumerate(data_df.index) if name.endswith(original_suffix)
    ]
    original_names = [
        data_df.index[row][: -len(original_suffix)] for row in original_rows
    ]
    contig_to_row = {name: row for row, name in enumerate(original_names)}
    contig_to_id = {name: idx for idx, name in enumerate(original_names)}
    usable_groups = {
        marker: sorted({name for name in names if name in contig_to_row})
        for marker, names in marker_groups.items()
    }
    usable_groups = {
        key: value for key, value in usable_groups.items() if len(value) >= 2
    }
    memberships = Counter(name for names in usable_groups.values() for name in names)
    rng = random.Random(seed)
    pairs, marker_ids, contig_ids, weights = [], [], [], []
    for marker_id, marker in enumerate(sorted(usable_groups)):
        names = usable_groups[marker]
        if len(names) > batch_size:
            names = sorted(rng.sample(names, batch_size))
        for left, right in itertools.combinations(names, 2):
            pairs.append((contig_to_row[left], contig_to_row[right]))
            marker_ids.append(marker_id)
            contig_ids.append((contig_to_id[left], contig_to_id[right]))
            weights.append(1.0 / (memberships[left] * memberships[right]))
    if not pairs:
        raise ValueError("No SCG cannot-link pairs overlap the training features")
    auxiliary_features = None
    if auxiliary_df is not None:
        if not auxiliary_df.index.equals(data_df.index):
            raise ValueError("SCG auxiliary features must align with canonical features")
        auxiliary_features = auxiliary_df.iloc[original_rows].to_numpy(
            dtype=np.float32, copy=True
        )
    dataset = SCGPairDataset(
        features=data_df.iloc[original_rows].to_numpy(dtype=np.float32, copy=True),
        pairs=np.asarray(pairs),
        marker_ids=np.asarray(marker_ids),
        contig_ids=np.asarray(contig_ids),
        weights=np.asarray(weights),
        auxiliary_features=auxiliary_features,
    )
    if logger is not None:
        logger.info(
            f"SCG loader: {len(dataset)} cannot-link pairs from "
            f"{len(usable_groups)} marker groups."
        )
    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        num_workers=max(0, num_workers),
        worker_init_fn=worker_init_fn,
        pin_memory=torch.cuda.is_available(),
    )
