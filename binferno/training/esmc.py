"""Frozen ESM-C protein embeddings and trainable coordinate-aware view pooling.

Each protein is represented by the mean of its residue embeddings (excluding
special tokens). Long proteins are chunked without discarding residues. Each
view pools its complete genomic ORFs using a mean or learned query attention.
Only the pooling/projection is trained, using the trainer's existing objective.
"""

from dataclasses import dataclass
import re
import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from Bio import SeqIO
from pydantic import BaseModel, Field
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch.nn.functional as F

from ..clustering.marker_quality import resolve_orf_to_contig
from .scg import prepare_gene_annotations


@dataclass
class ProteinLatentTable:
    """Frozen protein rows with zero-based, half-open genomic coordinates.

    Unknown coordinates use -1 and are allowed only in shared augmentation.
    """
    proteins: np.ndarray
    offsets: np.ndarray
    orf_starts: np.ndarray
    orf_ends: np.ndarray
    orf_strands: np.ndarray

    def __iter__(self):
        # Preserve unpacking for callers that only need the frozen table.
        return iter((self.proteins, self.offsets))


def _orf_coordinates(record):
    # Pyrodigal write_translations retains Prodigal's coordinate-bearing header:
    # >contig_1 # begin # end # strand # attributes (one-based inclusive).
    match = re.search(r" # (\d+) # (\d+) # (-?1)(?: #|$)", record.description)
    if match is None:
        return -1, -1, 0
    begin, end, strand = map(int, match.groups())
    if begin < 1 or end < begin:
        raise ValueError(f"Invalid ORF coordinates for {record.id!r}")
    return begin - 1, end, strand


class ESMCConfig(BaseModel):
    enabled: bool = Field(default=False, description="Include the ESM-C protein modality.")
    model: Literal["esmc_300m", "esmc_600m"] = "esmc_300m"
    pooling: Literal["mean", "attention"] = "attention"
    fusion: Literal["residual", "concat"] = Field(
        default="concat",
        description="Late protein residual or concatenation before a joint MLP.",
    )
    augmentation: Literal["shared", "coordinate"] = Field(
        default="coordinate", description="Pool complete ORFs in each fragment; shared is the baseline."
    )
    num_heads: int = Field(default=4, ge=1)
    weight: float = Field(default=0.2, gt=0.0, description="Protein residual weight.")
    batch_size: int = Field(default=4, ge=1, description="Protein chunks per ESM-C batch.")
    max_residues: int = Field(default=1022, ge=1, le=2046)
    device: str = Field(default="auto", description="ESM-C inference device; auto, cpu or cuda.")


def _load_model(config, device):
    # Lazy import: ordinary training and cached extraction do not require esm.
    try:
        from esm.models.esmc import ESMC
    except ImportError as exc:
        raise ImportError(
            "ESM-C requires the optional dependency: pip install 'binferno[esmc]'. "
            "Use the EvolutionaryScale esm package, not fair-esm."
        ) from exc
    return ESMC.from_pretrained(config.model, device=device).eval().requires_grad_(False)


def _embed_proteins(sequences, config, device):
    """Batch residue chunks and return one equally weighted latent per protein."""
    model = _load_model(config, device)
    sums = [None] * len(sequences)
    counts = np.zeros(len(sequences), dtype=np.int64)
    pending = []

    def flush(): # where the actual embedding happens, and the latent sum is calculated for each protein
        tokens = model.tokenizer(
            [sequence for _, sequence in pending], padding=True, return_tensors="pt"
        )["input_ids"].to(device)
        with torch.inference_mode():
            output = model(sequence_tokens=tokens).embeddings
            for row, (protein_id, sequence) in enumerate(pending):
                # ESM-C adds BOS/EOS. Padding and both special tokens are excluded.
                latent_sum = output[row, 1 : len(sequence) + 1].float().sum(0).cpu()
                if sums[protein_id] is None:
                    sums[protein_id] = latent_sum
                else:
                    sums[protein_id] += latent_sum
                counts[protein_id] += len(sequence)
        pending.clear()

    for protein_id, sequence in enumerate(sequences):
        for start in range(0, len(sequence), config.max_residues):
            pending.append((protein_id, sequence[start : start + config.max_residues]))
            if len(pending) == config.batch_size:
                flush()
    if pending:
        flush()
    return torch.stack([total / count for total, count in zip(sums, counts)]).numpy() # return the mean latent for each protein


def build_protein_latents(contigs, known_contigs, output_dir, threads, config, logger=None):
    """Return a ragged protein table in known_contigs order, with a content cache."""
    contigs, output_dir = Path(contigs), Path(output_dir)
    names = list(known_contigs)
    if not names or len(set(names)) != len(names):
        raise ValueError("ESM-C needs nonempty, unique contig names")
    faa_path = prepare_gene_annotations(contigs, output_dir, threads)
    fingerprint = hashlib.sha256()
    with faa_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            fingerprint.update(block)
    fingerprint.update(json.dumps([2, config.model, config.max_residues, names]).encode())
    cache_path = output_dir / f"esmc_{fingerprint.hexdigest()}.npz"
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cache:
            return ProteinLatentTable(*(cache[key] for key in (
                "proteins", "offsets", "orf_starts", "orf_ends", "orf_strands"
            )))

    records = list(SeqIO.parse(faa_path, "fasta"))
    groups = [[] for _ in names]
    name_to_id = {name: index for index, name in enumerate(names)}
    if records:
        # Resolve against the entire assembly: length-filtered training contigs
        # may be a small subset of all predicted genes.
        assembly_names = [record.id for record in SeqIO.parse(contigs, "fasta")]
        mapping, report, _ = resolve_orf_to_contig(
            [record.id for record in records], assembly_names
        )
        for record in records:
            contig_id = name_to_id.get(mapping[record.id])
            if contig_id is None:
                continue
            sequence = str(record.seq).upper().rstrip("*")
            sequence = "".join(aa if aa in "ACDEFGHIKLMNPQRSTVWYBXZUO" else "X" for aa in sequence)
            if sequence:
                groups[contig_id].append((sequence, _orf_coordinates(record)))
        if logger is not None:
            logger.info(f"ESM-C: ORF-to-contig join {report.rate:.1%}.")
    entries = [entry for group in groups for entry in group]
    sequences = [entry[0] for entry in entries]
    coordinates = np.asarray([entry[1] for entry in entries], dtype=np.int64).reshape(-1, 3)
    offsets = np.concatenate(([0], np.cumsum([len(group) for group in groups]))) # offsets are used to know where each contig's proteins start and end in the flattened sequences list
    if sequences:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if config.device == "auto" else torch.device(config.device)
        proteins = _embed_proteins(sequences, config, device)
    else:
        dim = 960 if config.model == "esmc_300m" else 1152
        proteins = np.empty((0, dim), dtype=np.float32)
    # Atomic replacement prevents an interrupted extraction leaving a valid-looking cache.
    temporary = cache_path.with_suffix(".tmp.npz")
    np.savez(temporary, proteins=proteins, offsets=offsets,
             orf_starts=coordinates[:, 0], orf_ends=coordinates[:, 1],
             orf_strands=coordinates[:, 2])
    temporary.replace(cache_path)
    if logger is not None:
        logger.info(
            f"ESM-C: {len(sequences)} proteins across {np.count_nonzero(np.diff(offsets))}"
            f"/{len(names)} contigs; frozen latents cached at {cache_path}."
        )
    return ProteinLatentTable(proteins, offsets, coordinates[:, 0],
                              coordinates[:, 1], coordinates[:, 2])


class ContigProteinEncoder(nn.Module):
    """Pool a CPU-backed ragged table, moving only batch proteins to the device.

    Frozen protein tables live in the extraction cache; model checkpoints contain
    the trainable pooling and projection weights. Integer contig/view lookup
    metadata travels separately from canonical numerical features.
    """

    def __init__(self, proteins, offsets, output_dim, config, view_rows=None):
        super().__init__()
        self.proteins = torch.as_tensor(proteins, dtype=torch.float32, device="cpu") # CPU-backed table avoids GPU memory pressure and allows large batch sizes
        self.offsets = np.asarray(offsets, dtype=np.int64)
        if self.proteins.ndim != 2 or self.offsets.ndim != 1 or len(self.offsets) < 2:
            raise ValueError("Expected a protein matrix and contig offsets")
        if self.offsets[0] != 0 or self.offsets[-1] != len(self.proteins) or np.any(np.diff(self.offsets) < 0):
            raise ValueError("Invalid protein table offsets")
        self.view_rows = view_rows
        self.lengths = np.diff(self.offsets)
        self.pooling = config.pooling
        self.weight = config.weight
        self.projection = nn.Linear(self.proteins.shape[1], output_dim)
        if self.pooling == "attention":
            if output_dim % config.num_heads:
                raise ValueError("ESM-C attention heads must divide the representation dimension")
            self.query = nn.Parameter(torch.randn(1, 1, output_dim) * 0.02)
            self.attention = nn.MultiheadAttention(output_dim, config.num_heads, batch_first=True)
        else:
            self.means = torch.zeros((len(self.offsets) - 1, self.proteins.shape[1]))
            for index, (start, end) in enumerate(zip(self.offsets[:-1], self.offsets[1:])):
                if end > start:
                    self.means[index] = self.proteins[start:end].mean(0)

    def pool_with_presence(self, ids, view_ids=None):
        device = self.projection.weight.device

        if ids is None or ids.dtype != torch.long:
            raise ValueError("ESM-C requires explicit torch.long contig IDs")
        ids = ids.detach().cpu()
        
        if view_ids is None:
            view_ids = torch.zeros_like(ids)
        if view_ids.dtype != torch.long or view_ids.shape != ids.shape:
            raise ValueError("ESM-C view IDs must be torch.long and match contig IDs")
        
        pairs = torch.stack((ids, view_ids.detach().cpu()), dim=-1)
        unique, inverse = torch.unique(pairs, dim=0, return_inverse=True) # unique contains the unique contig/view pairs, and inverse maps each original pair to its index in unique
        indices = unique[:, 0].numpy()
        if np.any(indices < 0) or np.any(indices >= len(self.offsets) - 1):
            raise ValueError("ESM-C contig ID is outside the protein table")
        
        # deduplicate contig/view pairs.
        rows = []
        for contig_id, view_id in unique.tolist():
            if self.view_rows is None:
                selected = np.arange(self.offsets[contig_id], self.offsets[contig_id + 1])
            else:
                try:
                    selected = self.view_rows[(contig_id, view_id)]
                except KeyError as exc:
                    raise ValueError(f"Unknown ESM-C contig/view pair {(contig_id, view_id)}") from exc
            rows.append(selected)

        lengths = np.asarray([len(selected) for selected in rows], dtype=np.int64)
        valid = torch.as_tensor(lengths > 0, device=device)
        if self.pooling == "mean":
            if self.view_rows is None:
                means = self.means[unique[:, 0]]
            else:
                means = torch.stack([
                    self.proteins[selected].mean(0) if len(selected)
                    else self.proteins.new_zeros(self.proteins.shape[1])
                    for selected in rows
                ])
            pooled = self.projection(means.to(device))
        else:
            # Gather only real proteins on the CPU and project before padding.
            # Padding the original 960/1152-wide latents wastes host-to-device
            # bandwidth and projection work when contig lengths are uneven.
            packed_starts = np.cumsum(lengths) - lengths
            positions = np.arange(lengths.sum()) - np.repeat(packed_starts, lengths)
            protein_rows = np.concatenate(rows)
            packed = self.proteins.index_select(0, torch.from_numpy(protein_rows))
            projected = self.projection(packed.to(device))

            # Keep one unmasked dummy for empty contigs to avoid all-masked
            # attention. Their residual (and gradients) is zeroed below.
            width = max(int(lengths.max()), 1)
            destinations = np.repeat(np.arange(len(unique)), lengths) * width + positions
            tokens = projected.new_zeros((len(unique) * width, projected.shape[-1]))
            tokens = tokens.index_copy(
                0, torch.as_tensor(destinations, device=device), projected
            ).view(len(unique), width, -1)
            mask = torch.arange(width, device=device)[None, :] >= torch.as_tensor(
                np.maximum(lengths, 1), device=device
            )[:, None]
            # Masked, single-query pooling under AMP can hit fused CUDA
            # attention kernel failures. Math SDPA is inexpensive for one query
            # and keeps this workaround local to protein pooling (also backward).
            with sdpa_kernel(SDPBackend.MATH):
                pooled, _ = self.attention(
                    self.query.expand(len(unique), -1, -1), tokens, tokens,
                    key_padding_mask=mask, need_weights=False,
                )
            pooled = pooled[:, 0]
        pooled = pooled * valid[:, None] # zero out empty contigs so they do not contribute to the residual
        
        inverse = inverse.to(device)

        return pooled[inverse], valid[inverse]
    
    def forward(self, ids, view_ids=None):
        pooled, _ = self.pool_with_presence(ids, view_ids)
        return pooled

    def fuse(self, representation, ids, view_ids=None):
        protein = self(ids, view_ids)
        # Match representation scale so model-specific encoder norms do not
        # accidentally erase or amplify the protein residual.
        scale = representation.norm(dim=-1, keepdim=True).detach().clamp_min(1e-6)
        return representation + self.weight * scale * F.normalize(protein, dim=-1) # weight is used to weigh protein's latent when fusing with canonical learnt features.


def prepare_esmc_features(data_df, contigs, known_contigs, output_dir, threads,
                          config, output_dim, logger=None, fragment_dict=None):
    """Attach exact-header integer lookup metadata without adding feature columns.

    Coordinate mode requires interval metadata for sequence fragments. Feature-only
    views must explicitly declare ``feature_only=True`` to retain all proteins.
    View 0 always denotes the full contig, including Barlow's .original view.
    """
    if not config.enabled:
        return data_df, None
    names = list(known_contigs)
    name_to_id = {name: index for index, name in enumerate(names)}
    table = build_protein_latents(contigs, names, output_dir, threads, config, logger)
    proteins, offsets = table
    if config.augmentation == "coordinate" and not isinstance(table, ProteinLatentTable):
        raise ValueError("Coordinate augmentation requires ORF coordinate metadata")
    if config.augmentation == "coordinate" and np.any(table.orf_starts < 0):
        raise ValueError("Coordinate augmentation requires coordinate-bearing gene caller headers; "
                         "regenerate the cached SCG .faa annotations")
    lookup, view_rows = {}, {}
    headers = {}
    for name in names:
        headers[f"{name}.original"] = (name, 0, None)
        if f"{name}.original" not in data_df.index:
            headers[f"{name}.0"] = (name, 0, None)
    for name, info in (fragment_dict or {}).items():
        if name not in name_to_id:
            continue
        for view_id, (header, interval) in enumerate(info.get("fragment_info", {}).items(), 1):
            original = header == f"{name}.original" or (
                header == f"{name}.0" and f"{name}.original" not in data_df.index
            )
            headers[header] = (name, 0 if original else view_id, interval)
    for header in data_df.index:
        if header not in headers:
            name = header.rsplit(".", 1)[0]
            if config.augmentation == "coordinate" or name not in name_to_id:
                raise ValueError(f"Cannot map ESM-C view {header!r}; provide fragment metadata")
            headers[header] = (name, len(headers) + 1, None)
        name, view_id, interval = headers[header]
        contig_id = name_to_id[name]
        lookup[header] = (contig_id, view_id)
        start, end = offsets[contig_id:contig_id + 2]
        rows = np.arange(start, end, dtype=np.int64)
        if config.augmentation == "coordinate" and view_id != 0:
            if interval is None:
                raise ValueError(f"Missing genomic interval for ESM-C view {header!r}")
            if not interval.get("feature_only", False):
                if "start_pos" not in interval or "length" not in interval:
                    raise ValueError(f"Missing genomic interval for ESM-C view {header!r}")
                a, length = interval["start_pos"], interval["length"]
                if not isinstance(a, (int, np.integer)) or not isinstance(length, (int, np.integer)) or a < 0 or length <= 0:
                    raise ValueError(f"Invalid genomic interval for ESM-C view {header!r}")
                rows = rows[(table.orf_starts[rows] >= a) & (table.orf_ends[rows] <= a + length)]
        view_rows[(contig_id, view_id)] = rows
    # Inference uses view 0 even if a feature table contains only fragments.
    for contig_id in range(len(names)):
        view_rows[(contig_id, 0)] = np.arange(offsets[contig_id], offsets[contig_id + 1])
    encoder = ContigProteinEncoder(proteins, offsets, output_dim, config,
                                  view_rows if config.augmentation == "coordinate" else None)
    if logger is not None:
        subsets = [rows for (_, view), rows in view_rows.items() if view != 0]
        logger.info(
            f"ESM-C {config.augmentation}: {len(subsets)} fragment views; "
            f"{sum(len(rows) == 0 for rows in subsets)} without eligible proteins."
        )
    result = data_df.copy()
    result.attrs["esmc_lookup"] = lookup
    return result, encoder
