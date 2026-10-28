"""Frozen ESM-C protein embeddings and trainable per-contig pooling.

Each protein is represented by the mean of its residue embeddings (excluding
special tokens). Long proteins are chunked without discarding residues. A
contig then uses either the mean of its proteins or learned query attention.
Only the pooling/projection is trained, using the trainer's existing objective.
"""

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
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from ..clustering.marker_quality import resolve_orf_to_contig
from .scg import prepare_gene_annotations

# if you want to add annotations in addition to predictions:
        # run_hmmsearch_pyhmmer(
        #     hmm_path=marker_hmm,
        #     faa_path=faa_path,
        #     domtblout_path=hmmout_path,
        #     threads=threads,
        #     e_value=max_evalue,
        # )

class ESMCConfig(BaseModel):
    enabled: bool = Field(default=False, description="Include the ESM-C protein modality.")
    model: Literal["esmc_300m", "esmc_600m"] = "esmc_300m"
    pooling: Literal["mean", "attention"] = "mean"
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
    fingerprint.update(json.dumps([1, config.model, config.max_residues, names]).encode())
    cache_path = output_dir / f"esmc_{fingerprint.hexdigest()}.npz"
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cache:
            return cache["proteins"], cache["offsets"]

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
                groups[contig_id].append(sequence)
        if logger is not None:
            logger.info(f"ESM-C: ORF-to-contig join {report.rate:.1%}.")
    sequences = [sequence for group in groups for sequence in group]
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
    np.savez(temporary, proteins=proteins, offsets=offsets)
    temporary.replace(cache_path)
    if logger is not None:
        logger.info(
            f"ESM-C: {len(sequences)} proteins across {np.count_nonzero(np.diff(offsets))}"
            f"/{len(names)} contigs; frozen latents cached at {cache_path}."
        )
    return proteins, offsets


class ContigProteinEncoder(nn.Module):
    """Pool a CPU-backed ragged table, moving only batch proteins to the device.

    Frozen protein tables live in the extraction cache; model checkpoints contain
    the trainable pooling and projection weights. IDs are carried in the final
    feature column so all existing paired-view and SCG loaders preserve alignment.
    """

    def __init__(self, proteins, offsets, output_dim, config):
        super().__init__()
        self.proteins = torch.as_tensor(proteins, dtype=torch.float32, device="cpu")
        self.offsets = np.asarray(offsets, dtype=np.int64)
        if self.proteins.ndim != 2 or self.offsets.ndim != 1 or len(self.offsets) < 2:
            raise ValueError("Expected a protein matrix and contig offsets")
        if self.offsets[0] != 0 or self.offsets[-1] != len(self.proteins) or np.any(np.diff(self.offsets) < 0):
            raise ValueError("Invalid protein table offsets")
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

    def forward(self, ids): # here lacks one potential change: what if the proteins across the views are lacking or subsampled?
        device = self.projection.weight.device
        # Pool each contig only once even when a batch contains multiple views.
        unique, inverse = torch.unique(ids.detach().long().cpu(), return_inverse=True)
        if torch.any(unique < 0) or torch.any(unique >= len(self.offsets) - 1):
            raise ValueError("ESM-C contig ID is outside the protein table")
        lengths = self.offsets[unique.numpy() + 1] - self.offsets[unique.numpy()]
        valid = torch.as_tensor(lengths > 0, device=device)
        if self.pooling == "mean":
            pooled = self.projection(self.means[unique].to(device))
        else:
            # One masked dummy token makes empty contigs safe for attention;
            # their contribution is explicitly zeroed below.
            sequences = [
                self.proteins[self.offsets[index] : self.offsets[index + 1]]
                if length else self.proteins.new_zeros((1, self.proteins.shape[1]))
                for index, length in zip(unique.tolist(), lengths)
            ]
            padded = pad_sequence(sequences, batch_first=True).to(device)
            tokens = self.projection(padded)
            mask = torch.arange(tokens.shape[1], device=device)[None, :] >= torch.as_tensor(
                np.maximum(lengths, 1), device=device
            )[:, None]
            pooled, _ = self.attention(
                self.query.expand(len(unique), -1, -1), tokens, tokens,
                key_padding_mask=mask, need_weights=False,
            )
            pooled = pooled[:, 0]
        pooled = pooled * valid[:, None] # zero out empty contigs so they do not contribute to the residual
        return pooled[inverse.to(device)]

    def fuse(self, representation, ids):
        protein = self(ids)
        # Match representation scale so model-specific encoder norms do not
        # accidentally erase or amplify the protein residual.
        scale = representation.norm(dim=-1, keepdim=True).detach().clamp_min(1e-6)
        return representation + self.weight * scale * F.normalize(protein, dim=-1) # weight is used to weigh protein's latent when fusing with canonical learnt features.


def prepare_esmc_features(data_df, contigs, known_contigs, output_dir, threads,
                          config, output_dim, logger=None, fragment_dict=None):
    """Append a contig lookup ID while preserving canonical feature ordering."""
    if not config.enabled:
        return data_df, None
    names = list(known_contigs)
    # Existing datasets cast to float32; integral IDs remain exact up to 2**24.
    if len(names) > 2**24:
        raise ValueError("ESM-C feature IDs support at most 2**24 contigs")
    name_to_id = {name: index for index, name in enumerate(names)}
    view_to_contig = {}
    if fragment_dict is not None:
        for name, info in fragment_dict.items():
            view_to_contig[f"{name}.original"] = name
            view_to_contig.update({header: name for header in info.get("fragment_info", {})})
    ids = []
    for header in data_df.index:
        name = view_to_contig.get(header, header.rsplit(".", 1)[0])
        if name not in name_to_id:
            raise ValueError(f"Cannot map ESM-C view {header!r} to a training contig")
        ids.append(name_to_id[name])
    proteins, offsets = build_protein_latents(
        contigs, names, output_dir, threads, config, logger
    )
    encoder = ContigProteinEncoder(proteins, offsets, output_dim, config)
    result = data_df.copy()
    result["__esmc_contig_id"] = np.asarray(ids, dtype=np.float32)
    return result, encoder
