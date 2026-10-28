import hashlib
import itertools
import os
from collections import OrderedDict
from pathlib import Path

from typing import Dict, Tuple, List, Optional, Sequence
from pydantic import Field
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.backends import cudnn
import torch.nn.functional as F
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
import pyfastx
import loguru

from .models.barlow_twins import NeuralNetwork
from .models.new_htax_encoder import HierarchicalTaxonomyEncoder
from .models.taxonomy_gate import validate_rank_taxonomy_table
from ..features.taxonomy import TAXONOMIC_RANKS
from ..features import BinfernoFeatureBuilder
from .base import (
    BaseTrainer,
    TAX_NODES,
    TAX_NODE_INDICES,
    PARENTAL_ARRAY,
    TAXONOMY_DF,
    label_augmented_views,
)
from .WeightedBTLoss import WeightedBTLoss
from .esmc import prepare_esmc_features
from .protein_batch import feature_tensor, ProteinBatch
from .scg import (
    SCGMarginLoss,
    build_scg_groups as build_shared_scg_groups,
    prepare_scg_group_loader,
)

class PositivePairDataset(Dataset):
    """Gather paired views from one canonical feature matrix."""

    def __init__(self, features, pair_indices, base_ids, weights,
                 taxonomy=None, taxonomy_mask=None, protein_metadata=None):
        self.features = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
        if protein_metadata is not None:
            self.features = ProteinBatch(self.features, *protein_metadata)
        index_dtype = np.int32 if len(features) < 2**31 else np.int64
        self.pair_indices = torch.from_numpy(
            np.ascontiguousarray(pair_indices, dtype=index_dtype)
        )
        self.base_ids = base_ids
        self.weights = weights
        self.taxonomy = (
            None if taxonomy is None
            else torch.from_numpy(np.ascontiguousarray(taxonomy, dtype=np.float32))
        )
        self.taxonomy_mask = taxonomy_mask

    def __len__(self):
        return len(self.pair_indices)

    def __getitem__(self, index):
        left, right = self.pair_indices[index].tolist()
        batch = [self.features[left], self.features[right]]
        if self.taxonomy is not None:
            batch.extend([self.taxonomy[left], self.taxonomy[right]])
        if self.taxonomy_mask is not None:
            batch.append(self.taxonomy_mask[index])
        batch.extend([self.base_ids[index], self.weights[index]])
        return tuple(batch)

    def __getitems__(self, indices):
        """Gather one DataLoader batch with vectorized row lookup."""
        pairs = self.pair_indices[indices].long()
        left, right = pairs[:, 0], pairs[:, 1]
        first = self.features[left]
        second = self.features[right]
        tax_first = tax_second = None
        if self.taxonomy is not None:
            tax_first = self.taxonomy[left]
            tax_second = self.taxonomy[right]
        samples = []
        for offset, index in enumerate(indices):
            sample = [first[offset], second[offset]]
            if tax_first is not None:
                sample.extend([tax_first[offset], tax_second[offset]])
            if self.taxonomy_mask is not None:
                sample.append(self.taxonomy_mask[index])
            sample.extend([self.base_ids[index], self.weights[index]])
            samples.append(tuple(sample))
        return samples


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
        # Measured beta(tau=0) = 1.16 on real data: pair multiplicity already
        # weights contigs superlinearly in length, past MetaCAT's L^1. Rescale
        # so each contig's pairs keep their tau=0 total -- tau then only
        # redistributes WITHIN a contig (crop vs .original), which is the part
        # multiplicity cannot express.
        g = np.asarray(per_contig_group, dtype=np.int64)
        assert g.shape == prec.shape, "per_contig_group must be one id per pair"
        tot = np.bincount(g, weights=prec, minlength=g.max() + 1)
        cnt = np.bincount(g, minlength=g.max() + 1)
        prec = prec * (cnt[g] / np.maximum(tot[g], 1e-12))

    prec = prec / prec.mean()  # mean 1.0 keeps the loss on the tau=0 scale
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

class BarlowTwinsTrainer(BaseTrainer):
    # Basic parameters.
    learning_rate: float = Field(
        default=2.5e-3, description="Learning rate for training."
    )
    view_num: int = Field(
        default=8, description="Number of augmented views to generate for each contig."
    )
    warmup_epochs: int = Field(
        default=10,
        description="Number of epochs to linearly increase the learning rate from a small value to the initial learning rate at the start of training.",
    )
    max_lr_epochs: int = Field(
        default=0,
        description="Number of epochs to wait before resuming learning rate scheduling after reaching the maximum learning rate.",
    )
    proj_head_dim: int = Field(
        default=2048,
        description="Dimension of the projection head used in the Barlow Twins architecture. The projection head is a small neural network that maps the output of the main encoder to a space where the Barlow Twins loss is applied. A larger proj_head_dim can allow for more complex transformations and potentially better performance, but it also increases the number of parameters and computational cost of the model.",
    )
    # Weighted-barlowtwins loss terms.
    weight_tau: float = Field(
        default=0.0,
        description="Length-precision exponent for the Barlow Twins loss. 0 "
        "reproduces the unweighted loss exactly. 0.5 is the recommended "
        "operating point; above 0.75 the effective batch size collapses.",
    )
    weight_clip_percentile: float = Field(
        default=99.0,
        description="Percentile cap on raw pair precision before tempering.",
    )
    weight_per_contig_norm: bool = Field(
        default=True,
        description="Renormalise weights so each contig's pairs keep their "
        "unweighted total, so tau acts within contigs only.",
    )
    stratified_cap: bool = Field(
        default=True,
        description="Guarantee every contig keeps one positive pair when the "
        "pair set exceeds max_positive_pairs. False reproduces the uniform "
        "sampler, which starved 7,423 of 131,042 contigs on a real run.",
    )
    # Weighting taxonomy embeddings inside barlowtwins model.
    alpha_tax: float = Field(
        default=0.10,
        ge=0.0,
        le=0.20, 
        description="A small weight used to constrain taxonomy embedding fusion."
    )
    lambda_taxonomy: float = Field(
        default=0.1,
        ge=0.0,
        description="Weight of the masked multi-rank taxonomy prediction loss. "
        "Zero disables the auxiliary loss while retaining taxonomy embedding fusion.",
    )
    # SCG cannot-link loss.

    def __call__(
        self,
        contigs: Path,
        bam: Path,
        output_dir: Path,
        features: BinfernoFeatureBuilder = None,
        threads: int = 4,
        logger: loguru._logger.Logger = None,
    ) -> pd.DataFrame:
        """
        Args:
         - contigs: path to the contigs fasta file.
         - bam: path to the bam file for coverage information.
         - output_dir: directory to save the model and logs.
         - view_num: number of augmented views to generate for each contig.
         - contig_length: minimum length of contigs to consider for augmentation.
        """
        output_aug_dir = output_dir / "augmentation"
        output_aug_dir.mkdir(parents=True, exist_ok=True)
        output_train_dir = output_dir / "training"
        annotation_dir = output_train_dir / "scg"
        if self._esmc.enabled and self._esmc.fusion == "concat":
            raise ValueError(
                "Protein concat fusion is currently supported only by SimCLRNet."
            )
            
        if self._esmc.enabled:
            # Keep existing checkpoints separate from models with a protein branch.
            import hashlib
            key = hashlib.sha256(self._esmc.model_dump_json().encode()).hexdigest()[:12]
            output_train_dir = output_train_dir / f"esmc-{key}"
        output_train_dir.mkdir(parents=True, exist_ok=True)
        embedding_path = output_train_dir / "embeddings.tsv"
        if embedding_path.exists():
            logger.info(
                f"Embeddings already exist at {embedding_path}. Loading embeddings from file."
            )
            embeddings = pd.read_csv(embedding_path, sep="\t", header=0, index_col=0)
            return embeddings

        fragment_dict, all_fragments, namelist = self.gen_views(
            contigs=contigs,
            output_dir=output_aug_dir,
            view_num=self.view_num,
            contig_length=self._common.contig_length,
        )

        self._frag_lengths = build_fragment_length_map(fragment_dict)

        feature_result = features(
            FragMent_Dict=fragment_dict,
            All_Fragments=all_fragments,
            contigs=contigs,
            bam_path=bam,
            threads=threads,
            logger=logger,
            output_dir=output_aug_dir,
        )

        if isinstance(feature_result, tuple):
            feature_df, kmer_dim = feature_result
        else:
            feature_df = feature_result
            if hasattr(features, "_get_kmer_dim"):
                kmer_dim = features._get_kmer_dim()
            else:
                kmer_dim = 0

        feature_df, taxonomy_df = feature_df

        feature_df = self.sort_features_by_method(
            feature_df,
            contig_ids=namelist,
            simclr=False,
        )

        taxonomy_vocab_sizes = None
        if isinstance(taxonomy_df, pd.DataFrame):
            tax_columns = [f"tax_{rank}" for rank in TAXONOMIC_RANKS]
            mask_columns = [f"mask_{rank}" for rank in TAXONOMIC_RANKS]
            expected_columns = tax_columns + mask_columns
            if set(taxonomy_df.columns) != set(expected_columns):
                raise ValueError(
                    "Taxonomy columns must contain rank IDs and rank masks; "
                    f"expected {expected_columns}, got {taxonomy_df.columns.tolist()}."
                )
            missing_rows = feature_df.index.difference(taxonomy_df.index)
            if len(missing_rows):
                raise ValueError(
                    "Taxonomy features are missing augmented fragments: "
                    f"{missing_rows[:5].tolist()}"
                )
            taxonomy_df = taxonomy_df.reindex(feature_df.index)[expected_columns]
            taxonomy_vocab_sizes = validate_rank_taxonomy_table(
                taxonomy_df, tax_columns, mask_columns
            )

        self.unify_seeds(
            self._common.seed
        )  # ensure reproducibility by unifying seeds across random, numpy, and torch.

        taxonomy_model = None
        if taxonomy_vocab_sizes is not None:
            taxonomy_model = HierarchicalTaxonomyEncoder(
                vocab_sizes=taxonomy_vocab_sizes,
                output_dim=self._common.output_dim,
            )

        if torch.cuda.is_available():
            device = torch.device("cuda")
            cudnn.deterministic = True
            cudnn.benchmark = False
        else:
            device = torch.device("cpu")
            torch.set_num_threads(threads)
        logger.info(f"Using device: {device}")

        cov_dim = sum([1 for col in feature_df.columns if "cov" in col])
        composition_dim = feature_df.shape[1] - cov_dim

        corgi_dim = composition_dim - kmer_dim

        feature_df, protein_encoder = prepare_esmc_features(
            feature_df, contigs, namelist, annotation_dir, threads, self._esmc,
            self._common.output_dim, logger, fragment_dict=fragment_dict,
        )

        h_tax = isinstance(taxonomy_df, dict)

        model = NeuralNetwork(
            protein_encoder=protein_encoder,
            h_tax=h_tax,
            tax_model=taxonomy_model,
            tax_emb_dim=taxonomy_df.shape[1] if taxonomy_model is not None else 0,
            tnf_dim=composition_dim,
            coverage_dim=cov_dim,
            kmer_dim=kmer_dim,
            proj_head_dim=self.proj_head_dim,
            corgi_dim=corgi_dim,
            output_dim=self._common.output_dim,
            alpha_tax=self.alpha_tax,
        ).to(
            device
        )  # can weights be reproducible?

        logger.info(f"Model architecture:\n{model}")
        if taxonomy_model is not None and self.lambda_taxonomy > 0.0:
            logger.info(
                "Adding canonical-to-taxonomy auxiliary loss to Barlow Twins "
                f"training (lambda_taxonomy={self.lambda_taxonomy:g})."
            )

        dataloader, optimizer, scheduler = self.prepare_trainer(
            model=model,
            data_df=feature_df,
            batch_size=self._common.batch_size,
            epochs=self._common.epochs,
            num_workers=threads,
            logger=logger,
            taxonomy_features=taxonomy_df,
        )

        scg_dataloader = None
        if self._scg.lambda_scg > 0.0:
            logger.info("Adding SCG cannot-link loss to Barlow Twins training.")
            scg_groups = self.build_scg_groups(
                contigs=contigs,
                known_contigs=namelist,
                output_dir=annotation_dir,
                threads=threads,
                logger=logger,
            )
            scg_dataloader = self.prepare_scg_loader(
                data_df=feature_df,
                taxonomy_features=taxonomy_df,
                marker_groups=scg_groups,
                num_workers=threads,
                logger=logger,
            )

        model_path = output_train_dir / "best_model.pt"
        if not model_path.exists():
            model = self.train(
                dataloader=dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                method="barlow_twins",
                epochs=self._common.epochs,
                model=model,
                output_dir=output_train_dir,
                logger=logger,
                scg_dataloader=scg_dataloader,
            )
        else:
            logger.info(
                f"Model already exists at {model_path}. Loading model from file."
            )
            model.load_state_dict(torch.load(model_path, map_location=device))

        if not embedding_path.exists():
            orig_df = feature_df[feature_df.index.str.endswith(".original")]
            namelist = [index.rsplit(".", 1)[0] for index in orig_df.index]
            embeddings = self.get_embeddings(
                model=model,
                data_df=feature_df,
                output_path=embedding_path,
                batch_size=self._common.batch_size,
                save_individual_embs=self._common.get_individual_embs,
                taxonomy_features=taxonomy_df,
            )

        embeddings = pd.DataFrame(
            F.normalize(torch.from_numpy(embeddings.values).float(), dim=1).numpy(),
            index=embeddings.index,
            columns=embeddings.columns,
        )

        return embeddings

    def get_embeddings(
        self,
        model=None,
        data_df=None,
        output_path=None,
        batch_size=1024,
        save_individual_embs=False,
        taxonomy_features=None,
    ) -> pd.DataFrame:
        """Encode original views, optionally with the separate taxonomy table."""
        if not isinstance(taxonomy_features, pd.DataFrame):
            return super().get_embeddings(
                model=model,
                data_df=data_df,
                output_path=output_path,
                batch_size=batch_size,
                save_individual_embs=save_individual_embs,
            )

        if not taxonomy_features.index.equals(data_df.index):
            raise ValueError(
                "Taxonomy features must have the same ordered indices as "
                "canonical features."
            )
        output_path = Path(output_path)
        if not output_path.parent.exists():
            raise ValueError(
                f"Output path {output_path} does not exist. "
                "Please create it before calling get_embeddings."
            )

        model.eval()
        device = next(model.parameters()).device
        original_mask = data_df.index.str.endswith(".original")
        suffix = ".original"
        if not original_mask.any():
            original_mask = data_df.index.str.endswith(".0")
            suffix = ".0"
        original_features = data_df.loc[original_mask]
        original_taxonomy = taxonomy_features.loc[original_mask]

        embeddings = {}
        kmer_embs = {}
        cov_embs = {}
        with torch.no_grad():
            for start in range(0, len(original_features), batch_size):
                feature_batch = feature_tensor(
                    original_features.iloc[start : start + batch_size]
                ).to(device)
                taxonomy_batch = torch.as_tensor(
                    original_taxonomy.iloc[start : start + batch_size].values,
                    dtype=torch.float32,
                    device=device,
                )
                batch_embeddings = F.normalize(
                    model.get_embedding(feature_batch, taxonomy_batch), p=2, dim=-1
                )
                if save_individual_embs:
                    kmer_batch, cov_batch = model.get_separate_modality_embs(
                        feature_batch
                    )
                for row, header in enumerate(
                    original_features.iloc[start : start + batch_size].index
                ):
                    clean_header = header[: -len(suffix)]
                    embeddings[clean_header] = batch_embeddings[row].cpu().numpy()
                    if save_individual_embs:
                        kmer_embs[clean_header] = kmer_batch[row].cpu().numpy()
                        cov_embs[clean_header] = cov_batch[row].cpu().numpy()

        embeddings_df = pd.DataFrame.from_dict(embeddings, orient="index")
        embeddings_df.to_csv(output_path, sep="\t")
        if save_individual_embs:
            pd.DataFrame.from_dict(kmer_embs, orient="index").to_csv(
                output_path.parent / "tnf_embeddings.tsv", sep="\t"
            )
            pd.DataFrame.from_dict(cov_embs, orient="index").to_csv(
                output_path.parent / "cov_embeddings.tsv", sep="\t"
            )
        return embeddings_df

    class BT_loss(nn.Module):
        def __init__(self, lambda_bt=5e-3, alpha=1.0, eps=1e-6):
            super().__init__()
            self.lambda_bt = lambda_bt  # lambda barlow twins tunes the importance of the redundancy reduction term relative to the invariance term. A higher lambda_bt encourages the model to produce more decorrelated features, while a lower lambda_bt allows for more redundancy in the features.
            self.alpha = alpha  # alpha tunes the importance of the invariance loss relative to the redundancy loss. A higher alpha emphasizes the importance of making the diagonal elements of the correlation matrix close to 1, while a lower alpha allows for more flexibility in the diagonal elements.
            self.eps = eps
            self._cache = {}

        def forward(
            self,
            feature1,
            feature2,
            w: Optional[torch.Tensor] = None,
            return_stats=False,
        ):
            N, D = feature1.size()
            device = feature1.device
            cache_key = (D, str(device))
            if cache_key not in self._cache:
                # We cache an identity matrix to avoid allocating it on every step
                self._cache[cache_key] = torch.eye(D, device=device)
            eye = self._cache[cache_key]

            z1 = (feature1 - feature1.mean(dim=0)) / (
                feature1.std(dim=0, unbiased=False) + self.eps
            )
            z2 = (feature2 - feature2.mean(dim=0)) / (
                feature2.std(dim=0, unbiased=False) + self.eps
            )

            C = torch.mm(z1.T, z2) / N

            loss_matrix = C.pow(2)
            diag = torch.diagonal(C)
            invariance_loss = (
                (diag - 1.0).pow(2).sum()
            )  # invariance loss

            redundancy_loss = loss_matrix.sum() - torch.diagonal(loss_matrix).sum()
            loss = self.alpha * invariance_loss + self.lambda_bt * redundancy_loss

            if return_stats:
                with torch.no_grad():
                    # Isolate off-diagonals cleanly using the cached identity matrix
                    off_diag_abs = C.abs().masked_fill(eye.to(torch.bool), 0.0)

                    # Bundle tensors together into a single structure before sending to CPU
                    # This minimizes the overhead of calling .item() sequentially
                    stats_tensor = torch.stack(
                        [
                            diag.mean(),
                            diag.std(),
                            off_diag_abs.sum()
                            / (D * D - D),  # True mean of off-diagonals
                            off_diag_abs.max(),
                        ]
                    )
                    stats_cpu = stats_tensor.cpu().tolist()

                    stats = {
                        "mean_diagonal": stats_cpu[0],
                        "std_diagonal": stats_cpu[1],
                        "mean_abs_off_diagonal": stats_cpu[2],
                        "max_abs_off_diagonal": stats_cpu[3],
                        "invariance_loss": invariance_loss.item(),
                        "redundancy_loss": redundancy_loss.item(),
                    }
                return loss, stats

            return loss

    class BuildDataset:
        def __init__(
            self,
            data_df: pd.DataFrame,
            max_positive_pairs=700000,
            taxonomy_features: Optional[pd.DataFrame | Dict | None] = None,
            namelist: Optional[list[str]] = None,
            fragment_lengths: Optional[Dict[str, int]] = None,
            weight_tau: float = 0.0,
            weight_clip_percentile: float = 99.0,
            weight_per_contig_norm: bool = True,
            stratified_cap: bool = True,
        ):
            self.data_df = data_df
            self.max_positive_pairs = max_positive_pairs
            self.fragment_contig_ids = list(data_df.index)
            self.taxonomy_features = taxonomy_features
            self.namelist = namelist
            self.h_tax = False
            self.fragment_lengths = fragment_lengths
            self.weight_tau = weight_tau
            self.weight_clip_percentile = weight_clip_percentile
            self.weight_per_contig_norm = weight_per_contig_norm
            self.stratified_cap = stratified_cap

            grouped = self._group_fragments_by_contig()
            self.base_name_to_id = {
                contig_name: idx for idx, contig_name in enumerate(grouped)
            }

            self.index_to_base_id = {}
            for contig_name, frag_indices in grouped.items():
                contig_id = self.base_name_to_id[contig_name]
                for frag_idx in frag_indices:
                    self.index_to_base_id[frag_idx] = contig_id

            self.contig_aug_dict = {
                contig_name: frag_indices
                for contig_name, frag_indices in grouped.items()
                if len(frag_indices) >= 2
            }

            if not self.contig_aug_dict:
                raise ValueError(
                    "No base contigs found with multiple fragments. Cannot generate positive pairs."
                )

            positive_pairs = self._build_all_positive_pairs()
            self.training_pairs = self._cap_positive_pairs(
                positive_pairs,
                self.max_positive_pairs,
            )

            if not self.training_pairs:
                raise ValueError("No positive pairs selected. Training cannot proceed.")

            random.shuffle(self.training_pairs)

        def _extract_base_name(self, fragment_header):
            """
            Infer the source contig name from a fragment header.
            """
            if fragment_header.split("_")[-1] == "original":
                return fragment_header.split("_")[0]
            return fragment_header.split(".")[0]

        def _group_fragments_by_contig(self):
            """
            Build a dictionary mapping base contig names to row indices in data_df.
            """
            grouped = {}

            for row_idx, fragment_header in enumerate(self.fragment_contig_ids):
                base_name = self._extract_base_name(fragment_header)
                grouped.setdefault(base_name, []).append(row_idx)

            return grouped

        def _build_all_positive_pairs(self):
            """
            Generate all fragment-fragment positive pairs within each base contig.
            """
            pair_list = []

            for frag_indices in self.contig_aug_dict.values():
                pair_list.extend(itertools.combinations(frag_indices, 2))

            return list(pair_list)

        def _cap_positive_pairs(self, candidate_pairs, max_cap):
            """Cap the pair set while guaranteeing every contig keeps a pair.

            Uniform sampling silently starves short contigs: a contig with a
            single pair loses it outright at the cap ratio. Measured on a real
            run, 7,423 of 131,042 contigs contributed no gradient at all.
            """
            if not self.stratified_cap:
                return random.sample(candidate_pairs, max_cap)
            candidate_pairs = list(candidate_pairs)
            if len(candidate_pairs) <= max_cap:
                return candidate_pairs

            by_contig = {}
            for pair in candidate_pairs:
                by_contig.setdefault(self.index_to_base_id[pair[0]], []).append(pair)

            contigs = list(by_contig)
            random.shuffle(contigs)

            kept, remainder = [], []
            for contig in contigs:
                group = by_contig[contig]
                pick = random.randrange(len(group))
                if len(kept) < max_cap:
                    kept.append(group[pick])
                    remainder.extend(group[:pick] + group[pick + 1 :])
                else:
                    remainder.extend(group)

            if len(kept) < max_cap:
                need = max_cap - len(kept)
                kept.extend(
                    remainder
                    if len(remainder) <= need
                    else random.sample(remainder, need)
                )
            return kept

        def pair_frgs_to_contigs(self) -> dict:
            """
            Kept for compatibility with the original interface.
            """
            return self._group_fragments_by_contig()

        def construct_tensordatasets_from_pairs(
            self, logger: loguru._logger.Logger = None
        ) -> PositivePairDataset:
            """Keep one feature matrix and gather selected positive pairs on demand."""
            X = self.data_df.to_numpy(dtype=np.float32, copy=False)
            if isinstance(self.taxonomy_features, pd.DataFrame):
                tax_X = self.taxonomy_features.to_numpy(dtype=np.float32, copy=False)
            elif isinstance(self.taxonomy_features, dict):
                self.h_tax = True
                tax_X = label_augmented_views(
                    self.taxonomy_features[TAXONOMY_DF],
                    self.data_df.index.tolist(),
                    node_indices=self.taxonomy_features[TAX_NODE_INDICES],
                )
                taxonomy_df = self.taxonomy_features[TAXONOMY_DF]
                node_indices = self.taxonomy_features[TAX_NODE_INDICES]
                tax_X = tax_X.to_numpy(dtype=np.float32, copy=False)
            else:
                tax_X = None

            assert not np.isnan(
                X
            ).any(), "data_df contains NaN values before TensorDataset construction."
            assert not np.isinf(
                X
            ).any(), "data_df contains Inf values before TensorDataset construction."

            pair_dtype = np.int32 if len(X) < 2**31 else np.int64
            pair_indices = np.asarray(self.training_pairs, dtype=pair_dtype)
            idx1 = pair_indices[:, 0]
            idx2 = pair_indices[:, 1]

            base_ids_np = np.fromiter(
                (self.index_to_base_id[i] for i in idx1),
                dtype=np.int64,
                count=len(idx1),
            )
            base_ids = torch.from_numpy(base_ids_np)
            if self.fragment_lengths is not None and self.weight_tau > 0.0:
                group = base_ids_np if self.weight_per_contig_norm else None
                w = torch.from_numpy(
                    make_pair_weights(
                        self.data_df.index,
                        idx1,
                        idx2,
                        self.fragment_lengths,
                        tau=self.weight_tau,
                        clip_percentile=self.weight_clip_percentile,
                        per_contig_group=group,
                    )
                )
            else:
                w = torch.ones(len(idx1), dtype=torch.float32)

            if logger is not None:
                logger.info(
                    f"pair weights: tau={self.weight_tau} "
                    f"min={w.min():.3f} max={w.max():.3f} "
                    f"n_eff_ratio={(1.0/((w/w.sum())**2).sum()/len(w)):.3f}"
                )
            mask = None
            if tax_X is not None and self.h_tax:
                # Keep the legacy hierarchical mask aligned with selected pairs.
                mask = torch.zeros(len(idx1), dtype=torch.bool)
                for i, idx in enumerate(idx1):
                    base_name = self._extract_base_name(self.data_df.index[idx])
                    tax = (
                        taxonomy_df.loc[base_name]["tax_node_idx"]
                        if base_name in taxonomy_df.index
                        else node_indices.get("root", -1)
                    )
                    if tax != node_indices.get("root", -1):
                        mask[i] = True
            return PositivePairDataset(
                features=X,
                pair_indices=pair_indices,
                base_ids=base_ids,
                weights=w,
                taxonomy=tax_X,
                taxonomy_mask=mask,
                protein_metadata=(tuple(feature_tensor(self.data_df))[1:]
                                  if "esmc_lookup" in self.data_df.attrs else None),
            )

    def build_scg_groups(
        self,
        contigs: Path,
        known_contigs: Sequence[str],
        output_dir: Path,
        threads: int,
        logger,
    ) -> Dict[str, List[str]]:
        return build_shared_scg_groups(
            contigs=contigs,
            known_contigs=known_contigs,
            output_dir=output_dir,
            threads=threads,
            marker_hmm=self._scg.scg_marker_hmm,
            max_evalue=self._scg.scg_max_evalue,
            min_hmm_coverage=self._scg.scg_min_hmm_coverage,
            logger=logger,
        )

    def prepare_scg_loader(
        self,
        data_df: pd.DataFrame,
        marker_groups: Dict[str, List[str]],
        taxonomy_features=None,
        num_workers: int = 0,
        logger=None,
    ) -> DataLoader:
        return prepare_scg_group_loader(
            data_df=data_df,
            marker_groups=marker_groups,
            original_suffix=".original",
            batch_size=self._scg.scg_batch_size,
            seed=self._common.seed,
            # The iterator is recycled often; worker copies and restarts cost memory.
            num_workers=0,
            logger=logger,
            worker_init_fn=self.seed_worker,
            auxiliary_df=(
                taxonomy_features if isinstance(taxonomy_features, pd.DataFrame) else None
            ),
        )

    def prepare_trainer(
        self,
        model,
        data_df,
        batch_size: int = 1024,
        logger=None,
        epochs: int = 300,
        num_workers: int = 0,
        taxonomy_features=None,
        **kwargs,
    ):
        dataset_builder = self.BuildDataset(
            data_df=data_df,
            taxonomy_features=taxonomy_features,
            fragment_lengths=getattr(self, "_frag_lengths", None),
            weight_tau=self.weight_tau,
            weight_clip_percentile=self.weight_clip_percentile,
            weight_per_contig_norm=self.weight_per_contig_norm,
            stratified_cap=self.stratified_cap,
        )
        train_dataset = dataset_builder.construct_tensordatasets_from_pairs(
            logger=logger
        )

        effective_batch_size = batch_size
        # a way of adaptively changing the batch sizes.
        frag_lengths = getattr(self, "_frag_lengths", None)
        if self.weight_tau > 0.0 and frag_lengths is None:
            raise RuntimeError(
                "weight_tau > 0 but no fragment length map was built. "
                "self._frag_lengths is set in __call__ after gen_views."
            )

        while len(train_dataset) < effective_batch_size and effective_batch_size > 1:
            effective_batch_size = effective_batch_size // 2
            if logger is not None:
                logger.warning(
                    f"Dataset size ({len(train_dataset)}) < Batch size ({effective_batch_size * 2}). "
                    f"Automatically reducing batch size to {effective_batch_size}."
                )

        has_enough_data = len(train_dataset) > effective_batch_size * 10

        dataloader_kwargs = {
            "batch_size": effective_batch_size,
            "shuffle": True,
            "drop_last": not has_enough_data,
            "worker_init_fn": self.seed_worker,
            "num_workers": max(0, num_workers),
            "pin_memory": torch.cuda.is_available(),
        }

        if dataloader_kwargs["num_workers"] > 0:
            dataloader_kwargs.update(
                {
                    "persistent_workers": True,
                    "prefetch_factor": 2,
                }
            )

        dataloader = DataLoader(train_dataset, **dataloader_kwargs)
        optimizer = optim.AdamW(
            model.parameters(), lr=1e-4, weight_decay=0.05, betas=(0.9, 0.95)
        )
        scheduler = self.warmup_cosine_scheduler(
            optimizer,
            learning_rate=self.learning_rate,
            min_learning_rate=1e-4,
            epochs=epochs,
            warmup_epochs=self.warmup_epochs,
            warmup=True,
            max_lr_hold_epochs=self.max_lr_epochs,
        )
        return dataloader, optimizer, scheduler

    def train(
        self,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        method: str = "barlow_twins",
        epochs: int = 300,
        model=None,
        output_dir: Path = None,
        logger=loguru._logger.Logger,
        scg_dataloader: Optional[DataLoader] = None,
    ):
        MODEL_PATH = output_dir / "best_model.pt"
        if logger is None:
            raise ValueError(
                "Logger is not provided for training. Please provide a logger in the kwargs with key 'logger'."
            )

        epoch_iter = tqdm(range(epochs), desc="Training Epochs")
        early_stopping = self.EarlyStoppingManager(patience=10)
        for epoch in epoch_iter:

            avg_loss, stats = self.train_an_epoch(
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                device=device,
                scg_dataloader=scg_dataloader,
                epoch=epoch,
            )

            scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            logger.info("Current learning rate: {:.2e}".format(current_lr))

            if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}, LR: {current_lr:.2e}"
                )

            if stats is not None:
                self.print_stats(stats, epoch, method=method, logger=logger)

            early_stopping.check_improvement(avg_loss, model.state_dict())
            if early_stopping.should_stop():
                logger.info(
                    f"Early stopping after {epoch+1} epochs (patience: {early_stopping.patience})"
                )
                break

            best_state, best_loss = early_stopping.get_best_state()

        if best_state is not None:
            model.load_state_dict(best_state)
            logger.info(f"Loaded best model with loss: {best_loss:.4f}")

            torch.save(model.state_dict(), MODEL_PATH)
            logger.info(f"Model saved to {MODEL_PATH}")

        return model

    def train_an_epoch(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scg_dataloader: Optional[DataLoader] = None,
        epoch: int = 0,
    ):
        criterion = WeightedBTLoss(lambda_bt=5e-3, alpha=1.0).to(device)
        model.train()

        sum_loss = 0.0
        stats = None
        is_htax = model.hierarchical_taxonomy
        num_batches = len(dataloader)
        scg_criterion = SCGMarginLoss(self._scg.cosine_margin).to(device)
        scg_iterator = iter(scg_dataloader) if scg_dataloader is not None else None
        scg_loss_sum = 0.0
        scg_updates = 0
        scg_scale = (
            1.0
            if self._scg.scg_loss_warmup == 0
            else min(1.0, float(epoch + 1) / self._scg.scg_loss_warmup)
        )

        for batch_i, batch_data in enumerate(
            dataloader
        ):  # here it determines taxonomy data to be yes or no.
            last_batch = batch_i == num_batches - 1
            *batch_data, w = batch_data  # weights always last
            w = w.to(device, non_blocking=True)

            if len(batch_data) == 3:
                features1, features2, base_ids = batch_data
                features1 = features1.to(device, non_blocking=True)
                features2 = features2.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                total_loss, stats = self.calculate_loss(
                    criterion=criterion,
                    model=model,
                    features1=features1,
                    features2=features2,
                    tax_1=None,
                    tax_2=None,
                    mask=None,
                    last_batch=last_batch,
                    w=w,
                )
            elif not is_htax:
                features1, features2, tax_1, tax_2, base_ids = batch_data
                features1 = features1.to(device, non_blocking=True)
                features2 = features2.to(device, non_blocking=True)
                tax_1 = tax_1.to(device, non_blocking=True)
                tax_2 = tax_2.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                total_loss, stats = self.calculate_loss(
                    criterion=criterion,
                    model=model,
                    features1=features1,
                    features2=features2,
                    tax_1=tax_1,
                    tax_2=tax_2,
                    mask=None,
                    last_batch=last_batch,
                    w=w,
                )
            else:
                feature1, feature2, label1, label2, mask, base_ids = batch_data
                feature1 = feature1.to(device, non_blocking=True)
                feature2 = feature2.to(device, non_blocking=True)
                label1 = label1.to(device, non_blocking=True)
                label2 = label2.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                total_loss, stats = self.calculate_loss(
                    criterion=criterion,
                    model=model,
                    features1=feature1,
                    features2=feature2,
                    tax_1=label1,
                    tax_2=label2,
                    mask=mask,
                    last_batch=last_batch,
                    w=w,
                )

            if scg_iterator is not None and batch_i % self._scg.scg_update_freq == 0:
                try:
                    scg_batch = next(scg_iterator)
                except StopIteration:
                    scg_iterator = iter(scg_dataloader)
                    scg_batch = next(scg_iterator)
                if len(scg_batch) == 5:
                    scg_features, scg_taxonomy, marker_ids, contig_ids, scg_weights = scg_batch
                    scg_taxonomy = scg_taxonomy.to(device, non_blocking=True)
                else:
                    scg_features, marker_ids, contig_ids, scg_weights = scg_batch
                    scg_taxonomy = None
                scg_features = scg_features.to(device, non_blocking=True)
                marker_ids = marker_ids.to(device, non_blocking=True)
                contig_ids = contig_ids.to(device, non_blocking=True)
                scg_weights = scg_weights.to(device, non_blocking=True)
                scg_embeddings = model.encode(scg_features, scg_taxonomy)
                scg_loss = scg_criterion(
                    scg_embeddings,
                    marker_ids,
                    contig_ids,
                    pair_weights=scg_weights,
                    max_hard_pairs=self._scg.scg_hard_pairs,
                )
                total_loss = total_loss + self._scg.lambda_scg * scg_scale * scg_loss
                scg_loss_sum += scg_loss.detach().item()
                scg_updates += 1

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )  # prevent the exploding gradient.

            optimizer.step()
            sum_loss += total_loss.item()  # sum loss.

        avg_loss = sum_loss / len(dataloader)
        if stats is not None and scg_updates:
            stats["scg_loss"] = scg_loss_sum / scg_updates
            stats["scg_weight_scale"] = self._scg.lambda_scg * scg_scale
            stats["scg_updates"] = scg_updates
        return avg_loss, stats

    def calculate_loss(
        self,
        criterion,
        model,
        features1,
        features2,
        tax_1=None,
        tax_2=None,
        mask=None,
        last_batch=False,
        w=None,
    ):
        is_htax = model.hierarchical_taxonomy

        if tax_1 is not None or tax_2 is not None:
            if getattr(model, "uses_rank_taxonomy", False):
                output1, output2, canonical1, canonical2 = model(
                    features1,
                    features2,
                    tax_emb1=tax_1,
                    tax_emb2=tax_2,
                    return_canonical=True,
                )
                if self.lambda_taxonomy > 0.0:
                    taxonomy_loss = self.rank_taxonomy_loss(
                        model, canonical1, tax_1, w
                    ).add(
                        self.rank_taxonomy_loss(model, canonical2, tax_2, w)
                    ).mul(0.5)
                else:
                    taxonomy_loss = output1.new_zeros(())
                if last_batch:
                    loss, stats = criterion(
                        feature1=output1,
                        feature2=output2,
                        return_stats=True,
                        w=w,
                    )
                    stats["taxonomy_loss"] = taxonomy_loss.detach().item()
                    stats["weighted_taxonomy_loss"] = (
                        self.lambda_taxonomy * taxonomy_loss.detach().item()
                    )
                    return loss + self.lambda_taxonomy * taxonomy_loss, stats
                loss = criterion(feature1=output1, feature2=output2, w=w)
                return loss + self.lambda_taxonomy * taxonomy_loss, None

            elif not is_htax:
                output1, output2, tax_enc1, tax_enc2 = model(
                    features1, features2, tax_1, tax_2
                )
                if last_batch:
                    loss, stats = criterion(
                        feature1=output1, feature2=output2, return_stats=True, w=w
                    )
                    tax_vec_loss, tax_stats = criterion(
                        feature1=tax_enc1, feature2=tax_enc2, return_stats=True, w=w
                    )
                    loss = loss + tax_vec_loss

                    # merge stats from both losses
                    stats = {**stats, **{f"tax_{k}": v for k, v in tax_stats.items()}}
                    return loss, stats
                else:
                    loss = criterion(feature1=output1, feature2=output2, w=w)
                    tax_vec_loss = criterion(feature1=tax_enc1, feature2=tax_enc2, w=w)
                    loss = loss + tax_vec_loss
                    return loss, None

            else:
                output1, output2, tax_feature1, tax_feature2 = model(
                    features1, features2, tax_emb1=tax_1, tax_emb2=tax_2
                )
                logit1, logit2 = tax_feature1["logits"], tax_feature2["logits"]
                tax_hloss1 = model.taxonomy_encoder.loss(
                    logit1, tax_1, mask, weights=None
                )
                tax_hloss2 = model.taxonomy_encoder.loss(
                    logit2, tax_2, mask, weights=None
                )
                tax_hloss = (tax_hloss1 + tax_hloss2) / 2
                if last_batch:
                    loss, stats = criterion(
                        feature1=output1, feature2=output2, return_stats=True, w=w
                    )
                    total_loss = loss + tax_hloss
                    return total_loss, stats
                else:
                    loss = criterion(feature1=output1, feature2=output2, w=w)
                    total_loss = loss + tax_hloss
                    return total_loss, None
        else:
            output1, output2 = model(features1, features2)
            if last_batch:
                loss, stats = criterion(
                    feature1=output1, feature2=output2, return_stats=True, w=w
                )
                return loss, stats
            else:
                loss = criterion(feature1=output1, feature2=output2, w=w)
                return loss, None

    @staticmethod
    def rank_taxonomy_loss(model, canonical_embedding, taxonomy, weights=None):
        """Masked cross-entropy over every available rank.

        The prediction heads only see canonical composition/coverage features.
        Each rank is normalized by its random-classifier baseline and independently
        averaged, so populous or well-annotated ranks cannot dominate the loss. Pair precision
        weights are normalized within valid rows to preserve that scale.
        """
        n_ranks = len(model.taxonomy_vocab_sizes)
        expected_width = 2 * n_ranks
        if taxonomy.ndim != 2 or taxonomy.shape[1] != expected_width:
            raise ValueError(
                f"Expected taxonomy tensor [B, {expected_width}], "
                f"got {tuple(taxonomy.shape)}"
            )

        tax_ids = taxonomy[:, :n_ranks].long()
        tax_masks = taxonomy[:, n_ranks:] > 0
        logits_by_rank = model.taxonomy_logits(canonical_embedding)
        rank_losses = []

        for rank, logits in enumerate(logits_by_rank):
            valid = tax_masks[:, rank] & (tax_ids[:, rank] > 0)
            if not torch.any(valid):
                continue
            per_row = F.cross_entropy(
                logits[valid], tax_ids[valid, rank], reduction="none"
            )
            # A random classifier has CE=log(n_classes). Normalizing by that
            # baseline prevents species/genus vocabularies from dominating
            # broad ranks solely because they contain more categories.
            per_row = per_row / np.log(logits.shape[1])
            if weights is None:
                rank_losses.append(per_row.mean())
            else:
                valid_weights = weights[valid].to(
                    device=per_row.device, dtype=per_row.dtype
                ).clamp_min(0.0)
                rank_losses.append(
                    (per_row * valid_weights).sum()
                    / valid_weights.sum().clamp_min(1e-8)
                )

        if not rank_losses:
            return canonical_embedding.sum() * 0.0
        return torch.stack(rank_losses).mean()

    def barlow_twins_augmentation(
        self,
        sequence: str,
        header: str,
        min_contig_length: int,
        num_augmentations: int,
        max_overlap: float = 0.25,
    ) -> list[tuple[str, str, int, int]]:
        """
        Per-sequence augmentation.

        Returns:
            List of (fragment_header, fragment_sequence, global_start, fragment_length).
        """
        seq_len = len(sequence)
        fragments: list[tuple[str, str, int, int]] = []

        if seq_len < min_contig_length:
            return fragments

        fragments.append((f"{header}.original", sequence, 0, seq_len))

        if num_augmentations <= 0:
            return fragments

        split_threshold = 50000

        if seq_len > split_threshold:
            mid = seq_len // 2

            left_fragments = self._generate_half_augmentations(
                sequence=sequence[:mid],
                base_header=header,
                half_id="h1",
                global_offset=0,
                min_contig_length=min_contig_length,
                num_augmentations=num_augmentations,
                max_overlap=max_overlap,
            )
            right_fragments = self._generate_half_augmentations(
                sequence=sequence[mid:],
                base_header=header,
                half_id="h2",
                global_offset=mid,
                min_contig_length=min_contig_length,
                num_augmentations=num_augmentations,
                max_overlap=max_overlap,
            )

            fragments.extend(left_fragments)
            fragments.extend(right_fragments)

        else:
            generated = self._generate_half_augmentations(
                sequence=sequence,
                base_header=header,
                half_id="",
                global_offset=0,
                min_contig_length=min_contig_length,
                num_augmentations=num_augmentations,
                max_overlap=max_overlap,
            )
            fragments.extend(generated)

            # print(
            #     f"Generated {len(fragments)} fragments for {header} (length {seq_len})"
            # )

        return fragments

    def _generate_half_augmentations(
        self,
        sequence: str,
        base_header: str,
        half_id: str,
        global_offset: int,
        min_contig_length: int,
        num_augmentations: int,
        max_overlap: float,
    ) -> list[tuple[str, str, int, int]]:
        """
        Generate augmented fragments for a sequence half (or a full smaller contig).

        Returns:
            List of (fragment_header, fragment_sequence, global_start, fragment_length).
        """
        seq_len = len(sequence)
        if seq_len < min_contig_length:
            return []

        seed_material = f"{base_header}_{half_id}".encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:4], "big")
        rng = random.Random(seed)

        fragments: list[tuple[str, str, int, int]] = []

        # Store accepted intervals separately for faster overlap checks:
        # (global_start, global_end, length)
        accepted_intervals: list[tuple[int, int, int]] = []

        selected = 0
        attempts = 0
        max_attempts = max(1, num_augmentations * 10)

        while selected < num_augmentations and attempts < max_attempts:
            attempts += 1

            # Same strategy weighting as the original:
            # 50% "both", otherwise choose left/right/center uniformly.
            if rng.getrandbits(1):  # 50% chance to choose "both" edges masking
                strategy = "both"
            else:
                strategy = rng.choice(("left", "right", "center"))

            if strategy == "left":
                max_mask = seq_len - min_contig_length
                if max_mask <= 0:
                    continue
                start_pos = rng.randint(1, max_mask)
                end_pos = seq_len

            elif strategy == "right":
                max_mask = seq_len - min_contig_length
                if max_mask <= 0:
                    continue
                start_pos = 0
                end_pos = seq_len - rng.randint(1, max_mask)

            elif strategy == "both":
                max_total_mask = seq_len - min_contig_length
                if max_total_mask <= 0:
                    continue
                total_mask = rng.randint(1, max_total_mask)
                left_mask = rng.randint(0, total_mask)
                right_mask = total_mask - left_mask
                start_pos = left_mask
                end_pos = seq_len - right_mask

            else:  # center
                if seq_len < 2 * min_contig_length:
                    continue

                max_center_mask = seq_len - 2 * min_contig_length
                if max_center_mask <= 0:
                    continue

                center_mask_size = rng.randint(1, max_center_mask)
                center_start = rng.randint(
                    min_contig_length,
                    seq_len - min_contig_length - center_mask_size,
                )
                center_end = center_start + center_mask_size

                if rng.getrandbits(1):
                    start_pos = 0
                    end_pos = center_start
                else:
                    start_pos = center_end
                    end_pos = seq_len

            frag_len = end_pos - start_pos
            if frag_len < min_contig_length:
                continue

            global_start = start_pos + global_offset
            global_end = end_pos + global_offset

            # Faster overlap scan over compact interval data.
            valid = True
            for prev_start, prev_end, prev_len in accepted_intervals:
                overlap_start = (
                    global_start if global_start > prev_start else prev_start
                )
                overlap_end = global_end if global_end < prev_end else prev_end
                overlap_len = overlap_end - overlap_start

                if overlap_len > 0:
                    denom = frag_len if frag_len > prev_len else prev_len
                    if (overlap_len / denom) >= max_overlap:
                        valid = False
                        break

            if not valid:
                continue

            fragment_header = (
                f"{base_header}.{half_id}.{selected}"
                if half_id
                else f"{base_header}.{selected}"
            )
            fragment_seq = sequence[start_pos:end_pos]

            fragments.append((fragment_header, fragment_seq, global_start, frag_len))
            accepted_intervals.append((global_start, global_end, frag_len))
            selected += 1

        return fragments

    def augmentation(
        self,
        fasta_file=None,
        num_views=6,
        length_threshold=1000,
    ):
        # initialize the augmentation.
        seqs = pyfastx.Fastx(fasta_file)
        all_fragments = []
        fragment_to_contig = {}
        short_fragment_count = 0
        sequence_count = 0
        # apply augmentation.
        fragments_dict = OrderedDict()
        fragment_count = 0

        # the loop over sequences to generate augmentations.
        for header, seq in seqs:
            if len(seq) < length_threshold:
                continue
            clean_header = str(header.split()[0])
            fragments_dict[clean_header] = {
                "sequence": seq,
                "fragments": [],
                "fragment_info": {},
            }

            augmented_fragments = self.barlow_twins_augmentation(
                sequence=seq,
                header=clean_header,
                min_contig_length=length_threshold,
                num_augmentations=num_views,
            )

            for (
                fragment_header,
                fragment_seq,
                start_pos,
                frag_len,
            ) in augmented_fragments:
                fragments_dict[clean_header]["fragments"].append(fragment_header)
                fragments_dict[clean_header]["fragment_info"][fragment_header] = {
                    "start_pos": start_pos,
                    "length": frag_len,
                }

                if len(fragment_seq) < length_threshold:
                    short_fragment_count += 1
                    continue

                all_fragments.append((fragment_header, fragment_seq))
                fragment_to_contig[fragment_header] = clean_header
                fragment_count += 1

            sequence_count += 1

        print(f"Total sequences processed: {sequence_count}")
        print(f"Total fragments generated: {len(all_fragments)}")
        return fragments_dict, all_fragments

    def augment_views(
        self,
        contigs: Path,
        view_num: int,
        contig_length: int,
        output_dir: Path,
    ) -> Dict[str, Dict]:
        """
        Augmentation of contigs only. The returned data is a dictionary storing: raw contig -> ( seq, header, (start, end) )
        """
        # three things used across the stage:
        output_dir.mkdir(parents=True, exist_ok=True)
        FragMent_Dict: OrderedDict = OrderedDict()
        All_Fragments: list = []

        FragMent_Dict, All_Fragments = self.augmentation(
            fasta_file=str(contigs),
            num_views=view_num,
            length_threshold=contig_length,
        )

        for view in range(view_num):
            output_fasta = os.path.join(
                output_dir, f"augmented_fragments_view{view}.fasta"
            )
            view_fragments = [
                (header, seq)
                for header, seq in All_Fragments
                if header.endswith(f".{view}")
            ]
            with open(output_fasta, "w") as f:
                for fragment_header, fragment_seq in view_fragments:
                    f.write(f">{fragment_header}\n")
                    f.write(f"{fragment_seq}\n")

        return FragMent_Dict, All_Fragments

    def gen_views(
        self,
        contigs=None,
        output_dir=None,
        view_num=6,
        contig_length=1000,
    ) -> Tuple[Dict[str, pd.DataFrame], list, list]:
        # process input contigs into augmented fragments.
        FragMent_Dict, All_Fragments = self.augment_views(
            contigs=contigs,
            view_num=view_num,
            contig_length=contig_length,
            output_dir=output_dir,
        )

        namelist = list(
            FragMent_Dict.keys()
        )  # the ordered contig names which will be used for indexing the features.

        for view in range(view_num):
            output_fasta = os.path.join(
                output_dir, f"augmented_fragments_view{view}.fasta"
            )
            view_fragments = [
                (header, seq)
                for header, seq in All_Fragments
                if header.endswith(f".{view}")
            ]
            with open(output_fasta, "w") as f:
                for fragment_header, fragment_seq in view_fragments:
                    f.write(f">{fragment_header}\n")
                    f.write(f"{fragment_seq}\n")

        return FragMent_Dict, All_Fragments, namelist

