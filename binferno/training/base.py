import math
import os
import random
from pathlib import Path

import numpy as np
from pydantic import BaseModel
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru._logger import Logger

from ..features import BinfernoFeatureBuilder

TAX_NODES = "tax_node"
TAX_NODE_INDICES = "tax_node_index"
PARENTAL_ARRAY = "parental_array"
TAXONOMY_DF="contig_taxonomy_df"


class BaseTrainer(BaseModel):

    def __call__(
        self,
        contigs: Path,
        bam: Path,
        output_dir: Path,
        features: BinfernoFeatureBuilder = None,
        threads: int = 4,
        logger: Logger = None,
    ) -> pd.DataFrame:
        raise NotImplementedError(
            "The __call__ method must be implemented by the subclass."
        )

    def unify_seeds(self, seed):
        # synchronize random seeds for reproducibility.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    class EarlyStoppingManager:
        def __init__(self, patience=10):
            self.patience = patience
            self._lowest_loss = float("inf")
            self._saved_state = None
            self._stagnant_epochs = 0

        def check_improvement(self, loss, model_state):
            improved = loss < self._lowest_loss

            if improved:
                self._lowest_loss = loss
                self._saved_state = model_state
                self._stagnant_epochs = 0
                return

            self._stagnant_epochs += 1

        def should_stop(self):
            return self._stagnant_epochs >= self.patience

        def get_best_state(self):
            """Return the best checkpoint state and its associated loss."""
            return self._saved_state, self._lowest_loss

    def print_stats(self, stats, epoch, method="barlow_twins", logger=None):
        if logger is None:
            raise ValueError(
                "Logger is not provided for printing stats. Please provide a logger."
            )
        if method == "barlow_twins":
            logger.info(
                f"Epoch {epoch+1} - mean diagonal: {stats['mean_diagonal']:.4f}, std_diagonal: {stats['std_diagonal']:.4f}, "
                f"Mean off-diagonal: {stats['mean_abs_off_diagonal']:.4f}, max off-diagonal: {stats['max_abs_off_diagonal']:.4f}, "
                f"Invariance Loss: {stats['invariance_loss']:.4f}, Redundancy Loss: {stats['redundancy_loss']:.4f}"
            )
        else:
            logger.info(
                f"Epoch {epoch+1} - SimCLR Loss: {stats['loss']:.4f} - Top-1 Accuracy: {stats['top1_acc']:.2f}%"
            )

    def warmup_cosine_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        min_learning_rate: float,
        epochs: int,
        warmup: bool = False,
        warmup_epochs: int = 10,
        max_lr_hold_epochs: int = 0,
        learning_rate: float = 2.5e-3,
    ):
        max_lr = learning_rate
        # warmup_start_lr = max_lr * 0.1
        start_lr = max_lr * 0.1

        hold_end = warmup_epochs + max_lr_hold_epochs  # epoch where cosine decay begins
        for param_group in optimizer.param_groups:
            param_group["lr"] = start_lr  # base for LambdaLR

        def lr_lambda(current_epoch):

            if current_epoch < warmup_epochs:
                if warmup:
                    # Linear ramp from start_lr to max_lr
                    progress = current_epoch / max(warmup_epochs - 1, 1)
                    lr = start_lr + progress * (max_lr - start_lr)
                else:
                    if current_epoch < hold_end:
                        lr = max_lr

            else:
                decay_epochs = epochs - hold_end
                if decay_epochs <= 0:
                    lr = min_learning_rate
                else:
                    progress = (current_epoch - hold_end) / decay_epochs
                    progress = min(progress, 1.0)
                    cosine = 0.5 * (1 + math.cos(math.pi * progress))
                    lr = min_learning_rate + (max_lr - min_learning_rate) * cosine

            return lr / start_lr

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return scheduler

    def get_embeddings(
        self,
        model: nn.Module,
        data_df: pd.DataFrame,
        taxonomy_features: pd.DataFrame,
        output_path: Path = None,
        batch_size: int = 1024,
        save_individual_embs: bool =False,
    ) -> pd.DataFrame:

        if not os.path.exists(os.path.dirname(output_path)):
            raise ValueError(
                f"Output path {output_path} does not exist. Please create it before calling get_embeddings."
            )

        embeddings_path = output_path
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        embeddings = dict()
        kmer_embs = dict()
        cov_embs = dict()
        tax_embs = dict()
        
        orig_contigs = data_df[data_df.index.str.endswith(".original")].copy()
        suffix = ".original"

        if orig_contigs.empty:
            # reformat the data_df index by finding those not containing ".", rename them as .original.
            orig_contigs = data_df[data_df.index.str.endswith(".0")].copy()
            suffix = ".0"
        
        if taxonomy_features is not None:
            taxonomy_features = taxonomy_features[taxonomy_features.index.str.endswith(suffix)].copy()

        with torch.no_grad():
            for i in range(0, len(orig_contigs), batch_size):
                batch_encode = orig_contigs.iloc[i : i + batch_size]
                batch_features = torch.tensor(
                    batch_encode.values, dtype=torch.float32
                ).to(device)

                if taxonomy_features is not None:
                    batch_taxonomy = taxonomy_features.iloc[i : i + batch_size]
                    batch_taxonomy = torch.tensor(
                        batch_taxonomy.values, dtype=torch.float32
                    ).to(device)
                else:
                    batch_taxonomy = None

                batch_embeddings = model.get_embedding(batch_features, batch_taxonomy)
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                for j, header in enumerate(batch_encode.index):
                    clean_header = header.replace(suffix, "")
                    embeddings[clean_header] = batch_embeddings[j].cpu().numpy()

                    # Save encoder embeddings if requested
                    if save_individual_embs:
                        if taxonomy_features is not None:
                            kmer_emb, cov_emb, tax_emb = model.get_separate_modality_embs(
                                batch_features, batch_taxonomy
                            )
                            kmer_embs[clean_header] = kmer_emb[j].cpu().numpy()
                            cov_embs[clean_header] = cov_emb[j].cpu().numpy()
                            tax_embs[clean_header] = tax_emb[j].cpu().numpy()
                        else:
                            kmer_emb, cov_emb = model.get_separate_modality_embs(
                                batch_features
                            )
                            kmer_embs[clean_header] = kmer_emb[j].cpu().numpy()
                            cov_embs[clean_header] = cov_emb[j].cpu().numpy()

        embeddings_df = pd.DataFrame.from_dict(embeddings, orient="index")
        embeddings_df.to_csv(str(embeddings_path), sep="\t")

        # note: if architecture changes, this save_individual_embs may no longer be compatible.
        if save_individual_embs:
            kmer_embeddings = os.path.join(
                os.path.dirname(output_path), "tnf_embeddings.tsv"
            )
            coverage_embeddings = os.path.join(
                os.path.dirname(output_path), "cov_embeddings.tsv"
            )
            kmer_emb_df = pd.DataFrame.from_dict(kmer_embs, orient="index")
            cov_emb_df = pd.DataFrame.from_dict(cov_embs, orient="index")

            kmer_emb_df.to_csv(kmer_embeddings, sep="\t")
            cov_emb_df.to_csv(coverage_embeddings, sep="\t")
            if taxonomy_features is not None:
                tax_embeddings = os.path.join(
                    os.path.dirname(output_path), "tax_embeddings.tsv"
                )
                tax_emb_df = pd.DataFrame.from_dict(tax_embs, orient="index")
                tax_emb_df.to_csv(tax_embeddings, sep="\t")

        return embeddings_df

    def sort_features_by_method(
        self,
        features_df: pd.DataFrame,
        simclr: bool = False,
        contig_ids: list = None,
    ) -> pd.DataFrame:
        if not contig_ids:
            raise ValueError(
                "contig_ids is required for sorting features. Please provide the list of contig names in the "
                "order they were processed during augmentation."
            )
        if not simclr:
            # sort the features_df by index.
            features_df = features_df.sort_index()
            return features_df
        else:
            # for simclr, group by view, stack the features of same view together, then concatenate the features.
            features_df["view"] = (
                features_df.index.str.rsplit(".", n=1).str[-1].astype(str)
            )
            # replace "original" as "0".
            # features_df["view"] = features_df["view"].replace("original", "0")
            features_df["view"] = features_df["view"].astype(int)

            features_df = features_df.sort_values(by=["view"], ascending=True)
            for i, view in enumerate(features_df["view"].unique()):
                view_df = features_df[features_df["view"] == view]
                # reindex view_df to make sure the order of features is the same across views.
                view_df.index = [ind.rsplit(".", 1)[0] for ind in view_df.index]
                view_df = view_df.reindex(
                    contig_ids
                )  # reindexing to ensure the same order of contigs across views
                view_df.index = [
                    f"{idx}.{view}" for idx in view_df.index
                ]  # restore the original index format after reindexing
                if i == 0:
                    sorted_features_df = view_df
                else:
                    sorted_features_df = pd.concat(
                        [sorted_features_df, view_df], axis=0
                    )
            # for each view, make sure to sort the features by index.
            features_df = sorted_features_df.drop(columns=["view"])
            return features_df
        
    class taxonomy_hloss(nn.Module):
        def __init__(self, lambda_tax=1.0):
            super().__init__()
            self.lambda_tax = lambda_tax

        def forward(self, tax_logits, base_ids):
            # tax_logits: (batch_size, num_taxa)
            # base_ids: (batch_size,) with values in [0, num_base_contigs-1]
            batch_size = tax_logits.size(0)
            device = tax_logits.device

            # Create a mask for pairs of samples that come from the same base contig
            mask = torch.eq(base_ids.unsqueeze(1), base_ids.unsqueeze(0)).float().to(device)

            # Compute pairwise distances between taxonomic logits
            diff = tax_logits.unsqueeze(1) - tax_logits.unsqueeze(0)  # (batch_size, batch_size, num_taxa)
            dist_squared = (diff ** 2).sum(dim=-1)  # (batch_size, batch_size)

            # We want to minimize distances for pairs from the same contig and maximize for different contigs
            loss_same = (mask * dist_squared).sum() / (mask.sum() + 1e-8)  # Avoid division by zero
            loss_diff = ((1 - mask) * F.relu(1.0 - dist_squared.sqrt())).sum() / ((1 - mask).sum() + 1e-8)

            loss = self.lambda_tax * (loss_same + loss_diff)
            return loss