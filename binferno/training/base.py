import math
import os
import random
import numpy as np
from pydantic import BaseModel
import pandas as pd
from pathlib import Path
import torch
import torch.nn.functional as F
from loguru._logger import Logger

from ..features import BinfernoFeatureBuilder


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
        optimizer,
        min_learning_rate,
        epochs,
        warmup_epochs=10,
        max_lr_hold_epochs=0,
        learning_rate=1e-3,
    ):
        max_lr = learning_rate
        # warmup_start_lr = max_lr * 0.1w
        start_lr = max_lr

        hold_end = warmup_epochs + max_lr_hold_epochs  # epoch where cosine decay begins
        for param_group in optimizer.param_groups:
            param_group["lr"] = start_lr  # base for LambdaLR

        def lr_lambda(current_epoch):
            # if current_epoch < warmup_epochs:
            #     # Linear ramp from start_lr to max_lr
            #     progress = current_epoch / max(warmup_epochs - 1, 1)
            #     lr = start_lr + progress * (max_lr - start_lr)

            if current_epoch < hold_end:
                # Flat hold at max_lr — scheduler steps but LR doesn't change
                lr = max_lr

            else:
                # Cosine decay from max_lr to min_learning_rate
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
        model=None,
        data_df=None,
        output_path=None,
        batch_size=1024,
        save_individual_embs=False,
    ):

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
        orig_contigs = data_df[data_df.index.str.endswith(".original")].copy()
        suffix = ".original"

        if orig_contigs.empty:
            # reformat the data_df index by finding those not containing ".", rename them as .original.
            orig_contigs = data_df[data_df.index.str.endswith(".0")].copy()
            suffix = ".0"

        with torch.no_grad():
            for i in range(0, len(orig_contigs), batch_size):
                batch_encode = orig_contigs.iloc[i : i + batch_size]
                batch_features = torch.tensor(
                    batch_encode.values, dtype=torch.float32
                ).to(device)

                batch_embeddings = model.get_embedding(batch_features)
                batch_embeddings = F.normalize(
                    batch_embeddings, p=2, dim=1
                )
                for j, header in enumerate(batch_encode.index):
                    clean_header = header.replace(suffix, "")
                    embeddings[clean_header] = batch_embeddings[j].cpu().numpy()

                    # Save encoder embeddings if requested
                    if save_individual_embs:
                        kmer_emb, cov_emb = model.get_separate_modality_embs(
                            batch_features
                        )
                        kmer_embs[clean_header] = kmer_emb[j].cpu().numpy()
                        cov_embs[clean_header] = cov_emb[j].cpu().numpy()

        embeddings_df = pd.DataFrame.from_dict(embeddings, orient="index")
        embeddings_df.to_csv(str(embeddings_path), sep="\t")

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


""" class info_nce_loss(nn.Module):
        def __init__(self, temperature=0.15):
            super().__init__()
            self.temperature = temperature
            self._cache = {}

        def forward(self, features, device, batch_size=1024, n_views=6):
            "
            Ultra-fast multi-view InfoNCE loss matching the 'comebin' style.
            Computes (n_views - 1) independent losses per anchor view against all negatives,
            leveraging fused mathematical operators and memory-bound optimizations.
            "
            total = batch_size * n_views
            cache_key = (batch_size, n_views, str(device))

            # --- Efficient Persistent Caching ---
            if cache_key not in self._cache:
                sample_ids = torch.arange(batch_size, device=device).repeat(n_views)
                
                # Construct masks for structured extraction
                pos_mask = sample_ids.unsqueeze(0).eq(sample_ids.unsqueeze(1))
                diag_mask = torch.eye(total, dtype=torch.bool, device=device)
                pos_mask_no_diag = pos_mask & ~diag_mask
                
                # Fast index matrix for gathering positive scores: [total, n_views - 1]
                pos_indices = pos_mask_no_diag.nonzero()[:, 1].view(total, n_views - 1)
                
                # Complete negative mask to isolate true negative elements
                neg_mask = pos_mask
                
                self._cache[cache_key] = (pos_indices, neg_mask, sample_ids)

            pos_indices, neg_mask, sample_ids = self._cache[cache_key]

            # 1. High-speed Matrix Multiplication
            features = F.normalize(features, dim=1)
            sim = torch.matmul(features, features.T) / self.temperature

            # 2. Extract Positives and Flatten: Shape [total * (n_views - 1), 1]
            # This gathers exactly the 5 positive pairs per row
            positives = torch.gather(sim, dim=1, index=pos_indices).view(-1, 1)

            # 3. Mask out true positives and self-diagonal targets to isolate negatives
            negatives_only = sim.masked_fill(neg_mask, float('-inf'))
            
            # Virtual repetition to pair each row's negatives against each individual positive view
            # Shape: [total * (n_views - 1), total]
            negatives_repeated = torch.repeat_interleave(negatives_only, n_views - 1, dim=0)

            # 4. Fused Mathematical Log-Sum-Exp Reduction (Avoids huge torch.cat calls)
            max_val = torch.clamp(positives, min=0.0) 
            pos_exp = torch.exp(positives - max_val)
            neg_sum_exp = torch.sum(torch.exp(negatives_repeated - max_val), dim=1, keepdim=True)
            
            # Final objective calculation matching comebin's multi-view behavior
            loss = -(positives - max_val - torch.log(pos_exp + neg_sum_exp)).mean()

            # 5. Optimized Metric Computation (Top-1 and Top-5 Accuracy)
            with torch.no_grad():
                # Extract pure negative scores across the batch
                # Shape: [total, total] -> max negative per row: [total, 1]
                max_negatives = negatives_only.max(dim=1, keepdim=True).values
                max_negatives_repeated = torch.repeat_interleave(max_negatives, n_views - 1, dim=0)
                
                # Top-1: Check if the individual positive view score beats the best negative score
                top1_acc = (positives > max_negatives_repeated).float().mean().item() * 100.0

                # Top-5: Sort or partition the negative space to find the 5th highest negative baseline
                # topk(5) returns sorted negative tensor of shape [total, 5]. The 5th item (index 4) is our threshold.
                top5_neg_threshold = torch.topk(negatives_only, k=5, dim=1).values[:, 4:5]
                top5_neg_repeated = torch.repeat_interleave(top5_neg_threshold, n_views - 1, dim=0)
                
                # Top-5: True if the positive score beats the 5th highest negative score
                top5_acc = (positives > top5_neg_repeated).float().mean().item() * 100.0

            return loss, top1_acc, top5_acc"""
