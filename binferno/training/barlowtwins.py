import hashlib
import itertools
import os
from collections import OrderedDict
from pathlib import Path

from typing import Dict, Tuple
from pydantic import Field
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.backends import cudnn
import torch.nn.functional as F
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
import pyfastx
import loguru

from .models.barlow_twins import NeuralNetwork
from ..features import BinfernoFeatureBuilder
from .base import BaseTrainer


class BarlowTwinsTrainer(BaseTrainer):

    learning_rate: float = Field(
        default=2.5e-3, description="Learning rate for training."
    )
    batch_size: int = Field(default=1024, description="Batch size for training.")
    epochs: int = Field(default=300, description="Number of epochs for training.")
    weight_decay: float = Field(default=1e-4, description="Weight decay for training.")
    output_dim: int = Field(
        default=128, description="Output dimension for the embeddings."
    )
    contig_length: int = Field(
        default=1000, description="Minimum contig length to be included in training."
    )
    view_num: int = Field(
        default=8, description="Number of augmented views to generate for each contig."
    )
    seed: int = Field(
        default=42, description="Random seed for training and augmentation."
    )
    dense_mode: bool = Field(
        default=False,
        description="Whether to use dense layers in the model architecture. If False, the model will use a more lightweight architecture without dense layers.",
    )
    warmup_epochs: int = Field(
        default=10,
        description="Number of epochs to linearly increase the learning rate from a small value to the initial learning rate at the start of training.",
    )
    max_lr_epochs: int = Field(
        default=0,
        description="Number of epochs to wait before resuming learning rate scheduling after reaching the maximum learning rate.",
    )
    get_individual_embs: bool = Field(
        default=False,
        description="Whether to save individual embeddings for each view separately. If False, only the final embeddings after fusion are saved.",
    )
    auxiliary_loss: bool = Field(
        default=False,
        description="Whether to include an auxiliary loss term that encourages the model to produce similar embeddings for the original and augmented views of the same contig. This can help improve the quality of the learned embeddings by providing an additional training signal that reinforces the relationship between the original and augmented data.",
    )
    proj_head_dim: int = Field(
        default=512,
        description="Dimension of the projection head used in the Barlow Twins architecture. The projection head is a small neural network that maps the output of the main encoder to a space where the Barlow Twins loss is applied. A larger proj_head_dim can allow for more complex transformations and potentially better performance, but it also increases the number of parameters and computational cost of the model.",
    )

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
            contig_length=self.contig_length,
        )

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

        feature_df = self.sort_features_by_method(
            feature_df,
            contig_ids=namelist,
            simclr=False,
        )

        if torch.cuda.is_available():
            device = torch.device("cuda")
            cudnn.deterministic = True
            cudnn.benchmark = True
        else:
            device = torch.device("cpu")
            torch.set_num_threads(threads)
        logger.info(f"Using device: {device}")

        cov_dim = sum([1 for col in feature_df.columns if "cov" in col])
        composition_dim = feature_df.shape[1] - cov_dim

        corgi_dim = composition_dim - kmer_dim
        model = NeuralNetwork(
            tnf_dim=composition_dim,
            coverage_dim=cov_dim,
            kmer_dim=kmer_dim,
            proj_head_dim=self.proj_head_dim,
            corgi_dim=corgi_dim,
            dense_mode=self.dense_mode,
            output_dim=self.output_dim,
        ).to(device)

        logger.info(f"Model architecture:\n{model}")

        dataloader, optimizer, scheduler = self.prepare_trainer(
            model=model,
            data_df=feature_df,
            batch_size=self.batch_size,
            epochs=self.epochs,
            num_workers=threads,
        )

        model_path = output_train_dir / "best_model.pt"
        if not model_path.exists():
            model = self.train(
                dataloader=dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                method="barlow_twins",
                epochs=self.epochs,
                model=model,
                output_dir=output_train_dir,
                logger=logger,
                auxiliary_loss=self.auxiliary_loss,
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
                batch_size=self.batch_size,
                save_individual_embs=self.get_individual_embs,
            )

        embeddings = pd.DataFrame(
            F.normalize(torch.from_numpy(embeddings.values).float(), dim=1).numpy(),
            index=embeddings.index,
            columns=embeddings.columns,
        )

        return embeddings

    def seed_worker(self, worker_id: int):
        """Seed worker processes for DataLoader reproducibility."""
        worker_seed = torch.initial_seed() % 2**10
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    class BT_loss(nn.Module):
        def __init__(self, lambda_bt=5e-3, alpha=1.0, eps=1e-6):
            super().__init__()
            self.lambda_bt = lambda_bt  # lambda barlow twins tunes the importance of the redundancy reduction term relative to the invariance term. A higher lambda_bt encourages the model to produce more decorrelated features, while a lower lambda_bt allows for more redundancy in the features.
            self.alpha = alpha  # alpha tunes the importance of the invariance loss relative to the redundancy loss. A higher alpha emphasizes the importance of making the diagonal elements of the correlation matrix close to 1, while a lower alpha allows for more flexibility in the diagonal elements.
            self.eps = eps
            # self.register_buffer(
            #     'off_diag_mask',
            #     ~torch.eye(proj_dim, dtype=torch.bool)
            # )
            self._cache = {}

        def forward(self, feature1, feature2, return_stats=False):
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
            )  # this is the invariance loss, which encourages the diagonal elements of the correlation matrix to be close to 1, meaning that the corresponding features in the two views are highly correlated.

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

    class modality_BT_loss(nn.Module):
        def __init__(self, lambda_bt=5e-3, alpha=1.0, eps=1e-6):
            super().__init__()
            self.lambda_bt = lambda_bt  # lambda barlow twins tunes the importance of the redundancy reduction term relative to the invariance term. A higher lambda_bt encourages the model to produce more decorrelated features, while a lower lambda_bt allows for more redundancy in the features.
            self.alpha = alpha  # alpha tunes the importance of the invariance loss relative to the redundancy loss. A higher alpha emphasizes the importance of making the diagonal elements of the correlation matrix close to 1, while a lower alpha allows for more flexibility in the diagonal elements.
            self.eps = eps
            self._cache = {}

        def forward(self, feature1, feature2):
            N, D = feature1.size()
            device = feature1.device
            cache_key = (D, str(device))
            if cache_key not in self._cache:
                self._cache[cache_key] = torch.eye(D, device=device)

            z1 = (feature1 - feature1.mean(dim=0)) / (
                feature1.std(dim=0, unbiased=False) + self.eps
            )
            z2 = (feature2 - feature2.mean(dim=0)) / (
                feature2.std(dim=0, unbiased=False) + self.eps
            )

            C = torch.mm(z1.T, z2) / N

            loss_matrix = C.pow(2)
            diag = torch.diagonal(C)
            invariance_loss = (diag - 1.0).pow(2).sum()

            redundancy_loss = loss_matrix.sum() - torch.diagonal(loss_matrix).sum()
            loss = self.alpha * invariance_loss + self.lambda_bt * redundancy_loss
            return loss

    class BuildDataset:
        def __init__(self, data_df, max_positive_pairs=700000):
            self.data_df = data_df
            self.max_positive_pairs = max_positive_pairs
            self.fragment_contig_ids = list(data_df.index)

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
            """
            Randomly subsample positive pairs if they exceed the allowed cap.
            """
            if len(candidate_pairs) <= max_cap:
                return list(candidate_pairs)

            return random.sample(candidate_pairs, max_cap)

        def form_positive_pairs(self):
            """
            Kept for compatibility with the original interface.
            """
            return self._build_all_positive_pairs()

        def _select_positive_pairs(self, all_potential_pairs, max_cap):
            """
            Kept for compatibility with the original interface.
            """
            return self._cap_positive_pairs(all_potential_pairs, max_cap)

        def pair_frgs_to_contigs(self) -> dict:
            """
            Kept for compatibility with the original interface.
            """
            return self._group_fragments_by_contig()

        def construct_tensordatasets_from_pairs(self):
            """
            Convert selected positive pairs into a TensorDataset of:
                (feature1, feature2, base_id)
            """
            X = self.data_df.to_numpy(dtype=np.float32, copy=False)

            if not np.isfinite(X).all():
                bad_rows, bad_cols = np.where(~np.isfinite(X))
                print("First bad entries:")
                for r, c in zip(bad_rows[:20], bad_cols[:20]):
                    print(
                        "row:",
                        r,
                        "index:",
                        self.data_df.index[r],
                        "col:",
                        self.data_df.columns[c],
                        "value:",
                        self.data_df.iloc[r, c],
                    )
                raise ValueError(
                    "data_df contains NaN or Inf before TensorDataset construction."
                )

            pair_indices = np.asarray(self.training_pairs, dtype=np.int64)
            idx1 = pair_indices[:, 0]
            idx2 = pair_indices[:, 1]

            features1 = torch.from_numpy(X[idx1].copy())
            features2 = torch.from_numpy(X[idx2].copy())

            base_ids_np = np.fromiter(
                (self.index_to_base_id[i] for i in idx1),
                dtype=np.int64,
                count=len(idx1),
            )
            base_ids = torch.from_numpy(base_ids_np)

            return TensorDataset(features1, features2, base_ids)

    def prepare_trainer(
        self,
        model,
        data_df,
        batch_size: int = 1024,
        logger=None,
        epochs: int = 300,
        num_workers: int = 0,
        **kwargs,
    ):
        dataset_builder = self.BuildDataset(data_df=data_df)
        train_dataset = dataset_builder.construct_tensordatasets_from_pairs()

        effective_batch_size = batch_size
        # a way of adaptively changing the batch sizes.
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
        auxiliary_loss=False,
        logger=loguru._logger.Logger,
    ):
        self.unify_seeds(
            self.seed
        )  # ensure reproducibility by unifying seeds across random, numpy, and torch.
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
                auxiliary_loss=auxiliary_loss,
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
        self, model, dataloader, optimizer, device, auxiliary_loss=False
    ):
        criterion = self.BT_loss(lambda_bt=5e-3, alpha=1.0).to(device)
        # if using kmer or coverage.
        auxiliary_criterion = self.modality_BT_loss(lambda_bt=5e-3, alpha=1.0).to(
            device
        )
        model.train()
        total_loss = 0.0
        stats = None

        for batch_i, batch_data in enumerate(dataloader):
            features1, features2, base_ids = batch_data
            features1 = features1.to(device, non_blocking=True)
            features2 = features2.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            output1, output2, kmer_z1, kmer_z2, cov_z1, cov_z2 = model(
                features1, features2
            )

            last_batch = batch_i == len(dataloader) - 1

            # fused_embedding derived loss.
            if last_batch:
                fuse_loss, stats = criterion(
                    feature1=output1, feature2=output2, return_stats=True
                )

            else:
                fuse_loss = criterion(feature1=output1, feature2=output2)

            if auxiliary_loss:
                L_kmer = 1.0
                L_cov = 1.0
                aux_loss_kmer = auxiliary_criterion(feature1=kmer_z1, feature2=kmer_z2)
                aux_loss_cov = auxiliary_criterion(feature1=cov_z1, feature2=cov_z2)
                loss = fuse_loss + L_kmer * aux_loss_kmer + L_cov * aux_loss_cov
            else:
                loss = fuse_loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )  # prevent the exploding gradient.

            optimizer.step()
            total_loss += fuse_loss.item()  # sum loss.

        avg_loss = total_loss / len(dataloader)
        return avg_loss, stats

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

        # copy raw fasta file to augmented_fragments_view0.fasta.

        return FragMent_Dict, All_Fragments, namelist
