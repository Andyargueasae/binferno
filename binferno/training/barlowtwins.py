import hashlib
import itertools
from typing import Dict
from pydantic import Field
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.backends import cudnn
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from .models.barlow_twins import NeuralNetwork
from ..features import BinfernoFeatureBuilder
from .base import BaseTrainer
import pyfastx
from collections import OrderedDict


class BarlowTwinsTrainer(BaseTrainer):
    features: BinfernoFeatureBuilder

    learning_rate: float = Field(
        default=2.5e-3, description="Learning rate for training."
    )
    batch_size: int = Field(default=1024, description="Batch size for training.")
    epochs: int = Field(default=300, description="Number of epochs for training.")
    weight_decay: float = Field(default=1e-4, description="Weight decay for training.")
    seed: int = Field(default=42, description="Random seed for training.")
    output_dim: int = Field(
        default=128, description="Output dimension for the embeddings."
    )
    n_layers: int = Field(
        default=2, description="Number of layers for the neural network."
    )
    contig_length: int = Field(
        default=1000, description="Minimum contig length to be included in training."
    )
    view_num: int = Field(
        default=6, description="Number of augmented views to generate for each contig."
    )
    seed: int = Field(
        default=42, description="Random seed for training and augmentation."
    )

    def __call__(
        self,
        contigs: str,
        bam: str,
        output_dir: Path,
        logger=None,
        threads: int = 4,
        multisample: bool = False,
        **kwargs,
    ):
        """
        Args:
         - contigs: path to the contigs fasta file.
         - bam: path to the bam file for coverage information.
         - output_dir: directory to save the model and logs.
         - view_num: number of augmented views to generate for each contig.
         - contig_length: minimum length of contigs to consider for augmentation.
        """

        fragment_dict, all_fragments, namelist = self.gen_views(
            contigs=contigs,
            simclr=False,
            barlow_twins=True,
            output_dir=output_dir,
            view_num=self.view_num,
            contig_length=self.contig_length,
        )

        feature_df = self.features(
            FragMent_Dict=fragment_dict,
            All_Fragments=all_fragments,
            contigs=contigs,
            bam_path=bam,
            threads=threads,
            logger=logger,
            output_dir=output_dir,
        )

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
        kmer_dim = 136
        proj_head_dim = 512
        model = NeuralNetwork(
            input_dim=feature_df.shape[1],
            cov_dim=cov_dim,
            composition_dim=composition_dim,
            kmer_dim=kmer_dim,
            proj_head_dim=proj_head_dim,
            n_layers=self.n_layers,
            output_dim=self.output_dim,
        ).to(device)

        dataloader, optimizer, scheduler = self.prepare_trainer(
            model=model,
            data_df=feature_df,
            batch_size=self.batch_size,
            epochs=self.epochs,
        )

        model_path = output_dir / "best_model.pt"
        if not model_path.exists():
            model = self.train(
                dataloader=dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                method="barlow_twins",
                epochs=self.epochs,
                model=model,
                output_dir=output_dir,
                logger=logger,
            )
        else:
            logger.info(
                f"Model already exists at {model_path}. Loading model from file."
            )
            model.load_state_dict(torch.load(model_path, map_location=device))

        embedding_path = output_dir / "embeddings.tsv"
        if not embedding_path.exists():
            orig_df = feature_df[feature_df.index.str.endswith(".original")]
            namelist = [index.rsplit(".", 1)[0] for index in orig_df.index]
            embeddings = self.get_embeddings(
                model=model,
                data_df=feature_df,
                output_path=embedding_path,
                batch_size=self.batch_size,
                save_individual_embs=False,
            )

        else:
            logger.info(
                f"Embeddings already exist at {embedding_path}. Loading embeddings from file."
            )
            embeddings = pd.read_csv(embedding_path, sep="\t", header=0, index_col=0)

        return embeddings

    def seed_worker():
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

        def forward(self, feature1, feature2, return_stats=False):
            N, D = feature1.size()

            z1 = (feature1 - feature1.mean(dim=0)) / (feature1.std(dim=0) + self.eps)
            z2 = (feature2 - feature2.mean(dim=0)) / (feature2.std(dim=0) + self.eps)

            C = torch.mm(z1.T, z2) / N

            diag = torch.diagonal(C)
            invariance_loss = (
                (diag - 1.0).pow(2).sum()
            )  # this is the invariance loss, which encourages the diagonal elements of the correlation matrix to be close to 1, meaning that the corresponding features in the two views are highly correlated.

            off_diag_mask = ~torch.eye(D, dtype=torch.bool, device=feature1.device)
            redundancy_loss = (
                C[off_diag_mask].pow(2).sum()
            )  # off diagonal elements of the correlation matrix to be close to 0, meaning that different features are decorrelated and carry unique information.

            loss = self.alpha * invariance_loss + self.lambda_bt * redundancy_loss
            if return_stats:
                with torch.no_grad():
                    off_diag = C[off_diag_mask]
                    stats = {
                        "mean_diagonal": diag.mean().item(),
                        "std_diagonal": diag.std().item(),
                        "mean_abs_off_diagonal": off_diag.abs().mean().item(),
                        "max_abs_off_diagonal": off_diag.abs().max().item(),
                        "invariance_loss": invariance_loss.item(),
                        "redundancy_loss": redundancy_loss.item(),
                    }
                return loss, stats

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
            pair_records = []

            X = self.data_df.to_numpy(dtype=np.float64)

            print("Before tensor construction:")
            print("X shape:", X.shape)
            print("NaN count:", np.isnan(X).sum())
            print("Inf count:", np.isinf(X).sum())
            print("Finite:", np.isfinite(X).all())

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

            for idx1, idx2 in self.training_pairs:
                feat1 = self.data_df.iloc[idx1].to_numpy(dtype=np.float64)
                feat2 = self.data_df.iloc[idx2].to_numpy(dtype=np.float64)
                base_id = self.index_to_base_id[idx1]
                pair_records.append((feat1, feat2, base_id))

            features1 = torch.tensor(
                np.array([record[0] for record in pair_records]),
                dtype=torch.float32,
            )
            features2 = torch.tensor(
                np.array([record[1] for record in pair_records]),
                dtype=torch.float32,
            )
            base_ids = torch.tensor(
                np.array([record[2] for record in pair_records]),
                dtype=torch.long,
            )

            return TensorDataset(features1, features2, base_ids)

    def prepare_trainer(
        self, model, data_df, batch_size: int = 1024, logger=None, **kwargs
    ):
        dataset_builder = self.BuildDataset(data_df=data_df)
        train_dataset = dataset_builder.construct_tensordatasets_from_pairs()

        effective_batch_size = batch_size
        epochs = kwargs.get("epochs", 300)
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
        }

        dataloader = DataLoader(train_dataset, **dataloader_kwargs)
        optimizer = optim.AdamW(
            model.parameters(), lr=1e-4, weight_decay=0.05, betas=(0.9, 0.95)
        )
        scheduler = self.warmup_cosine_scheduler(
            optimizer,
            learning_rate=2.5e-3,
            min_learning_rate=1e-4,
            epochs=epochs,
            batch_size=effective_batch_size,
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
        logger=None,
        **kwargs,
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
        early_stopping = self.EarlyStoppingManager(patience=15)
        for epoch in epoch_iter:

            avg_loss, stats = self.train_an_epoch(
                model=model, dataloader=dataloader, optimizer=optimizer, device=device
            )
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

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

    def train_an_epoch(self, model, dataloader, optimizer, device):
        criterion = self.BT_loss(lambda_bt=5e-3, alpha=1.0).to(device)
        model.train()
        total_loss = 0.0
        stats = None

        for batch_i, batch_data in enumerate(dataloader):
            features1, features2, base_ids = batch_data
            features1, features2, base_ids = (
                features1.to(device),
                features2.to(device),
                base_ids.to(device),
            )

            optimizer.zero_grad()

            output1, output2 = model(features1, features2)

            last_batch = batch_i == len(dataloader) - 1

            if last_batch:
                loss, stats = criterion(
                    feature1=output1, feature2=output2, return_stats=True
                )

            else:
                loss = criterion(feature1=output1, feature2=output2)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )  # prevent the exploding gradient.
            # then, do the optimization.
            optimizer.step()
            total_loss += loss.item()  # sum loss.

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

            print(
                f"Generated {len(fragments)} fragments for large contig {header} "
                f"(length {seq_len}): {len(left_fragments)} from first half, "
                f"{len(right_fragments)} from second half"
            )
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

            print(
                f"Generated {len(fragments)} fragments for {header} (length {seq_len})"
            )

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
        method: str = "barlow_twins",
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

        assert method in [
            "barlow_twins",
            "simclr",
        ], "Method must be either 'barlow_twins' or 'simclr'. \
            Please set the method parameter accordingly."

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
            output_dir=str(output_dir),
            num_views=view_num,
            length_threshold=contig_length,
            method="barlow_twins",
        )

        return FragMent_Dict, All_Fragments

    def gen_views(
        self,
        contigs=None,
        simclr=False,
        barlow_twins=False,
        output_dir=None,
        view_num=6,
        contig_length=1000,
    ) -> Dict[str, pd.DataFrame]:
        # process input contigs into augmented fragments.
        FragMent_Dict, All_Fragments = self.augment_views(
            contigs=contigs,
            view_num=view_num,
            contig_length=contig_length,
            output_dir=output_dir,
            simclr=simclr,
            barlow_twins=barlow_twins,
        )

        namelist = list(
            FragMent_Dict.keys()
        )  # the ordered contig names which will be used for indexing the features.

        return FragMent_Dict, All_Fragments, namelist
