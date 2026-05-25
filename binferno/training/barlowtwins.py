import itertools
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
from .neuralnetwork import NeuralNetwork
from ..features import BinfernoFeatures
from .base import BaseTraining


class BarlowTwinsTraining(BaseTraining):
    features: BinfernoFeatures

    def __call__(
        self,
        contigs: str,
        bam: str,
        output_dir: Path,
        view_num: int = 6,
        contig_length: int = 1000,
        logger=None,
        threads: int = 4,
        epochs: int = 300,
        n_layers: int = 2,
        output_dim: int = 128,
        multiscale: bool = False,
        batch_size: int = 1024,
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
            view_num=view_num,
            contig_length=contig_length,
        )

        feature_df = self.features(
            FragMent_Dict=fragment_dict,
            All_Fragments=all_fragments,
            contigs=contigs,
            bam_path=bam,
            threads=threads,
            logger=logger,
            output_dir=output_dir
        )

        feature_df = self.features.sort_features_by_method(
            feature_df, contig_ids=namelist, simclr=False,
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
            n_layers=n_layers,
            output_dim=output_dim,
            multiscale=multiscale,
        ).to(device)

        dataloader, optimizer, scheduler = self.prepare_training(
            model=model, data_df=feature_df, batch_size=batch_size, epochs=epochs
        )

        model_path = output_dir / "best_model.pt"
        if not model_path.exists():
            model = self.Train(
                dataloader=dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                method="barlow_twins",
                epochs=epochs,
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
                batch_size=batch_size,
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

    def prepare_training(
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

    def Train(
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
