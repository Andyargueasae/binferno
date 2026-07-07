from pathlib import Path
import random
import os
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
from torch.backends import cudnn
from pydantic import Field
import pyfastx
import loguru
import pandas as pd
from typing import Dict, List, Optional, Tuple
# from torch.cuda.amp import GradScaler, autocast

from ..utils import calc_n50
from ..features import BinfernoFeatureBuilder
from .models.simclr import NeuralNetwork
from .InfoNCEloss import info_nce_loss, info_nce_loss_hard_weight
from .base import (
    BaseTrainer, 
    label_augmented_views, build_mask, 
    TAXONOMY_DF, TAX_NODES, TAX_NODE_INDICES, PARENTAL_ARRAY
)

KMER_DIM = 136

class SimCLRTrainer(BaseTrainer):
    learning_rate: float = Field(
        default=1e-3, description="Learning rate for training."
    )
    batch_size: int = Field(default=1024, description="Batch size for training.")
    epochs: int = Field(default=300, description="Number of epochs for training.")
    output_dim: int = Field(
        default=128, description="Output dimension for the embeddings."
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
    temperature: float | None = Field(
        default=None,
        description="Temperature parameter for the InfoNCE loss. "
        "By default, it is set based on the N50 of the input contigs. "
        "If the N50 is greater than 10,000, the temperature is set to 0.07; otherwise, it is set to 0.15.",
    )
    max_lr_epochs: int = Field(
        default=0,
        description="Number of epochs to wait before resuming learning rate scheduling after reaching the maximum learning rate.",
    )
    hard_weighted_nce: bool = Field(
        default=False,
        description="Whether to use the hard weighted InfoNCE loss instead of the standard InfoNCE loss. This loss gives more weight to hard negative samples during training.",
    )
    get_individual_embs: bool = Field(
        default=False,
        description="Whether to save individual embeddings for each view separately. If False, only the final embeddings after fusion are saved.",
    )
    hidden_dim: int = Field(
        default=1024,
        description="Hidden dimension for the dense layers in the NeuralNetwork architecture when dense_mode is True. This parameter is ignored if dense_mode is False.",
    )
    patience: int = Field(
        default=10,
        description="Number of epochs to wait for improvement in the validation loss before stopping training early.",
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
        """
        # Set the temperature based on the N50 of the input contigs if not provided.
        if self.temperature:
            temperature = self.temperature
        else:
            N_50 = calc_n50(contigs)
            logger.info(f"N50 of the input contigs: {N_50}")
            temperature = 0.15
            if N_50 > 10000:
                temperature = 0.07

        # the first step is to generate the augmented views and then turn views into vectors.
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

        features, kmer_dim = feature_result
        canonical_feature_df, taxonomy_features = features
        canonical_feature_df = self.sort_features_by_method(
            canonical_feature_df,
            contig_ids=namelist,
            simclr=True,
        )

        if torch.cuda.is_available():
            device = torch.device("cuda")
            cudnn.deterministic = True
            cudnn.benchmark = True
        else:
            device = torch.device("cpu")
            torch.set_num_threads(threads)
        logger.info(f"Using device: {device}")

        cov_dim = sum([1 for col in canonical_feature_df.columns if "cov" in col])
        kmer_dim = kmer_dim if kmer_dim != KMER_DIM else KMER_DIM
        
        hierarchical_taxonomy = False
        if taxonomy_features is None:
            taxonomy_dim = 0
        elif isinstance(taxonomy_features, pd.DataFrame):
            taxonomy_dim = taxonomy_features.shape[1]
        elif isinstance(taxonomy_features, dict):
            taxonomy_dim = 1 # one integer for each contig representing label.
            hierarchical_taxonomy = True
        
        if hierarchical_taxonomy:
            tax_nodes = taxonomy_features[TAX_NODES]
            node_indices = taxonomy_features[TAX_NODE_INDICES]
            parental_array = taxonomy_features[PARENTAL_ARRAY]
        else:
            tax_nodes = None
            node_indices = None
            parental_array = None

        model = NeuralNetwork(
            tnf_dim=kmer_dim,
            coverage_dim=cov_dim,
            output_dim=self.output_dim,
            tax_emb_dim=taxonomy_dim,
            hidden_dim=self.hidden_dim,
            hierarchical_taxonomy=hierarchical_taxonomy,
            tax_nodes=tax_nodes,
            node_indices=node_indices,
            parental_array=parental_array,
        ).to(device)

        logger.info(f"Model architecture:\n{model}")
        # prepare training.
        train_loader, optimizer, scheduler = self.prepare_trainer(
            data_df=canonical_feature_df,
            batch_size=self.batch_size,
            contig_length=self.contig_length,
            model=model,
            n_views=self.view_num,
            contigs=contigs,
            num_workers=threads,
            taxonomy_features=taxonomy_features,
        )

        model_path = output_train_dir / "best_model.pt"
        if model_path.exists():
            logger.info(
                f"Model already exists at {model_path}. Loading the model and skipping training."
            )
            model.load_state_dict(torch.load(model_path))
        else:
            model = self.train(
                dataloader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                method="simclr",
                model=model,
                epochs=self.epochs,
                logger=logger,
                n_cov=cov_dim,
                output_dir=output_train_dir,
                temperature=temperature,
            )

        if isinstance(taxonomy_features, pd.DataFrame):
            taxonomy_df = taxonomy_features.loc[canonical_feature_df.index]
        elif isinstance(taxonomy_features, dict):
            taxonomy_df = taxonomy_features[TAXONOMY_DF].set_index("contig_id")
        else: 
            taxonomy_df = None

        embeddings = self.get_embeddings(
            model=model,
            data_df=canonical_feature_df,
            taxonomy_df=taxonomy_df,
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
    
    def prepare_trainer(
        self,
        model,
        data_df: pd.DataFrame,
        n_views: int=6,
        contigs: Path=None,
        num_workers: int = 4,
        batch_size: int = 1024,
        contig_length: int = 1000,
        taxonomy_features: Optional[Dict|pd.DataFrame|None]=None,
    ):
        # determine the namelist for the dataset.
        orig_df = data_df[data_df.index.str.endswith("0")]
        namelist = [index.rsplit(".", 1)[0] for index in orig_df.index]
        # slice the data_df by the length of namelist, make sure to assert length for sliced data to be same.
        dataset = list(
            torch.split(torch.from_numpy(data_df.values).float(), len(namelist))
        )
        length_weight = self.length_weight(contig=contigs)
        dataset = [
            dataset[i][
                np.array([length_weight[name] for name in namelist])
                >= contig_length
            ] for i in range(n_views)
        ]
        epochs = self.epochs 
        assert (
            len(dataset) == n_views
        ), f"Number of views in the dataset ({len(dataset)}) does not match the expected number of views ({n_views}). \
            Please check the data_df and the n_views parameter."

        length_weight = self.length_weight(contig=contigs)
        if not length_weight:
            raise ValueError(
                "Length weight is empty. Please check the contigs input and the length_weight function."
            )

        if taxonomy_features is not None:
            if isinstance(taxonomy_features, pd.DataFrame):
                
                taxonomy_features = taxonomy_features.reindex(data_df.index)

                tax_tensor = list(
                    torch.split(torch.from_numpy(taxonomy_features.values).float(), len(namelist))
                ) # should have been the same size at axis=0 as dataset.
                tax_tensor = [
                    tax_tensor[i][
                        np.array([length_weight[name] for name in namelist])
                        >= contig_length
                    ] for i in range(n_views)
                ][:n_views] # make sure the number of views is the same as n_views
                
                train_dataset = TensorDataset(
                    *dataset, *tax_tensor 
                )

            elif isinstance(taxonomy_features, dict):
                taxonomy_labels = label_augmented_views(
                    taxonomy_features[TAXONOMY_DF],
                    data_df.index.tolist(),
                    node_indices = taxonomy_features[TAX_NODE_INDICES]
                )
                taxonomy_labels = self.sort_features_by_method(
                    taxonomy_labels,
                    contig_ids=namelist,
                    simclr=True,
                )
                _, mask = build_mask(
                    contig_names=namelist,
                    contig_taxonomy=dict(zip(taxonomy_features[TAXONOMY_DF]["contig_id"], taxonomy_features[TAXONOMY_DF]["tax_node_idx"])),
                    node_indices=taxonomy_features[TAX_NODE_INDICES],
                )

                mask = mask[np.array([length_weight[name] for name in namelist]) >= contig_length]
                
                tax_label_dataset = list(
                    torch.split(torch.from_numpy(taxonomy_labels.values.astype(np.int32)).float(), len(namelist))
                )
                target_idx = tax_label_dataset[0][
                    np.array([length_weight[name] for name in namelist])
                    >= contig_length
                ]
                del tax_label_dataset

                train_dataset = TensorDataset(
                    *dataset,
                    target_idx, # can it be vanished because tax_label_dataset are all redundantly the same label? they are from the same contig!
                    mask
                )
                
        else:
            train_dataset = TensorDataset(
            *[
                dataset[i][
                    np.array([length_weight[name] for name in namelist])
                    >= contig_length
                ]
                for i in range(n_views)
            ]
        )

        requested_workers = num_workers
        if requested_workers < 0:
            requested_workers = 0

        dataloader_kwargs = {
            "dataset": train_dataset,
            "batch_size": batch_size,
            "shuffle": True,
            "num_workers": requested_workers,
            "pin_memory": torch.cuda.is_available(),
            "drop_last": True,
        }

        if requested_workers > 0:
            dataloader_kwargs.update(
                {
                    "persistent_workers": True,
                    "prefetch_factor": 2,
                    "multiprocessing_context": "spawn",
                }
            )

        train_loader = DataLoader(**dataloader_kwargs)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-4, weight_decay=0.05, betas=(0.9, 0.95)
        )
        scheduler = self.warmup_cosine_scheduler(
            optimizer,
            learning_rate=self.learning_rate,
            min_learning_rate=1e-4,
            epochs=epochs,
            warmup=False,
            max_lr_hold_epochs=self.max_lr_epochs,
        )

        return train_loader, optimizer, scheduler

    def length_weight(self, contig: str):
        seq_obj = pyfastx.Fastx(contig)
        contig_length = dict()
        for name, seq in seq_obj:
            contig_length[name] = len(seq)
        return contig_length

    def train(
        self,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        logger: loguru._logger.Logger,
        method: str = "barlow_twins",
        epochs: int = 300,
        model=None,
        n_cov: int = 0,
        output_dir: Path = None,
        temperature: float = 0.15,
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
        early_stopping = EarlyStoppingManagerSimCLR(patience=self.patience)
        for epoch in epoch_iter:
            avg_loss, stats = self.train_an_epoch(
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                device=device,
                temperature=temperature,
                n_cov=n_cov,
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

            early_stopping.check_improvement(
                avg_loss, stats["top1_acc"], model.state_dict()
            )
            if early_stopping.should_stop():
                logger.info(
                    f"Early stopping after {epoch+1} epochs (patience: {early_stopping.patience})"
                )
                break

            best_state, best_loss, best_acc = early_stopping.get_best_state()

        if best_state is not None:
            model.load_state_dict(best_state)
            logger.info(
                f"Loaded best model with loss: {best_loss:.4f} and accuracy: {best_acc:.4f}"
            )

        torch.save(model.state_dict(), MODEL_PATH)
        logger.info(f"Model saved to {MODEL_PATH}")

        return model

    def train_an_epoch(
        self,
        model,
        dataloader,
        optimizer,
        device,
        temperature=0.15,
        n_cov=0,
    ):
        # get the info nce loss, and see if auxiliary loss is needed.
        if not self.hard_weighted_nce:
            info_nceloss_fn = info_nce_loss(temperature=temperature).to(device)
        else:
            info_nceloss_fn = info_nce_loss_hard_weight(temperature=temperature).to(
                device
            )

        model.train()
        stats = None
        total_loss = 0.0
        total_acc = 0.0
        if n_cov < 0 or n_cov == 0:
            raise ValueError("n_cov must be non-negative.")

        h_tax = model.hierarchical_taxonomy

        for i, features in enumerate(dataloader):
            contig_features, tax_features = features[:self.view_num], features[self.view_num:]
            contig_features = torch.cat(contig_features, dim=0).to(device, non_blocking=True)
            if tax_features and not h_tax:
                tax_features = torch.cat(tax_features, dim=0).to(device, non_blocking=True)
            elif tax_features and h_tax:
                continue
            else:
                tax_features = None
            optimizer.zero_grad(set_to_none=True)
            actual_batch_size = contig_features.shape[0] // self.view_num
            enc_feature, cov_emb, tax_emb = model(contig_features, tax_emb=tax_features) 
            loss, batch_acc, _ = info_nceloss_fn(
                enc_feature,
                batch_size=actual_batch_size,
                device=device,
                n_views=self.view_num,
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += batch_acc
            last_batch = i == len(dataloader) - 1

            if last_batch:
                stats = {
                    "loss": total_loss / len(dataloader),
                    "top1_acc": total_acc / len(dataloader),
                }

        return total_loss / len(dataloader), stats

    def gen_views(
        self,
        contigs=None,
        output_dir=None,
        view_num=6,
        contig_length=1000,
    ) -> Tuple[Dict[str, pd.DataFrame], List, List]:
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

        return FragMent_Dict, All_Fragments, namelist

    def augmentation(self, fasta_file=None, num_views=6, length_threshold=1000):
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

            augmented_fragments = self.simclr_augmentation(
                seqs=seq,
                clean_header=clean_header,
                length_threshold=length_threshold,
                num_views=num_views,
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

    def simclr_augmentation(
        self,
        seqs: str,
        clean_header: str,
        length_threshold: int,
        num_views: int,
    ) -> List:
        seq_len = len(seqs)
        fragments = []
        fragments.append((f"{clean_header}.0", seqs, 0, seq_len))
        for i in range(1, num_views):
            try:
                start = random.randint(0, seq_len - (length_threshold + 1))
            except ValueError:
                start = 0
            sim_len = random.randint(length_threshold, seq_len - start)
            end = start + sim_len - 1
            sequence = str(seqs[start : end + 1])
            seq_id = f"{clean_header}.{i}"
            fragments.append((seq_id, sequence, start, sim_len))
        return fragments

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


class EarlyStoppingManagerSimCLR:
    def __init__(self, patience=10):
        self.patience = patience
        self._lowest_loss = float("inf")
        self._top_acc = 0.0
        self._saved_state = None
        self._stagnant_epochs = 0

    def check_improvement(self, loss, top_acc, model_state):
        acc_over_99 = (
            top_acc > 99.0
        )  # at least 0.5% gain or accuracy above 99% to consider it an improvement
        # improved = loss < self._lowest_loss or top_acc > self._top_acc

        if acc_over_99:
            self._stagnant_epochs += 1

    def should_stop(self):
        return self._stagnant_epochs >= self.patience

    def get_best_state(self):
        """Return the best checkpoint state and its associated loss and top accuracy."""
        return self._saved_state, self._lowest_loss, self._top_acc
