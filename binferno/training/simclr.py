from pathlib import Path
import random
from collections import OrderedDict
import os

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.backends import cudnn
from pydantic import Field
import pyfastx
from torch.amp import GradScaler
import loguru
import pandas as pd
from typing import Dict, List, Tuple


from ..utils import calc_n50
from .base import BaseTrainer
from ..features import BinfernoFeatureBuilder
from .models.simclr import SimCLRNet, EmbeddingNet, NeuralNetwork, EmbNet2
from .InfoNCEloss import info_nce_loss, info_nce_loss_hard_weight
from .base import (
    BaseTrainer, 
    label_augmented_views, build_mask,
    TAXONOMY_DF, TAX_NODES, TAX_NODE_INDICES, PARENTAL_ARRAY
)

KMER_DIM = 136


class SimCLRTrainer(BaseTrainer):
    simclrnet: bool = Field(
        default=False,
        description="Whether to use SimCLRNet architecture for training. If False, use a simpler MLP architecture.",
    )
    proj: bool = Field(
        default=False,
        description="Whether to use a projection head in the NeuralNetwork architecture.",
    )
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
        default=6, description="Number of augmented views to generate for each contig."
    )
    seed: int = Field(
        default=42, description="Random seed for training and augmentation."
    )
    gated: bool = Field(
        default=False,
        description="Whether to use gated fusion in the Fusion module instead of attention fusion.",
    )
    attention: bool = Field(
        default=False,
        description="Whether to use attention fusion in the Fusion module instead of gated fusion.",
    )
    temperature: float | None = Field(
        default=None,
        description="Temperature parameter for the InfoNCE loss. "
        "By default, it is set based on the N50 of the input contigs. "
        "If the N50 is greater than 10,000, the temperature is set to 0.07; otherwise, it is set to 0.15.",
    )
    embnet: bool = Field(
        default=False,
        description="Whether to use the EmbNet2 architecture for training instead of SimCLRNet or NeuralNetwork.",
    )
    max_lr_epochs: int = Field(
        default=10,
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
    auxiliary_loss: bool = Field(
        default=False,
        description="Whether to include an auxiliary loss on the individual view embeddings in addition to the main InfoNCE loss on the fused embeddings. This can help encourage the model to learn useful representations in each view before fusion.",
    )
    dense_mode: bool = Field(
        default=False,
        description="Whether to use dense mode in the NeuralNetwork architecture, which adds additional fully connected layers after concatenating the kmer and coverage embeddings. If False, the model will directly output the final embedding after fusion without additional layers.",
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

        # after getting the augmented views, pass to binfernofeatures to get the features pooled.
        feature_result = features(
            FragMent_Dict=fragment_dict,
            All_Fragments=all_fragments,
            contigs=contigs,
            bam_path=bam,
            threads=threads,
            logger=logger,
            output_dir=output_aug_dir,
            max_norm_stacked=True,  # for simclr, we found that normalizing the coverage features before feeding into the model can lead to better performance, so we set std_first to True.
        )

        if isinstance(feature_result, tuple):
            feature_df, kmer_dim = feature_result
        else:
            feature_df = feature_result
            if hasattr(features, "_get_kmer_dim"):
                kmer_dim = features._get_kmer_dim()
            else:
                kmer_dim = 0

        # after getting the features, sort the features by method to ensure integrity before training.
        feature_df = self.sort_features_by_method(
            feature_df,
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

        cov_dim = sum([1 for col in feature_df.columns if "cov" in col])

        composition_dim = feature_df.shape[1] - cov_dim
        kmer_dim = kmer_dim if kmer_dim != KMER_DIM else KMER_DIM
        corgi_dim = composition_dim - kmer_dim
        proj_head_dim = 128

        # prepare model.
        if self.simclrnet:
            cov_model = EmbeddingNet(
                in_sz=cov_dim,
                out_sz=self.output_dim,  # 128
                emb_szs=[2048] * 3,
                ps=[0.2] * 2,
                actn=nn.LeakyReLU(),
                use_bn=True,
            ).to(device)
            model = SimCLRNet(
                in_sz=composition_dim + self.output_dim,
                out_sz=self.output_dim,
                emb_szs=[2048] * 3,
                use_bn=True,
                ps=[0.2] * 2,
                actn=nn.LeakyReLU(),
                cov_model=cov_model,
            ).to(device)
        elif self.embnet:
            model = EmbNet2(
                tnf_dim=composition_dim,
                coverage_dim=cov_dim,
                output_dim=self.output_dim,
                kmer_dim=kmer_dim,
            ).to(device)
        else:
            model = NeuralNetwork(
                tnf_dim=composition_dim,
                coverage_dim=cov_dim,
                output_dim=self.output_dim,
                kmer_dim=kmer_dim,
                gated=self.gated,
                attention=self.attention,
                corgi_dim=corgi_dim,
                proj=self.proj,
                dense_mode=self.dense_mode,
            ).to(device)

        logger.info(f"Model architecture:\n{model}")
        # prepare training.
        train_loader, optimizer, scheduler, dataset = self.prepare_trainer(
            data_df=feature_df,
            batch_size=self.batch_size,
            contig_length=self.contig_length,
            model=model,
            n_views=self.view_num,
            simclrnet=self.simclrnet,
            contigs=contigs,
            num_workers=threads,
        )

        model_path = output_train_dir / "best_model.pt"
        if model_path.exists():
            logger.info(
                f"Model already exists at {model_path}. Loading the model and skipping training."
            )
            model.load_state_dict(torch.load(model_path))
        elif self.simclrnet:
            model = self.simclrnet_train(
                dataloader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                epochs=self.epochs,
                model=model,
                n_cov=cov_dim,
                n_views=self.view_num,
                temperature=temperature,
                output_dir=output_train_dir,
                logger=logger,
            )
        elif self.embnet:
            model = self.train(
                dataloader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                epochs=self.epochs,
                model=model,
                n_cov=cov_dim,
                n_views=self.view_num,
                temperature=temperature,
                output_dir=output_train_dir,
                logger=logger,
            )
        else:
            model = self.train(
                dataloader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                method="simclr",
                model=model,
                logger=logger,
                n_cov=cov_dim,
                output_dir=output_train_dir,
                auxiliary_loss=self.auxiliary_loss,
                temperature=temperature,
            )

        if self.gated:
            with torch.no_grad():
                sample_batch = next(iter(train_loader))
                x = sample_batch[0][:32].to(device)
                tnf = x[:, : model.tnf_enc_dim]
                cov = x[:, model.tnf_enc_dim :]
                kmer_emb = model.tnf_encoder(tnf)
                cov_emb = model.coverage_encoder(cov)
                cat = torch.cat([kmer_emb, cov_emb], dim=1)
                kg = model.fusion_network.kmer_gate(cat)
                cg = model.fusion_network.coverage_gate(cat)
                logger.info(
                    f"kmer_gate mean: {kg.mean():.3f}, coverage_gate mean: {cg.mean():.3f}"
                )
                logger.info(
                    f"kmer_emb norm: {kmer_emb.norm(dim=1).mean():.3f}, cov_emb norm: {cov_emb.norm(dim=1).mean():.3f}"
                )

        # in case embeddings isn't exist, get embeddings from the trained model and save to file.
        if self.simclrnet:
            embeddings = self.simclr_get_embs(
                model=model,
                data=dataset,
                output_path=embedding_path,
                device=device,
                kmer_len=kmer_dim,
            )
        else:
            embeddings = self.get_embeddings(
                model=model,
                data_df=feature_df,
                output_path=embedding_path,
                batch_size=self.batch_size,
                save_individual_embs=self.get_individual_embs,
            )

        # normalize embeddings, but here the embedding is pd.dataframe, so we can use F.normalize from torch after converting the dataframe to tensor, then convert back to dataframe.
        embeddings = pd.DataFrame(
            F.normalize(torch.from_numpy(embeddings.values).float(), dim=1).numpy(),
            index=embeddings.index,
            columns=embeddings.columns,
        )

        return embeddings

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

    def prepare_trainer(
        self,
        data_df,
        batch_size: int = 1024,
        contig_length: int = 1000,
        model=None,
        n_views=6,
        simclrnet=False,
        contigs=None,
        num_workers=4,
        **kwargs,
    ):
        # determine the namelist for the dataset.
        orig_df = data_df[data_df.index.str.endswith("0")]
        namelist = [index.rsplit(".", 1)[0] for index in orig_df.index]
        # slice the data_df by the length of namelist, make sure to assert length for sliced data to be same.
        dataset = list(
            torch.split(torch.from_numpy(data_df.values).float(), len(namelist))
        )
        epochs = kwargs.get("epochs", 300)
        assert (
            len(dataset) == n_views
        ), f"Number of views in the dataset ({len(dataset)}) does not match the expected number of views ({n_views}). \
            Please check the data_df and the n_views parameter."

        length_weight = self.length_weight(contig=contigs)
        if not length_weight:
            raise ValueError(
                "Length weight is empty. Please check the contigs input and the length_weight function."
            )

        # breakpoint()
        train_dataset = torch.utils.data.TensorDataset(
            *[
                dataset[i][
                    np.array([length_weight[name] for name in namelist])
                    >= contig_length
                ]
                for i in range(n_views)
            ]
        )

        # Keep multiprocessing options conservative to avoid native worker crashes
        # on some PyTorch/HPC stacks (fork + pinned memory + persistent workers).
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

        train_loader = torch.utils.data.DataLoader(**dataloader_kwargs)

        if simclrnet:
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=1e-3, weight_decay=1e-4
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=0, last_epoch=-1
            )
        else:
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

        return train_loader, optimizer, scheduler, dataset

    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

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
        auxiliary_loss: bool = False,
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

        # Create once so the index cache persists across all epochs

        epoch_iter = tqdm(range(epochs), desc="Training Epochs")
        early_stopping = EarlyStoppingManagerSimCLR(patience=15)
        for epoch in epoch_iter:
            avg_loss, stats = self.train_an_epoch(
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                device=device,
                temperature=temperature,
                n_cov=n_cov,
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
        auxiliary_loss=False,
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

        for i, features in enumerate(dataloader):
            contig_features = torch.cat(features, dim=0).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            actual_batch_size = contig_features.shape[0] // self.view_num
            enc_feature, kmer_emb, cov_emb = model.forward_one(contig_features)
            loss, batch_acc, _ = info_nceloss_fn(
                enc_feature,
                batch_size=actual_batch_size,
                device=device,
                n_views=self.view_num,
            )

            if auxiliary_loss:
                L_kmer = 1.0
                L_cov = 1.0
                kmer_loss, _, _ = info_nceloss_fn(
                    kmer_emb,
                    batch_size=actual_batch_size,
                    device=device,
                    n_views=self.view_num,
                )
                cov_loss, _, _ = info_nceloss_fn(
                    cov_emb,
                    batch_size=actual_batch_size,
                    device=device,
                    n_views=self.view_num,
                )
                loss = loss + L_kmer * kmer_loss + L_cov * cov_loss

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

    def simclrnet_train(
        self,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        epochs: int = 300,
        model=None,
        n_cov: int = 0,
        n_views: int = 6,
        temperature: float = 0.15,
        output_dir: str = None,
        **kwargs,
    ):
        self.unify_seeds(
            self.seed
        )  # ensure reproducibility by unifying seeds across random, numpy, and torch.
        info_nceloss_fn = self.info_nce_loss(temperature=temperature).to(device)
        if n_cov <= 0:
            raise ValueError(
                "n_cov must be a positive integer representing the number of coverage features in the input data."
            )

        scaler = GradScaler(enabled=device.type == "cuda")
        logger = kwargs.get("logger", None)
        if logger is None:
            raise ValueError(
                "Logger is not provided for training. Please provide a logger in the kwargs with key 'logger'."
            )

        logger.info("Starting SimCLR training with {} epochs.".format(epochs))
        earlystop_epoch = 0
        for epoch_counter in tqdm(range(epochs), desc="SimCLR Training Epochs"):
            total_loss = 0.0
            total_acc1 = 0.0
            for _, contig_features in enumerate(dataloader):
                contig_features = torch.cat(contig_features, dim=0)
                contig_features = contig_features.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(
                    enabled=device.type == "cuda", device_type=device.type
                ):
                    features, _ = model(
                        contig_features[:, :-n_cov], contig_features[:, -n_cov:]
                    )
                    actual_batch_size = contig_features.shape[0] // n_views
                    loss, batch_acc = info_nceloss_fn(
                        features,
                        batch_size=actual_batch_size,
                        device=device,
                        n_views=n_views,
                    )
                    total_loss += loss.item()
                    total_acc1 += batch_acc

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if epoch_counter >= 10:
                scheduler.step()
            avg_loss = total_loss / len(dataloader)
            avg_acc1 = total_acc1 / len(dataloader)
            logger.info(
                f"Epoch {epoch_counter+1}/{epochs} - SimCLR Loss: {avg_loss:.4f}, Top-1 Accuracy: {avg_acc1:.2f}%, LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

            if epoch_counter >= 10 and avg_acc1 >= 99.0:
                earlystop_epoch += 1
            else:
                earlystop_epoch = 0

            if earlystop_epoch >= 5:
                logger.info(
                    f"Early stopping at epoch {epoch_counter+1} with top-1 accuracy: {avg_acc1:.2f}%"
                )
                break
        logger.info("Finished SimCLR training.")
        # save model.
        torch.save(model.state_dict(), os.path.join(output_dir, "simclr_model.pt"))
        return model

    def embnet_get_embs(self, model, data, namelist, output_path, device):
        with torch.no_grad():
            model.eval()
            bs_ = 1024

            print(len(data))

            out = np.concatenate(
                [
                    model(
                        data[0][i : i + bs_, :].to(device),
                    )[0]
                    .to("cpu")
                    .numpy()
                    for i in range(0, len(data[0]), bs_)
                ],
                axis=0,
            )
            embeddings_df = pd.DataFrame(out, index=namelist)

            embeddings_df.to_csv(str(output_path), sep="\t", header=True, index=True)
        return embeddings_df

    def simclr_get_embs(self, model, data, namelist, output_path, device, kmer_len):
        with torch.no_grad():
            model.eval()
            bs_ = 1024

            print(len(data))

            out = np.concatenate(
                [
                    model(
                        data[0][i : i + bs_, -kmer_len:].to(device),
                        data[0][i : i + bs_, :-kmer_len].to(device),
                    )[0]
                    .to("cpu")
                    .numpy()
                    for i in range(0, len(data[0]), bs_)
                ],
                axis=0,
            )
            embeddings_df = pd.DataFrame(out, index=namelist)

            embeddings_df.to_csv(str(output_path), sep="\t", header=True, index=True)

            covout = np.concatenate(
                [
                    model(
                        data[0][i : i + bs_, -kmer_len:].to(device),
                        data[0][i : i + bs_, :-kmer_len].to(device),
                    )[1]
                    .to("cpu")
                    .numpy()
                    for i in range(0, len(data[0]), bs_)
                ],
                axis=0,
            )
            embeddings_df = pd.DataFrame(covout, index=namelist)
            outfile = output_path.parent / "cov_embeddings.tsv"
            embeddings_df.to_csv(str(outfile), sep="\t", header=True, index=True)
        return embeddings_df

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

        # save the fragments to file for potential reuse in the future stages, to save time for augmentation in the future.

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
        # raw_fasta = os.path.join(output_dir, f"augmented_fragments_view0.fasta")
        # if not os.path.exists(raw_fasta):
        #     os.symlink(os.path.abspath(contigs), raw_fasta) # create a symbolic link to the original contigs fasta file for view 0, which is the unaugmented view.

        return FragMent_Dict, All_Fragments


class EarlyStoppingManagerSimCLR:
    def __init__(self, patience=10):
        self.patience = patience
        self._lowest_loss = float("inf")
        self._top_acc = 0.0
        self._saved_state = None
        self._stagnant_epochs = 0

    def check_improvement(self, loss, top_acc, model_state):
        loss_improved = loss < self._lowest_loss - 1e-4
        acc_improved = top_acc > self._top_acc + 0.05  # at least 0.05% gain
        improved = loss_improved or acc_improved
        # improved = loss < self._lowest_loss or top_acc > self._top_acc

        if improved:
            if loss < self._lowest_loss:
                self._lowest_loss = loss
            if top_acc > self._top_acc:
                self._top_acc = top_acc
            self._saved_state = model_state
            self._stagnant_epochs = 0
            return

        self._stagnant_epochs += 1

    def should_stop(self):
        return self._stagnant_epochs >= self.patience

    def get_best_state(self):
        """Return the best checkpoint state and its associated loss and top accuracy."""
        return self._saved_state, self._lowest_loss, self._top_acc
