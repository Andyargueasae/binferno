from pathlib import Path
import random
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.backends import cudnn
from .models.simclr import SimCLRNet, EmbeddingNet, NeuralNetwork
from .base import BaseTrainer
from ..features import BinfernoFeatureBuilder
from pydantic import Field
import pyfastx
from torch.amp import GradScaler
import os
import pandas as pd
from typing import Dict, List
from collections import OrderedDict

KMER_DIM = 136


class SimCLRTrainer(BaseTrainer):
    features: BinfernoFeatureBuilder
    simclrnet: bool = Field(
        default=False,
        description="Whether to use SimCLRNet architecture for training. If False, use a simpler MLP architecture.",
    )
    no_proj: bool = Field(
        default=False,
        description="Whether to use a projection head in the NeuralNetwork architecture.",
    )
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
        temperature: float = 0.15,
        multisample: bool = False,
        **kwargs,
    ):
        """
        Args:
         - contigs: path to the contigs fasta file.
         - bam: path to the bam file for coverage information.
        """

        # the first step is to generate the augmented views and then turn views into vectors.
        fragment_dict, all_fragments, namelist = self.gen_views(
            contigs=contigs,
            simclr=True,
            barlow_twins=False,
            output_dir=output_dir,
            view_num=self.view_num,
            contig_length=self.contig_length,
        )

        # after getting the augmented views, pass to binfernofeatures to get the features pooled.
        feature_df = self.features(
            FragMent_Dict=fragment_dict,
            All_Fragments=all_fragments,
            contigs=contigs,
            bam_path=bam,
            threads=threads,
            logger=logger,
            output_dir=output_dir,
        )

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
        kmer_dim = KMER_DIM
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
        else:
            model = NeuralNetwork(
                tnf_dim=composition_dim,
                coverage_dim=cov_dim,
                output_dim=self.output_dim,
                kmer_dim=kmer_dim,
                n_layers=self.n_layers,
                proj_head_dim=proj_head_dim,
                apply_multiscale=True,
                no_proj=self.no_proj,
            ).to(device)

        # prepare training.
        train_loader, optimizer, scheduler, dataset = self.prepare_trainer(
            data_df=feature_df,
            batch_size=self.batch_size,
            contig_length=self.contig_length,
            model=model,
            n_views=self.view_num,
            simclrnet=self.simclrnet,
            **kwargs,
        )

        # train loop.
        model_path = output_dir / "best_model.pt"
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
                output_dir=output_dir,
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
                n_views=self.view_num,
                **kwargs,
            )

        embedding_path = output_dir / "embeddings.tsv"
        if not embedding_path.exists():
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
                    save_individual_embs=False,
                )
        else:
            logger.info(
                f"Embeddings already exist at {embedding_path}. Loading embeddings from file."
            )
            embeddings = pd.read_csv(embedding_path, sep="\t", header=0, index_col=0)
        return embeddings

    def prepare_trainer(
        self,
        data_df,
        batch_size: int = 1024,
        contig_length: int = 1000,
        model=None,
        n_views=6,
        simclrnet=False,
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

        length_weight = self.length_weight(contig=kwargs.get("contigs", None))
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

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=4,
        )

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
                learning_rate=2.5e-3,
                min_learning_rate=1e-4,
                epochs=epochs,
                batch_size=batch_size,
            )

        return train_loader, optimizer, scheduler, dataset

    class info_nce_loss(nn.Module):
        def __init__(self, temperature=0.15):
            super().__init__()
            self.temperature = temperature

        def forward(self, features, batch_size, n_views, device):
            features = F.normalize(features, dim=1)

            N = batch_size * n_views
            similarity_matrix = torch.matmul(features, features.T) / self.temperature

            labels = torch.arange(batch_size, device=device).repeat(n_views)

            same_sample = labels.unsqueeze(0) == labels.unsqueeze(1)
            self_mask = torch.eye(N, dtype=torch.bool, device=device)

            positives_mask = same_sample & ~self_mask
            negatives_mask = ~same_sample

            positives = similarity_matrix[positives_mask].view(N, n_views - 1)
            negatives = similarity_matrix[negatives_mask].view(N, N - n_views)

            logits = torch.cat(
                [
                    positives.reshape(-1, 1),
                    negatives.repeat_interleave(n_views - 1, dim=0),
                ],
                dim=1,
            )

            labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device)

            return logits, labels

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
        method: str = "barlow_twins",
        epochs: int = 300,
        model=None,
        n_cov: int = 0,
        n_views: int = 6,
        **kwargs,
    ):
        self.unify_seeds(
            self.seed
        )  # ensure reproducibility by unifying seeds across random, numpy, and torch.
        temperature = kwargs.get("temperature", 0.15)
        output_dir = kwargs.get("output_dir", None)
        MODEL_PATH = output_dir / "best_model.pt" if output_dir else "best_model.pt"
        logger = kwargs.get("logger", None)
        if logger is None:
            raise ValueError(
                "Logger is not provided for training. Please provide a logger in the kwargs with key 'logger'."
            )

        epoch_iter = tqdm(range(epochs), desc="Training Epochs")
        early_stopping = self.EarlyStoppingManager(patience=15)
        for epoch in epoch_iter:
            avg_loss, stats = self.train_an_epoch(
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                temperature=temperature,
                n_cov=n_cov,
                n_views=n_views,
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

    def train_an_epoch(
        self,
        model,
        dataloader,
        optimizer,
        device,
        batch_size=1024,
        temperature=0.15,
        n_views=6,
        **kwargs,
    ):
        criterion = nn.CrossEntropyLoss().to(device)
        info_nceloss_fn = self.info_nce_loss(temperature=temperature).to(device)
        model.train()
        stats = None
        total_loss = 0.0
        n_cov = kwargs.get("n_cov", 0)
        if n_cov < 0 or n_cov == 0:
            raise ValueError("n_cov must be non-negative.")

        for i, features in enumerate(dataloader):
            contig_features = torch.cat(features, dim=0).to(device)
            optimizer.zero_grad()

            enc_feature = model.forward_one(contig_features)
            logits, labels = info_nceloss_fn(
                enc_feature, batch_size=batch_size, device=device, n_views=n_views
            )
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            last_batch = i == len(dataloader) - 1
            top1_acc = self.accuracy(logits, labels, topk=(1,))[0].item()

            if last_batch:
                stats = {"loss": total_loss / len(dataloader), "top1_acc": top1_acc}

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
        criterion = torch.nn.CrossEntropyLoss().to(device)

        if n_cov <= 0:
            raise ValueError(
                "n_cov must be a positive integer representing the number of coverage features in the input data."
            )

        scaler = GradScaler(enabled=False)
        logger = kwargs.get("logger", None)
        if logger is None:
            raise ValueError(
                "Logger is not provided for training. Please provide a logger in the kwargs with key 'logger'."
            )

        def info_nce_loss(self, features, device, batch_size=1024, n_views=6, **kwargs):
            """
            Calculate the InfoNCE loss for SimCLR.

            :param features: Input features.
            :return: Logits and labels for the loss.
            """

            labels = torch.cat(
                [torch.arange(batch_size) for i in range(n_views)],
                dim=0,
            )

            # labels are labelled in arrange format: 0,1,2,...,batch_size-1,0,1,2,...,batch_size-1,... for n_views times.
            # so... ordering dataloader as 1,2,3,4,5,0 will not work, because the positive pairs will not be in the same order as the labels. We need to order the dataloader as 0,1,2,3,4,5 to match the labels.
            labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
            labels = labels.to(device)
            features = F.normalize(features, dim=1)

            similarity_matrix = torch.matmul(features, features.T)

            # discard the main diagonal from both: labels and similarities matrix
            mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
            labels = labels[~mask].view(labels.shape[0], -1)
            similarity_matrix = similarity_matrix[~mask].view(
                similarity_matrix.shape[0], -1
            )
            # assert similarity_matrix.shape == labels.shape

            # select and combine multiple positives
            positives = similarity_matrix[labels.bool()].view(-1, 1)
            # select only the negatives the negatives
            negatives = similarity_matrix[~labels.bool()].view(
                similarity_matrix.shape[0], -1
            )
            negatives = negatives[:, None].expand(-1, n_views - 1, -1).flatten(0, 1)

            logits = torch.cat([positives, negatives], dim=1)
            labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
            logits = logits / kwargs.get("temperature", 0.15)
            return logits, labels

        logger.info("Starting SimCLR training with {} epochs.".format(epochs))
        earlystop_epoch = 0
        for epoch_counter in tqdm(range(epochs), desc="SimCLR Training Epochs"):
            total_loss = 0.0
            total_acc1 = 0.0
            for _, contig_features in enumerate(dataloader):
                contig_features = torch.cat(contig_features, dim=0)
                contig_features = contig_features.to(device)

                with torch.amp.autocast(enabled=False, device_type=device.type):
                    features, _ = model(
                        contig_features[:, :-n_cov], contig_features[:, -n_cov:]
                    )
                    actual_batch_size = contig_features.shape[0] // n_views
                    logits, labels = info_nce_loss(
                        features,
                        device,
                        batch_size=actual_batch_size,
                        n_views=n_views,
                        temperature=temperature,
                    )
                    loss = criterion(logits, labels)
                    top1_acc = self.accuracy(logits, labels, topk=(1,))[0].item()
                    total_loss += loss.item()
                    total_acc1 += top1_acc

                optimizer.zero_grad()
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
        return

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

            augmented_fragments = self.simclr_augmentation(
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

    def simclr_augmentation(
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
            output_dir=str(output_dir),
            num_views=view_num,
            length_threshold=contig_length,
            method="simclr",
        )

        return FragMent_Dict, All_Fragments
