# from binferno.binferno.features.base import BaseFeatures
import hashlib
import math
import os
import random
import pyfastx
import numpy as np
from pydantic import BaseModel
import pandas as pd
from pathlib import Path
import torch
from typing import Dict, List
from collections import OrderedDict


class BaseTraining(BaseModel):
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
        output_dir=None,
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

            if method == "barlow_twins":
                augmented_fragments = self.barlow_twins_augmentation(
                    sequence=seq,
                    header=clean_header,
                    min_contig_length=length_threshold,
                    num_augmentations=num_views,
                )
            else:
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
        barlow_twins: bool = False,
        simclr: bool = True,
    ) -> Dict[str, Dict]:
        """
        Augmentation of contigs only. The returned data is a dictionary storing: raw contig -> ( seq, header, (start, end) )
        """
        # three things used across the stage:
        output_dir.mkdir(parents=True, exist_ok=True)
        FragMent_Dict: OrderedDict = OrderedDict()
        All_Fragments: list = []

        assert (
            barlow_twins != simclr
        ), "Only one of barlow_twins and simclr can be True. Please set one to False."

        FragMent_Dict, All_Fragments = self.augmentation(
            fasta_file=str(contigs),
            output_dir=str(output_dir),
            num_views=view_num,
            length_threshold=contig_length,
            method="barlow_twins" if barlow_twins else "simclr",
        )

        return FragMent_Dict, All_Fragments

    # Three helper functions for training.
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
        def __init__(self, patience=15):
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
        self, optimizer, learning_rate, min_learning_rate, epochs, batch_size
    ):
        warmup_epochs = 10  # 10 epochs to warm up, suitable for 200 epochs of training. We can also make this a tunable parameter if needed.
        for param_group in optimizer.param_groups:
            param_group["lr"] = min_learning_rate

        max_lr = (batch_size / 256) * learning_rate * 0.2
        warmup_start_lr = max_lr * 0.1

        # now, construct the warmup arrays.
        def lr_lambda(current_epoch):
            # Warmup phase: linearly increase from warmup_start_lr to max_lr
            if current_epoch < warmup_epochs:
                if warmup_epochs == 1:
                    lr = max_lr
                else:
                    progress = current_epoch / (warmup_epochs - 1)
                    lr = warmup_start_lr + progress * (max_lr - warmup_start_lr)
            else:
                if epochs == warmup_epochs:
                    lr = min_learning_rate
                else:
                    progress = (current_epoch - warmup_epochs) / (
                        epochs - warmup_epochs
                    )
                    progress = min(progress, 1.0)
                    cosine = 0.5 * (1 + math.cos(math.pi * progress))
                    lr = min_learning_rate + (max_lr - min_learning_rate) * cosine
            return lr / warmup_start_lr

        # create cosine annealing.
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
                kmer_emb, cov_emb = model.get_separate_modality_embs(batch_features)
                batch_embeddings = torch.nn.functional.normalize(
                    batch_embeddings, p=2, dim=1
                )
                for j, header in enumerate(batch_encode.index):
                    clean_header = header.replace(suffix, "")
                    embeddings[clean_header] = batch_embeddings[j].cpu().numpy()

                    # Save encoder embeddings if requested
                    if save_individual_embs:
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
