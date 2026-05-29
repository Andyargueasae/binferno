import pandas as pd
from pydantic import Field
from pathlib import Path

import torch
from .barlowtwins import BarlowTwinsTrainer
from .simclr import SimCLRTrainer
from .base import BaseTrainer
from ..features import BinfernoFeatureBuilder
import torch.nn.functional as F

class BinfernoTrainer(BaseTrainer):
    simclr: SimCLRTrainer
    use_simclr: bool = Field(
        default=False,
        description="Whether to use SimCLR for training. If False, only Barlow Twins will be used for training.",
    )
    barlowtwins: BarlowTwinsTrainer
    use_barlowtwins: bool = Field(
        default=False,
        description="Whether to use Barlow Twins for training. If False, only SimCLR will be used for training.",
    )
    multisample: bool = Field(
        default=False,
        description="Whether to use multisample augmentation for SimCLR training.",
    )
    concat: bool = Field(
        default=False,
        description="Whether to concatenate the embeddings from SimCLR and Barlow Twins. If False, the embeddings are added and normalized.",
    )

    def __call__(
        self,
        contigs: Path,
        bam: Path,
        output_dir: Path = None,
        temperature: float = 0.15,
        logger=None,
        threads: int = 4,
        features: BinfernoFeatureBuilder = None,
    ) -> pd.DataFrame:
        """
        The step incorporates both augmentation-data generation and training.
        Args:
            contigs: path to the input contigs in fasta format.
            bam: path to the input bam file directory.
            temperature: temperature parameter for contrastive learning.
            multisample: whether to use multisample augmentation for SimCLR training.
        Returns:
            embeddings: a dataframe containing the embeddings for each contig.
        """
        # determine the training method to be used.
        simclr, barlow_twins = self.use_simclr, self.use_barlowtwins
        train_modules = []
        output_dirs = []
        embeddings = pd.DataFrame()
        if not simclr and not barlow_twins:
            raise ValueError("At least one training method must be chosen. \
                            Please set training_method to (True, False) for SimCLR, \
                            (False, True) for Barlow Twins, or (True, True) for both.")
        if simclr:
            output_dirs.append(output_dir / "simclr")
            train_modules.append(self.simclr)
        if barlow_twins:
            output_dirs.append(output_dir / "barlow_twins")
            train_modules.append(self.barlowtwins)
        
        for out_dir, train_module in zip(output_dirs, train_modules):
            out_dir.mkdir(
                parents=True, exist_ok=True
            )  # ensure now output directory exists for the training method.
            logger.info(
                f"Start augmentation and training for {train_module.__class__.__name__}."
            )
            output_emb = train_module(
                contigs=contigs,
                bam=bam,
                output_dir=out_dir,
                temperature=temperature,
                multisample=self.multisample,
                logger=logger,
                threads=threads,
                features=features,
            )
            # normalized output_emb before concatenations, use F.normalize from torch.

            embeddings = pd.concat([embeddings, output_emb], axis=1)
        if not self.concat:
            # split the embeddings first, then add and normalize them.
            num_methods = len(train_modules)
            split_embs = torch.split(torch.tensor(embeddings.values), embeddings.shape[1] // num_methods, dim=1)
            summed_emb = torch.zeros_like(split_embs[0])
            for emb in split_embs:
                summed_emb += emb
            normalized_emb = F.normalize(summed_emb, p=2, dim=1)
            embeddings = pd.DataFrame(normalized_emb.numpy(), index=embeddings.index)
        return embeddings
