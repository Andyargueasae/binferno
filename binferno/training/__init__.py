import pandas as pd
from pydantic import Field
from pathlib import Path
from .barlowtwins import BarlowTwinsTraining
from .simclr import SimCLRTraining
from .base import BaseTraining


class BinfernoTraining(BaseTraining):
    simclr: SimCLRTraining
    barlowtwins: BarlowTwinsTraining
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
    model: str = Field(
        default="default",
        description="The model architecture to use for training. \
        Currently supports 'default' for NeuralNetwork and 'SimCLRNet' for the SimCLR architecture. \
        If 'default' is chosen, the NeuralNetwork architecture will be used for both SimCLR and \
        Barlow Twins training. If 'SimCLRNet' is chosen, the SimCLR architecture will be used \
        for SimCLR training, and the NeuralNetwork architecture will be used for Barlow Twins training.",
    )
    apply_multiscale: bool = Field(
        default=False, description="Whether to pass features to apply \
         multiscale transformation."
    )

    def __call__(
        self,
        contigs: str,
        bam: str,
        threads: int = 10,
        output_dir: Path = None,
        training_method: tuple = (True, False),  # (simclr, barlow_twins)
        temperature: float = 0.15,
        multisample: bool = False,
        view_num: int = 6,
        logger=None,
    ) -> pd.DataFrame:
        """
        The step incorporates both augmentation-data generation and training.
        Args:
            contigs: path to the input contigs in fasta format.
            bam: path to the input bam file directory.
            threads: number of threads to use for augmentations.
            training_method: a tuple indicating whether to use SimCLR and/or Barlow Twins for training.
            temperature: temperature parameter for contrastive learning.
            multisample: whether to use multisample augmentation for SimCLR training.
            view_num: number of views to generate for SimCLR training.
        Returns:
            embeddings: a dataframe containing the embeddings for each contig.
        """
        # determine the training method to be used.
        simclr, barlow_twins = training_method
        train_modules = []
        output_dirs = []
        embeddings = pd.DataFrame()
        if not simclr and not barlow_twins:
            raise ValueError("At least one training method must be chosen. \
                            Please set training_method to (True, False) for SimCLR, \
                            (False, True) for Barlow Twins, or (True, True) for both.")
        elif simclr:
            output_dirs.append(output_dir / "simclr")
            train_modules.append(self.simclr)
        else:
            output_dirs.append(output_dir / "barlow_twins")
            train_modules.append(self.barlowtwins)

        for out_dir, train_module in zip(output_dirs, train_modules):
            out_dir.mkdir(parents=True, exist_ok=True) # ensure now output directory exists for the training method.
            # and everything will be saved in this out_dir. 
            logger.info(
                f"Start augmentation and training for {train_module.__class__.__name__}."
            )
            output_emb = train_module(
                contigs=contigs,
                bam=bam,
                threads=threads,
                output_dir=out_dir,
                temperature=temperature,
                view_num=view_num,
                multisample=multisample,
                logger=logger,
                epochs=self.epochs,
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                model=self.model,
                n_layers=self.n_layers,
                output_dim=self.output_dim,
                multiscale=self.apply_multiscale,
                batch_size=self.batch_size,
            )
            embeddings = pd.concat([embeddings, output_emb], axis=1)

        return embeddings
