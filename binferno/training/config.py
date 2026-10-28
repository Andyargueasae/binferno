"""Configuration shared by Binferno's self-supervised trainers."""

from pathlib import Path

from pydantic import BaseModel, Field


class CommonTrainingConfig(BaseModel):
    batch_size: int = Field(default=1024, ge=1, description="Training batch size.")
    epochs: int = Field(default=300, ge=1, description="Number of training epochs.")
    output_dim: int = Field(default=128, ge=1, description="Embedding dimension.")
    contig_length: int = Field(default=1000, ge=1, description="Minimum training contig length.")
    seed: int = Field(default=42, description="Random seed for training and augmentation.")
    get_individual_embs: bool = Field(
        default=False, description="Save separate modality embeddings."
    )


class SCGConfig(BaseModel):
    lambda_scg: float = Field(default=0.0, ge=0.0, description="SCG cannot-link loss weight; zero disables SCG detection.")
    cosine_margin: float = Field(default=0.4, ge=-1.0, le=1.0)
    scg_batch_size: int = Field(default=128, ge=2)
    scg_update_freq: int = Field(default=2, ge=1)
    scg_loss_warmup: int = Field(default=10, ge=0)
    scg_marker_hmm: Path = Field(
        default=Path(__file__).resolve().parent.parent / "auxiliary" / "marker.hmm"
    )
    scg_max_evalue: float = Field(default=1e-10, gt=0.0)
    scg_min_hmm_coverage: float = Field(default=0.75, gt=0.0, le=1.0)
    scg_hard_pairs: int = Field(
        default=1024, ge=1,
        description="Maximum hard marker pairs per update (Barlow Twins only).",
    )
