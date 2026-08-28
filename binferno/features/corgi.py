import glob
from pathlib import Path
from .base import BaseFeatureBuilder
import pandas as pd
from corgi import Corgi
import numpy as np
from typing import Union, Optional, Tuple, List
import os
from pydantic import Field
from ..utils import calc_n50


def read_embeddings(
    embeddings_path: Union[Path, str],
    embeddings_index: Optional[Path] = None,
    merge: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Read embeddings from a memmap file.
    """
    embeddings_path = Path(embeddings_path)
    index_path = embeddings_index or embeddings_path.with_suffix(".index")
    index_path = Path(index_path)
    assert index_path.exists(), f"Index file '{index_path}' does not exist."
    # hit an error here. May need to think of contacting Rob.
    index_with_tabs = Path(index_path).read_text().strip().splitlines()
    embeddings = read_memmap(embeddings_path, len(index_with_tabs))

    if merge:
        df = pd.DataFrame(embeddings)
        df["index_with_tabs"] = index_with_tabs
        # Group by 'index_with_tabs' and average the embeddings
        grouped = df.groupby("index_with_tabs", sort=False).mean(numeric_only=True)
        # Extract the averaged embeddings and the corresponding unique ids
        embeddings = grouped.values
        index_with_tabs = grouped.index.to_list()

    index = [line.split("\t") for line in index_with_tabs]

    return embeddings, index


def read_memmap(path, count, dtype: str = "float16") -> np.memmap:
    file_size = os.path.getsize(path)
    dtype_size = np.dtype(dtype).itemsize
    num_elements = file_size // dtype_size
    # embedding size may vary according to num_elements.
    embedding_size = num_elements // count
    shape = (
        count,
        embedding_size,
    )  # embedding size fluctuates because of different contig subsequencing.
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


def run_corgi(
    input_fasta,
    output_csv: str = None,
    min_length: int = 0,
    model_path: str = None,
    embedding: str = None,
):
    corgi = Corgi()
    _ = corgi.predict(
        input=input_fasta,
        output_csv=output_csv,
        embeddings=embedding,
        min_length=min_length,
        checkpoint=model_path,
    )
    return


class CorgiFeatureBuilder(BaseFeatureBuilder):
    model_path: str = Field(
        default="",
        description="Path to the pretrained Corgi model checkpoint. If None, will use the default pretrained model.",
    )
    corgi_normalize: bool = Field(
        default=True,
        description="Whether to normalize the Corgi embeddings. If True, the embeddings will be L2 normalized. If False, the raw embeddings will be returned.",
    )

    def __call__(
        self,
        seq_dir: str,
        contigs: Path,
        min_length: int = 1000,
    ) -> pd.DataFrame:
        """
        Run corgi on the input fasta file, and return a pandas dataframe with the following columns:
        contig_id, embedding (as a list of floats), and any other metadata (e.g., length, coverage).
        """
        # 1. run corgi.
        # 2. read corgi embedding outputs, and return the pandas dataframe.
        aug_views = sorted(
            glob.glob(os.path.join(seq_dir, f"augmented_fragments_view*.fasta"))
        )

        # process raw contigs first, make sure every contigs has .original suffix for indexing the features later.
        if len(aug_views) == 0:
            raise ValueError(
                f"No augmented fragment fasta files found in the directory '{seq_dir}'. \
                            Please check the input directory and make sure it contains files named \
                            'augmented_fragments_view*.fasta'."
            )

        corgi_df = pd.DataFrame()
        # because barlow twins augmentation didn't include original as one of views, process raw first.
        # replace method-guided determination by checking: if aug0 isn't same raw contigs, then it is barlow_twins.
        aug_zero_fasta = os.path.join(seq_dir, f"augmented_fragments_view0.fasta")
        # check if aug_zero_fasta is the direct copy of raw contigs: n_sequences and N50 should be the same.
        N50_aug_zero = calc_n50(aug_zero_fasta, min_length=min_length)
        N50_raw = calc_n50(contigs, min_length=min_length)

        contig_basename = os.path.basename(contigs)
        if (
            N50_aug_zero != N50_raw
        ):  # view 0 isn't direct copy of raw contigs, run corgi on raw contigs first and add to the corgi_df.
            raw_emb = Path(f"{seq_dir}/{contig_basename.split('.')[0]}_embeddings.npy")
            raw_index = Path(
                f"{seq_dir}/{contig_basename.split('.')[0]}_embeddings.index"
            )
            if not os.path.exists(raw_emb) or not os.path.exists(raw_index):
                run_corgi(
                    input_fasta=contigs,
                    min_length=min_length,
                    model_path=self.model_path,
                    embedding=raw_emb,
                )

            raw_emb, raw_idx = read_embeddings(raw_emb, raw_index)
            raw_df = pd.DataFrame(
                raw_emb, index=[f"{idx}.original" for _, idx in raw_idx]
            )
            raw_df.columns = [str(col) for col in raw_df.columns]
            corgi_df = pd.concat([corgi_df, raw_df], axis=0)

            del (
                raw_df,
                raw_emb,
                raw_idx,
            )

        for fasta in aug_views:  # view 0 is the raw contigs, copy to rename.
            if os.stat(fasta).st_size == 0:
                raise ValueError(
                    f"Input fasta file '{fasta}' is empty. Please check the input fasta file and make sure it contains valid sequences."
                )

            embedding_path = Path(f"{fasta.replace('.fasta', '')}_embeddings.npy")
            embedding_index = Path(f"{fasta.replace('.fasta', '')}_embeddings.index")
            if not os.path.exists(embedding_path) or not os.path.exists(
                embedding_index
            ):
                run_corgi(
                    input_fasta=fasta,
                    min_length=0,  # here, min length must be 0, as views are all required.
                    model_path=self.model_path,
                    embedding=embedding_path,
                )

            # read the embeddings and construct the dataframe.
            view_emb, view_index = read_embeddings(embedding_path, embedding_index)
            view_df = pd.DataFrame(view_emb, index=[idx for _, idx in view_index])
            view_df.columns = [str(col) for col in view_df.columns]
            corgi_df = pd.concat([corgi_df, view_df], axis=0)

            del view_df, view_emb, view_index

        if self.corgi_normalize:
            corgi_df = corgi_df.apply(
                lambda x: x / np.linalg.norm(x), axis=1
            )  # L2 normalization by row.
        return corgi_df
