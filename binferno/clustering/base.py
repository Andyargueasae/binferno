from __future__ import annotations, print_function

import mimetypes
import os
from time import time
from pydantic import BaseModel
import gzip
from Bio import SeqIO
import functools
import logging
from typing import List, Sequence
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.cluster._kmeans import (
    check_random_state,
    euclidean_distances,
    row_norms,
    stable_cumsum,
)


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def save_result(result, filepath, namelist):
    filedir, filename = os.path.split(filepath)
    if not filename:
        filename = "result.tsv"
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    f = open(filepath, "w")
    for contigIdx in range(len(result)):
        f.write(namelist[contigIdx] + "\t" + str(result[contigIdx].item(0)) + "\n")
    f.close()


def gen_bins(fastafile, resultfile, outputdir):
    # read fasta file
    print("Processing file:\t{}".format(fastafile))
    sequences = {}
    if fastafile.endswith("gz"):
        with gzip.open(fastafile, "r") as f:
            for line in f:
                line = str(line, encoding="utf-8")
                if line.startswith(">"):
                    if " " in line:
                        seq, others = line.split(" ", 1)
                        sequences[seq] = ""
                    else:
                        seq = line.rstrip("\n")
                        sequences[seq] = ""
                else:
                    sequences[seq] += line.rstrip("\n")
    else:
        with open(fastafile, "r") as f:
            for line in f:
                if line.startswith(">"):
                    if " " in line:
                        seq, others = line.split(" ", 1)
                        sequences[seq] = ""
                    else:
                        seq = line.rstrip("\n")
                        sequences[seq] = ""
                else:
                    sequences[seq] += line.rstrip("\n")
    print("Reading Map:\t{}".format(resultfile))
    dic = {}
    with open(resultfile, "r") as f:
        for line in f:
            contig_name, cluster_name = line.strip().split("\t")
            try:
                dic[cluster_name].append(contig_name)
            except:
                dic[cluster_name] = []
                dic[cluster_name].append(contig_name)
    print("Writing bins:\t{}".format(outputdir))
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    bin_name = 0
    for _, cluster in dic.items():
        binfile = os.path.join(outputdir, "{}.fa".format(bin_name))
        with open(binfile, "w") as f:
            for contig_name in cluster:
                contig_name = ">" + contig_name
                try:
                    sequence = sequences[contig_name]
                except:
                    bin_name += 1
                    continue
                f.write(contig_name + "\n")
                f.write(sequence + "\n")
                bin_name += 1


class BaseClustering(BaseModel):

    def calculateN50(self, seqLens):
        thresholdN50 = sum(seqLens) / 2.0

        seqLens.sort(reverse=True)

        testSum = 0
        N50 = 0
        for seqLen in seqLens:
            testSum += seqLen
            if testSum >= thresholdN50:
                N50 = seqLen
                break
        return N50

    def get_length(self, fastx_file):
        file_type = mimetypes.guess_type(fastx_file)[1]
        if file_type == "gzip":
            f = gzip.open(fastx_file, "rt")
        elif not file_type:
            f = open(fastx_file, "rt")
        else:
            raise RuntimeError("Unknown type of file: '{}".format(fastx_file))
        length = {}
        if os.path.getsize(fastx_file) == 0:
            return length
        file_format = None
        line = f.readline()
        if line.startswith("@"):
            file_format = "fastq"
        elif line.startswith(">"):
            file_format = "fasta"
        f.seek(0)
        if not file_format:
            raise RuntimeError("Invalid sequence file: '{}".format(fastx_file))
        for seq_record in SeqIO.parse(f, file_format):
            length[seq_record.id] = len(seq_record.seq)

        f.close()
        return length

    def _filter_embeddings_by_length(
        self,
        emb_df: pd.DataFrame,
        contig_file: str,
        contig_len: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Keep only contigs with length >= contig_len.
        """
        lengths = self.get_length(contig_file)

        names = emb_df.index.to_numpy()

        features = emb_df.to_numpy(dtype=np.float32)

        contig_lengths = np.array([lengths[name] for name in names], dtype=np.float64)
        keep_mask = contig_lengths >= contig_len

        return (
            features[keep_mask],
            names[keep_mask],
            contig_lengths[keep_mask],
        )

    def _read_seed_count(self, seed_file: str) -> int:
        if not os.path.exists(seed_file):
            raise FileNotFoundError(f"Seed file '{seed_file}' does not exist.")
        with open(seed_file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            raise ValueError(f"Seed file '{seed_file}' is empty.")
        try:
            seed_num = len(set(lines))
        except ValueError:
            raise ValueError(f"Invalid seed file '{seed_file}'.")
        return seed_num

    def gen_seed_idx(self, seed_file: str, contig_id_list: Sequence[str]) -> List[int]:
        """
        Resolve seed contig names to row indices in contig_id_list.
        """
        contig_to_idx = {name: idx for idx, name in enumerate(contig_id_list)}
        resolved = []

        with open(seed_file) as handle:
            for line in handle:
                contig_name = line.rstrip("\n")
                idx = contig_to_idx.get(contig_name)
                if idx is not None:
                    resolved.append(idx)

        return resolved

    def seed_kmeans_full(
        self,
        logger: logging.Logger,
        contig_file: str,
        namelist: Sequence[str],
        out_path: str,
        X_mat: np.ndarray,
        bin_number: int,
        prefix: str,
        length_weight: np.ndarray,
        seed_file: str,
    ) -> None:
        """
        Run weighted seed KMeans and write result TSV + bins.
        """
        output_prefix = f"{out_path}{prefix}"
        output_tsv = f"{output_prefix}_k_{bin_number}_result.tsv"

        if os.path.exists(output_tsv):
            logger.info(f"Skipping existing KMeans result: {output_tsv}")
            return

        seed_indices = self.gen_seed_idx(seed_file, contig_id_list=namelist)
        if not seed_indices:
            logger.warning("No seed contigs matched namelist; skipping seed-kmeans.")
            return
        if bin_number > len(namelist):
            logger.warning(
                f"Requested bin_number={bin_number} exceeds number of contigs={len(namelist)}; skipping."
            )
            return

        start = time()
        km = KMeans(
            n_clusters=bin_number,
            random_state=7,
            algorithm="lloyd",
            init=functools.partial(self.partial_seed_init, seed_idx=seed_indices),
            n_init=1,
        )
        km.fit(X_mat, sample_weight=length_weight)

        save_result(km.labels_, output_tsv, namelist)
        gen_bins(contig_file, output_tsv, f"{output_tsv}_bins")

        logger.info(f"Weighted seed-kmeans time:\t{time() - start:.3f}s")

    def partial_seed_init(
        self,
        X,
        n_clusters: int,
        random_state,
        seed_idx,
        n_local_trials=None,
    ) -> np.ndarray:
        """
        KMeans init that fixes the first centers to seed_idx and fills the rest with
        a kmeans++-style procedure.
        """
        random_state = check_random_state(random_state)
        x_squared_norms = row_norms(X, squared=True)
        n_samples, n_features = X.shape

        if n_clusters <= 0:
            raise ValueError("n_clusters must be positive")
        if n_clusters > n_samples:
            raise ValueError("n_clusters cannot exceed number of samples")
        if not seed_idx:
            raise ValueError("seed_idx must contain at least one seed index")
        if len(seed_idx) > n_clusters:
            raise ValueError("number of seed indices exceeds n_clusters")

        centers = np.empty((n_clusters, n_features), dtype=X.dtype)

        if n_local_trials is None:
            n_local_trials = 2 + int(np.log(n_clusters))

        first_id = seed_idx[0]
        centers[0] = X[first_id].toarray().ravel() if sp.issparse(X) else X[first_id]

        closest_dist_sq = euclidean_distances(
            centers[0, np.newaxis],
            X,
            Y_norm_squared=x_squared_norms,
            squared=True,
        ).ravel()

        for c, center_id in enumerate(seed_idx[1:], start=1):
            centers[c] = (
                X[center_id].toarray().ravel() if sp.issparse(X) else X[center_id]
            )
            closest_dist_sq = np.minimum(
                closest_dist_sq,
                euclidean_distances(
                    centers[c, np.newaxis],
                    X,
                    Y_norm_squared=x_squared_norms,
                    squared=True,
                ).ravel(),
            )

        current_pot = float(closest_dist_sq.sum())

        for c in range(len(seed_idx), n_clusters):
            rand_vals = random_state.random_sample(n_local_trials) * current_pot
            candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq), rand_vals)
            np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

            distance_to_candidates = euclidean_distances(
                X[candidate_ids],
                X,
                Y_norm_squared=x_squared_norms,
                squared=True,
            )

            best_candidate = None
            best_pot = None
            best_dist_sq = None

            for trial in range(n_local_trials):
                candidate_dist_sq = distance_to_candidates[trial]
                new_dist_sq = np.minimum(closest_dist_sq, candidate_dist_sq)
                new_pot = float(new_dist_sq.sum())

                if best_candidate is None or new_pot < best_pot:
                    best_candidate = int(candidate_ids[trial])
                    best_pot = new_pot
                    best_dist_sq = new_dist_sq

            centers[c] = (
                X[best_candidate].toarray().ravel()
                if sp.issparse(X)
                else X[best_candidate]
            )
            closest_dist_sq = best_dist_sq
            current_pot = best_pot

        return centers

    def apply_dimension_reduction(self, X: np.ndarray) -> np.ndarray:
        """
        Apply dimension reduction to the input features.
        Placeholder for actual dimension reduction method (e.g., PCA, UMAP and SUDE).
        """

        return X
