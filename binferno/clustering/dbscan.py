"""Density-based clustering module for binferno (DBSCAN / HDBSCAN).

Drop-in replacement for the original dbscan.py. The marker-quality and
sweep machinery now live in `marker_quality.py` and `dbscan_sweep.py`;
this file is the binferno-facing wrapper only.

Outputs written to `output_path`:
  dbscan_clusters.tsv       contig_id -> cluster_id (-1 = unbinned)
  dbscan_bin_quality.tsv    per-bin completeness / contamination / bp
  dbscan_sweep_audit.tsv    every parameter set tried and what it scored
  dbscan_marker_report.txt  the marker-table diagnostic
  cluster_result.pkl        pickled ClusterResult
"""

import os
import pickle
from pathlib import Path
from typing import Optional

import loguru
import numpy as np
import pandas as pd
from pydantic import Field
from sklearn.preprocessing import normalize

from .base import BaseClustering, ClusterResult
from ..utils import gen_seed
from .cluster_utils import (
    _filter_embeddings_by_length,
    seed_kmeans_full,
    _read_seed_count,
)
from .marker_quality import build_marker_table, diagnose_marker_table
from .dbscan_sweep import (
    SweepConfig,
    bins_to_dataframe,
    density_sweep,
    quality_dataframe,
    recruit_unbinned,
)


class DbscanClustering(BaseClustering):
    """Sweep DBSCAN (or HDBSCAN) parameters, scoring each clustering by the
    number of near-complete, low-contamination bins it produces. DEPRECATED: use Leiden clustering instead."""

    hdbscan: bool = Field(
        default=False,
        description="Use HDBSCAN instead of DBSCAN. [DEPRECATED: HDBSCAN is not recommended for binning]",
    )
    core_bp: Optional[int] = Field(
        default=100_000,
        description=(
            "Assembled bp required within eps for a contig to be a core "
            "point. Portable across assembly sizes; set None to fall back "
            "to classic contig-count min_samples."
        ),
    )
    min_bin_bp: int = Field(
        default=100_000, description="Discard candidate bins smaller than this."
    )
    contig_len: int = Field(
        default=1000,
        description="Contigs shorter than this are ignored for clustering.",
    )
    marker_hmm: Optional[Path] = Field(
        default=Path(__file__).parent.parent / "auxiliary" / "bacar_marker.hmm",
        description=(
            "Path to the .hmm file defining the marker vocabulary. Strongly "
            "recommended: without it, completeness is normalised by the "
            "markers that happened to hit this assembly and is not "
            "comparable between runs."
        ),
    )
    archaeal_hmm: Optional[Path] = Field(
        default=None,
        description="Optional second marker set; completeness takes the best domain.",
    )
    max_marker_evalue: float = Field(default=1e-5)
    recruit: bool = Field(
        default=True, description="Recruit leftover contigs into nearby bins."
    )

    def __call__(
        self,
        contigs: Path,
        output_path: Path,
        embeddings: pd.DataFrame,
        logger: "loguru._logger.Logger",
        threads: int = 1,
        plastid: bool = False,
        longread: bool = False,
    ) -> ClusterResult:

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        if embeddings is None or embeddings.empty:
            raise ValueError("embeddings must be a non-empty DataFrame")
        if contigs is None:
            raise ValueError("contigs is required")

        marker_name = "pltd_marker" if plastid else "bacar_marker"
        n_genome = gen_seed(
            logger,
            contig_file=contigs,
            output_dir=output_path,
            threads=threads,
            contig_length_threshold=self.contig_len + 1,
            marker_name=marker_name,
            quarter="2quarter",
        )
        base = os.path.basename(str(contigs))
        seed_file = (
            output_path
            / f"{base}.{marker_name}.2quarter_lencutoff_{self.contig_len + 1}.seed"
        )

        result = ClusterResult(n_genome=n_genome, clusters={})

        logger.info("Start clustering.")
        emb_mat, namelist, length_weight = _filter_embeddings_by_length(
            emb_df=embeddings, contig_file=contigs, contig_len=self.contig_len
        )
        length_weight = np.asarray(length_weight, dtype=np.int64)
        logger.info(f"Contigs retained after length filter: {len(namelist)}")

        norm_embeddings = normalize(emb_mat).astype(np.float32, copy=False)

        # ------------------------------------------------------------------
        # marker table (this is where the old None/0 completeness came from)
        # ------------------------------------------------------------------
        hmmout_path = _resolve_hmmout_path(contigs, output_path, marker_name)
        hmm_files = {}
        if self.marker_hmm:
            hmm_files["bacteria" if not plastid else "plastid"] = self.marker_hmm
        if self.archaeal_hmm:
            hmm_files["archaea"] = self.archaeal_hmm

        table, info = build_marker_table(
            hmmout_path,
            namelist,
            contig_lengths=length_weight,
            hmm_files=hmm_files or None,
            max_evalue=self.max_marker_evalue,
            logger=logger,
        )
        with open(output_path / "dbscan_marker_report.txt", "w") as fh:
            diagnose_marker_table(table, info, out=fh)
        logger.info(
            f"Markers: format={info['hmmout_format']}, "
            f"ORF->contig rule={info['orf_id_style']} "
            f"({info['orf_join_rate']:.1%} joined), "
            f"{info['contigs_with_markers']}/{table.n_contigs} contigs carry markers, "
            f"ceiling completeness={info['max_achievable_completeness']}"
        )
        for sname, ceiling in info["max_achievable_completeness"].items():
            if ceiling < 0.9:
                logger.warning(
                    f"Marker set '{sname}': only {ceiling:.1%} of its markers occur "
                    "anywhere in this assembly. Any completeness threshold above "
                    "that value is unreachable and the sweep will score 0 everywhere."
                )

        # ------------------------------------------------------------------
        # seeded k-means, kept for downstream post-processing
        # ------------------------------------------------------------------
        seed_num = _read_seed_count(str(seed_file))
        logger.info(f"Bin_numbers:\t{seed_num}")
        kmeans_out = seed_kmeans_full(
            logger=logger,
            contig_file=contigs,
            namelist=namelist,
            out_path=str(output_path) + os.sep,
            X_mat=norm_embeddings,
            bin_number=seed_num,
            prefix="weight_seed_kmeans",
            length_weight=length_weight,
            seed_file=str(seed_file),
        )
        result.clusters.update(kmeans_out)

        # ------------------------------------------------------------------
        # sweep
        # ------------------------------------------------------------------
        cfg = SweepConfig(core_bp=self.core_bp, min_bin_bp=self.min_bin_bp)
        sweep = density_sweep(
            latent=norm_embeddings,
            contig_lengths=length_weight,
            table=table,
            cfg=cfg,
            use_hdbscan=self.hdbscan,
            longread=longread,
            do_rescue=True,
            n_jobs=threads if threads and threads > 0 else -1,
            logger=logger,
        )

        bins, unbinned = sweep.bins, sweep.unbinned
        if self.recruit and bins and unbinned:
            bins, unbinned = recruit_unbinned(
                bins, unbinned, norm_embeddings, table, cfg
            )

        quals = table.quality_many(bins)
        n_hq = sum(q.is_hq(cfg.hq_completeness, cfg.hq_contamination) for q in quals)
        n_mq = sum(q.is_mq(cfg.mq_completeness, cfg.mq_contamination) for q in quals)
        logger.info(
            f"Sweep best params={sweep.params}: {len(bins)} bins "
            f"({n_hq} HQ, {n_mq} MQ), {len(unbinned)} contigs unbinned; "
            f"gen_seed K={n_genome}"
        )

        assign = bins_to_dataframe(bins, namelist, unbinned, include_unbinned=True)
        assign.to_csv(output_path / "dbscan_clusters.tsv", index=False, sep="\t")
        quality_dataframe(bins, quals, namelist).to_csv(
            output_path / "dbscan_bin_quality.tsv", index=False, sep="\t"
        )
        sweep.audit.to_csv(
            output_path / "dbscan_sweep_audit.tsv", index=False, sep="\t"
        )

        result.clusters["dbscan_clusters.tsv"] = assign
        with open(output_path / "cluster_result.pkl", "wb") as fh:
            pickle.dump(result, fh)

        logger.info("Clustering completed.")
        return result


def _resolve_hmmout_path(contigs: Path, output_path: Path, marker_name: str) -> str:
    """Locate gen_seed's HMMER output. Tries the documented location first,
    then a few known variants, and fails with the list of what it tried
    rather than a bare FileNotFoundError on one guessed path."""
    base = os.path.basename(str(contigs))
    candidates = [
        Path(output_path) / f"{base}.{marker_name}.hmmout",
        Path(output_path) / f"{base}.{marker_name}.hmmout.gz",
        Path(output_path) / f"{base}.{marker_name}.domtblout",
        Path(contigs).parent / f"{base}.{marker_name}.hmmout",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    found = sorted(str(p.name) for p in Path(output_path).glob("*hmm*"))
    raise FileNotFoundError(
        "Could not find gen_seed's HMMER output. Tried:\n  "
        + "\n  ".join(str(c) for c in candidates)
        + f"\nFiles matching *hmm* in {output_path}: {found or 'none'}"
    )
