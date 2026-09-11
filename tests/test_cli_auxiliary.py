# tests/test_cli_auxiliary.py
import gzip
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from binferno import utils as bin_utils
from binferno.auxiliary import getmarker_2quarter as marker_mod


def _run(args):
    return subprocess.run(
        [sys.executable, "-m", *args],
        capture_output=True,
        text=True,
    )


def test_getmarker_cli_missing_required_args_fails():
    result = _run(["binferno.auxiliary.getmarker_2quarter"])
    assert result.returncode != 0
    combined = (result.stdout + "\n" + result.stderr).lower()
    assert "usage" in combined or "not enough arguments" in combined


def test_getmarker_cli_happy_path(tmp_path: Path):
    fasta = tmp_path / "contigs.fasta"
    hmm = tmp_path / "hits.hmmout"
    out = tmp_path / "seed.out"

    fasta.write_text(">contigA\n" + "A" * 1200 + "\n" ">contigB\n" + "C" * 1200 + "\n")

    # Need >=17 columns; indexes used by code:
    # arr[0]=target, arr[3]=marker, arr[5]=hmm_len, arr[15], arr[16]=covered span
    hmm.write_text(
        "contigA_1 x x TIGR00388 x 100 x x x x x x x x x 0 60\n"
        "contigB_1 x x TIGR00388 x 100 x x x x x x x x x 5 70\n"
    )

    result = _run(
        [
            "binferno.auxiliary.getmarker_2quarter",
            str(hmm),
            str(fasta),
            "1000",
            str(out),
        ]
    )

    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert out.exists()
    lines = [x.strip() for x in out.read_text().splitlines() if x.strip()]
    assert "contigA" in lines
    assert "contigB" in lines


def test_gethmmmarker_selects_matching_contigs(tmp_path: Path):
    fasta = tmp_path / "contigs.fasta"
    fasta.write_text(
        ">contigA\n" + "A" * 1200 + "\n"
        ">contigB\n" + "C" * 1500 + "\n"
        ">contigC\n" + "G" * 200 + "\n"
    )
    hmm = tmp_path / "hits.hmmout"
    hmm.write_text(
        "contigA_1 x x TIGR00388 x 100 x x x x x x x x x 10 60\n"
        "contigB_1 x x TIGR00388 x 100 x x x x x x x x x 20 80\n"
        "contigC_1 x x TIGR00388 x 100 x x x x x x x x x 5 50\n"
    )
    out = tmp_path / "seed.txt"

    rc = marker_mod.gethmmmarker(str(hmm), str(fasta), 1000, str(out))

    assert rc == 0
    assert out.exists()
    contents = [line.strip() for line in out.read_text().splitlines() if line.strip()]
    assert contents == ["contigA", "contigB"]


def test_gethmmmarker_returns_minus_one_with_no_valid_hit(tmp_path: Path):
    fasta = tmp_path / "contigs.fasta"
    fasta.write_text(">contigA\n" + "A" * 1200 + "\n")
    hmm = tmp_path / "hits.hmmout"
    hmm.write_text("contigA_1 x x TIGR00388 x 100 x x x x x x x x x 5 20\n")
    out = tmp_path / "seed.txt"

    rc = marker_mod.gethmmmarker(str(hmm), str(fasta), 1000, str(out))

    assert rc == -1
    assert not out.exists()


def test_extract_contig_base_handles_fragmented_names():
    assert marker_mod.extract_contig_base("contigA_1_1_+") == "contigA"
    assert marker_mod.extract_contig_base("contigA_1") == "contigA"
    assert marker_mod.extract_contig_base("contigA") is None


def test_calc_n50_uses_min_length_threshold(tmp_path: Path):
    fasta = tmp_path / "contigs.fasta"
    fasta.write_text(
        ">c1\n" + "A" * 6000 + "\n"
        ">c2\n" + "C" * 3000 + "\n"
        ">c3\n" + "G" * 200 + "\n"
    )

    assert bin_utils.calc_n50(fasta, min_length=500) == 3000


def test_get_length_parses_fasta_and_gzip(tmp_path: Path):
    fasta = tmp_path / "contigs.fasta"
    fasta.write_text(">a\nACGT\n>b\nAAAA\n")

    assert bin_utils.get_length(str(fasta)) == {"a": 4, "b": 4}

    gz_path = tmp_path / "contigs.fa.gz"
    with gzip.open(gz_path, "wt", encoding="utf-8") as fh:
        fh.write(">g1\nTTTT\n>g2\nGGGGGG\n")

    assert bin_utils.get_length(str(gz_path)) == {"g1": 4, "g2": 6}


def test_save_result_writes_ordered_tab_output(tmp_path: Path):
    out = tmp_path / "nested" / "result.tsv"
    result = [np.array([7.0]), np.array([9.5])]

    bin_utils.save_result(result, str(out), ["c1", "c2"])

    assert out.exists()
    assert out.read_text().splitlines() == ["c1\t7.0", "c2\t9.5"]


def test_gen_seed_invokes_external_steps_and_writes_seed(monkeypatch, tmp_path: Path):
    contigs = tmp_path / "tower.fasta"
    contigs.write_text(">contigA\n" + "A" * 1500 + "\n")
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    seed_path = output_dir / "tower.fasta.marker.3quarter_lencutoff_1000.seed"

    def fake_run_pyrodigal(contig_file, faa_path, threads):
        faa_path.write_text(">gene1\nAAA\n")

    def fake_run_hmmsearch_pyhmmer(
        hmm_path, faa_path, domtblout_path, threads, e_value
    ):
        domtblout_path.write_text("# domtblout\n")

    def fake_subprocess_run(cmd, shell, check):
        seed_path.write_text("contigA\n")
        return None

    monkeypatch.setattr(bin_utils, "run_pyrodigal", fake_run_pyrodigal)
    monkeypatch.setattr(bin_utils, "run_hmmsearch_pyhmmer", fake_run_hmmsearch_pyhmmer)
    monkeypatch.setattr(bin_utils, "file_len", lambda path: 1)
    monkeypatch.setattr(bin_utils.subprocess, "run", fake_subprocess_run)

    count = bin_utils.gen_seed(
        logger=type("Logger", (), {"info": lambda *a, **k: None})(),
        contig_file=contigs,
        output_dir=output_dir,
        threads=2,
        contig_length_threshold=1000,
        marker_name="marker",
        quarter="3quarter",
    )

    assert count == 1
    assert seed_path.exists()
    assert seed_path.read_text().splitlines() == ["contigA"]
