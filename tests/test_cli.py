# tests/test_cli.py
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from binferno.binferno_version import __version__


def _run(args, cwd=None):
    return subprocess.run(
        [sys.executable, "-m", *args],
        capture_output=True,
        text=True,
        cwd=cwd,
    )


def _norm(s: str) -> str:
    # Normalize enough to survive Tyro/Pydantic rendering differences.
    s = s.lower()
    s = re.sub(r"\x1b\[[0-9;]*m", "", s)  # strip ANSI
    s = s.replace("_", "-")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _assert_any(haystack: str, needles: list[str]):
    if not any(n in haystack for n in needles):
        raise AssertionError(f"Expected one of {needles!r} in output:\n{haystack}")


def test_cli_help_smoke():
    result = _run(["binferno.main", "--help"])
    assert result.returncode == 0, result.stderr
    out = _norm(result.stdout)

    # Top-level usage and required args
    assert "usage" in out
    _assert_any(out, ["contigs", "--contigs"])
    _assert_any(out, ["bam-dir", "bam_dir", "--bam-dir"])
    _assert_any(out, ["output-dir", "output_dir", "--output-dir"])

    # Global controls
    _assert_any(out, ["threads", "--threads"])
    _assert_any(out, ["train-only", "train_only", "--train-only"])

    # Nested Pydantic/Tyro groups should be present
    _assert_any(out, ["features", "--features."])
    _assert_any(out, ["training", "--training."])
    _assert_any(out, ["clustering", "--clustering."])
    _assert_any(out, ["postprocessor", "--postprocessor."])

    # A few representative nested options
    _assert_any(out, ["training.use-simclr", "training.use_simclr"])
    _assert_any(out, ["training.use-barlowtwins", "training.use_barlowtwins"])
    _assert_any(out, ["features.use-kmer", "features.use_kmer"])
    _assert_any(out, ["clustering.cluster-method", "clustering.cluster_method"])


def test_cli_missing_required_args_fails():
    result = _run(["binferno.main"])
    assert result.returncode != 0
    combined = _norm(result.stdout + "\n" + result.stderr)
    _assert_any(combined, ["required", "usage", "error"])


def test_cli_unknown_flag_fails():
    result = _run(["binferno.main", "--this-flag-does-not-exist"])
    assert result.returncode != 0
    combined = _norm(result.stdout + "\n" + result.stderr)
    _assert_any(combined, ["unrecognized", "unknown", "invalid", "usage"])


def test_cli_invalid_enum_value_fails(tmp_path: Path):
    contigs = tmp_path / "contigs.fna"
    contigs.write_text(">c1\n" + "A" * 1200 + "\n")
    bam_dir = tmp_path / "bam"
    bam_dir.mkdir()
    out_dir = tmp_path / "out"

    result = _run(
        [
            "binferno.main",
            "--contigs",
            str(contigs),
            "--bam-dir",
            str(bam_dir),
            "--output-dir",
            str(out_dir),
            "--clustering.cluster-method",
            "not-a-method",
        ]
    )
    assert result.returncode != 0
    combined = _norm(result.stdout + "\n" + result.stderr)
    _assert_any(
        combined, ["cluster-method", "invalid", "not-a-method", "leiden", "dbscan"]
    )


def test_cli_help_contains_pydantic_defaults():
    result = _run(["binferno.main", "--help"])
    assert result.returncode == 0
    out = _norm(result.stdout)

    # Don't assert exact full lines, just verify defaults are rendered somewhere.
    # Tyro + Pydantic usually include "(default: ...)".
    _assert_any(out, ["default", "(default:"])
    _assert_any(
        out, ["training.simclr.batch-size", "training.simclr.batch_size", "batch-size"]
    )
    _assert_any(out, ["training.barlowtwins.epochs", "training.barlowtwins.epochs"])


def test_version_module_format():
    parts = __version__.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)
