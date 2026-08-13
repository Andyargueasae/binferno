import subprocess
import sys

from binferno.binferno_version import __version__


def test_cli_help_smoke():
    result = subprocess.run(
        [sys.executable, "-m", "binferno.main", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    stdout = result.stdout.lower()
    assert "usage" in stdout
    assert "contigs" in stdout
    assert "bam" in stdout
    assert "output" in stdout
    assert "threads" in stdout
    assert "train_only" in stdout or "train-only" in stdout
    assert "training.use-simclr" in stdout


def test_cli_missing_required_args_fails():
    result = subprocess.run(
        [sys.executable, "-m", "binferno.main"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    combined = (result.stdout + "\n" + result.stderr).lower()
    assert "required" in combined or "usage" in combined


def test_version_module_format():
    parts = __version__.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)