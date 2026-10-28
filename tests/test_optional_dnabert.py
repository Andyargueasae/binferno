import subprocess
import sys


def test_default_import_without_transformers():
    # A fresh interpreter checks imports even if other tests loaded the builders.
    result = subprocess.run(
        [sys.executable, "-c", """
import builtins
original_import = builtins.__import__

def without_transformers(name, *args, **kwargs):
    if name.split(".")[0] == "transformers":
        raise ModuleNotFoundError(
            "No module named 'transformers'", name="transformers"
        )
    return original_import(name, *args, **kwargs)

builtins.__import__ = without_transformers
from binferno.features import BinfernoFeatureBuilder
from binferno.features.dnabert_s import DnaBertSFeatureBuilder

builder = DnaBertSFeatureBuilder()
try:
    builder(logger=None)
except ImportError as exc:
    assert "DNABERT-S requires transformers" in str(exc)
else:
    raise AssertionError("DNABERT-S should explain its missing dependency")
"""],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
