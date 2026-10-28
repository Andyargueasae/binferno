import subprocess
import sys

import pytest


@pytest.mark.parametrize("missing", ["transformers", "umap"])
def test_default_import_without_optional_dependencies(missing):
    # A fresh interpreter checks imports even if other tests loaded the builders.
    result = subprocess.run(
        [sys.executable, "-c", """
import sys
missing = sys.argv[1]
import builtins
original_import = builtins.__import__

def without_transformers(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".")[0] == missing:
        raise ModuleNotFoundError(
            f"No module named {missing!r}", name=missing
        )
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = without_transformers
from binferno.features import BinfernoFeatureBuilder
from binferno.features.dnabert_s import DnaBertSFeatureBuilder

from binferno.main import binferno
from binferno.reducer.umap import UmapReducer
builder = DnaBertSFeatureBuilder() if missing == "transformers" else UmapReducer()
try:
    builder(logger=None) if missing == "transformers" else builder(data=None)
except ImportError as exc:
    assert ("DNABERT-S requires transformers" if missing == "transformers"
            else "UMAP reduction requires umap-learn") in str(exc)
else:
    raise AssertionError("Optional feature should explain its missing dependency")
""", missing],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

