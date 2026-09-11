import pytest
import random
import numpy as np
import pandas as pd
import pyfastx
from pathlib import Path

CONTIG = Path(__file__).parent / "data" / "tiny.fasta"
MARKER = Path(__file__).parent / "data" / "marker.fna"
BAM = Path(__file__).parent / "data" / "bam"
KMER_FEATURE_DIM = 136


@pytest.fixture(autouse=True)
def set_test_seed() -> None:
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


def pytest_addoption(parser):
    parser.addoption(
        "--long",
        action="store_true",
        default=False,
        help="run long-running tests",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--long"):
        return
    skip_long = pytest.mark.skip(reason="needs --long option to run")
    for item in items:
        if "long" in item.keywords:
            item.add_marker(skip_long)


@pytest.fixture
def tiny_fasta(tmp_path: Path) -> Path:
    path = tmp_path / "contigs.fna"
    path.write_text(
        ">contig_1\n" + "ATGC" * 900 + "\n"
        ">contig_2\n" + "GCAT" * 900 + "\n"
        ">contig_3\n" + "AAAA" * 900 + "\n"
    )
    return path


@pytest.fixture
def bam_dir(tmp_path: Path) -> Path:
    bam_path = tmp_path / "bam"
    bam_path.mkdir()
    (bam_path / "sample1.bam").write_text("")
    return bam_path


def toy_embeddings(contig: Path) -> pd.DataFrame:
    seq = pyfastx.Fasta(str(contig), build_index=False)
    namelist = [s for s, _ in seq]
    x = np.random.default_rng(42).normal(size=(len(namelist), 128)).astype("float32")
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return pd.DataFrame(x, index=namelist)


def toy_features(
    n_cov: int,
    view_num: int = 6,
    simclr: bool = False,
) -> pd.DataFrame:
    rows = 30
    cols = KMER_FEATURE_DIM + 2 * n_cov
    x = np.random.default_rng(42).normal(size=(rows, cols)).astype("float32")
    x /= np.linalg.norm(x, axis=1, keepdims=True)

    columns = (
        [f"{i}" for i in range(KMER_FEATURE_DIM)]
        + [f"cov_{i}_mean" for i in range(n_cov)]
        + [f"cov_{i}_std" for i in range(n_cov)]
    )

    if simclr:
        index = [f"contig_{i}.{j}" for i in range(5) for j in range(view_num)]
    else:
        index = [f"contig_{i}.original" for i in range(5)] + [
            f"contig_{i}.{j}" for i in range(5) for j in range(view_num - 1)
        ]
    return pd.DataFrame(x, index=index, columns=columns)


@pytest.fixture
def dummy_all_fragments():
    seq1 = "ATGC" * 900
    seq2 = "GCAT" * 900
    seq3 = "AAAA" * 900

    frag_dict = {
        "contig_1": {"sequence": seq1, "fragments": [], "fragment_info": {}},
        "contig_2": {"sequence": seq2, "fragments": [], "fragment_info": {}},
        "contig_3": {"sequence": seq3, "fragments": [], "fragment_info": {}},
    }
    all_frags = []
    namelist = []

    for i in range(6):
        subseq_start = random.randint(0, 1000)
        subseq_end = random.randint(1000, 4 * 900)
        frag_len = subseq_end - subseq_start

        for contig_name, seq in [
            ("contig_1", seq1),
            ("contig_2", seq2),
            ("contig_3", seq3),
        ]:
            frag_id = f"{contig_name}_frag_{i}"
            frag_dict[contig_name]["fragments"].append(frag_id)
            frag_dict[contig_name]["fragment_info"][frag_id] = {
                "start": subseq_start,
                "end": subseq_end,
                "start_pos": subseq_start,
                "length": frag_len,
            }
            all_frags.append((frag_id, seq[subseq_start:subseq_end]))
            namelist.append(frag_id)

    return frag_dict, all_frags, namelist


def dummy_train_simclr():
    return {
        "simclrnet": False,
        "proj": True,
        "learning_rate": 0.001,
        "batch_size": 3,
        "epochs": 3,
        "temperature": 0.15,
        "contig_length": 1000,
        "gated": True,
        "hard_weighted_nce": True,
        "auxiliary_loss": True,
        "seed": 42,
        "tnf_dim": 136,
        "attention": True,
        "view_num": 6,
    }


def dummy_train_barlowtwins_args():
    return {
        "batch_size": 3,
        "epochs": 3,
        "learning_rate": 0.001,
        "seed": 42,
        "tnf_dim": 136,
        "view_num": 6,
    }


def sim_cluster_fasta(n_contigs: int = 30) -> Path:
    path = Path("sim_cluster_contigs.fna")
    local_rng = np.random.default_rng(42)
    with open(path, "w") as f:
        for i in range(n_contigs):
            f.write(f">contig_{i}\n")
            length_multiplier = int(local_rng.integers(240, 2500))
            f.write("ATGC" * length_multiplier + "\n")
    return path
