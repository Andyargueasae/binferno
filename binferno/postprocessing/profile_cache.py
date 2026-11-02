"""Input signatures for reusing completed CheckM profiling steps."""

import hashlib
import json
from pathlib import Path


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def database_signature(database):
    """Track database replacement/update without rereading multi-GB HMM files.

    Database files are identified by path, size, mtime and ctime. In-place
    database updates must retain the normal filesystem change timestamps.
    """
    root = Path(database).resolve()
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            stat = path.stat()
            entries.append((
                str(path.relative_to(root)), stat.st_size,
                stat.st_mtime_ns, stat.st_ctime_ns,
            ))
    return {"root": str(root), "files": entries}


def job_signature(bin_dir, bin_ext, marker_set_file, database, tools):
    bins = sorted(
        path for path in Path(bin_dir).iterdir()
        if path.is_file() and path.name.endswith("." + bin_ext)
    )
    if not bins:
        raise ValueError(f"No .{bin_ext} bins found in {bin_dir}")
    inputs = {
        "schema": 1,
        "bins": [(path.name, file_digest(path)) for path in bins],
        "markers": file_digest(marker_set_file),
        "database": database,
        "tools": tools,
    }
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()


def read_state(path):
    try:
        state = json.loads(path.read_text())
        return state if isinstance(state, dict) else {}
    except (OSError, ValueError):
        return {}


def write_state(path, state):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, sort_keys=True))
    temporary.replace(path)
