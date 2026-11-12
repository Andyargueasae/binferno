"""Check package/module versions and, optionally, a release tag (Python 3.11+)."""

import argparse
from pathlib import Path
import runpy
import tomllib


def check_version(root: Path, tag: str | None = None) -> str:
    with (root / "pyproject.toml").open("rb") as stream:
        package_version = tomllib.load(stream)["project"]["version"]
    module_version = runpy.run_path(
        str(root / "binferno" / "binferno_version.py")
    )["__version__"]
    if module_version != package_version:
        raise ValueError(
            f"Version mismatch: pyproject.toml={package_version}, "
            f"binferno_version.py={module_version}"
        )
    if tag is not None and tag != f"v{package_version}":
        raise ValueError(f"Release tag {tag!r} must be 'v{package_version}'")
    return package_version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="Release tag, e.g. v0.1.0")
    args = parser.parse_args()
    try:
        version = check_version(Path(__file__).resolve().parents[1], args.tag)
    except ValueError as error:
        parser.exit(1, f"{error}\n")
    print(f"Version verified: {version}")


if __name__ == "__main__":
    main()
