# Publish Binferno with conda

This recipe packages Binferno 0.1.0 from the immutable GitHub commit
`b6207a7eaea2f192ca2cde9f2c50cc2c9a0d202c`, including CORGI removal and PR #7's
regression fixes. The archive URL and verified SHA256 are recorded in `meta.yaml`.
This deliberately does not use the older `v0.1.0` tag or PyPI 0.1.0 source, which
predate these changes. The first conda build number is 0. Future source changes
require a new source hash and an appropriate version/build-number update.

All runtime dependencies are declared as conda dependencies. `bio-corgi` is no
longer required by current main. PyPI names are mapped to conda names:
`torch` -> `pytorch`, `igraph` -> `python-igraph`, and
`matplotlib` -> `matplotlib-base`. FragGeneScan and HMMER are installed too.
ESM-C's SDK/weights and user-selected reference databases remain optional,
separate downloads. The initial validation platform is Linux x86_64.
`noarch: python` describes Binferno's package, not every dependency's availability.

## Build and test

Create the tooling environment and run the full build from the repository root:

```bash
conda create -n binferno-build --strict-channel-priority --override-channels \
  -c conda-forge -c bioconda python=3.11 conda-build
conda activate binferno-build
python -c "from conda_build import api; api.build('recipe', anaconda_upload=False, channel_urls=['conda-forge', 'bioconda'], override_channels=True)"
```

The build downloads and verifies the pinned source archive, installs without pip
dependency downloads, and tests the installed package outside the source tree.
Tests cover CLI/imports, bundled marker files, external executables, and the
existing pytest suite. CUDA-only and long-running tests retain their normal skip
conditions. No model weights are downloaded. This is not a full biological
benchmark or GPU validation.

The Conda packaging workflow performs this full build automatically, in addition
to Python sdist/wheel checks and recipe/metadata parity checks. It uploads the
result as `binferno-conda-package` in the workflow's artifacts. It never uploads
to Anaconda.org automatically. Successful rendering alone is not a build test.

## Publish through Bioconda (recommended)

1. Review and merge the packaging PR after the full conda CI build passes.
2. In a fork of `bioconda/bioconda-recipes`, copy this directory's `meta.yaml` to
   `recipes/binferno/meta.yaml`. It is self-contained: the source URL/hash and
   tests are included, with no reference to a local checkout.
3. Run Bioconda lint/build checks and open a pull request in that repository.
   Bioconda review, merge and upload make the package publicly installable.

No Anaconda.org token is needed for a Bioconda recipe contribution. Adding this
recipe to Binferno's repository does not itself publish a package.

After acceptance/upload, users can run:

```bash
conda create -n binferno --strict-channel-priority --override-channels \
  -c conda-forge -c bioconda binferno
conda activate binferno
binferno --help
```

Do not advertise that command as available before the package appears in Bioconda.
A tagged source release is preferable for future versions; recompute the hash for
its exact archive rather than substituting a tag URL under an existing checksum.

## Optional personal Anaconda.org channel

The same built artifact can be uploaded to an account you control. Install
`anaconda-client` in a separate upload environment, authenticate using your own
credentials, then upload the exact tested package:

```bash
anaconda upload --user YOUR_CHANNEL /path/to/binferno-0.1.0-*.conda
```

Users then include `-c YOUR_CHANNEL` as well as conda-forge and Bioconda. An
Anaconda.org account/channel and authentication are required only for this path.
Never commit upload tokens. No upload has been configured or performed here.

References:
- https://bioconda.github.io/contributor/guidelines.html
- https://bioconda.github.io/contributor/workflow.html
