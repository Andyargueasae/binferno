# Conda packaging for Binferno

## Status

This repository contains a development conda recipe, not a published package.
Bioconda is the appropriate community channel for this bioinformatics application.
The initial build target is Linux x86_64; `noarch: python` describes Binferno's
Python files, not the availability of every compiled dependency on every platform.

The `bio-corgi >=0.5.1,<0.6.0` runtime dependency was absent from both conda-forge
and Bioconda when checked on 2 November 2026. It is imported when the CLI starts,
so it cannot simply be omitted. Its dependency chain (including torchapp,
hierarchicalsoftmax, polytorch, seqbank and multitax) also needs auditing and, where
missing, separate conda recipes. A full conda build cannot succeed until this is
resolved. Do not put `pip install bio-corgi` in the recipe's build or test script:
that would hide dependencies from conda and is not a Bioconda publication solution.

The recipe preserves the dependency constraints in `pyproject.toml`, maps PyPI
names to conda names (`torch` -> `pytorch`, `igraph` -> `python-igraph`,
`matplotlib` -> `matplotlib-base`), and adds directly imported rich, tqdm and
typing_extensions plus the external FragGeneScan and HMMER executables.
The existing constraints may require older builds; the latest version of a conda
package alone is not sufficient evidence that the full environment is solvable.

## Install from a checkout now (conda + pip)

From the repository root:

```bash
conda env create -f environment.yml
conda activate binferno
python -m pip check
binferno --help
command -v run_FragGeneScan.pl
hmmsearch -h
```

`environment.yml` uses Python 3.11, conda-forge/Bioconda external tools, then
installs this checkout with pip and its Python dependencies. This is a mixed
installation, not `conda install binferno`. Pip may download large PyTorch/CUDA
wheels; allow sufficient disk space. This environment is a proposed installation
path, not a claim that a full pipeline or GPU run has been validated.

ESM-C remains optional. If required, install the matching SDK after the core
installation with `python -m pip install '.[esmc]'` from the checkout, then run
`python -m pip check`. Model weights and user-selected reference databases are
separate downloads and are not bundled by this recipe. Verify CUDA availability
with `python -c "import torch; print(torch.cuda.is_available())"` on the target host.

## Validate and build

Fast checks (no model downloads or training):

```bash
python -m pip install build 'poetry-core>=2,<3' jinja2 pyyaml packaging
python -m build --no-isolation
python scripts/check_conda_packaging.py dist/*.whl
```

With conda-build installed:

```bash
conda render recipe --no-finalize
# Once all runtime dependencies are available in these channels:
conda build recipe --override-channels -c conda-forge -c bioconda
```

The Conda packaging workflow builds the sdist/wheel, checks recipe dependency and
version parity, checks the console entry point and bundled HMM/marker-set files,
and renders the conda recipe without solving. Passing these checks does **not**
mean the conda dependency solve or installed CLI smoke tests passed. The manual
`full_build` option enables the real build and the recipe's installed tests;
it is expected to fail while bio-corgi is unavailable in the configured channels.
There is no automatic upload step.

## Publish to Bioconda

1. Package missing dependencies first, including their transitive dependencies;
   confirm the pinned versions can solve on Linux x86_64. Include missing
   bioinformatics recipes in a Bioconda submission and general packages in
   conda-forge as appropriate.
2. Merge/release the desired Binferno source after reviewing this branch. Build
   and validate that release's sdist; use a new version if the published source
   has changed. Do not silently overwrite an existing release.
3. Copy `recipe/meta.yaml` to `recipes/binferno/meta.yaml` in a fork of
   `bioconda/bioconda-recipes`. Replace `source: path: ..` with an immutable
   release/sdist URL and the SHA256 of those exact bytes. Set the recipe version
   to that release. A local source path is only for development in this repository.
4. Run Bioconda lint/build tests, including `binferno --help` and resource checks,
   and open a pull request there. Bioconda's review/merge/build process publishes
   the package. Adding a recipe to this repository alone does not publish it.

After acceptance and upload, users can create an environment with:

```bash
conda create -n binferno --strict-channel-priority --override-channels \
  -c conda-forge -c bioconda binferno
```

That command is a future installation command, not currently advertised as working.
A personal Anaconda.org channel is an alternative, but still needs all runtime
packages to be available and correctly declared; uploads require an account/token.

References:
- https://bioconda.github.io/contributor/guidelines.html
- https://bioconda.github.io/contributor/workflow.html
