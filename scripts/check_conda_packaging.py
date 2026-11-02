"""Check recipe/metadata parity and built wheel contents without runtime dependencies."""
from pathlib import Path
import sys
import re
import tomllib
import zipfile
from configparser import ConfigParser
from email.parser import BytesParser

import jinja2
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
import yaml

root = Path(__file__).resolve().parents[1]
project = tomllib.loads((root / 'pyproject.toml').read_text())['project']
recipe = yaml.safe_load(jinja2.Template((root / 'recipe/meta.yaml').read_text()).render(PYTHON='python'))
assert recipe['package']['name'] == project['name']
assert str(recipe['package']['version']) == project['version']
assert recipe['build']['noarch'] == 'python'
assert 'path' not in recipe['source'], 'Publication requires an immutable source URL'
assert re.search(r'/archive/[0-9a-f]{40}\.tar\.gz$', recipe['source']['url'])
assert re.fullmatch(r'[0-9a-f]{64}', recipe['source']['sha256'])
assert 'bio-corgi' not in {canonicalize_name(Requirement(d).name) for d in project['dependencies']}
assert recipe['build']['entry_points'] == ['binferno = binferno.main:main']

# Conda uses different names for these PyPI distributions.
mapping = {'torch': 'pytorch', 'igraph': 'python-igraph', 'matplotlib': 'matplotlib-base'}
# The recipe narrows upstream's broad range to the supported private sklearn API.
conda_constraints = {'scikit-learn': '>=1.7.2,<1.8'}
run = {}
for spec in recipe['requirements']['run']:
    name, _, constraint = spec.partition(' ')
    run[canonicalize_name(name)] = constraint.replace(' ', '')
for dependency in project['dependencies']:
    req = Requirement(dependency)
    name = mapping.get(req.name, req.name)
    assert canonicalize_name(name) in run, f'Missing runtime dependency: {name}'
    # Compare individual specifiers: their order is not significant.
    expected = conda_constraints.get(req.name, str(req.specifier))
    assert set(run[canonicalize_name(name)].split(',')) == set(expected.split(',')), dependency
assert run['python'] == project['requires-python']
for name in ('fraggenescan', 'hmmer', 'rich', 'tqdm', 'typing-extensions'):
    assert name in run, f'Missing executable/direct import: {name}'

wheel_path = Path(sys.argv[1])
with zipfile.ZipFile(wheel_path) as wheel:
    names = wheel.namelist()
    metadata_path = next(n for n in names if n.endswith('.dist-info/METADATA'))
    metadata = BytesParser().parsebytes(wheel.read(metadata_path))
    assert metadata['Name'] == project['name']
    assert metadata['Version'] == project['version']
    entry = ConfigParser()
    entry.read_string(wheel.read(metadata_path.replace('METADATA', 'entry_points.txt')).decode())
    assert entry['console_scripts']['binferno'] == 'binferno.main:main'
    required = [
        *(f'binferno/auxiliary/{name}.hmm' for name in ('marker', 'bacar_marker', 'pltd_marker')),
        *(f'binferno/postprocessing/scripts/checkm_ms/{name}.ms' for name in ('archaea', 'bacteria', 'plastid')),
    ]
    for path in required:
        assert path in names and wheel.getinfo(path).file_size > 0, f'Missing/empty resource: {path}'
print('Recipe dependencies, version, entry point, and wheel resources verified.')
