# Contributing to Pepsy

Pepsy requires Python 3.12+ and uses a `src/` layout. Follow any device-local
environment override before running the commands below.

## Set up and run the smoke tests

For a new checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
python -m pyflakes src tests
python -m ruff check src tests
```

The default test command runs the small `smoke` selection. It does not run
every core or domain test.

## Choose a test scope

Install extended dependencies before running optional domain tests:

```bash
python -m pip install -e ".[dev,test-extended]"
```

| Scope | Command |
| --- | --- |
| Full collection, including slow tests | `python -m pytest -q -o addopts=""` |
| Core without optional dependencies | `python -m pytest -q -o addopts="" -m "core and not optional"` |
| Optional features | `python -m pytest -q -o addopts="" -m optional` |
| One domain | `python -m pytest -q -o addopts="" -m vmc` (or `bp`, `tree`, `peps`, etc.) |

The `core` marker alone includes some tests that need optional backends.
Tests skip when a required dependency or upstream capability is unavailable;
a skipped test is not validation of that path.

MPS and tree tests are split by responsibility:

| Area | Modules |
| --- | --- |
| MPS compression and FIT | `test_mps_compression_modes.py`, `test_mps_fit_kernels.py` |
| Native MPS symmetry | `test_mps_fermions.py` (optional) |
| MPS layout, scale, measurements | `test_mps_layout.py`, `test_mps_normalization.py`, `test_mps_controls.py` |
| Tree layout and state | `test_tree_layout.py`, `test_tree_state.py` |
| Tree native arrays and measurements | `test_tree_native.py`, `test_tree_controls.py` |

Run a complete domain, including its replay tests, with:

```bash
MPLBACKEND=Agg python -m pytest -q -o addopts="" tests/test_optimize_mps.py tests/test_mps_*.py
MPLBACKEND=Agg python -m pytest -q -o addopts="" tests/test_optimize_tree.py tests/test_tree_*.py
```

Use the smallest [installation profile](docs/installation.md#optional-features)
needed for your change. The full VMC profile requires `.[vmc]` in addition to
`test-extended`.

## What CI checks

[CI](.github/workflows/ci.yml) uses Python 3.12:

| Job | Coverage |
| --- | --- |
| Core minimums | Smoke and core tests without optional/slow cases; the five direct core dependencies are pinned to minimums read from `pyproject.toml`. |
| Extended | Full collection with `.[dev,test-extended,contraction,vmc]` and a 60% whole-package coverage gate. |
| MPI | Two- and three-rank integration, including stabilizer trajectories through `.[dev,mpi,stabilizer]`. |
| Other checks | Packaging, strict documentation build, type checks, and agent guidance. |

Development tools and transitive dependencies resolve normally. The extended
job checks backend imports first, stops at the first failure, and attaches its
traceback to the run. The smoke job has no whole-package coverage gate.

Reproduce the type check with:

```bash
python -m mypy src/pepsy/tensors/validation.py src/pepsy/vmc/api.py
```

For import profiling, use `python -X importtime -c "import pepsy"`; see the
[import-cost notes](docs/development/import-weight.md).

## Keep changes focused

- Put new APIs in their [owning namespace](docs/development/package_layout.md).
  `pepsy.experimental` provides discovery, not separate implementations.
- Update the handwritten API guide when public behavior changes. Deprecated
  imports must warn and keep working for the [compatibility window](docs/stability.md).
- Test observable behavior with a deterministic regression and, when useful,
  a distinct boundary or invariant case. Record backend, dtype, seed, and
  tolerance assumptions for numerical changes.
- Check aliases at the configuration boundary, then test numerical behavior
  through the canonical mode. Use parameter cross-products only when the
  interaction itself can fail.
- Mark optional-backend, differential, and stress coverage as `integration`
  or `slow`. Keep them out of the smoke loop.
- Share small MPS/tree reference builders in `_mps_test_helpers.py` and
  `_tree_test_helpers.py`; keep single-suite setup in its own test module.
- Keep caches, build output, notebook execution artifacts, and local
  environment files out of commits.

See the [documentation guide](docs/development/README.md#writing-markdown)
for where to put new Markdown and how to keep it readable.

## Validate agent guidance

After editing skills, their catalog, or linked files, run:

```bash
python .github/skills/pepsy-maintainer/scripts/validate_catalog.py
```

This checks skill metadata, local skill links, and catalog/bundle coverage.
It does not check instruction meaning or all documentation links. Follow the
[skill policy](.github/skills/SKILL_POLICY.md) for the full workflow.
