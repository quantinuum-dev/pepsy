# Contributing to Pepsy

Pepsy uses a `src/` layout and requires Python 3.12 or newer. CI runs on
Python 3.12. Create an isolated environment, install the development profile, and
run the fast test suite before opening a change:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -q
python -m pyflakes src tests
python -m ruff check src tests
```

The default suite excludes integration and stress tests to keep local
iteration short. Run the complete suite when changing an advanced domain:

```bash
python -m pip install -e ".[dev,test-extended]"
pytest -q -o addopts=""
```

Test tiers and domain markers are available when a narrower pass is useful:

```bash
pytest -q -o addopts="" -m "core and not optional"
pytest -q -o addopts="" -m optional
pytest -q -o addopts="" -m vmc       # replace with bp, tree, peps, or another domain
```

The `core and not optional` profile is the dependency-light API contract;
`core` without the exclusion includes stable APIs exercised through optional
backends. The full command includes all domain and slow tests.

MPS and tree tests are organized by responsibility. The `test_optimize_*`
files cover replay and public optimizer behavior; related suites live beside
them rather than importing tests from those files:

| Area | Focused test modules |
| --- | --- |
| MPS compression and FIT | `test_mps_compression_modes.py`, `test_mps_fit_kernels.py` |
| Native MPS symmetry | `test_mps_fermions.py` |
| MPS layout, scale, and measurements | `test_mps_layout.py`, `test_mps_normalization.py`, `test_mps_controls.py` |
| Tree layout and state invariants | `test_tree_layout.py`, `test_tree_state.py` |
| Tree native arrays and measurements | `test_tree_native.py`, `test_tree_controls.py` |

Run a whole domain, including the smaller existing regression modules, with:

```bash
MPLBACKEND=Agg pytest -q -o addopts="" tests/test_optimize_mps.py tests/test_mps_*.py
MPLBACKEND=Agg pytest -q -o addopts="" tests/test_optimize_tree.py tests/test_tree_*.py
```

The extracted modules declare their tier/domain markers explicitly. Native
MPS symmetry tests are marked `optional` because every case requires Symmray.
Small shared reference builders live in `_mps_test_helpers.py` and
`_tree_test_helpers.py`; keep setup used by only one suite in that suite.

CI runs `-m "smoke or (core and not optional and not slow)"` on Python 3.12
with the five direct core dependencies pinned to their declared minimums.
Constraints are generated from `pyproject.toml`; development tools and
transitive dependencies resolve normally. This checks the actual lower-bound
contract without maintaining a second version list.
One extended job installs current compatible releases through
`.[dev,test-extended,contraction,vmc]` and runs the
complete collection with a 60% whole-package coverage gate. Combining optional
profiles keeps JAX coverage in MPS, tree, and operator tests as well as VMC,
without a second full-suite VMC job. Optional tests skip only when their
dependency or required upstream capability is absent. The smoke job checks
contracts without a whole-package coverage gate.
The extended job checks backend imports before the numerical suite, stops at
the first failing test, and publishes its traceback
as a GitHub annotation; successful runs still execute the full collection.
MPI integration CI installs `.[dev,mpi,stabilizer]` to exercise stabilizer
trajectories as well as ordinary MPS and tree execution.

The type-check configuration targets Python 3.12 so installed NumPy stubs
use the same language version. Reproduce it with:

```bash
python -m mypy src/pepsy/tensors/validation.py src/pepsy/vmc/api.py
```

To measure the local import boundary without importing an advanced domain:

```bash
python -X importtime -c "import pepsy" 2> /tmp/pepsy-importtime.txt
tail -n 1 /tmp/pepsy-importtime.txt
```

Use the smallest relevant optional profile when developing a backend:

- `.[contraction]` for accelerated Cotengra path search;
- `.[symmetry]` for Symmray-backed tensors;
- `.[stabilizer]` for Stim-backed stabilizer workflows;
- `.[vmc-torch]` or `.[vmc-netket]` for VMC integrations;
- `.[solvers]` and `.[layout]` for external solver and layout backends.

Keep the stable API small and add new advanced functionality under its
responsibility-based module. The `pepsy.experimental` namespace provides lazy
discovery of existing domains; it does not own their implementations. Add a
regression test for observable behavior and update the handwritten Markdown documentation when a
public API changes. Deprecated imports should emit `DeprecationWarning` and
remain functional during the documented compatibility window.

Keep algorithm tests focused: add one deterministic regression test for the
changed behavior, plus a boundary or invariant case only when it covers a
distinct failure mode. Avoid large parameter matrices, repeated seeds, and
duplicate integration tests. Mark optional-backend, differential, and stress
coverage as `integration` or `slow`; do not make it part of the default loop.
Check aliases at the public configuration boundary, then run numerical cases
through the canonical mode. Cross-product parameterization is appropriate
when the interaction itself can fail independently, not merely to repeat the
same algorithm under every accepted spelling.

Do not commit generated caches, build output, notebook execution artifacts, or
local environment files. For numerical changes, include the backend, dtype,
seed, and tolerance assumptions in the test or documentation.

## Agent guidance validation

After changing agent skills, their catalog, or linked files, run from the
repository root:

```bash
python .github/skills/pepsy-maintainer/scripts/validate_catalog.py
```

This standard-library-only check validates skill names and frontmatter,
required metadata-file presence, local link targets in `SKILL.md`, and catalog
and bundle-manifest path coverage. It does not validate instruction meaning or
links throughout all documentation. See the [skill policy](.github/skills/SKILL_POLICY.md)
for ownership and review requirements.

The independent **Agent guidance** job in [CI](.github/workflows/ci.yml) runs
this command on pull requests and pushes to `main` or `develop`, without
installing Pepsy or its numerical dependencies.
