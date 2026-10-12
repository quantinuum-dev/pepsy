# Test profiles and CI

This page owns Pepsy's test commands, backend coverage and CI scope.
Activate the session's environment or device-local override before every
Python command and run from the repository root. For setup and contributor
workflow, see [CONTRIBUTING](../../CONTRIBUTING.md).

The default test command selects the small `smoke` profile. Use a focused
module for the behavior being changed; broaden only when its dependencies or
unresolved failures require it. Skips describe unavailable coverage.

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
The test harness defaults `JAX_DEFAULT_MATMUL_PRECISION=highest` so numerical
references use full float32 accumulation on GPUs, and
`XLA_PYTHON_CLIENT_PREALLOCATE=false` so JAX can share memory with other tested
backends. Explicit environment settings take precedence. These are test
defaults; importing Pepsy does not change either setting.
Tests also restore the public JAX mesh on exit when that API is available,
so importing NetKet in one test does not pin subsequent CPU tests to a GPU mesh.

Tests skip when a required dependency or upstream capability is unavailable;
a skipped test is not validation of that path.
CI prints skip reasons with `-ra`. Add that flag locally when investigating
coverage gaps; check whether each skip requires a dependency, upstream
capability, multiple MPI ranks, or hardware unavailable on the runner before
changing the test. Installing optional packages alone does not provide GPU
or multi-process coverage.

To exercise MPI with an installed MPI runtime and `mpi4py`, run both rank
counts (nightly uses the same integration module):

```bash
mpiexec -n 2 python -m pytest -q -ra -o addopts="" tests/test_mpi_integration.py
mpiexec -n 3 python -m pytest -q -ra -o addopts="" tests/test_mpi_integration.py
```

The JAX non-default-device check can use two logical CPU devices. Set the
flag before starting Python; this checks device placement, not GPU execution:

```bash
XLA_FLAGS=--xla_force_host_platform_device_count=2 python -m pytest -q -ra -o addopts="" tests/test_peps_sampler.py::test_peps_sampler_jax_nondefault_device
```

CUDA/CuPy tests require a compatible NVIDIA GPU and runtime. Metal tests
require an available Torch MPS device; an individual operation can still be
unsupported. Keep these skip reasons visible rather than enabling CPU
fallback and counting it as GPU validation.

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

Use the smallest [installation profile](../installation.md#optional-features)
needed for your change. The full VMC profile requires `.[vmc]` in addition to
`test-extended`.

## What CI checks

[CI](../../.github/workflows/ci.yml) runs two lightweight jobs on pushes and pull
requests, using Python 3.12 on Linux:

| Job | Coverage |
| --- | --- |
| Core checks | The same small smoke selection as `python -m pytest -q`, Ruff, focused type checks, and agent guidance. The five direct core dependencies use minimums read from `pyproject.toml`. |
| Package | Build a source archive and a wheel from that archive; validate metadata, imports, and a small numerical example in clean installations. |

Pip downloads are cached separately per job. Superseded runs on the same ref
are cancelled, and each job has a ten-minute timeout. Tools and transitive
dependencies resolve normally. Routine CI does not install Torch, JAX,
NetKet, MPI, or documentation-builder packages. Users install backend extras
only when their own workflows need them; the ordinary package install stays
on the base dependency profile.

The [nightly workflow](../../.github/workflows/nightly.yml) runs the full collection
with `.[test-extended,contraction,vmc]` and a **60% whole-package coverage gate**.
It checks backend imports first and stops on the first failure. A second job
runs MPI unit tests, then integration tests with two and three ranks,
sharing one installation. Full tests have a 90-minute timeout;
MPI has ten minutes.

The nightly schedule is **03:30 UTC** on the repository's default branch.
To check `develop` or another branch before merging, select **Nightly full
suite → Run workflow → branch** in GitHub Actions. Both workflows also support
manual runs. Optional and slow regressions may wait until this full check;
a green routine CI run does not establish full-suite success.

Reproduce the type check with:

```bash
python -m mypy src/pepsy/tensors/validation.py src/pepsy/vmc/api.py
```

For import profiling, use `python -X importtime -c "import pepsy"`; see the
[import-cost notes](import-weight.md).
