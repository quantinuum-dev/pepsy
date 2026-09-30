# Dependency minimum audit — September 2026

This is dated compatibility evidence, not an instruction to upgrade dependencies
in another environment. `pyproject.toml` owns the current requirements.
Classification: **adopt supported dependency bounds**. No numerical algorithm,
vendor source, or compatibility guard was changed.

## Core requirements

| Dependency | Previous floor | Supported floor | Evidence |
| --- | --- | --- | --- |
| NumPy | 1.24 | 1.26 | Python 3.12 support starts at 1.26; exact 1.26.0 tested. |
| Quimb | 1.8 | 1.15 | Pepsy uses `tn2d.core`, `TensorNetworkGenOperator`, and modern decomposition APIs absent in 1.12.1. Both 1.13 and 1.14 still fail existing small cyclic PEPS3D tests. 1.15 passes. |
| Autoray | 0.6 | 0.9 | Tree compression uses `get_namespace`; MPO assembly uses dispatched `astype`. 0.8.0/0.8.10 fail the latter with NumPy 1.26. 0.9's composed dispatch works. |
| Cotengra | 0.8.0 | unchanged | Exact 0.8.0 tested with reusable optimizers and core contraction contracts. |
| tqdm | 4.65 | unchanged | Exact 4.65.0 tested in the core profile. |

The joint core-floor run used NumPy 1.26.0, Quimb 1.15.0, Autoray 0.9.0,
Cotengra 0.8.0, and tqdm 4.65.0: **1370 passed, 319 skipped**, with 2188
deselected, using `smoke or (core and not optional and not slow)`.
Optional packages were absent from this isolated import path; these skips
are not a claim that optional integrations were tested at core-only minimums.
SciPy resolved to 1.16.3 and Numba to 0.67.0; transitive dependencies were
not pinned to their historical minimums.

Failure evidence came from existing tests, without weakening expectations:

- Quimb 1.13/1.14: three cases of
  `test_ps_to_3dpeps_cyclic_small_dimensions_keep_valid_topology` raise
  `IndexError` inside the upstream PEPS3D constructor.
- Autoray 0.8/0.8.10 with NumPy 1.26: MPO assembly/exponential tests fail
  because `do("astype", ...)` seeks a missing NumPy module-level function.

Sources: [NumPy 1.26 release notes](https://numpy.org/doc/1.26/release/1.26.0-notes.html),
[Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray 0.9 distribution](https://pypi.org/project/autoray/0.9.0/).
Published wheels were compared directly against Pepsy imports and call sites;
API-name presence alone was insufficient to establish compatibility.

## Optional requirements

| Dependency | Previous floor | Supported floor | Evidence / scope |
| --- | --- | --- | --- |
| Torch | 2.0 | 2.4 | 2.2 supports Python 3.12 but its wheel fails NumPy 2 conversion. 2.4 passes conversion with NumPy 1.26 and 2.5 plus 60 backend tests; one unavailable-device test skipped. |
| SciPy | 1.10 | 1.16 | Matches Quimb 1.15's dependency floor. SciPy 1.10 also explicitly excludes Python 3.12. The core-floor run resolved 1.16.3. |
| NLopt | 2.7 | 2.9 | 2.8 lacks `runtime_error`, used in Pepsy's optimizer-recovery contract. 2.9 passes seven NLopt solver/recovery tests. |
| Matplotlib | 3.7 | 3.9 | 3.8.0's wheel fails with NumPy 2; 3.9 renders an Agg PNG and passes eight Pepsy drawing tests with NumPy 2.5.3. |
| Nevergrad | 1.0 | 1.0.3 | 1.0–1.0.2 use removed `np.float_` during import. 1.0.3 passes a real Pepsy hybrid layout-search test with NumPy 2. |
| mpi4py | 3 | 3.1.5 | Upstream explicitly added Python 3.12 support here. This exact source-build minimum was not locally exercised. |
| Stim | 1.12 | 1.13 | Python 3.12 wheels; 14 stabilizer-sampler tests pass with 1.13.0, plus a tableau-to-NumPy probe with NumPy 2. |
| NetKet | 3.10 | 3.22 | Older declared versions lack stable fermionic/SR namespaces. 3.21 still fails to import with allowed JAX 0.11.0 (`concrete_or_error` removed). 3.22 supports the tested JAX interval. |
| JAX | 0.4 | 0.7 | Aligns with NetKet's requirement; the existing `<0.11.1` bound remains. |
| Flax | 0.8 | 0.10.6 | Aligns with NetKet 3.22's requirement. |
| Optax | 0.2 | 0.2.2 | Aligns with NetKet's requirement. |
| Autograd (test extra) | 1.6 | 1.7 | 1.6 fails to import on this Mac (`np.float128` absent), even with its `future` dependency supplied. 1.7 passes a gradient probe and Pepsy's BP autodiff integration test. |

The exact NetKet lower-bound combination (3.22.0 / JAX 0.7.0 / Flax 0.10.6 /
Optax 0.2.2) passed **57 tests** in `test_netket_flat_z2.py` and
`test_vmc_api.py`, using released tensor dependencies. Other transitives
resolved normally. NetKet 3.20 also passed these tests with the installed
JAX 0.8.2, but that alone did not justify allowing it with newer JAX.
The earlier repair separately validated NetKet 3.22.4 / JAX 0.11.0 in the full
suite; JAX 0.11.1/0.11.2 reproduce the NetKet annotation import failure.
Torch 2.4 also passed 49 VMC API and batching/compile tests with released
Quimb and Symmray; only the existing fallback-performance warnings appeared.

Sources: [mpi4py changes](https://mpi4py.readthedocs.io/en/stable/changes.html#release-3-1-5-2023-10-04),
[NetKet 3.22 metadata](https://pypi.org/project/netket/3.22.0/),
[Torch 2.4 wheels](https://pypi.org/project/torch/2.4.0/),
[Matplotlib 3.9 wheels](https://pypi.org/project/matplotlib/3.9.0/),
[Nevergrad 1.0.3](https://pypi.org/project/nevergrad/1.0.3/),
[NLopt 2.9](https://pypi.org/project/nlopt/2.9.0/),
[Stim 1.13](https://pypi.org/project/stim/1.13.0/).
These are supported floors validated for the described cases, not claims
that every earlier patch or every optional-version cross-product was tested.

## Requirements retained and audit limits

- Symmray 0.4.0 remains the floor. Released 0.4.0 was used in the NetKet and
  native Torch backend checks. Its newer capabilities retain existing guards.
- CMA-ES 0.10.0 completed an ask/tell iteration. Cotengrust remains optional
  at 0.1, with tested Cotengra fallback contracts. Exact Cotengrust 0.1 could
  not be installed on this Mac without a Rust toolchain; no API evidence
  justified raising its floor. This source-build requirement differs from
  a demonstrated unsupported Python/API version.
- Guppy remains `>=1.0,<2`; the adapter consumes the HUGR interface and no
  newly required API was identified. Exact 1.0.0 was not rerun here.
- Build, development, and documentation tool requirements are unchanged.
  The audit found no newly required tool API; it did not run every historical
  version of those tools.
- Upstream checks included Quimb, Autoray, Cotengra, and Symmray. Symmray's
  array documentation was unavailable; official source and released wheels
  supplied the compatibility evidence.

## Ongoing validation

The existing core CI job now generates exact constraints from the five
`project.dependencies` lower bounds and runs its existing contract selection.
The extended job continues resolving current compatible releases. Both run
`pip check`. This adds no test cases, dependencies, or full-suite jobs.
Optional floors have the focused evidence above; they are not all pinned in
CI. Preserve capability checks for features beyond these baseline contracts.
