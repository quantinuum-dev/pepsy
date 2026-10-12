# 2026-09-26 — PepsSampler inferred and explicit array backends

Status: implemented and validated in the working tree on develop / 80f451a.
This supersedes the implementation status in the earlier
[Torch/Autoray study](peps_sampler_torch_audit.md), which remains a record of
the original behavior.

## Implemented behavior

- PepsSampler accepts a callable to_backend. With no override it infers the
  dense array backend, dtype, and device from the PEPS using Pepsy's shared
  conversion helpers. An override converts private arrays; multiplication by
  one isolates mutable inputs while preserving Torch autograd history.
- refresh repeats an explicit converter or re-infers the current source.
  Inconsistent tensor backends/devices/dtypes fail with a conversion hint;
  native Symmray remains unsupported.
- Density matrices stay as backend arrays. Autoray dispatches diagonal
  extraction, validation, diagnostics, clipping, probability normalization,
  categorical draws, and proposal bookkeeping to the tensor backend.
- Identity future caps now use the source dtype/backend/device. This fixes
  the reproduced Torch mixed-array failure with marginal_chi=None/0.
- Negative-diagonal tolerance uses the real dtype's epsilon and local scale.
  Substantially negative diagonals, non-finite matrices, and invalid traces
  still raise. Small positive unnormalized rhos are accepted.
- An explicitly evaluated zero-probability branch returns zero immediately,
  avoiding further contractions of its zero conditional network.
- Final amplitudes are evaluated once per distinct final prefix group.
  Returned result shapes and mantissa/exponent conventions are preserved.

Implementation: [samplers.py](../../../src/pepsy/sampling/samplers.py).
Public behavior/examples: [sampler API](../../api/sampling/samplers.md).

## Dependency audit and decisions

Reused the unchanged upstream/environment audit from the linked study:
Autoray 0.11.1.dev3+g1b476b305, Quimb 1.15.1.dev66+ge927f06e1,
Torch 2.6.0+cu124, JAX 0.10.2, Cotengra 0.8.3.dev7+g1d7fd333f,
Cotengrust 0.2.1, Symmray 0.4.1.dev7+g83fb22865.

- **Adopt:** public Autoray get_namespace(array), array operations, and
  namespace.random.default_rng(seed).choice. No library upgrades, monkey
  patches, or installed-library edits.
- **Compatibility shim:** a scoped JAX default_device context around identity
  creation and random key/draw construction. The installed namespace infers
  CpuDevice(1), but eye does not inject its device argument; the installed
  JAX RNG similarly creates its key using the process default. This failed
  with two CPU devices before the context fix. The context applies only to
  JAX and restores the caller's setting; the dedicated non-default-device
  regression passes.
- **Defer:** tensor batch axes, fully compiled sampling loops, Symmray PEPS,
  distributed/sharded JAX arrays, and large-system performance claims.

Official API references inspected:
[Autoray namespace](https://autoray.readthedocs.io/en/latest/autoapi/autoray/index.html#autoray.get_namespace),
[JAX default_device](https://docs.jax.dev/en/latest/_autosummary/jax.default_device.html).
Installed signatures and dispatch were also inspected directly. Minimum
dependency-version testing was not performed in this shared environment.

## New validation

All commands activated the existing py312 development environment and used
the checkout's src directory on PYTHONPATH. JAX tests used JAX_PLATFORMS=cpu;
BLAS/OpenMP thread counts were one.

1. Full PEPS sampler selection with
   XLA_FLAGS=--xla_force_host_platform_device_count=2:
   python -m pytest -q -o addopts='' tests/test_peps_sampler.py
   → **35 passed**. Covers NumPy/Torch/JAX, exact/identity-future/Quimb-future/
   DMRG-FIT proposals, dense amplitudes, reproducibility, dtype/device,
   source preservation, explicit conversion/refresh, non-leaf Torch gradients,
   dtype-sensitive rho validation, duplicate amplitudes, and JAX placement.
2. Broader sampler/public API/package layout selection before the final
   JAX identity-context adjustment:
   python -m pytest -q -o addopts='' tests/test_peps_sampler.py tests/test_sampler.py tests/test_public_api.py tests/test_package_layout.py
   → **195 passed, 1 failed**. The final PEPS-only rerun above includes the
   device-context fix and its additional regression.
3. Final repository smoke:
   python -m pytest -q
   → **152 passed, 1 failed**.
4. Both broader selections fail only
   tests/test_package_layout.py::test_package_version_matches_installed_distribution:
   installed metadata/runtime report 0.4.0; the unchanged pyproject declares
   0.5.0. No shared environment/package metadata was modified for this task.
5. python -m ruff check src tests and git diff --check pass.

### Torch CPU/CUDA reproduction matrix

Re-ran the original 2x3 D=2 seed-83 audit matrix, four batch shots seed 11:
exact; Quimb future with marginal_chi=8; DMRG/FIT future with marginal_chi=8;
and Quimb identity future with marginal_chi=0. Conditioned sample_chi=4,
one FIT sweep, greedy contraction optimizer.

**12/12 passed:** Torch CPU complex128 and cuda:0 complex128/complex64
across all four paths. Inspected centers, future environments, and
conditioned boundaries preserve dtype/device. Source tensors are unchanged.
Maximum amplitude errors against independent dense contractions:
2.45e-14 for CUDA complex128 and 7.69e-6 for CUDA complex64 on this
unnormalized random state.

Explicit Autoray-to-NumPy calls during sampling now contain integer choice
arrays only; no full local-rho or probability-vector transfers were observed.
Scalar .item reads and upstream compression synchronization remain.
Temporary evidence: /tmp/peps_sampler_backend_cuda.py, .json, .log;
pytest logs /tmp/peps_sampler_validation.log,
/tmp/peps_sampler_backend_final.log, and /tmp/peps_sampler_smoke.log.

## Remaining limits

Python handles Quimb projections and prefix groups; results remain host lists.
Each conditional validates through a scalar synchronization, diagnostics move
to host on access, and compression routines may also read device scalars.
This evidence does not establish a production speedup or absence of all host
synchronization. Seeds are reproducible within a backend/device/method, not
across different RNG implementations. JAX GPU and distributed arrays were
not tested; JAX CPU placement was checked on two devices.

No commits, staging, publication, dependency changes, or production-job
changes were made.
