# 2026-10-01 — Fix metric precedence and final gate compression

- Scope: user requested fixing the two findings in the
  [extended review](2026-10-01-extended-peps-review.md).
- Branch / baseline: `develop` / `7773d2e`, including the earlier uncommitted
  [initial normalization-cap fix](2026-10-01-sweep-initial-normalization-cap.md).
- Commit status: working-tree edits only; no staging, commit or push.

## Changes

- One resolver selects normalization/evaluation caps for metric calls, run
  records and delegated defaults. Per-call mapping chi wins, then named
  per-call cap, named constructor cap, stored metric mapping chi, shared
  boundary mapping chi and the automatic pair. Explicit backend-specific
  normalization mappings remain overrides.
- Exact targets disable inherited final gate chi and reject explicit target
  caps, keeping the untruncated target independent of output compression.
- Final gate compression supplies the canonical form only to 1D networks,
  leaving PEPS/PEPO compression on its dimensional API. Backend conversion,
  gradients, cutoff policies and native sectors remain in existing routines.
- Added deterministic routing/record, independent Schmidt reconstruction,
  NumPy/Torch compression and native fermionic overlap regressions. Updated
  owning API guides and changelog. The prior constructor fix is preserved.

Upstream sources, installed versions/signatures and numerical limits are in
the [compatibility note](../docs/development/notes/2026-10-01-metric-gate-compatibility.md).

## Fresh validation

All Python commands activated the existing `envs/py312` environment.

- Initial affected selection: **534 passed**, no skips, 22.08 seconds. This
  preceded the native regression and final override check.
- Added native fermionic operator regression: **1 passed**.
- A subsequent affected rerun: **533 passed, 2 failed** from CuPy allocation
  failure and JAX CUDA initialization failure under GPU memory pressure.
  Failing tests were `test_apply_gate_1d_infers_cupy_backend_from_network` and
  `test_bdymps_build_to_jax_casts_complex_numpy_input_to_real`. No tests or
  numerical tolerances were weakened.
- Final CPU selection with `CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q -ra
  -o addopts='' tests/test_optimize_peps.py tests/test_gate.py
  tests/test_gate_cutoff.py tests/test_optimize_global.py
  tests/test_prepare_boundary_inputs.py tests/test_public_api.py
  tests/test_package_layout.py`: **535 passed, 1 skipped**, 17 warnings,
  21.53 seconds. Skip is the CuPy CUDA check. Includes the added native
  regression and backend normalization override case.
- Five real sweep constructor/override cases also passed independently.
- Full-suite attempt was interrupted after backend CUDA memory failures:
  **283 passed, 3 failed, 1 skipped**, 41.14 seconds. Failures were
  `test_cupy_namespace_creation_on_available_device`,
  `test_torch_backend_and_linalg_on_available_devices[cuda]`, and
  `test_jax_single_device_sharding_has_compatible_backend_metadata`.
  CUDA allocation/init errors are retained in
  `/tmp/pepsy-20261001-metric-gate-full-suite.log`. No full-suite success claim.
- Final Ruff, whitespace and relevant local documentation links passed.

## Remaining limits

GPU/full-suite validation needs an available device with enough free memory.
No dependency upgrades, sibling edits, or unrelated TreeSampler fixes.
Existing review documents and the previous fix remain uncommitted and intact.
