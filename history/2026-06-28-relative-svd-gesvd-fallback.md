# 2026-06-28 — Relative torch SVD registration and `gesvd` fallback

- Milestone: M6 — AD decomposition prototype / full-SVD hardening
- Branch / commit: `main` and `develop` / `acf207e`

## What changed
- Added `pepsy.tensors.reg_rel_svd_torch()` as the preferred torch full-SVD
  registration.
- Kept `reg_complex_svd_torch()` and `register_torch_linalg(mode="complex")`
  as compatibility paths to the same relative-regularized rule.
- Documented the backward rule as Townsend rectangular full-SVD terms with
  Lorentzian relative broadening and the complex SVD gauge correction.
- Added a CPU SciPy `gesvd` forward fallback for the relative/complex SVD path
  when `torch.linalg.svd` fails, including batched real and complex inputs.

## Why
- The first registration pass stabilized the backward rule but still used plain
  CPU `torch.linalg.svd` in forward, so ill-conditioned high-chi bonds could
  still crash on `gesdd` failures.
- The real-mode registration already had a forward fallback; the relative and
  complex registration paths now have the same robustness.

## How it was validated
- `pytest -q tests/test_core_seed.py::test_reg_rel_svd_torch_forward_falls_back_to_scipy_gesvd` -> `2 passed`.
- SVD-focused tests in `tests/test_core_seed.py` -> `9 passed`.
- `pytest -q tests/test_public_api.py tests/test_package_layout.py tests/test_core_seed.py` -> `470 passed`.
- `python -m pyflakes src/pepsy/backends/linalg_torch.py src/pepsy/backends/linalg.py tests/test_core_seed.py` -> passed.

## Decisions / findings
- This remains a full-SVD autodiff shim. It is not the true truncated-SVD
  pullback planned in `PLAN.md`.
- For real float64 forward-only runs, `register_torch_linalg(mode="real")`
  remains appropriate. For complex/autodiff full-SVD paths, use
  `reg_rel_svd_torch()` or `register_torch_linalg(mode="complex")`.

## Next step (do this first next time)
- Keep true truncated-SVD AD work separate from this full-SVD robustness layer.
  Add a dedicated split driver or custom VJP before changing Quimb truncation
  behavior.

## Open questions / blockers
- None for the full-SVD fallback. True truncated-SVD AD remains open.
