# 2026-10-05 — Clarify MPS precision defaults

- Scope: user requested complex128 as the default initialization precision and
  automatic cutoffs 1e-12/1e-6 for complex128/complex64 respectively.
- Branch / baseline: `develop` / `bec773a`, with existing working-tree changes.
- Status: documentation only for this task; nothing staged, committed or pushed.

Verified that `ps_to_mps` already defaults to complex128, `MpsOptimizer.run`
already defaults to cutoff="auto", and the shared cutoff resolver already has
the requested mapping. The optimizer inherits a supplied state's dtype; no
implicit conversion of explicitly selected complex64 arrays was introduced.
Updated the [MPS API guide](../docs/api/optimizers/mps.md) with the exact table,
complex128 GPU examples, and explicit complex64 opt-in guidance.

Checked the constructor/default signature, six NumPy/Torch CUDA/CuPy dtype
combinations, and numeric cutoff overrides in the selected environment. All
passed. Relative API links and `git diff --check` passed. No numerical suite
was needed for this documentation-only clarification.
