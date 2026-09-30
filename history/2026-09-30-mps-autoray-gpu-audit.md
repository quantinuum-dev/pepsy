# 2026-09-30 — Autoray and DMRG GPU audit

- Scope: user requested a careful assessment of Autoray for GPU consistency
  in MPS `dmrg`, `dmrg2`, `dmrg3`.
- Branch / baseline: `develop` / `f01f596`; reviewed current working tree.
- Status: audit note and this handoff only; runtime/defaults/dependencies
  unchanged by this task. Nothing staged, committed or published. Existing
  edits, including concurrent unrelated gate/test changes, were preserved.

See [findings and measured evidence](../docs/development/notes/2026-09-30-mps-autoray-gpu-audit.md).
Pepsy already adopts Autoray's principal device/dtype/RNG facilities.
The concrete new finding is JAX GPU arithmetic precision: the default failed
7/8 existing backend tests; `JAX_DEFAULT_MATMUL_PRECISION=highest` passed 8/8.
All-mode highest-precision JAX probes and CPU/CUDA Torch probes passed their
exact-state comparisons. CUDA-only existing regressions passed 8/8.

Recommended explicit JAX precision configuration; not installed as a new
package default. CuPy performance parity and new Autoray conversion APIs
remain proposed work, not authorized follow-on implementation.
No full-suite, performance, multi-GPU, native Symmray GPU, or autodiff claim.
Documentation links checked locally and `git diff --check` run at handoff.
