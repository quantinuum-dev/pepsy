# 2026-10-05 — Implement Autoray MPS Kraus batching

- Scope: user-authorized GPU trajectory optimization with consistent Autoray
  dispatch for Torch and CuPy.
- Branch / baseline: `develop` / `bec773a`, plus prior uncommitted corrections.
- Commit status: working-tree implementation/tests/docs only; nothing staged,
  committed or published. Existing unrelated work preserved.

Implemented one shared amplitude block per dense Kraus channel, bounded
on-backend outcome batches and cached converted operators. Stable reductions
retain tested complex64 probabilities of 1e-60, and only the outcome norm vector
and existing base scalar reach the host. Native/symmetric and other optimizer
routes retain their existing behavior. Added 38 regressions and API/changelog
documentation. See the [implementation audit](../docs/development/notes/2026-10-05-mps-kraus-autoray-batching.md).

Measured nonadjacent probability speedup about 3.8x on both Torch CUDA and CuPy;
eight-shot independent nonadjacent-channel replay improved 1.24x/1.22x.
One-site CuPy amplitude-damping replay was essentially unchanged. These are
synchronized implementation measurements, not earlier prototype numbers.

Validation: 411 focused passes, one second-JAX-device skip, 30 slow deselections;
54 API/layout passes. Separate CUDA/CuPy checks: 16 passes and the same two
known infidelity-comparison failures. Ruff, documentation links and whitespace
checks pass. No full-suite claim. Template reuse/worker-policy changes remain
deferred; no additional permission step is introduced by this handoff.
