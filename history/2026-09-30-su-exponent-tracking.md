# 2026-09-30 — SU gauge exponent tracking implemented

- Scope: user requested normalization of SU gauges with their scale retained
  in tensor-network exponent metadata, fixing the diagnosed overflow.
- Pepsy branch / baseline: develop / 1b4ca92. Examples: main / 40cadd7.
- Status: uncommitted working-tree edits; nothing staged or published.
- Preserved unrelated package example-removal work and earlier examples edits.

Implemented optional `gate_simple(..., renorm=False, strip_exponent=True)`
using the existing `renorm_gauge` helper after each gate/routed SWAP. The
roughening PEPS engine enables it. Actual scale is preserved; no extra
decomposition or gauge-equilibration sweep is needed. Docs and tests updated.

The previous default 3x3 D=2/4 overflow tests now pass through t=6 on NumPy
and Torch CPU. Earlier valid physical norms agree within 1.5e-14. Exact,
BP, and C=9 norm scale propagation, Torch derivatives, native fermionic
Symmray tensors, and norms beyond floating-point range were checked.

Validation: 175 passed / 1 GPU skip in existing gate/scale/SU tests;
73 passed / 1 preexisting environment-version mismatch in new scale,
BP reference, public API and layout checks; 35 passed in roughening PEPS
tests. Actual 3x3 D=4 CLI measurements and samples at t=4/6 completed and
sample amplitudes matched exact contractions. Ruff and diff checks pass.

See [implementation, audit, results and limitations](../docs/development/notes/2026-09-30-su-exponent-tracking.md).
No shared-environment upgrade, full repository suite, GPU production run,
or SimpleUpdateGen gate-option allowlist extension was performed.
