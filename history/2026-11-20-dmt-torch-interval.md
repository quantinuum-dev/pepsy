# 2026-11-20 — Public Torch DMT intervals and 6×6 physics

- Scope: finish Torch-first interval integration and 6×6 comparisons in both
  bases; CuPy explicitly deferred, XX† fitting explicitly excluded.
- Branch / baseline: `develop`, `eaedf789`; all preceding uncommitted work
  preserved. No staging, commit or push performed.

Implemented public transactional `apply_transfer_mpo` for `PauliDMT` and
`MpoOptimizer`, reusing canonical workspaces with interval/full controls.
Prepare ordinary oversized inputs exactly before the map; cap afterwards.
Retain update-level warnings without a bond report, strict final panel checks
once, and cheap one-site guards. API docs distinguish physical transfer axes
from the existing gate-stream transpose convention and `.run()` routing.

Added `examples/dmt_torch_lattice.py` and independent public-path tests. The
example evolves an open tilted-field Ising lattice with all physical edges,
plaquette protection and a depolarized tilted product initial state. It stores
sampled panels, raw scalar diagnostics and errors, with bounded CPU runs and
matched-time summary mode. No global 36-spin density/statevector is built.

Torch CPU here also uses Accelerate. A finite captured residual yields NaNs
under native Torch SVD; long native controls also fail. Add an explicit
`TorchLinalgConfig.cpu_qr="scipy_pivoted"` forward-only CPU policy, leaving
native defaults and CUDA selection unchanged. Delegate CPU SVD separately via
existing policy. The pivot permutation is absorbed in R, which is no longer
triangular. Fix subnormal complex QR phase normalization without clipping.
This does not prove a common root cause for all observed Torch failures.

Real-Pauli double precision chi96 reaches t=2 and chi192 reaches t=1 with
trace defects around 3e-10 and no preservation warnings. Matched plaquette
gaps remain large. Complex128 ordinary/Pauli and full/interval controls are
recorded separately; GESDD still fails later in complex Pauli even with
pivoted QR. See the [evidence record](../docs/development/notes/2026-11-20-dmt-torch-interval.md)
and linked JSON for exact endpoints, failures, physical model and results.
A slower complex128 GESVD/pivoted-QR chi96 control reaches t=0.6 without
warnings and stops at its timestep-boundary runtime budget. All 20 recorded
runs have finished; failed and native-fault controls are retained explicitly.

Validation: 425 focused DMT/backend passes, two skips; subsequent public
interval/CPU-QR selection has 26 passes including new one-site warning and
rollback cases. Full suite stops at the known baseline BP namespace failure
(104 passes, seven skips). Public API/layout has 54 passes and known installed
version mismatch. Ruff and whitespace checks pass. CUDA/CuPy, 6×6 timestep
convergence and long-time chi convergence remain unverified.
