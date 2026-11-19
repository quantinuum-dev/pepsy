# 2026-11-19 — Exact-reference 2D DMT dynamics

- Scope: user requested long-time physical 2D Hamiltonian accuracy, chi
  convergence, positivity and comparison with variational compression.
  Their preceding DMRG-style question was conceptual; no new constrained
  variational solver was authorized or implemented by this benchmark.
- Branch / baseline: `develop`, `f5d31006d02821028c890a04b17e4b15b9683d0b`.
- Commit status: uncommitted; extensive preceding DMT/MPS work preserved.

## Changes and rationale

Added [the 2D benchmark](../examples/dmt_2d_dynamics.py) and focused tests.
All physical lattice edges enter the Ising Hamiltonian; fields are shared
over incident edges without double counting. Exact continuous-time and
identical-circuit references separate timestep error from compression.
Compare existing ordinary DMT and unconstrained two-site FIT, at equal
chi and zero target/output cutoffs. No production solver/default changed.

Audit current gated targets at three timesteps; keep raw trace, energy,
local/global errors and Hermitian-part spectra. FIT's live local per-gate
state differs from `self.p`, so its oracle owns preceding physical cores.
No trace/PSD repair hides unphysical trajectories. Short Torch checks use
native Torch gates and states in complex128.

## Evidence and validation

The [scientific record](../docs/development/notes/2026-11-19-dmt-2d-dynamics.md)
owns completed runs, raw artifacts and limitations. Exact six-qubit chi=64
controls recover the uncompressed circuit through t=10. Truncated DMT
preserves gated plaquettes near roundoff while accumulating large trajectory
errors and negative global eigenvalues. Finite-size evidence does not
establish native PEPO behavior or universal thermodynamic convergence.

Completed 27 trajectory rows: 26 completed, χ=20 strict-budget rejection
retained. Four NumPy families run through t=10; Torch checks run through t=2.
The nine-qubit χ=256 DMT control agrees with its exact circuit to 1.41e-13
relative HS error at t=10. At χ=40/64/128, sampled mean plaquette errors are
0.18090/0.14707/0.09796 for DMT versus 0.23824/0.20757/0.13994 for ordinary
FIT; endpoint errors need not be monotonic. Global physicality fails at the
truncated caps despite local preservation. Raw JSON/CSV and four PNG/SVG
pairs are linked by the scientific record. FIT trace-normalized energy is
a separate readout diagnostic, not an alteration to the evolved operator.

Focused benchmark + DMT refinement + MPO replay + Pauli lattice selection:
**161 passed**, with one existing rank-policy warning. After a readout-only
energy-contraction optimization: **12 focused tests passed**. Ruff, DMT skill
validation and the 13-skill catalog pass. No full-suite or GPU claim.
Installed versions/signatures match the preceding active-task audit.

Unconstrained FIT can be backend-sensitive at a near-degenerate cutoff;
retain the probe evidence and qualify agreement. No tolerances were weakened
and no dependency/environment changes were made.
