# `pepsy.optimizers.sym_dmrg`

`SymDMRG2` is the public entry point for the symmetric two-site DMRG path.
Ordinary quimb MPOs are delegated directly to `quimb.tensor.DMRG2`.

The main solve controls intentionally mirror quimb's `DMRG2`: `bond_dims`,
`cutoffs`, `sweep_sequence`, `max_sweeps`, `verbosity`, and
`suppress_warnings` are accepted by `solve`, and `sweep(direction, ...)`
accepts `"R"`/`"L"` or `"right"`/`"left"`. Pepsy's older `chi`, `cutoff`, and
`sweeps` names remain accepted as aliases. `solve` keeps the Pepsy optimizer
convention of returning `self`; the quimb-style convergence boolean is stored
on `converged`. With `verbosity > 0`, Symmray sweeps use quimb's `progbar`
helper for the per-site progress display and print quimb-style pre/post sweep
energy lines.

For scale studies, `profile=True` enables JSON-friendly profiling events in
`profile_diagnostics`. `profile_summary()` aggregates counts and elapsed time
by phase, including canonicalization, environment setup/update, norm checks,
local eigensolves, `H_eff` matvecs, SVD splits, enrichment, sweeps, and solves.
The development harness `benchmarks/symdmrg2_fh_u1u1.py` runs deterministic
open-chain Fermi-Hubbard U1U1 cases and emits this profiling data as JSON. By
default the benchmark skips the startup `initial_energy` estimate and samples
local residual and matvec diagnostics, so runs spend their extra correctness
and profiling budget on the two-site solves being timed.

For Symmray Fermi-Hubbard MPOs, Pepsy assumes an OBC MPS/MPO chain and a
bosonic/Jordan-Wigner Symmray MPO whenever the input MPS uses fermionic
Symmray arrays that need bosonization. Periodic lattice edges should be
encoded as long-range terms in that OBC MPO, not as a cyclic MPS. Pepsy builds
dense left/right environments for `<psi|MPO|psi>`, block-sparse environments
for the projected `H_eff`, and dense debug environments for `N_eff`. The
active local basis is exactly the current `theta` block layout. By default,
Symmray `H_eff` matvecs use a block-native projected contraction;
`matvec_backend="dense_reference"` keeps the older
NumPy dense-aligned matvec available as a validator. During a local
dense/Lanczos solve, the block-native path caches the static projected
problem for the active two-site window, including reindexed MPO tensors and
left/right environment projectors. The cached projected problem also
precomputes the static block-contraction routing used by repeated Lanczos
matvecs. `profile_summary()` reports projected problem cache hits and misses
so scale runs can confirm the hot matvec path is reusing this setup work.
Symmray sweeps
canonicalize the MPS center before using H-only dense/Lanczos solves; a
non-identity effective norm is treated as a canonicalization/alignment error
unless the explicit diagnostic `local_solver="generalized_dense"` mode is
requested. If a requested state canonicalization method is unavailable,
SymDMRG2 forces `N_eff ~= I` checks for that H-only sweep even when
`norm_check="off"`, so the wrong metric cannot be used silently. The default
`norm_check="strict"` validates every two-site window.
For larger trusted runs, `norm_check="sampled"` checks boundary windows plus
every `norm_check_interval`-th interior window, `norm_check="first_sweep"`
checks only the first sweep, and `norm_check="off"` skips this expensive
debug assertion. Skipped checks are still recorded in
`norm_identity_diagnostics` with `skipped=True` so benchmark logs remain
auditable. `residual_check` accepts the same schedule modes and records
normalized local eigensolver residuals in `residual_diagnostics` without
raising by default; `residual_check_tol` marks diagnostics as passed/failed
when supplied. `matvec_diagnostics` records sampled `H_eff` matvec elapsed
time, cache-hit status, theta block size, projector block counts, and cached
contraction routing counts in `matvec_diagnostic_records`.
`compute_initial_energy=True` preserves the historical eager startup estimate,
`False` skips it, and `"lazy"` defers it until the `initial_energy` or
pre-sweep `energy` property is requested. Sweep setup
builds only the static side environments needed for the current direction,
updates the moving side incrementally after each two-site writeback, and
reuses the completed norm environments for normalized energies.
Every Symmray two-site writeback records an entry in
`svd_diagnostics`, including the split direction, bond name, `chi`, cutoff,
truncation error when reported by the split backend, and the left/right charge
sectors kept by Symmray's SVD. `compression_summary()` aggregates these split
records into maximum kept bond dimensions and maximum/summed truncation error,
matching the kind of compression health metrics used by mature MPS workflows.
`last_svd_diagnostic` exposes the most recent entry. The Symmray path also
records `norm_identity_diagnostics` for the per-window `N_eff ~= I` canonical
check, `residual_diagnostics` for scheduled `H theta - E theta` (or
generalized `H theta - E N theta`) residuals, `matvec_diagnostic_records` for
sampled projected matvec cost metadata, and `local_solve_diagnostics` for the
resolved dense/Lanczos solver, theta-space dimension, local energy, residual
status, and matvec backend used at each two-site solve. For narrow initial MPS
sector layouts, `sector_enrichment="template"`
can expand virtual-bond charge maps from a same-charge random template MPS and
seed newly valid blocks with `sector_noise` before the first sweep.
`sector_enrichment="adaptive"` repeats that template enrichment before every
sweep, which can reintroduce valid sectors after SVD truncation prunes them.
`sector_enrichment_diagnostics` records the added blocks and template bond
sectors.

```{eval-rst}
.. automodule:: pepsy.optimizers.sym_dmrg
   :members:
   :undoc-members:
   :show-inheritance:
```
