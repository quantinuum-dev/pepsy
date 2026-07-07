# `pepsy.optimizers.sym_dmrg`

`SymDMRG2` is the public entry point for the symmetric two-site DMRG path.
Ordinary quimb MPOs are delegated directly to `quimb.tensor.DMRG2`.

The main solve controls intentionally mirror quimb's `DMRG2`: `bond_dims`,
`cutoffs`, `sweep_sequence`, `max_sweeps`, `verbosity`, and
`suppress_warnings` are accepted by `solve`, and `sweep(direction, ...)`
accepts `"R"`/`"L"` or `"right"`/`"left"`. Pepsy's older `chi`, `cutoff`, and
`sweeps` names remain accepted as aliases. `min_sweeps` controls the Symmray
convergence gate and defaults to one comparison sweep before convergence can be
accepted. `solve` keeps the Pepsy optimizer convention of returning `self`; the
quimb-style convergence boolean is stored on `converged`. With `verbosity > 0`,
Symmray sweeps use quimb's `progbar`
helper for the per-site progress display and print quimb-style pre/post sweep
energy lines. For Pepsy-style call sites, `progbar=True` is accepted by
`solve` and `sweep` as a clear alias for at least `verbosity=1`; conflicting
`progbar=False, verbosity>0` settings are rejected.

For scale studies, `profile=True` enables JSON-friendly profiling events in
`profile_diagnostics`. `profile_summary()` aggregates counts and elapsed time
by phase, including canonicalization, environment setup/update, norm checks,
local eigensolves, `H_eff` matvecs, SVD splits, enrichment, sweeps, and solves.
The development harness `benchmarks/symdmrg2_fh_u1u1.py` runs deterministic
chain and mapped-square-lattice Fermi-Hubbard U1U1 cases and emits this
profiling data as JSON. By default the benchmark skips the startup
`initial_energy` estimate and samples local residual and matvec diagnostics,
so runs spend their extra correctness and profiling budget on the two-site
solves being timed. Use `--periodic`/`--pbc` with `--lattice-shape LX LY` to
encode periodic square-lattice edges as long-range OBC-MPO terms.

For Symmray Fermi-Hubbard MPOs, Pepsy assumes an OBC MPS/MPO chain and a
bosonic/Jordan-Wigner Symmray MPO whenever the input MPS uses fermionic
Symmray arrays that need bosonization; fermionic Symmray tensors in that MPO
are rejected before the state is bosonized. Periodic lattice edges should be
encoded as long-range terms in that OBC MPO, not as a cyclic MPS. Pepsy builds
dense left/right environments for `<psi|MPO|psi>`, block-sparse environments
for the projected `H_eff`, and dense debug environments for `N_eff`. Before
Symmray sweeps that are not already using explicit sector enrichment or the
mixer's sector expansion, SymDMRG2 widens additive U(1)-style MPS virtual
charge maps with the minimal zero-valued sectors reachable from both the left
prefix and the right suffix. Local dense/Lanczos solves then use a two-site
variational template whose active physical legs include every charge-compatible
local sector, so the subsequent SVD can nucleate bond sectors that were absent
from a narrow product or low-bond initial state. The default
`variational_sector_basis="adaptive"` reopens those legal zero sectors whenever
later sweeps need them. `variational_sector_basis="bond_dim"` reopens them only
when the requested sweep bond dimension increases, allowing equal-chi
turnarounds to reuse maintained environments but narrowing the later search
space. By default, Symmray `H_eff` matvecs use a block-native projected contraction;
`matvec_backend="dense_reference"` keeps the older
NumPy dense-aligned matvec available as a validator. During a local
dense/Lanczos solve, the block-native path caches the static projected
problem for the active two-site window, including reindexed MPO tensors and
left/right environment projectors. The cached projected problem also
precomputes the static block-contraction routing used by repeated Lanczos
matvecs. `profile_summary()` reports projected problem cache hits and misses
so scale runs can confirm the hot matvec path is reusing this setup work.
`matvec_layout="fused"` is available as an opt-in prototype for the
block-native path. It attempts to fuse multiple shared contraction legs inside
each cached projected problem, using Symmray's fused-index support when the
resulting charge maps are compatible. When the fused total-charge index would
collapse distinct charge-combination slots, the contraction falls back to the
exact unfused blockwise route and records candidate, attempt, and fallback
counts in matvec diagnostics; incompatible fused layouts are cached per
projected problem so repeated Lanczos matvecs do not keep retrying the same
failed fused contraction. This keeps the switch safe for benchmarking while
leaving the default `matvec_layout="unfused"` unchanged.
Default Symmray Lanczos solves keep Krylov vectors as block tensors and use
dense NumPy only for block dot products and the small Rayleigh-Ritz projected
matrix, avoiding a flat-vector Symmray-to-NumPy-to-Symmray round trip for every
`H_eff` application. The native block path stops when the residual estimate
meets `local_eig_tol`, when the Krylov basis reaches `local_eig_ncv`, or when
the TeNPy-style Ritz convergence gate passes. The Ritz gate uses the
Ritz-energy change (`local_eig_energy_tol`) together with the state-error
estimate `(RitzRes / gap)**2` (`local_eig_p_tol`, with gap floor
`local_eig_min_gap`). Set either tolerance to `None` to disable that side of
the gate. The defaults use `local_eig_energy_tol=inf`,
`local_eig_p_tol="auto"` (resolved to `1e-8`), `local_eig_min_steps=2`, and
`local_eig_ncv=20`, so a local solve may stop early only when the Ritz vector
is well resolved by the residual/gap criterion. For compatibility,
`local_eig_p_tol="auto"` resolves to `None` when
`local_eig_energy_tol=None`, preserving the explicit full-Krylov-cap mode.
When `local_eig_p_tol` is automatic and active, the default
`local_eig_p_tol_to_trunc="auto"` resolves to TeNPy's `0.05` coupling: after
each sweep, SymDMRG2 sets the next sweep's P-error tolerance from
`max_truncation_error * local_eig_p_tol_to_trunc`, clamped by
`local_eig_p_tol_min` and `local_eig_p_tol_max` (default `1e-4`). If
`local_eig_p_tol_min` is omitted, the floor is derived from the active SVD
cutoff as `max(1e-30, cutoff**2 * factor)`. Explicit fixed
`local_eig_p_tol` values stay fixed unless `local_eig_p_tol_to_trunc` is set
explicitly. Update records are exposed in `summary()` and the per-sweep
convergence diagnostic.
`local_eig_ncv` may also be a sweep schedule whose last entry repeats, matching
`bond_dims` and `cutoffs`. Local solve diagnostics record `stop_reason`,
`ritz_energy_delta`, `ritz_gap`, `ritz_p_error`, `num_steps`, and
`num_matvecs`. Real block data remains real unless the state or MPO data is
complex. Before each local solve,
SymDMRG2 also probes the projected
block-native `H_eff` support and drops widened zero blocks that are neither
structurally live nor already populated by the current MPS.
Symmray sweeps canonicalize the MPS center before using H-only dense/Lanczos
solves; a
non-identity effective norm is treated as a canonicalization/alignment error
unless the explicit diagnostic `local_solver="generalized_dense"` mode is
requested. `local_solver="auto"` now defaults to the matrix-free Lanczos path
for Symmray local solves; set `dense_threshold` above zero or request
`local_solver="dense"` only when deliberately building dense reference local
Hamiltonians. If a requested state canonicalization method is unavailable,
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
time, cache-hit status, theta block size, projector block counts, cached
contraction routing counts, and block-native substep timings in
`matvec_diagnostic_records`. The cached projected problem also records whether
the block-native path used the original right-first contraction route or the
left-first route selected for strongly imbalanced projector sizes.
`compute_initial_energy=True` preserves the historical eager startup estimate,
`False` skips it, and `"lazy"` defers it until the `initial_energy` or
pre-sweep `energy` property is requested. Sweep setup
builds only the static side environments needed for the current direction,
updates the moving side incrementally after each two-site writeback, and reuses
that maintained side on direction turnarounds. It rebuilds only when a pre-sweep
sector-layout mutation, a same-direction sweep, or an increased bond-dimension
sector-basis pass invalidates the cached side.
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
sampled projected matvec cost metadata, `variational_sector_diagnostics` for
automatic zero-valued sector-basis widening, and `local_solve_diagnostics` for
the resolved dense/Lanczos solver, theta-space dimension, local energy,
residual status, and matvec backend used at each two-site solve. The automatic
diagnostic records `map_source="prefix_closure"` when this minimal path is
available, and falls back to `map_source="template"` for unsupported symmetry
types. For especially fragile narrow initial MPS sector layouts,
`sector_enrichment="template"` can also expand virtual-bond charge maps from a
same-charge random template MPS and seed newly valid blocks with `sector_noise`
before the first sweep.
`sector_enrichment="adaptive"` repeats that template enrichment before every
sweep, which can reintroduce valid sectors after SVD truncation prunes them.
`sector_enrichment_diagnostics` records the added blocks and template bond
sectors.

For a genuinely random but well-conditioned Symmray initial MPS, use
`SymMPS.random_unitary_evolution(...)` or
`SymMPS.random_unitary_for_model(...)`. These constructors grow a
charge-correct product state with charge-preserving two-site random unitaries
and canonicalize the result before DMRG. Raw `SymMPS.random(bond_dim=chi)`
still exists, but it is a lower-level block-fill constructor and should be
treated as a robustness stress test on hard mapped-2D PBC cases.

When the optional subspace mixer is active, SymDMRG2 uses it as a temporary
exploration aid rather than a final convergence state. If the sweep convergence
criteria are met while the mixer was still active, the Symmray path records a
`mixer_lifecycle` diagnostic, disables the mixer, and requires a subsequent
no-mixer sweep before setting `converged=True`. This matches the TeNPy practice
of turning the mixer off and doing final clean sweeps once the state has found
the right Schmidt subspace.

For a hard periodic benchmark, the 3 by 3 PBC U1U1 Fermi-Hubbard sector-ED
reference currently used during development is
`E0 = -7.824105712954`. This case is intentionally not a normal unit test:
with `chi=128` and no mixer it can stall around `8.7e-3` above ED, while the
subspace mixer with `chi` up to the exact Schmidt bound of `256` reaches the
sector ED value to about `6e-12`. A representative command is:

```sh
python benchmarks/symdmrg2_fh_u1u1.py \
  --lattice-shape 3 3 --periodic \
  --chi 256 --initial-bond-dim 2 --sweeps 4 --sweep-sequence RL \
  --local-solver lanczos --dense-threshold 0 --local-eig-ncv 16 \
  --mixer subspace --mixer-bond-dim 256 --mixer-amplitude 1e-4 \
  --norm-check sampled --norm-check-interval 2 \
  --residual-check sampled --residual-check-interval 2 \
  --matvec-diagnostics sampled --matvec-diagnostics-interval 4 \
  --expected-energy -7.824105712954 --expected-energy-tol 1e-10 \
  --exact-schmidt-bound 256
```

```{eval-rst}
.. automodule:: pepsy.optimizers.sym_dmrg
   :members:
   :undoc-members:
   :show-inheritance:
```
