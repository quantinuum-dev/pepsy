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
Applications can serialize this profiling summary directly for their own
benchmark harnesses. By default, production callers can skip the startup
`initial_energy` estimate, skips per-window dense norm identity checks after
successful canonicalization, and samples local residual and matvec
diagnostics, so runs spend their extra correctness and profiling budget on the
two-site solves being timed. Use `--norm-check strict` when profiling the full
dense `N_eff` debug guard. Use periodic square-lattice edges as long-range
OBC-MPO terms rather than cyclic MPS bonds.

For Symmray Fermi-Hubbard MPOs, Pepsy assumes an OBC MPS/MPO chain and a
bosonic/Jordan-Wigner Symmray MPO whenever the input MPS uses fermionic
Symmray arrays that need bosonization; fermionic Symmray tensors in that MPO
are rejected before the state is bosonized. Periodic lattice edges should be
encoded as long-range terms in that OBC MPO, not as a cyclic MPS.

`SymDMRG2` optimizes in this bosonic Jordan-Wigner representation, so its
`state` holds ordinary abelian Symmray arrays. `SymDMRG2.fermionic_state()`
converts that converged state back into a native fermionic Symmray `SymMPS`
with exactly the same bond dimension, ready for fermionic gate streams (for
example real-time light-pulse evolution) and fermionic observables. The
conversion is the exact inverse of the bosonization; it normalizes each site
tensor to physical-leg-last before rebuilding the fermionic array, because
quimb's `DMRG2` can leave the physical leg first on the boundary tensor and the
fermionic reconstruction is leg-order sensitive. It requires the optimizer to
have been built from a fermionic `SymMPS` `init_mps` template, which supplies
the fermionic metadata (physical sectors, edges).

Pepsy builds
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
space. Adaptive zero-sector widening retargets already maintained block
environments into the expanded bond charge maps, then any actual zero-sector
widening recanonicalizes the MPS before H-only local solves. This means
equal-chi turnarounds with new adaptive sectors rebuild block environments in
the refreshed gauge rather than trusting a stale `N_eff = I` assumption.
Noisy sector enrichment still clears environments because it changes the
represented state. By default, Symmray `H_eff` matvecs use a block-native projected contraction;
`matvec_backend="dense_reference"` keeps the older
NumPy dense-aligned matvec available as a validator. During a local
dense/Lanczos solve, the block-native path caches the static projected
problem for the active two-site window, including reindexed MPO tensors and
left/right environment projectors. The cached projected problem also
precomputes the static block-contraction routing used by repeated Lanczos
matvecs. After the first application for a cache-compatible block layout, the
NumPy-backed Symmray path compiles the block-sector pair schedule for each
static-left contraction, flattens each output sector to a dense matrix product,
and groups equal `(M, K, N)` products into batched `numpy.matmul` calls. This
avoids rediscovering block routing and collapses repeated small dense products
inside every hot-loop `H_eff` application. The normal Fermi-Hubbard DMRG path
is already bosonized before these contractions, so it uses this fast path.
For unfused bosonic NumPy plans, output blocks with an identical dynamic right
source schedule can additionally use a source-fanout GEMM: Pepsy stacks their
static left maps even when their row count `M` differs, builds the shared right
matrix once, and scatters the GEMM rows back to their sector blocks. The
additional stacked maps are collectively capped at 32 MiB per compiled pair;
all unclaimed outputs retain the existing batched or single-matmul routes.
Compatible native fermionic arrays can still use the compiled route for
unfused, shared-leg, NumPy-only contractions, but retain their existing
per-output batching: Pepsy caches Symmray's sector phases with the plan and
checks fermionic metadata before each reuse. Fused, outer-product, mixed, or
non-NumPy fermionic contractions retain Symmray's exact `tensordot` path.
`profile_summary()` aggregates the new batch timing phases, while sampled
matvec diagnostics report the batch-plan shape and call counters so scale runs
can confirm the hot matvec path is reusing this setup work. Fanout diagnostics
use the `*_compiled_block_plan_fanout_*` prefix and report eligible/enabled
groups, output coverage, static bytes, predicted output-product savings, and
actual fanout GEMM calls; timing totals use
`*_compiled_block_fanout_pack_elapsed` and
`*_compiled_block_fanout_matmul_elapsed`.
The private dense block-sector effective-Hamiltonian cache is disabled by
default. Its composed matmuls change summation order and did not amortize their
setup cost within bounded local Krylov solves. The experimental path remains
available to focused benchmarks and still requires first-result agreement with
the streamed contractions at `1e-12`; its diagnostic record exposes the cache
state, size, block count, reuse count, validation error, and disabled reason.
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
`norm_check="off"` uses this fast canonical-assumption path after successful
OBC canonicalization, matching production DMRG usage. Set
`norm_check="strict"` to validate every two-site window when debugging
canonicalization or charge-alignment changes.
Strict and generalized-dense diagnostics build explicit dense norm
environments; skipped checks in canonical H-only modes are recorded as
canonical assumptions rather than as measured `N_eff` validations.
For audit runs, `norm_check="sampled"` checks boundary windows plus every
`norm_check_interval`-th interior window, and `norm_check="first_sweep"` checks
only the first sweep. Skipped checks are still recorded in
`norm_identity_diagnostics` with `skipped=True` so benchmark logs remain
auditable. `residual_check` accepts the same schedule modes and records
normalized local eigensolver residuals in `residual_diagnostics` without
raising by default; `residual_check_tol` marks diagnostics as passed/failed
when supplied. `matvec_diagnostics` records sampled `H_eff` matvec elapsed
time, cache-hit status, theta block size, projector block counts, cached
contraction routing counts, compiled block-plan build/use counts, and
block-native substep timings in
`matvec_diagnostic_records`. The cached projected problem also records whether
the block-native path used the original right-first contraction route or the
left-first route selected for strongly imbalanced projector sizes.
`compute_initial_energy="lazy"` is the default, deferring the startup estimate
until the `initial_energy` or pre-sweep `energy` property is requested.
`compute_initial_energy=True` preserves the historical eager startup estimate,
and `False` skips it entirely. Sweep setup
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

## Initialization robustness (benchmark findings)

Under the recommended setting (a product or `random_unitary` initial state plus
the `density_matrix` mixer) SymDMRG2 is variationally correct, monotone in `chi`,
and reproduces exact diagonalization to machine precision at the exact Schmidt
bound. On a 6-site Fermi-Hubbard chain at `chi=64` it reaches the exact
fixed-sector ground state to `~1e-14` for `U in {1, 4, 8}` from product,
`random_unitary`, and even raw block-fill starts; on the frustrated `3x3` PBC
torus the three initial states converge together as `chi` grows (init spread
`5.8e-2 -> 1.0e-2 -> 1.0e-3` for `chi in {16, 32, 64}`), always above ED.

On the hard periodic `6x6` `U/t=8` torus at `chi=32`, the recommended inits are
robust and beat a matched-`chi` TeNPy run across all of TeNPy's own settings
(`<H>/N - U/4`):

| initial state                    | SymDMRG2 + `density_matrix` mixer |
| -------------------------------- | --------------------------------- |
| product (ramped)                 | `-2.43436`                        |
| `random_unitary`                 | `-2.43137`                        |
| raw `SymMPS.random` block-fill   | `-2.40812` (stuck)                |

For comparison, TeNPy at `chi=32` lands in `[-2.42502, -2.43095]` across
mixer-on/off and product/random-unitary starts. So the recommended-init results
sit at or below TeNPy's best. The one caveat is the raw block-fill start: on this
far-from-converged large case it locks into a poor charge-sector layout that the
mixer cannot escape even with a `30x` larger, longer-lived amplitude. TeNPy has
no equivalent pathological start because its random init is well-conditioned by
construction; the Pepsy analog is `SymMPS.random_unitary_for_model`. Prefer a
product or `random_unitary` start; reserve raw `SymMPS.random(bond_dim=chi)` for
deliberate stress testing.

For speed and seed-stability together, pair the density-matrix mixer with
`variational_sector_basis="off"`. The adaptive zero-sector basis is redundant
once the mixer performs the sector exploration, and profiling shows it is both
the dominant per-sweep overhead and a source of seed-to-seed variance. Turning
it off is variationally identical where it matters (it still reaches exact ED to
`~1e-14` on solvable cases), makes SymDMRG2 ~1.5-3.7x faster per sweep, and
tightens the random-init seed spread to TeNPy's level: on a mixed `6x6` `U/t=8`
scan across `chi in {8, 12, 32}` the `random_unitary` seed spread is
`6.2e-4 / 9.7e-7 / 5.5e-4` versus TeNPy's `5.7e-3 / 2.3e-2 / 2.9e-4`, with
Pepsy's best energy below TeNPy at every `chi`. After this change the block-
sparse local matvec is the remaining cost (~50% of wall time); TeNPy is still
faster in absolute wall time per run.

When the optional subspace mixer is active, SymDMRG2 uses it as a temporary
exploration aid rather than a final convergence state. If the sweep convergence
criteria are met while the mixer was still active, the Symmray path records a
`mixer_lifecycle` diagnostic, disables the mixer, and requires a subsequent
no-mixer sweep before setting `converged=True`. This matches the TeNPy practice
of turning the mixer off and doing final clean sweeps once the state has found
the right Schmidt subspace.

`mixer="density_matrix"` (aliases `"dm"`) selects the White/TeNPy-style
density-matrix mixer, which is the recommended mode for escaping mixer-off
convergence plateaus. It perturbs the two-site reduced density matrix with the
environment-projected Hamiltonian directions (keeping the MPO virtual bond
open), `rho_mix = rho_theta + mixer_amplitude * rho_pert`, and eigendecomposes
to a truncated canonical isometry, opening the charge sectors a plain SVD
truncation would miss without changing the current two-site state. This mirrors
`tenpy.algorithms.mps_common.DensityMatrixMixer.mix_rho`. On the periodic
Fermi-Hubbard benchmarks it converges *below* a matched-`chi` TeNPy run: at
`chi=32` the 5x6 torus improves from the mixer-off plateau `-2.42248` to
`-2.42881` (TeNPy `-2.42674`) and the 6x6 torus from `-2.42977` to `-2.43428`
(TeNPy `-2.42962`), energy densities in the `<H>/N - U/4` convention. Use a slow
`mixer_decay` (default `0.9`) so the mixer stays active across the early sweeps
and a `mixer_disable_after` budget that leaves a few final clean sweeps.

.. automodule:: pepsy.optimizers.sym_dmrg
   :members:
   :undoc-members:
   :show-inheritance:
```
