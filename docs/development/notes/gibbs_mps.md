# GibbsMps implementation note

## 2026-10-27 pre-commit review

An explicit final entry in `measure_at` now uses the supplied `beta` object,
even when the entry itself is a Python float. This preserves beta gradients
and backend metadata at the endpoint. The expanded analytic-gradient regression
and 354 focused pre-commit tests passed; Ruff and whitespace checks passed.
The user authorized committing and pushing this work to `develop`. Earlier
statements below about uncommitted work describe those validation checkpoints.

## 2026-10-27 shared observable readout

**Adopt:** the existing `MpsEnergyOptimizer` owns canonical/environment
route selection and joint energy/observable measurement. `GibbsMps.energy`
and `measure` use that layer, retaining the prepared Hamiltonian operators
and one reusable `build_optimizer` instance. Gibbs maps requested flat physical
supports to even purification sites. The default remains canonical readout,
second-order integration with `beta_step=0.05`, and `measure_every=5`.
The existing standalone MPS energy objective keeps its exact default.

Dense MPS measurement shares one RDM per ordered support. The installed
Quimb environment route builds one sweep across all enclosing intervals;
canonical readout uses one disposable MPS and trusted center metadata when
provided. `TreeEnergyOptimizer.measure` likewise uses one disposable canonical
tree and shared support RDMs. Numerical RDMs/environments live only for the
measurement call. Native MPS/tree arrays keep their graded readout paths.

**Adopt, opt-in:** Torch einsum/vmap groups final RDM/operator contractions
by compatible shape, dtype and device, with bounded chunks. Planning, QR,
environment construction, and mutable tensor-network methods stay outside
vmap. Ordinary loops remain the non-Torch/native fallback. This is exact
readout batching, not hypercompressed amplitude sampling; it does not change
truncation policy. No performance ranking or production speedup was measured.

**Compatibility shim:** Quimb builds without `compute_partial_traces` use
exact readout with an explicit warning and effective-route metadata. A focused
regression covers the capability gate. Quimb's canonical unnormalized RDM
omits the global exponent while the environment route retains it; the shared
boundary cancels scale before normalized readout and restores it once for
unnormalized values. A large-exponent regression prevents inf/inf normalization.
The earlier upstream/version audit remains applicable; installed signatures
and implementations of `compute_partial_traces`, its canonical/environment
routes, and `partial_trace_exact` were inspected for this change.

Validation: 350 focused tests passed before the final compatibility additions;
the final energy/Gibbs/public-API/layout selection passed 161 tests, including
Torch CPU/CUDA batching, state/operator gradients, JAX fallback gradients,
native MPS/tree readout, cyclic MPS environments, and state preservation.
The broad repository run was interrupted in unrelated JAX compilation after
572 passes and two skips (310.55 seconds); it is **not** full-suite validation.
Ruff and `git diff --check` passed. No production simulations, commits, or
publication. Public usage and limitations are in the
[energy guide](../../api/optimizers/energy.md#batched-mps-and-tree-readout).

## 2026-10-27 final review and measurement cadence

**Adopt:** `measure_every=5` is now the `evolve` default. Beta zero and the
final beta remain included; `measure_every=1` explicitly requests every step.
Explicit `measure_at` targets remain supported with the default cadence.
Energy reuses its `build_optimizer(progbar=False, parallel=False)` instance
and bound term dictionary. It now supplies a copy of the optimizer's canonical
center metadata to Quimb's non-inplace readout. Installed Quimb source confirms
that readout copies the state and metadata and forwards `optimize` to local
contractions.

Review confirmed the dense non-unitary direct/DMRG overlap formulas against
general complex gate sequences, reversed supports, one-site physical filters,
moving centers, and repeated normalization, with and without truncation.
No change to the non-unitary formula or unitary behavior was needed.
Measurement tests reject canonical-center rescans, check cache reuse, and
assert that live arrays and metadata are unchanged by energy readout.
Latest validation: 303 tests passed across Gibbs, MPS replay/normalization,
public API and package layout; Ruff and `git diff --check` passed. No new
production runs or timing claims.

## 2026-10-27 trajectory measurement and canonical-overlap follow-up

**Adopt:** `evolve(..., measure_every=N)` runs N integration steps between
yields; `measure_at=[...]` hits explicit beta values on one evolving state.
The iterator always includes zero and the final beta. Observable evaluation
is explicit in the caller's loop. Equal uniform batches reuse their installed
gates; adjacent layers fuse inside each measurement interval. A finite-chi
final-only regression matches the fully fused `prepare` circuit. Existing
experiment supervisor scripts still need to switch to this API; old results
remain independent preparations.

**Adopt:** automatic non-unitary direct/DMRG diagnostics replace the earlier
Gibbs FIT-only opt-in getter below. Ordinary dense arrays contract only the
active canonical interval. DMRG uses the terminal projection identity and
reuses `FIT.final_norm`, requiring only a target-norm contraction. Direct
compression contracts target norm and cross-overlap, then reuses the retained
center norm. The common exponent cancels before contractions. Readout is
`MpsOptimizer.get_nonunitary_diagnostics()` and `GibbsMps.local_infidelity`;
the unitary norm-survival path is unchanged. Native symmetry/fermionic arrays
use full graded overlaps, explicitly reported as `scope="full_native"`.
This supersedes the earlier statement that non-unitary diagnostics are absent.

Validation: 427 tests passed across Gibbs, MPS replay/normalization, public
API, layout, Quimb compatibility and samplers; nine focused native checks
passed. A subsequent four-case Torch gradient check includes explicit-beta
and every-N-step measurement schedules. References cover truncated adjacent/non-adjacent overlaps for
direct/DMRG/DMRG2/DMRG3, batched FIT targets, explicit beta measurements,
large represented exponents, Torch gradients, and native graded fallback.
The dense-window tests reject outside site tags and verify one selected
network for FIT versus two for direct. No production jobs or timing
comparisons were run. The previous environment/upstream audit below applies;
the installed Trotter scheduler was re-inspected for interval fusion.

An exploratory diagonal-gate Torch DMRG gradient probe returned NaN both with
and without the new diagnostic. A nondegenerate dense Hermitian gate passes
the gradient regression. The pre-existing degenerate-SVD gradient issue is
outside this change and remains unresolved.

**Defer:** repeated gate-factorization caching, different lattice traversals,
adaptive chi, and adaptive integration require separate accuracy/performance
evidence. The earlier CPU profile was dominated by state SVD/QR work, so
smaller scheduling costs alone do not establish a production speedup.

## 2026-10-27 default step and infidelity audit

`prepare(beta)` now uses second-order evolution with thermal increments
bounded by `beta_step=0.05`, selecting `ceil(beta/beta_step)` equal steps.
The physical-side purification step is half the thermal increment. Explicit
`dt` retains its original purification-time meaning, and explicit `n_steps`
overrides either limit. The [API guide](../../api/optimizers/gibbs_mps.md)
defines both units. This supersedes this session's initial interpretation of
the requested `0.05` as a purification-time step.

**Adopt:** `evolve()` reuses `MpsOptimizer.set_gates()` and `run()` on the same
purification and yields β=0 plus each completed step. Gate payloads are built
once. Fusion stops at yielded boundaries, so finite-χ compression can differ
from fully fused `prepare()`. Dense checks cover intermediate energy/log-Z,
isolated terms, and unfused second-order equivalence; Torch checks preserve
β and coefficient gradients. No production Gibbs jobs were launched.

`MpsOptimizer`'s inexpensive retained-norm fidelity ledger applies when the
exact target norm is known, as for unitary gates. A Gibbs imaginary-time gate
changes that norm. The installed Quimb split `info["error"]` is an absolute
discarded-singular-value norm, not a normalized gate overlap; the direct
compressor does not retain a complete per-gate fidelity report. Therefore
`GibbsMps` does not expose that value as infidelity. FIT's opt-in
`fit_overlap_diagnostics` supplies a target-overlap measurement with added
contraction cost.
`GibbsMps.local_infidelity` forwards the last measured FIT overlap loss
without another contraction, returning `None` when no measurement exists.

A two-site exact probe used `0.1*diag(2,1,1,2)` on a normalized product state,
without truncation. Both `direct` and `dmrg` returned zero norm events and
`local_infidelity=None` with `non_unitary=True`. Incorrectly treating the
same filter as unitary yielded `local_infidelity=0.975`, measuring physical
norm change. The existing automatic ledger cannot be forwarded as a Gibbs
compression infidelity.

The existing CPU and GPU experiment supervisors launch a fresh `run_gibbs.py`
process for each β, whose runner creates a new Gibbs object and calls
`prepare`. The new class iterator does not automatically change those runner
scripts. A follow-up removed repeated stream installation from `evolve` and
cached bound energy operators across its measurements. Dense/gradient and
coefficient-rebinding regressions passed (45 Gibbs tests); no runtime speedup
was measured. The optional FIT overlap helper still calls `tn_fidelity` on
full target/fitted networks rather than cancelling the canonical exterior.

## 2026-10-26 corner-removed ITF benchmark review

The downstream 10×10 removed-corner ITF benchmark has 98 active spins and 176
bonds. With NumPy `float64`, one BLAS thread, `cutoff="auto"`,
`cutoff_mode="auto"`, second-order Trotterization, and `mode="direct"`, its
β=0.2, ten-step, χ=32 point took 35.41 s to prepare and 0.58 s to measure
local energy. Pepsy timing counted 2,758 gate events, 1.44 s in explicit
canonicalization, and 35.22 s in the inclusive direct replay stage. At
β=0.466 and χ=64, preparation took 232.18 s while local energy took 1.92 s.
These are single-run timings, not a scaling law or a converged thermal curve.
The replay stage still includes gate decomposition and MPS compression;
the timing does not isolate their separate costs. Torch CUDA `float64` was
slower than NumPy CPU for the measured small and 10×10 configurations, so no
GPU-wide ranking is inferred.

**Adopt:** `GibbsMps.energy()` now uses Quimb's installed public
`MatrixProductState.compute_local_expectation` to contract the prepared
Hamiltonian directly on the purification, passing one reusable Pepsy
`build_optimizer(progbar=False, parallel=False)` as `optimize`. It avoids a
reduced-density MPO and retains the backend scalar. A 3×4, β=0.2, ten-step, χ=64 replay returned
`energy()/10 = -0.4525734277850143`, matching the earlier runner's local
energy. Small dense and Torch CPU checks cover the package API.
`prepare` now rejects nonfinite β and `dt` before gate scheduling.

The installed `MatrixProductState.compute_local_expectation(...,
**contract_opts)` forwards `optimize` to local tensor contractions. The
reusable optimizer is built on the first successful energy call and retained
across preparations; a focused test verifies the forwarded object and dense
energy. A separate 10×10 β=0 Torch CPU `float64` probe with one thread measured
0.93 s for the energy call, including the first optimizer construction. This
probe does not establish the timing for an entangled χ=128 state. A 3×4,
β=0.2, χ=64 Torch CPU probe returned the same `energy()/10` as above and
measured 0.38 s for energy.

**Defer:** reusable factorization of repeated two-site thermal gates could
reduce per-event work, but the current timing does not show its share of
direct replay. A prototype must preserve the exact Trotter stream, automatic
cutoff policy, backend/device/dtype, and Torch/JAX gradients; it needs a
same-circuit energy comparison and a timed 10×10 point before adoption.

The active environment was Quimb `1.15.1.dev96+g5656a9bb2`, Autoray
`0.11.1.dev14+g014a3f69a`, Cotengra `0.8.3.dev8+g8954240f2`, Symmray
`0.4.1.dev20+ge8c907157`, and Torch `2.14.1+cu126`. Installed signatures
for `GibbsMps.prepare`, `MpsOptimizer.run`,
`LocalHamGen.get_trotter_gates`, and
`MatrixProductState.compute_local_expectation` were inspected. The official
Quimb changelog, Autoray repository, Cotengra documentation/changelog, and
Symmray repository were checked; the Symmray array page was unavailable via
the documentation browser. Symmray remains outside this dense MPS change.

## 2026-09-02 scheduler and replay audit

The active shared Python 3.12 environment was probed directly:

- Quimb `1.15.1.dev39+g369d09b9d`
- Autoray `0.11.1.dev1+gc56f64427`
- Cotengra `0.8.3.dev7+g1d7fd333f`
- Symmray `0.3.2.dev7+gd63bb4e3f`

The relevant callable signatures are:

- `LocalHamGen.get_trotter_gates(x, order=2, steps=1,
  ordering="sort", fuse_adjacent=True, alternate=True)`
- `LocalHamGen.get_auto_ordering(order="sort", group=False, **kwargs)`
- `MatrixProductState.gate_nonlocal_(G, where, dims=None, method="direct",
  transpose=False, info=None, *, inplace=True, **compress_opts)`
- `MatrixProductState.partial_trace_to_mpo(keep, upper_ind_id="b{}",
  rescale_sites=True)`

The scheduler implementation constructs each local gate as
`expm(frac * x * H_where)`, with the product-formula fractions summing to one
per Trotter step. GibbsMps supplies `x=-beta/(2*n_steps)` and
`steps=n_steps`, so the physical-side replay has total exponent `-beta/2`.
The `order=2` schedule is symmetric, `order=1` is a single ordered sweep,
and `order=4` uses Quimb's Suzuki recursion. `fuse_adjacent` only combines
consecutive applications of the same layer; it does not change the product
formula. `alternate` reverses within-layer traversal for MPS center movement
without changing the commuting-layer result.

The supplied edge list defines the interaction graph, so square and
triangular coordinate graphs use the same generic scheduler and need no
geometry-name switch. Integer coordinate locations can infer 1D/2D/3D
regular shapes and then use `OneDMap`; Quimb geometries with non-integer
sublattice labels such as `(x, y, "A")` remain outside the regular coordinate
inference path.

Connected one-site terms are combined per logical site and distributed evenly
over the incident pair terms before `LocalHamGen` construction. This preserves
the Hamiltonian while allowing the pair scheduler to fuse onsite contributions.
One-site terms on sites with no incident pair remain exact one-site gates.

The public GibbsMps default is now `mode="direct"`, which normalizes to
MpsOptimizer's `quimb-direct` path. The legacy `mode="mpo"` spelling maps to
the same compressor. `non_unitary=True` marks the replay as norm-changing and
disables unitary stabilization; it does not normalize after every gate by
itself. With `normalize_every=True`, each replay step first retains the
canonical center, canonicalizes the active span only when needed, divides one
center tensor by its norm, and accumulates the removed base-10 scale in
`p.exponent`. `normalize_final=True` performs the same operation once for a
trailing unnormalized step.

## 2026-09-01 compatibility audit

The first Gibbs purification route uses the installed public tensor-network
surfaces:

- Quimb `1.15.1.dev39+g369d09b9d`, with
  `MatrixProductState.gate_nonlocal_(G, where, dims=None, method="direct", ...)`
  for long-range physical gates and
  `MatrixProductState.partial_trace_to_mpo(keep, upper_ind_id="b{}", ...)`
  for ancilla tracing.
- Autoray `0.11.1.dev1+gc56f64427`, with `linalg.expm`, `multiply`, `kron`,
  `reshape`, and backend-aware array dispatch for generated gates.
- Quimb's `LocalHamGen.get_trotter_gates(x, order, steps, ordering,
  fuse_adjacent, alternate)` is used as the graph-aware product-formula
  scheduler. The purified evolution passes `x=-beta/(2 * n_steps)` and
  `steps=n_steps`, so the total physical-side exponent is `-beta / 2`.
- Cotengra `0.8.3.dev6+g08fe1a3a1`, used indirectly by Quimb contractions.
- Symmray `0.3.2.dev6+ga17699db6`; native Symmray is deferred for this first
  class because an interleaved ordinary MPS needs a charge-aware Bell-pair
  constructor and symmetry-preserving imaginary-time gates.

The installed Quimb `partial_trace_to_mpo` signature is
`(keep, upper_ind_id='b{}', rescale_sites=True)` and does not expose a
`contract_tags` keyword. Its implementation nevertheless performs the
reduction through local tag contractions. It also propagates the live MPS
global exponent to the returned MPO. `GibbsMps` therefore uses this native
reducer for both unscaled and Pepsy-rescaled states, keeping the ordinary
readout on Quimb's optimized path. If explicit contraction options are
supplied, it uses the equivalent native `contract_tags` and
`contract_cumulative` operations directly with `strip_exponent=True`, because
the installed public reducer does not expose those options. This preserves
scale-aware readout without ever densifying the purification.

The resolved graph-layer ordering is passed back to Quimb explicitly rather
than asking `LocalHamGen` to color the same interaction graph twice. This is
important for randomized ordering: the schedule retained in
`GibbsMps.trotter_layers` is now exactly the schedule used for replay. Stable
non-random orderings are cached on the Gibbs object because the Hamiltonian
topology is immutable while coefficients and imaginary time are rebound for
each preparation.

The compatibility decision is **adopt** for the ordinary dense/Autoray path:
`bell_to_mps` uses Quimb's `MatrixProductState` constructor and
`apply_to_arrays`; term operators, graph-combined local Hamiltonians,
exponential gates, MPS replay, and the traced MPO all stay on the selected
backend. One-site lifting is performed at the Pepsy boundary with Autoray
Kronecker products because the installed Quimb `H1` absorption helper builds
host identities. Generated gate streams are passed through the same converter
before installation. The class uses `MpsOptimizer`'s non-unitary path and
deliberately does not request unitary stabilization or overlap-fidelity
diagnostics.

Quimb's native `TrotterGate` objects are retained as `GibbsMps.trotter_gates`;
the optimizer receives a separate normalized `(gate, physical_where)` stream
because its public stream contract uses the interleaved purification positions.
Adjacent layer fusion remains enabled by default, so a many-step second-order
stream is intentionally shorter than a literal half-step replay while having
the same product formula. Isolated one-site terms are exact and are appended
as one-site gates rather than represented by dummy edges.

Deferred work includes multi-site gates for explicit string terms across gaps,
native Symmray purification, and observables that contract directly against
the purification without first producing an MPO.
