---
name: belief-propagation
description: 'Run, build, extend, or debug Pepsy belief propagation and its loop, cluster, and partitioned corrections in `pepsy.bp`, including one- and two-norm BP, loop cluster/series expansions, PNE, relay BP, GBP region graphs, message reuse, and convergence or error-bound questions. Use for Pepsy BP workflows and its Tensy DEM-decoder integration; not for decoder-specific glue.'
---

# Belief propagation, loop/cluster corrections, and PNE in pepsy

`pepsy.bp` wraps quimb's belief propagation and exposes three deliberately
different correction families: edge-resolved loop series, tensor-region loop
clusters, and partitioned network expansions (PNE). Keep BP wrappers **thin**
over `quimb.tensor.belief_propagation`; the annotated paper trail lives in
[`docs/development/references/belief_propagation.md`](../../../docs/development/references/belief_propagation.md).

## When to use
- Run / wrap / extend BP contraction of a quimb `TensorNetwork` in pepsy.
- Add or debug a **loop / cluster / generalized** correction to BP.
- Add or debug a partitioned network expansion, including explicit or
  higher-rank projectors, recursive schedules, residue diagnostics, and open
  outputs.
- Reason about **convergence** (does a given correction need a BP fixed point?),
  the cluster-vs-series distinction, or error bounds.
- Expose a clean pepsy BP/loop-correction API that **tensy** consumes for DEM
  decoding (keep decoder glue in tensy — see the tensy `bp-decoding` skill).

## Do NOT use for
- QEC-decoder-specific construction (DEM Tanner graph, syndrome conditioning,
  logical marginals, BP-OSD/MWPM baselines) → tensy `bp-decoding` skill.
- Plain MPS/PEPS gate application or boundary contraction → pepsy core APIs.

## Public API (`import pepsy.bp`)
- `one_norm_bp(tn, *, method="l1bp"|"hv1bp"|"d1bp", max_iterations, tol, damping,
  update, diis, init_messages, ...) -> RelayBPResult` — plain 1-norm BP to a
  fixed point (partition-function / nonnegative contractions, e.g. decoding).
- `two_norm_bp(tn, *, max_iterations, tol, damping, update, diis,
  init_messages, ...) -> RelayBPResult` — native D2BP for wavefunction and
  norm contractions. Symmray-backed fermionic PEPS retain their native charge
  blocks throughout message updates and distance evaluation; DIIS
  automatically falls back to native sequential updates because Symmray does
  not expose Quimb's dense vectorizer concatenation.
- `relay_bp(tn, *, method, num_relays, max_iterations, gamma_range, tol, damping,
  update, memory_first_leg, init_messages, seed, ...) -> RelayBPResult` —
  disordered-memory / relay-BP: per-node random memory (incl. negative) applied
  around quimb `iterate` on the public `messages` dict (quimb `damping` is a
  *uniform* `(old, new)` hook, so per-node disorder is driven here), relayed
  warm-started legs, best-of returned. Forces `local_convergence=False`.
- `loop_cluster_expand(tn, gloops, *, norm="2norm"|"1norm", combine="prod"|"sum",
  messages, run_bp, max_iterations, tol, damping, ...) -> LoopClusterResult` —
  loop **cluster** expansion (quimb `contract_gloop_expand`). `.expand(gloops)`
  reuses the converged messages; `.messages` exposes them.
- `loop_series_expand(tn, gloops, *, norm="2norm"|"1norm", ...)` —
  edge-resolved loop **series** with explicit excited-bond terms. Its integer
  cutoff is a maximum excited-bond degree, not a tensor-region size.
- Local D2BP observables: `partial_trace_loop_series_expand` and
  `partial_trace_loop_cluster_expand` return local reduced density matrices;
  `compute_local_expectation_loop_series` and
  `compute_local_expectation_loop_cluster` accept Quimb-style
  ``{site_or_sites: operator}`` mappings and return scalar expectations. Share
  one D2BP solve across terms. Use the scalar APIs for fermionic observables.
- Explicit-edge local D2BP observables:
  `partial_trace_edge_loop_series_expand` and
  `compute_local_expectation_edge_loop_series` use the same canonical
  `LoopSeriesTerm` Q-edge sets and edge-degree cutoff as
  `loop_series_expand`, rather than Quimb's local-region cutoff. The scalar
  path inserts the gate before forming the graded bra and is the supported
  fermionic expectation route. Terms that put Q on a bond wholly internal to
  `where` are currently rejected explicitly. Nonzero-Q fermionic scalar
  corrections are currently restricted to one-site gates; reject multi-site
  graded-Q gates clearly rather than treating a dense rho trace as a sign
  oracle.
- `partitioned_expand(tn, partition_inds=... | partitions=..., *,
  norm="2norm"|"1norm", form="linear"|"combinatorial", ...)` — PNE from
  Evenbly, Gray, and Chan (arXiv:2512.10910). It inserts complementary
  `P`/`Q=I-P` projectors on selected pairwise virtual indices, optionally
  retains the residue, supports explicit higher-rank projectors, D1/D2 open
  outputs, and does not require a converged BP fixed point when projectors or
  messages are supplied explicitly.
- `recursive_partitioned_expand(tn, partition_levels, ...)` — fixed recursive
  PNE schedule. The paper's cost-driven repartitioning is a policy layer; do
  not silently infer a contraction budget or claim automatic cost optimality.
- `select_pne_partitions(...)`, `pne_projectors(...)`, and
  `pne_projector_diagnostics(...)` — residue-based partition selection and
  projector inspection. Selection is a heuristic, not a rigorous error bound.
- `weight_pass(tn, *, alpha=0.8, ...) -> WeightPassingResult` — Appendix-C
  positive-weight passing on a closed pairwise network. Call
  `result.projectors(rank=r)` and pass the returned projectors to PNE on the
  returned gauge-transformed network.
- Long-range PEPS observable helpers: `compute_boundary_expectation(tn, terms,
  max_bond=chi, ...)` batches one- and two-site terms, including separated
  support, through Quimb's boundary environment. For a connected local
  approximation, `compute_path_cluster_expectation(tn, terms,
  max_distance=..., gauges=su_gauges, ...)` joins a two-site support by a graph
  path and uses simple-update bond vectors to close the cluster boundary.
  `compute_bp_path_expectation(...)` is the safe fermionic convenience route:
  native D2BP -> Pepsy BP-to-SU conversion -> path-cluster expectation.
  Terms use Quimb's mapping form, e.g. `{((x0, y0), (x1, y1)): operator}`.
- SU/simple-gauge bridge helpers: `simple_update_messages_from_gauges`,
  `d1bp_from_simple_update_gauges`, `run_d1bp_from_simple_update_gauges`,
  `simple_update_bp_residual`, `d2bp_from_simple_update_gauges`,
  `run_d2bp_from_simple_update_gauges`,
  `simple_update_core_and_gauges_from_d2bp`, and `norm1_gloop_expand`. Use
  these for scalar 1-norm or PEPS 2-norm work with externally supplied simple
  gauges. The D2BP bridge supports dense tensors and native Symmray fermionic
  `U1`, `U1U1`, and `Z2` block-sparse tensors.
- Result dataclasses: `RelayBPResult` (`.bp`, `.converged`, `.iterations`,
  `.max_mdiff`, `.contract()`, `.messages`, `.snapshot()`), `LoopClusterResult`
  (`.estimate`, `.bp_converged`, `.bp_iterations`, `.expand()`, `.messages`).

## Keep the correction APIs distinct

Use `loop_expand` only as a selector; do not translate one cutoff into
another:

| expansion | object being expanded | cutoff / selection |
| --- | --- | --- |
| `series` | individual excited bonds and generalized-loop supports | `gloops` = excited-bond degree |
| `cluster` | tensor regions with counting numbers | `gloops` = tensor-region size or explicit regions |
| `pne` | selected index partitions into `P` and `Q` subspaces | `partition_inds` or factorized `partitions` |

The original *local-RDM* APIs deliberately follow Quimb's `get_local_gloops` region
convention: their integer `gloops` is a local generalized-loop region cutoff,
not the edge-degree cutoff of `loop_series_expand`. Local loop series inserts
`Q` on the eligible internal bonds of each selected region; local loop cluster
contracts BP-closed regions with inclusion--exclusion counts. Do not compare
either term-for-term with a brute-force edge-subset enumeration. For that
case, use `partial_trace_edge_loop_series_expand` or
`compute_local_expectation_edge_loop_series`; do not silently reinterpret a
region cutoff.

PNE terms are not loop-cluster regions. A PNE residue is the all-`Q` network;
retaining it gives the exact projector identity, while dropping it is the
approximation whose error should be monitored.

For native Symmray D2BP, relay, loop-series, and PNE paths, directed messages
can temporarily omit charge
sectors after an update. Pepsy aligns message charge maps before distance
checks, D2BP normalization, loop-cluster/series expansion, and PNE projector
construction, while keeping the messages and tensor data block-sparse. Do not
replace this D2 path with a dense copy: dense eigendecomposition can reorder
eigenvectors across charge sectors and produce invalid Symmray gates.

Native fermionic arrays also require their global-phase metadata. An odd
Symmray array with a site `label` must carry the corresponding implicit
`dummy_modes`; `FermionicArray.new_with(...)` intentionally drops those modes.
The D2 entry points repair labelled odd arrays on a private network copy before
constructing BP or finite clusters, preserving blocks and lazy phases. This is
what makes the result invariant under Cotengra contraction-tree choices. Do
not treat `optimize="auto-hq"` as a fermionic sign fix: it is a Cotengra path
preset, just like a reusable `PathOptimizer`, and a path-dependent sign means
the input metadata is invalid or incomplete.

The missing-`dummy_modes` defect and the cyclic open-series defect are related,
but they are not the same bug. The former is a metadata-repair failure: a
labelled odd array created with `new_with(...)` has lost the implicit modes
needed to preserve its global fermionic phase. The latter can occur even when
every local array is parity-even and has `dummy_modes=()`. In that case the
failure is the native Symmray representation of a cyclic open correction: the
unexcited `P` projectors and excited `Q` projectors can require incompatible
mixed bra/ket orientations, so relabelling an open `Q` to make pairwise
contractions run is not an algebraically safe fermionic contraction. It can
produce a non-convergent series even though BP has converged and the dense
shadow is correct.

When a native fermionic open scalar or rho calculation has a cyclic graph,
use the graded loop-cluster-compatible route. It keeps the rho native, inserts
the gate in the graded ket/bra contraction for scalar observables, and avoids
using `trace(rho @ gate)` as an oracle. A useful diagnosis is to check
`parity`, `dummy_modes`, and `label` first: if all arrays are even with no
dummy modes, do not blame the missing-dummy repair. Compare against the native
exact oracle and the native loop-cluster result instead. Keep the explicit open
edge-series route for dense/ordinary cases and fermionic trees, where the
mixed-orientation cyclic obstruction is absent. If contraction budgets are
provided, apply them to the cluster contractions as well and inspect the
reported FLOP/peak-memory decisions; never use the budgeted call as a reason
to fall back to the unsafe mixed-orientation route. Use the route-independent
diagnostics `open_*_edge_term_costs` and
`open_*_cluster_region_costs` when consuming cost metadata; the older
`open_*_term_costs` fields are route-specific compatibility aliases.

### Fermionic local observables: required construction and oracle

Do **not** evaluate a native fermionic observable as `trace(rho @ gate)`, even
when a partial-trace API returns a native Symmray rho: opening and fusing the
physical legs loses the graded gate-routing convention. Treat those rhos as
diagnostics (charge support, trace, Hermiticity), not as an observable oracle.

For scalar local observables, insert the native gate into the ket with
`tensor_network_gate_inds(..., contract=False)` *before* forming the double
layer. Build the bra from D2BP's `tensor_dual_map` and attach messages using
`index_dual_map`; do not replace this with `tensor.conj()` carrying ket virtual
labels. Normalize with the directly contracted gate-free BP network, not a
trace of the fused rho. This works for both adjacent and separated supports.

Before trusting a new fermionic BP observable path, add a native-Symmray tree
test where D2BP is exact: compare a parity-even two-site gate (hopping,
pairing, or eta-pairing) against `compute_local_expectation_exact`, then repeat
with reversed site order and a correspondingly transposed native gate. A dense
or trace-only test does not establish fermionic sign correctness. Use a
brute-force edge-subset implementation only as a loop-enumeration oracle; if
it is dense/non-graded, it is not by itself a fermionic-sign oracle.

For valid closed scalar Symmray networks, the 1-norm APIs (`L1BP`, `HV1BP`,
and `D1BP`), their loop/series/PNE corrections, D1 SU bridge, and
`weight_pass` use a topology-identical dense BP shadow because Quimb's D1
initializers and weight SVDs require dense scalar operations. A raw fermionic
PEPS with dangling physical legs is not a direct 1-norm input: use D2BP for
its wavefunction norm (or explicitly construct a valid closed scalar network).

## quimb substrate (do not reimplement)
`quimb.tensor.belief_propagation`:
- BP families: **1-norm** `L1BP` / `HV1BP` / `D1BP` (partition-function /
  decoding); **2-norm** `L2BP` / `D2BP` (wavefunction / operator norms).
  **Pick 1-norm for decoding-style nonnegative contractions.**
- Loop corrections (methods on `D1BP` / `D2BP`):
  - `contract_gloop_expand(gloops=C, combine="prod"|"sum")` — loop **cluster**
    expansion (Gray et al. 2510.05647). `combine` is `D1BP`-only; `D2BP` is
    product-only and takes `progbar`. `D1BP.__init__` takes **no** `optimize`.
  - `contract_loop_series_expansion(gloops=C, multi_excitation_correct=True)` —
    loop **series** (Evenbly et al. 2409.03108); more fixed-point-sensitive.
  - `contract_with_loops(...)` — simple loop enumeration.
- Region-graph scaffolding: `RegionGraph`, `gen_region_counts` — Kikuchi region
  topology + counting numbers **only** (no `.run()` solver). A packaged **GBP**
  solver (Tindall et al. 2604.24760) is the natural next wrap → a pepsy
  `region_bp`; check whether it has landed in quimb before hand-rolling.
- Convergence toolkit: `run(max_iterations, tol, diis=True, damping, info=...)`;
  `info` returns `converged` / `iterations` / `max_mdiff`. `damping` is uniform.

## Convergence — the key mental model (keep straight)
Loop corrections split by how much they need a **converged BP fixed point**:
- **Loop cluster expansion** (`contract_gloop_expand`): exact contractions of
  growing regions; messages only close each region's **boundary**. A
  system-covering region is exact and message-independent, but finite loop-only
  cancellations and tree/dangling reductions assume fixed-point messages.
  With arbitrary messages, interpret the result as a boundary-closure cluster
  approximation and sweep cluster size.
- **Loop series expansion** (`contract_loop_series_expansion`): inserts
  `I − m⊗m` projectors that vanish **only at the fixed point**; also combinatorially
  divergent in the thermodynamic limit. **Fixed-point-sensitive.**
- **Free-energy cluster expansion** (Midha–Zhang 2510.02290, not in quimb):
  expands `log Z`, only connected clusters, **rigorous** `|log Z − F̃_m| ≤
  n e^{−d(m+1)}` under loop decay `|Z_l| ≤ e^{−c|l|}` (Kotecký–Preiss). Built
  **around** the fixed point → fixed-point-sensitive but gives an *a-priori error
  bar*. Useful as a diagnostic: estimate the loop-decay rate → predicted error.
- **Stochastic loop corrections** (Sim et al. 2603.08427): MCMC-sample the
  `Z = Z_BP × loop-factor` factorization → **unbiased**, statistical (not
  truncation) error.
- **GBP** (Tindall et al. 2604.24760): region messages; structural fix (absorb
  loops into regions), but generalized messages can be **harder to converge**.
- **Relay-BP** (`relay_bp`): the *dynamical* fixed-point hardener (disordered
  memory + relay). Complements the above; does **not** guarantee convergence.
- **PNE** (`partitioned_expand`): complementary projector identities are exact
  for any projectors, so a converged BP fixed point is not a mathematical
  prerequisite. BP messages provide a convenient rank-one choice, while
  explicit projectors or `weight_pass` provide non-BP and higher-rank choices.

## SU gauges, Gray's correction, and quimb `norm_gloop_expand`
- Corrected point from J. Gray: for the **same closed scalar/norm TN**,
  converged simple-update gauges correspond to a BP / super-orthogonal fixed
  point up to message-gauge normalization. Thus tree-like correlations are
  trivial after `normalize_simple(gauges)`, and `autoreduce=True` is valid.
- Do **not** claim converged SU gauges generally fail tree reductions. The real
  caveats are: gauges must be converged for the same projected scalar network;
  borrowed/open-PEPS gauges are only a warm start; signed amplitude networks can
  be outside the natural nonnegative 1-norm BP regime.
- Gray identified the actual quimb bug we reproduced: in
  `TensorNetworkGenVector.norm_gloop_expand(gauges=...)`, `nfactor` returned by
  `normalize_simple(gauges)` is already the norm scaling, so the final formula
  should effectively be `nfactor * sqrt(loop_product)`, not
  `sqrt(nfactor * loop_product)`. On a two-site tree, current quimb produced
  `z**2 == exact_norm`; the fixed formula gives `z == exact_norm`.
- For scalar BP residual tests from SU gauges, the tight convention is often
  `message_power=1.0`; the symmetric default `sqrt(gauge)` is the clean
  round-trip convention for mapping opposite messages back to an SU-like gauge.

## Gotchas
- `D2BP` = one super-site per lattice site (reaches exact at moderate cluster
  size); `D1BP` on a raw `peps.H & peps` (double the tensors) needs larger `C`.
- Raw DEM factor TNs are nonnegative and good for D1BP, but their factor graph
  can be huge and mostly tree-like. Small generalized-loop corrections may
  barely change rare logical-sector probabilities; use exact/MPS references
  and inspect log-likelihood margins before treating BP+LCE as a decoder.
- `norm="2norm"` + `combine="sum"` is rejected (D2BP is product-only).
- PNE selected indices must be pairwise internal virtual indices. Use
  `form="combinatorial"` for factorized multi-index partitions.
- D1 open PNE currently requires `allow_open=True` and explicit messages or
  projectors with `run_bp=False`; D2 open outputs are physical ket/bra pairs.
- `weight_pass` is intentionally restricted to closed pairwise networks; for a
  D2 calculation, obtain the environment on the appropriate closed
  double-layer network before supplying projectors to D2 PNE.

## Long-range fermionic PEPS observables

Use the native two-site Symmray operator built by `Fermion` or
`Fermion.operator_term`; never construct separated odd operators with a plain
Kronecker product or add a Jordan--Wigner string to the native path. Symmray
supplies the graded signs when the complete ordered native operator is
contracted with the native PEPS.

For a finite path cluster, `gauges` must be SU/simple-update *bond vectors*.
Do **not** pass D2BP's directed positive-semidefinite matrix messages to
Quimb's `compute_local_expectation_cluster`. Use `compute_bp_path_expectation`
or `simple_update_core_and_gauges_from_d2bp` to convert them first. The
returned `core` and external gauge vectors represent the same state only after
`core.copy().gauge_simple_insert(gauges)`; the cluster routine uses the vectors
only to close its cut boundary.

For native Symmray PEPS, path-cluster compression (`max_bond=chi`) uses Pepsy's
graded adapter around Quimb's public compressed-contraction API. It keeps the
observable RDM physical legs unfused, aligns zero-weight virtual charge sectors
on a private cluster copy, and uses Symmray's fermionic `squeeze` before
Quimb's QR/SVD steps. Thus standard Quimb/Cotengra path optimizers can be
supplied through `optimize`. A finite cluster or finite-chi boundary discrepancy
is an environment approximation, not by itself a fermionic-sign failure;
enlarge the region/chi against an exact small reference.

### Required sign regression

Any change to native fermionic D2BP, SU gauging, BP-to-SU conversion, or
long-range measurement must preserve these tests in
`tests/test_bp_symmray.py`:

- `test_fermionic_long_range_hopping_sign_survives_su_and_bp_gauges` prepares
  a controlled spinless U1 2x2 Fock PEPS whose diagonal hopping correlator
  crosses an occupied mode in row-major Jordan--Wigner order.
- `test_spinful_long_range_hopping_sign_survives_su_and_bp_gauges` repeats the
  parity-sensitive up-fermion correlator for spinful U1 and U1U1 PEPS.
- `test_spinful_eta_pair_measurement_survives_su_and_bp_gauges` prepares a
  long-range eta-pair observable with the public routed 2D gate path and
  verifies its imaginary-time expectation for spinful U1 and U1U1 PEPS.

Each has an independent dense JW oracle; the hopping cases also compare it to
the deliberately no-string bosonic control, which has the opposite sign. The
native exact contraction, a distinct contraction path, direct SU
reconstruction, D2BP-to-SU reconstruction, and the public BP path helper must
all match the JW value. The spinless regression also checks the native
compressed path-cluster route at a sufficiently large `max_bond`. Also assert
that native tensors/messages/vectors retain their Symmray types. This is the
sign oracle; matching only density observables or only two native contraction
routes is insufficient.

- Native Symmray fermionic BP coverage is in `tests/test_bp_symmray.py` and
  exercises `U1`, `U1U1`, and `Z2` SU↔D2BP round trips, D2 relay/loop-series/
  PNE corrections, and closed-scalar 1-norm compatibility. Preserve the array
  class and charge blocks when adding new D2 BP or gauge paths; route valid
  scalar D1 paths through the documented dense shadow.
- Keep `pepsy.bp` out of the lazy top-level namespace (import `pepsy.bp`); do
  **not** edit `src/pepsy/__init__.py` for it.
- Tests: `tests/test_bp_relay.py` and `tests/test_simple_update_gen.py`. Env: py312
  (`source ~/envs/py312/bin/activate`, `NUMBA_CACHE_DIR=/tmp/numba_cache`).

## References
Full annotated list: [`docs/development/references/belief_propagation.md`](../../../docs/development/references/belief_propagation.md).
Core: Evenbly–Gray–Chan 2512.10910 · Gray et al. 2510.05647 · Midha–Zhang
2510.02290 · Evenbly et al. 2409.03108 · Tindall et al. 2604.24760 · Sim et al.
2603.08427 · Müller et al. 2506.01779 · Alkabetz–Arad PRR 3 023073 ·
Tindall–Fishman SciPost 15 222.
