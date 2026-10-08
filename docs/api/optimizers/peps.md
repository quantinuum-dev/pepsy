# `pepsy.optimizers.peps`

For global mode and fixed-cap sweeps (`boundary_convergence=False`),
`PepsOptimizer` has separate chi controls for different jobs:

- `chi` caps the optimized PEPS/PEPO virtual bonds.
- `boundary_chi` controls sweep/global optimizer environments; when omitted,
  it defaults to `(4*D, 5*D)`, where `D=chi`.
- `normalize_chi` controls PEPS normalization contractions.
- `evaluation_chi` controls pre/post infidelity diagnostics used for accepting
  or rejecting a candidate.

With no cap overrides, all three configured starting settings resolve to `(4*D, 5*D)`.
Each accepts a scalar or a `(norm cap, overlap cap)` pair. Normalization uses
only the first entry; infidelity uses the first for both state norms and the
second for their overlap. For example, `chi=4` gives `(16, 20)` throughout.
Automatic normalization/evaluation caps remain independent of an explicit
environment override. Explicit constructor and per-run metric caps still
take precedence. A scalar applies equally to all contractions for that setting.
Metric cap precedence is: per-call mapping `chi`, named per-call cap,
named constructor cap, stored metric mapping `chi`, shared `boundary_kwargs`
`chi`, then the automatic pair. The same resolved metric caps are used in run
records and as defaults for delegated normalization. Explicit backend-specific
normalization mappings remain available through `sweep_kwargs`/`global_kwargs`.
An explicit mapping `chi=None` is preserved for exact metrics or reuse of
existing DMRG boundary handles. Named caps require positive integers or pairs.

With `boundary_convergence=False`, use `evaluation_chi` larger than `boundary_chi` when you want a stricter final
quality check without making every optimization environment more expensive.

## Adaptive boundary convergence before sweeps

`boundary_convergence=True` is the default for `PepsOptimizer` sweep mode.
Before refining a compressed target, it probes the unchanged warm start and
target at increasing `(chi_norm, chi_overlap)` caps. It measures both norms
and the complex overlap in **both x and y**. Norms must agree relatively;
the overlap comparison uses its complex amplitude divided by the square
root of the measured norm product. Comparing norms separately prevents a
constant fidelity ratio from concealing changing numerator/denominator errors.
Both successive-cap and cross-direction comparisons must pass. Nonpositive
norms and fidelity above one beyond dtype roundoff cannot establish convergence.
Norm reality is checked at the requested approximation accuracy:
`abs(Im(norm))/abs(norm) <= max(rtol, dtype_roundoff)` for each state norm.
Using only machine roundoff here would force cap growth even when residuals
are far below the requested contraction tolerance. Raw complex values remain
saved; this does not discard their imaginary parts or loosen fidelity guards.

```python
optimizer = PepsOptimizer(
    state, gates, chi=4, mode="sweep", fit_mode="eff",
    boundary_convergence={
        "start_chi": "auto", "max_chi": "auto", "rtol": 1e-5, "atol": 1e-8,
    },
)
```

The default `schedule="d2"` probes `D**2, 2*D**2, 3*D**2, ...`, where D is
the fitted PEPS cap, up to `max_chi="auto" = 8*D**2`. Thus D=4 probes
16,32,48,...,128 and D=2 probes 4,8,12,...,32. `start_chi` and `max_chi`
accept explicit scalar or norm/overlap pairs. These adaptive controls replace
the fixed `boundary_chi`, `normalize_chi`, and `evaluation_chi` settings for
this path, including initial normalization. A larger target still has its
own measured norm; D-squared is a cost heuristic, not an accuracy bound.

Defaults are `rtol=1e-5`, `atol=1e-8` and `patience=2`: both norms and the
complex overlap must pass on two consecutive cap increases and across x/y.
The selected pair stays fixed throughout this sweep fit. Final normalization
uses exactly the selected norm cap, preserving backend, dtype and device.
For the earlier multiplicative search, set `schedule="geometric"` and
`growth=2`; that schedule uses the resolved fixed caps as its starting minimum.

With `warm_start=True`, Pepsy DMRG probes reuse three boundary MPS guesses per
axis (state norm, target norm, overlap) on the unchanged states. One-site
boundaries expand through the existing boundary API; block FIT grows through
its local splits. Reused guesses use direct initialization and are always
refitted. Before accepting convergence, fresh contractions in both directions
must agree with the reused ones. This rejects a stale variational plateau.
Direct compressors and Quimb-MPS do not consume these variational guesses.
Validated boundary cuts may be handed to the fit and retained across gate
updates as described below; `reuse_environments=False` disables that reuse.

The checked warm-start norm and target norm are reused; the initial fidelity
check also reuses these measurements instead of contracting again. The
independent post-fit check, when enabled, remains separate. Explicit custom
normalization methods/options still run their requested normalization instead
of reusing the warm-start norm.

The cap floor survives `run(reset_traces=True)` and `set_gates()`. Each new
state is rechecked starting two D-squared increments below the retained cap,
bounded by `start_chi`. Larger caps are retained for subsequent fits.
Adaptive start/retained caps above `max_chi`
raise instead of silently exceeding the ceiling.

At the ceiling, an unsuccessful check emits a warning and continues the
sweep at that cap with `boundary_convergence.converged=False`. Unusable
nonfinite or nonpositive norms still raise. A warning does not certify
accuracy, and pre-fit convergence does not guarantee every later local
environment is accurate. Existing local-update safeguards remain active.
For reused DMRG guesses, the ceiling also gets a fresh contraction check;
its measured values are used for continuation without changing the failed
convergence status.

`get_boundary_convergence()` exposes the retained `chi_floor` and per-run
records. Each gate-batch record contains `boundary_convergence`, including
tested caps, actual boundary bond dimensions, warm/fresh initialization,
scalar norms/overlaps with their exponents, named comparisons, thresholds,
stable comparison counts, status and stopping reason.
`selection_status="converged"` means the checks passed;
`selection_status="limit_unconverged"` means the ceiling was used without
passing. `at_ceiling` is separate: the ceiling can itself pass. Per-axis
`validity` records contain relative imaginary residuals, their threshold,
fidelity validity and explicit rejection reasons.
The `boundary_convergence` timing phase includes these extra contractions.

This check runs even with `measure_infidelity=False`; that option disables
the separate acceptance check, not the new convergence probe. Set
`boundary_convergence=False` to retain the old fixed-cap behavior and its
unit-target-norm shortcut. Global mode and `optimize=False` do not probe.
Custom `boundary_options` or supplied sweep boundary handles currently require
disabling adaptive checks; otherwise a clear error prevents a misleading
comparison using a different boundary policy.
An explicit `fit_max_bond` is also incompatible with adaptive checks: it would
keep the actual FIT rank cap fixed while probe chi increases. This combination
raises before initial normalization or gate application. Leave `fit_max_bond=None`
and set adaptive `start_chi`/`max_chi`, or use `boundary_convergence=False` to
retain a fixed FIT cap. This applies to shared and sweep-specific FIT options.

`boundary_engine` controls the boundary implementation used when PEPS cleanup
delegates to `SweepOptimizer`. The default, `"auto"`, keeps dense inputs on the
Pepsy `BdyMPS`/`CompBdy` path and routes Symmray-looking inputs to Quimb MPS
boundaries. Use `boundary_engine="quimb-mps"` to force that path, and pass
Quimb environment controls with `boundary_options`. In particular,
`cutoff="auto"` uses the shared dtype-aware policy and `cutoff_mode` is
forwarded to Quimb's boundary SVD via `compress_opts`.

`PepsOptimizer.run()` now defaults to `k_2q_batch="auto"`. It absorbs gates
in circuit order until the next gate would make any target bond exceed its budget:
`2*chi` for diagonal two-qubit gates such as RZZ, increasing to `4*chi` when
another kind of two-site gate enters the batch.
Shared sites do not stop a batch. For a nearest-neighbor RZZ layer with one
gate per edge, every bond grows by at most two, so the entire layer fits in
one target within `2*chi`. Long-range routes or repeated gates can grow the
same bond again; actual target bond dimensions determine the stopping point.
The gate that ends a batch remains queued for the next batch.

For dense NumPy/Torch/CuPy diagonal qubit gates on nearest-neighbor coordinate
sites, the exact split uses the algebraic rank ceiling `2*current_bond`.
This removes redundant SVD dimensions without an approximation cutoff; a
diagonal qubit operator is a sum of at most two product operators. Other
gates use unrestricted exact splits. Native symmetry arrays and JAX/traced
or trainable gates conservatively use the `4*chi` budget; their zero entries
are not used to infer a smaller rank. Routed and physical-index gate splits
also retain the unrestricted exact path. These are upper thresholds, not
requests to pad every bond to twice or four times `chi`.

One-site gates inside an automatic batch are absorbed in their original order,
including after its last two-site gate; they do not count toward the two-site
limit. Leading one-site gates are applied directly. Gates are never reordered.
Use `k_2q_batch=1` for the previous behavior, or `2`, `3`, etc. to absorb up to
that many two-site gates plus intervening one-site gates. Integer batches may
contain overlapping gates and do not impose the automatic target bond budget.

A general two-qubit gate can require up to `4*chi`, and routed gates can grow
several bonds. If even the first gate exceeds its budget, it is processed alone
with its exact target intact. The batching budget never truncates a target.
Step records expose `k_2q_batch`, `batch_stop_reason`, `batch_target_bond_limit`,
`two_site_batch`, and `target_max_bond`, including this singleton exception.
Integer batch sizes and the singleton fallback can therefore exceed that budget
before compression. The retained state is still capped at `chi`.
A candidate target is built to check its dimensions, so the budget is a batching
threshold rather than a bound on peak temporary allocation.

Each accepted batch is compressed to `chi` and optionally refined by the
sweep/global optimizer. Two-site targets use
`cutoff=0`, no approximation bond cap, and no path compression, even when `run(cutoff=...)`
requests truncation for the warm start. Inherited `bond_dim` and final gate `chi` are disabled
for exact targets. An explicit truncating value, including final `chi`, in
`target_gate_kwargs` raises `ValueError`; this includes the `bond_dim` alias.
Warm-start `compress_all` receives both
the requested `cutoff` and `cutoff_mode`; one-site gates continue to use the
run's gate options directly. Infidelity estimates must be finite; a
substantially negative value is retried as described below, then raises if it
remains invalid. Physical-index strings such as `("k0,0", "k0,1")` are
supported without forwarding coordinate-routing options to the split routine.

## Profiling

### Reusing checked boundary environments

For dense Torch states without gradients and dense CuPy states, adaptive checking defaults to
`boundary_convergence={"reuse_environments": True}`. The final checked norm
and overlap boundary MPS are handed to the row/column fitter with the same
target index labels. The next sweep extends these cuts instead of starting
every boundary again. Use `reuse_environments=False` for a fresh-fit comparison.

Every left and right cut is validated against the tensors in its next slice,
its preceding boundary, and its compression policy. A changed row or column
invalidates all cuts that depend on it; cuts outside that region can be reused.
Changes to Torch data (including versioned in-place writes), indices, chi,
solver budget, cutoff, or boundary data invalidate the corresponding entries.
Network scale exponents remain outside this cache and are applied afresh.
Index/topology changes conservatively create new boundary guesses.

This does not skip chi probes, the final scalar contractions, or independent
fresh DMRG confirmation. Only the latest checked handles per axis are retained
across gate batches; no full trajectory is cached. A full row/column sweep can
change every tensor, so across-gate reuse is most effective for local updates.
NumPy, JAX, Symmray, trainable Torch arrays, and Torch inference tensors
currently use the original path because this cache needs reliable mutation
tracking. Do not mutate Torch arrays through unversioned `.data` writes.
Sweep result `environment_reuse` reports cache hits and rebuilds.

### Two-site full update

Select the update explicitly with `mode="sweep", update_style="two-site"`.
The default `update_style="row-column"` retains row/column variational fitting;
`"row"` selects x slices and `"column"` selects y slices. Those choices retain
the existing local solver and sweep budgets. `mode="full-update"` is also
accepted as a two-site alias. No separate one-site ALS mode is introduced.
The two-site path follows the reduced-tensor scheme
in [Lubasch, Cirac and Bañuls, Sec. III B](https://arxiv.org/pdf/1405.3259).
Currently it supports nearest-neighbor coordinate pairs on dense Torch or CuPy
complex64/complex128 PEPS, with Pepsy boundary MPS (direct or DMRG fitting).
It does not support differentiation through the update or native symmetry
arrays. `k_2q_batch="auto"` means one two-site gate in this mode; integers
greater than one are rejected. For row/column fitting, `k_2q_batch=1` explicitly
selects gate-by-gate fitting; `"auto"` retains automatic batching.

```python
optimizer = PepsOptimizer(
    state, gates, chi=4, mode="sweep", update_style="two-site",
    gate_order="column",
    fit_mode="eff",                         # DMRG boundary compression
    boundary_convergence={"max_chi": 128, "rtol": 1e-5},
    full_update_kwargs={
        "max_iterations": 50, "rtol": "auto",
        "rcond": 1e-12, "gauge": True, "solver": "auto",
    },
)
state = optimizer.run(
    k_2q_batch=1, normalize_final=True,
    measure_infidelity=False, measure_final_infidelity=False,
    accept_if_improved=False,
)
```

QR/LQ separates fixed external factors from the two reduced tensors. The
exact gated pair supplies the target; only its retained shared bond is
truncated to `chi`. The untouched row/column strip is contracted using the
checked boundary MPS. Its reduced norm matrix is Hermitianized, then negative
eigenvalues are set to zero. Independent gauges from its square root improve
conditioning when invertible; singular gauges use the ungauged path.

Reduction, PSD projection, and ALS reuse `pepsy.bp.reduced_update`; this path
does not run BP or use a BP environment. `solver="auto"` delegates to Quimb's
public ALS fitter, with the existing weighted-QR fallback. `solver="quimb"`
requires that route, while `solver="qr"` requires weighted least squares.
All new contractions receive Pepsy's Cotengra optimizer. The shared solver
starts from the gauged target's SVD. Its result is compared against both
ordinary and gauged SVD guesses in the same positive metric, retaining the
lowest cost. The final internal bond is balanced with SVD.
External PEPS tensors remain unchanged by the pair fit.
The environment gauges use two independent unfoldings of the same square root.
Internally `N = root.H @ root`, so the paper's QR/LQ construction for
`N = X @ X.H` is expressed as two QR factorizations. No inverse is applied
when either gauge fails the relative singular-value check. The weighted-QR
ALS route regauges reduced factors after each local solve. The Quimb route
uses its public native ALS routine for each complete sweep, retaining the same
open overlap networks, then applies QR/LQ regauging between sweeps. Tensor
arrays and linear algebra remain on the input backend; only scalar diagnostics
are read on the host.
Requested output normalization still runs at the selected norm chi, on the
same device and dtype. Independent pre/post checks remain controlled by the
usual `measure_infidelity`, `measure_final_infidelity`, and
`accept_if_improved` options. Adaptive calibration remains active without them.

`full_update_kwargs` owns the ALS controls, not SciPy/LBFGS sweep options.
With `boundary_convergence=False`, `boundary_chi` (first entry for a pair)
sets the reduced norm environment cap independently of `evaluation_chi`.
Its boundary cache persists across gates and `run()` calls. Disable both
independent infidelity measurements and acceptance checks for a norm-only
workflow; ALS still evaluates its local target in the positive norm metric.
`norm_environment_chi` records the cap actually used.
Paired `normalize_chi` settings use their first component, including the
default `(4*D, 5*D)` when adaptive convergence is disabled. Target normalization
honors `normalize_target` and its `non_unitary` default independently of final
output normalization. Its global magnitude is represented by a network
exponent shared by the reduced target and its reconstructions, without
rescaling exterior site tensors during the pair fit.

Full-update acceptance checks retain raw pre/post infidelities. Values clipped
for display cannot reject an ALS candidate, and a successful fixed-cap
precheck retry supplies the same effective chi to the postcheck. Step records
include `raw_pre_infidelity`, `raw_post_infidelity`, the effective evaluation
cap, and the corresponding evaluation records.

Within the active row or column, exact prefix/suffix contractions are cached
in addition to the transverse boundary MPS cuts. Each partial contraction is
validated against the source PEPS tensors, boundary tensors, their mutation
stamps and index layouts, and the preceding partial contraction. Torch uses
version counters. CuPy has no version counter, so it compares exact snapshots
on the device, transferring only a boolean. This adds comparison work and one
device snapshot per live tracked array; weak references release snapshots
when the corresponding arrays die. In-place writes through views invalidate
dependent entries. CuPy runs use the PEPS device context. Moving to
another strip discards its entries, keeping storage bounded to one strip.
All partial and final strip contractions use the supplied Cotengra optimizer;
the default reusable optimizer also caches contraction plans. No scalar norm
is cached. `optimizer_result['strip_environment_reuse']` reports hits and
rebuilds. Adaptive `reuse_environments=False` disables this strip cache too.
Fresh adaptive confirmation can replace boundary arrays and therefore invalidate
partial contractions even if their numerical values happen to agree.

`rcond=None` selects 1e-12 for complex128 and 1e-6 for complex64. Diagnostics
include the raw environment's anti-Hermitian residual, minimum eigenvalue,
negative spectral weight, gauge status, accepted ALS loss history, and cache
counts. `solver` reports the actual route, and `solver_costs` retains its cost
diagnostics. The default `rtol="auto"` reuses Pepsy's FIT tolerance policy:
1e-9 for complex128 and 1e-5 for complex64. After each complete ALS sweep, stop
when either the squared residual divided by the target norm, or the absolute
change in that normalized cost, is at most `rtol`. This is a local objective
criterion, not a bound on the global state error. `rtol=None` or zero disables
tolerance stopping; `max_iterations` always bounds the solve.

`iterations`, resolved `rtol`, `converged`, and `termination_reason` report
the observed stopping condition. Reasons include `residual_tolerance`,
`cost_tolerance`, `max_iterations`, and `solver_candidate_rejected`.
`solver_termination_reason` and `solver_converged` retain the underlying solver
status if its result is rejected in favor of a better SVD guess. A retained
guess that itself meets the residual tolerance reports convergence.
`success=True` means a finite usable update was returned, even if the iteration
budget was exhausted. Boundary convergence is recorded separately.
Positive projection stabilizes the local solve; it is **not** proof
that boundary contraction converged. The reported local positive-environment
fidelity, and its accumulated product, are not exact global fidelity.
Timing separates target preparation, chi calibration, reduced environment
construction, ALS, and output normalization.

Set `full_update_kwargs={"refine_sweeps": 1}` to add an optional one-site ALS
pass over a completed row or column after its two-site full updates. The
default is zero. Contiguous gates on the same strip share one exact,
untruncated block target, captured before their pair updates. A change of
strip, a repeated bond, or the end of the run closes the block. One-site
gates also close it unless smart scheduling retains in-strip rotations as
described below. Each bond appears at most once, bounding target bonds to at most
`4*chi` for qubit gates. No additional gate reordering is performed here.
Exterior site arrays and retained bond dimensions stay fixed. This is a
variational strip fit against the gated state, not a Hamiltonian ground-state
DMRG calculation. It can improve compression beyond the gated pair, but does
not guarantee smaller accumulated error against an ideal whole circuit.

The refinement uses Quimb's public native ALS with Hermitian local norm
matrices and positive eigensystem solves. Separate norm and overlap boundary
MPS and prefix/suffix caches are reused across site updates; contractions use
the supplied Cotengra optimizer. Torch/CuPy tensors remain on their device.
The exterior boundaries stay fixed within a refinement; adaptive calibration,
when enabled, is repeated against the block target beforehand. Dense local
solves can be expensive at large bond dimension. `refine_rtol="auto"` uses the
same dtype tolerance policy as pair ALS; zero or `None` runs the requested
pass budget. Successive passes alternate direction. Only a finite candidate
with lower cost and non-worsening fidelity in the fixed boundary estimate is
retained. Invalid or worsening later candidates preserve the best earlier
state. `strip_refinement` on the closing gate's step record contains the
target size, strip sites, pass count, costs, before/after infidelities,
acceptance, convergence, and cache statistics. Requested final normalization
and final metric measurements run after accepted refinement.

Every two-qubit full update retains `local_fidelity` and `local_infidelity`
in `get_step_records()`, even with both metric measurement flags disabled.
These use the already-built positive environment for the chosen pair result;
if outer acceptance rejects ALS, they describe the retained SVD warm start.
They precede any optional strip refinement and remain separate from its
block-target score. They require only reduced-tensor contractions, with no
additional whole-lattice metric evaluation. `local_fidelity_convention`
records this interpretation. An unavailable estimate is `None`.

`full_update_kwargs={"accumulate_local_infidelity": True}` (the default)
additionally records `accumulated_local_infidelity = 1 - product(F_gate)` and
`accumulated_local_gate_count`. Accumulation uses logarithms and `expm1` for
small errors; a zero fidelity produces accumulated infidelity one. A missing
estimate makes the cumulative result unavailable until traces are reset.
Set the option to `False` to retain only individual estimates. Accumulation
persists across runs with `reset_traces=False` and otherwise resets per run.
This product is a diagnostic, **not an exact global fidelity or error bound**.
Single-site gates do not add records. The older measured progress getters
retain their existing behavior; an accepted strip refinement without a final
measurement has `final_infidelity=None` and adds no measured progress point.

`gate_order="column"` or `"row"` traverses strips in a snake order, improving
the opportunity to retain cuts on either side of local updates. Each commuting
block finishes the preferred bond orientation before visiting transverse
strips. A block may contain arbitrary fixed two-qubit gates on disjoint sites;
overlapping gates may be reordered only when both are exactly diagonal.
Thus a disjoint circuit-depth layer can be sorted without assuming that
overlapping non-diagonal gates commute. Single-site, trainable, or unsupported
gates remain barriers, and a conflicting gate starts a new block.
This preserves the exact circuit; different truncation order can still change
the approximate result. The default `"input"` preserves the original order.
The traversal is a cache-friendly heuristic, not a globally optimal scheduling
search. When depth boundaries must remain strict, submit one depth per run.
`optimizer.last_gate_order` records one-based original gate positions in
execution order, and the supplied gate queue is restored even after failure.
`run(update_style=..., gate_order=...)` can override both options temporarily.

For more scheduling freedom, opt into `gate_order="smart"`:

```python
optimizer = PepsOptimizer(
    state, gates, chi=4, mode="full-update", gate_order="smart",
    full_update_kwargs={"refine_sweeps": 1, "accumulate_local_infidelity": True},
)
```

This policy builds ordering dependencies for overlapping gates whose
commutation is not established. Disjoint fixed gates can move independently.
Exact entry patterns recognize gates in `span(I, P)` for one- and two-qubit
Pauli products, including RX/RY/RZ and RXX/RYY/RZZ, plus general diagonal
gates. Pauli commutation uses the parity of different nonidentity axes on
shared sites: XX and YY commute on the same bond but generally do not on
bonds sharing only one site. There is no near-commuting numerical tolerance.
Unknown dense gates retain their overlapping dependencies; trainable gates,
unsupported arrays, selectors, and non-neighbor gates remain global barriers.
Smart scheduling requires dense NumPy, Torch, or CuPy state arrays.

Among legal next pair updates, the scheduler prefers the active strip and
unused bonds within its current refinement block. It pulls in required
single-site operations before their next dependent pair gate, while deferring
unrelated rotations. It compares greedy row-first and column-first schedules
by refinement-block count and then strip changes. This is a heuristic; it
does not globally minimize contraction cost. Dependency storage can grow
quadratically with the number of gates sharing sites. Submit one depth per
run when depth boundaries must remain strict.

Adjacent fixed single-site gates on the same site and native backend/dtype/
device are fused in execution order. Two-qubit gates are never fused, so each
retains its own local fidelity record. Transpose/dagger gate options disable
single-site fusion to preserve product order. In full-update mode, supplied
single-qubit matrices are converted to the PEPS backend/dtype/device before
absorption, just as pair gates are. No PEPS tensor is converted to NumPy.
Commutation checks use native small gate arrays; only scalar decisions reach
the host. Shared gate identities and comparison constants are reused within
compilation, and reclassified on each run to observe gate mutations.

With smart scheduling, a refinement block may span absorbed single-site
gates inside its active strip. These gates also act on the retained exact
block target. A repeated bond, change of strip, unsupported gate, or end of
the queue still closes the block. Single-site gates outside the strip also
close it. Two-qubit fidelity counts and the exact-target bond bound are
unchanged. Reordering still changes where truncations occur and can change
the approximate output, despite preserving the exact circuit.

`last_gate_order['original_step_groups']` maps each executed entry to all of
its original one-based gate IDs, including fused rotations. The existing
`original_steps` list contains the first original ID of each executed entry.
`input_gate_count`, `compiled_gate_count`, and `single_qubit_gates_fused`
summarize compilation. Step/refinement records retain their full applicable
original gate IDs. Execution `step`, `start_step`, and `end_step` use the
compiled queue positions. The user's original queue and gate arrays remain
unchanged, including after a failed run. Input, row, and column policies keep
their existing scheduling behavior.

When the driver performs a final acceptance check, the sweep's duplicate final
diagnostic is skipped unless custom debug diagnostics are requested. For
direct boundary compression with matching caps and the default metric policy,
the sweep also reuses the outer initial infidelity. Customized metrics,
different caps, and iterative boundary fitting keep separate initial checks.
The outer postcheck after candidate normalization remains authoritative.
For objective-only sweeps, pass `measure_infidelity=False`,
`measure_final_infidelity=False`, and `accept_if_improved=False` to `run`, plus
`sweep_optimize_kwargs={"compute_initial_loss": False, "compute_final_loss": False}`.
This disables independent whole-state overlap checks, retaining local fitting
safeguards and the default output normalization. There is no independent
comparison to the warm start or measured whole-state fidelity in this mode.

Sweep summaries mark `initial_loss_reused` and `final_loss_measured`; a skipped
internal diagnostic is not reported as a fresh fidelity measurement.

Use `run(timing=True)` to profile target construction (`target`), warm-start
compression (`compression`), normalization, fidelity checks (`fidelity`), and
variational cleanup (`sweep` or `global`). `get_timing()` returns the latest
run's `seconds`, `calls`, `total_seconds`, `status`, and `synchronized` flag,
including work completed before an exception. Each gate-batch record includes
its own `timing` phase deltas. Run totals also include normalization outside
gate batches; unclassified orchestration and one-site gates are included in
`total_seconds`, so phase sums need not equal the total.
`run(step_callback=callback)` sends a detached scalar record to the callback
after each completed two-site batch, so long runs can stream diagnostics before
the whole gate queue finishes. Callback errors propagate to the caller.

With timing enabled, each sweep `optimizer_result` includes a `timing` summary
with `boundary_seconds`, `optimize_seconds`, and scalar per-slice records.
These reuse SweepOptimizer's existing timers: boundary updates and local
objective/solver work are subsets of the outer sweep time and exclude setup,
whole-state diagnostics, and some environment preparation. Do not add these
inner totals to the outer phase totals.

`timing_sync_device=True` synchronizes supported accelerators at outer phase
boundaries, only when `timing=True`. This can affect runtime and is intended
for profiling. Inner slice timers remain unsynchronized host wall times and
are explicitly marked as such. Timing off adds no accelerator barriers and
does not enable extra contractions or alter optimization settings.

## Local solver defaults

The default sweep solver is NLopt `LD_LBFGS` for dense and Torch-backed Symmray
states. Torch supplies autograd derivatives for Torch parameters; NumPy
parameters use finite differences. The optional NLopt dependency is provided
by the `solvers` installation extra.

```python
from pepsy.optimizers import PepsOptimizer

optimizer = PepsOptimizer(
    state, gates, chi=32,
    optimizer="nlopt",
    optimizer_options={
        "algorithm": "LD_LBFGS",
        "maxeval": 50,
        "ftol_rel": 1e-9,
        "ftol_abs": 1e-9,
        "xtol_rel": 1e-9,
        "restore_best": True,
    },
)
result = optimizer.run(k_2q_batch="auto")  # or 1, 2, 3, ...
```

`maxeval` limits optimizer objective evaluations per local slice solve.
`ftol_rel` and `xtol_rel` set relative stopping tolerances for objective and
parameters; `ftol_abs` sets an absolute objective-change tolerance.
`restore_best` retains the best valid iterate. Finite-difference
gradients require extra underlying contractions per objective callback.
These limits do not bound the full gate stream. Sweep cleanup uses one global
cycle over `y,x`, four round trips per axis, and ten boundary FIT iterations
per move. Override the schedule with `sweep_optimize_kwargs`; override solver
controls with `optimizer_options` or per-run
`sweep_optimize_kwargs={"optimizer_options": {...}}`.

## Fidelity evaluation

With adaptive sweep checks, the target norm is measured during calibration
and reused throughout that fit, overriding the unitary shortcut below.

For global mode or `boundary_convergence=False`, default unitary evolution
passes `norm_target=1` to every
pre/post fidelity evaluation and to variational cleanup, avoiding the
enlarged target's norm contraction. This assumes that the incoming retained
PEPS is normalized and the gates are unitary. The compressed candidate's
norm is still measured at `evaluation_chi`.

Set `infidelity_kwargs={"norm_target": None}` to explicitly measure the
target norm, or supply a known scalar/scaled norm. Constructor settings and
per-run overrides are supported. `non_unitary=True` or
`normalize_final=False` defaults to measuring target norms. If initial
normalization is disabled, the caller must supply a normalized initial state
for the unit-norm assumption. Direct `estimate_infidelity(state, target)`
calls still measure both norms by default, since arbitrary input states need
not be normalized. Known or measured target norms from the precheck are
forwarded to variational cleanup.

The delegated sweep's initial/final diagnostics reuse that same target norm,
so optimization also avoids duplicate target norm contractions inside
the sweep. Explicit diagnostic overrides and exact debug metrics remain
available. With `measure_infidelity=False`, optimization still needs a valid
objective: an unknown target norm is measured once using `evaluation_chi`
and reused by sweep/global cleanup. A known norm, including the unitary
default of one, avoids that measurement. Disabling both measurement and
optimization needs no target norm contraction.

An inconsistent finite-cap contraction can still produce a negative
infidelity. Outside adaptive sweep fits, `evaluation_max_retries=2` allows up to two retries,
each using the same cap for both norms and overlap, twice the preceding
maximum cap. Every retry warns. Use `evaluation_max_retries=0` to enforce
strict caps. Exact contractions, nonfinite estimates and explicitly supplied
norms do not trigger automatic cap growth. Persistent invalid estimates raise.
Adaptive sweep postchecks use the calibrated cap without these extra retries,
so evaluation cannot silently exceed the configured convergence ceiling.
The default unit-target-norm path likewise does not retry or silently enable
target contraction. A negative estimate can indicate insufficient
`normalize_chi`; increase normalization/evaluation accuracy or opt into
target-norm measurement. Output normalization at finite chi is an
approximation to exact unit norm.
Approximate estimates outside [0,1] by at most `evaluation_negative_tol=1e-3`
are accepted immediately with a warning and clipped to [0,1] for
decisions and fidelity bookkeeping. The historical option name now covers
both ends of the interval. This absolute allowance is 0.1 percentage point and does not
certify a perfect overlap. It avoids retries and stopping for small boundary
contraction discrepancies, including when the target norm is supplied as one.
In sweep mode, a precheck clipped beyond ordinary dtype roundoff does not
satisfy `infidelity_tol` or become a zero baseline for improvement comparisons.
Its raw value is passed into delegated sweep diagnostics when reused; local
sweeps continue and choose best states from physically admissible raw losses.
This does not establish accuracy of the finite-chi metric or an improvement
over an unreliable precheck. A sweep with all local updates rejected reports
`optimizer_failed`, preserves the compressed warm start, and retains the
individual invalid-loss records; it does not abort gate-stream evolution.
The dtype roundoff scale (`1e-12` for double, `1e-6` for single precision)
remains a lower bound and is cleaned silently. Set `evaluation_negative_tol=0`
on `PepsOptimizer` for the previous roundoff-only policy. Exact contractions
always use that stricter policy. Larger invalid values retain the retry/error
behavior above; nonfinite values still raise.

`get_evaluation_records()` reports all attempted caps and raw errors, including
`clipped_negative`, `clipped_infidelity`, `raw_infidelity`, and `negative_tolerance` when the
allowance is used. Step records also embed their `evaluation_records`, so
streamed/saved batch diagnostics preserve these raw values. They retain the
requested `evaluation_chi` plus `effective_evaluation_chi`.
The same policy applies to the returned global and sweep optimizer losses,
so a small negative inner diagnostic cannot abort the gate stream before the
outer acceptance check. Optimizer summaries retain `raw_infidelity`, bounded
`infidelity`, and `clipped_infidelity` when clipping occurs. Raw solver losses
remain unchanged. This does not suppress solver errors or change solver
convergence tolerances; valid best-iterate recovery and acceptance still apply.
The tolerance is also forwarded to `SweepOptimizer` unless explicitly
overridden in `sweep_kwargs`. Sweep summaries retain `clipped_loss_records`
with raw and bounded local losses. Fidelity bookkeeping uses Autoray clipping
to `[0, 1]`; the differentiable sweep objective remains unclipped.
Records include excursions below zero and above one, at either the initial
or final local evaluation.

Row/column objectives use a shared reusable Cotengra optimizer from
`pepsy.tensors.build_optimizer()` (the canonical name of `build_contraction`),
rather than inheriting the boundary `contraction_opt` string preset. The same
search cache serves successive gate batches and `run()` calls. Override it
with `sweep_kwargs={"local_contraction_opt": optimizer}` when needed; boundary
contractions can be configured separately. The default `contraction_opt=None`
also builds this helper for normalization, overlap checks, boundary FIT, and
global refinement. The default driver shares that object with its local sweep
objectives; there is no `auto-hq` or `greedy` preset fallback in this policy.
Explicit caller overrides still take precedence.

**The clipping allowance is a continuation policy, not an accuracy bound.**
A negative approximate loss beyond dtype roundoff is excluded from sweep
early stopping and best-iterate selection even when its reported value is
clipped to zero. A small positive approximate loss can still be inaccurate.
For accurate
comparisons, inspect raw diagnostics and check convergence as both
normalization and norm/overlap evaluation caps increase, using exact small
systems as references where practical. Setting the clipping allowance to zero
restores stricter validity checks but does not certify finite-cap accuracy.

Candidate acceptance compares pre/post states at a common effective cap; if
the postcheck needs a larger cap, the saved warm start is remeasured there
without further retries. No automatic retry guarantees contraction accuracy,
and positive finite-cap errors still need convergence checks when precision
matters. Norm recomputation and retries increase diagnostic contraction cost.

`run()` defaults to `cutoff="auto"`, `cutoff_mode="auto"`, and
`infidelity_tol="auto"`. Each run resolves these from the current PEPS array
dtype, without transferring tensor data to the host:

| PEPS precision | Warm-start cutoff | Infidelity threshold |
| --- | --- | --- |
| float64 / complex128 | 1e-12 | 1e-9 |
| float32 / complex64 | 1e-6 | 1e-5 |
| 16-bit floating point | 1e-3 | 1e-3 |

Automatic cutoff mode is `"rsum2"`. The cutoff uses the shared MPS/gate
truncation policy, while the infidelity threshold uses the MPS FIT tolerance
scale. The latter decides whether to skip refinement of the compressed
warm start; it does not guarantee the final infidelity reaches that value.
Explicit nonnegative finite numbers retain precedence, including zero.
`cutoff_mode=None` remains equivalent to `"auto"`. Policies are resolved anew
after `set_state`, and gate/batch step records include the resolved `cutoff`,
`cutoff_mode`, and `infidelity_tol`. Exact post-gate targets remain untruncated.
The prior fixed settings remain available through
`run(cutoff=1e-12, cutoff_mode="rsum2", infidelity_tol=1e-10)`.

By default, the initial PEPS is normalized once on the first `run()` and
evolution is assumed unitary (`non_unitary=False`). Exact gate targets are
not rescaled. Warm starts and optimized candidates are normalized using
`normalize_chi`; so are targets retained directly because they fit within
`chi`, and the final output after standalone one-site gates. Rejected
optimization restores the already normalized warm start. Thus the retained
state has unit norm according to the selected finite-cap contraction, which
does not guarantee an exactly unit dense norm at insufficient boundary chi.
Increase `normalize_chi` to check convergence. `normalize_final=False` skips
candidate/direct-output normalization; warm starts are always normalized.
`normalize_target=None` follows `non_unitary`; set `non_unitary=True` for
nonunitary evolution or explicitly override `normalize_target=True/False`.
Adaptive sweeps measure the unchanged target norm before fitting. Outside
that path, default unitary fidelity evaluation uses target norm one without
contracting or rescaling that target; explicit target-norm measurement
remains available to account for finite-boundary normalization error.
The sweep receives an already normalized warm start and skips duplicate
constructor normalization. Explicit `sweep_kwargs={"renormalize_state": True}`
enables it, using `normalize_chi` independently of `boundary_chi` unless
`sweep_kwargs["renormalize_kwargs"]["chi"]` overrides that initial cap.
Standalone `SweepOptimizer` normalization and metric defaults are unchanged.

If sweep cleanup reports an invalid initial boundary estimate, the driver
retains the already-normalized warm start and records `reason="optimizer_failed"`
and `optimized=False`, even with `accept_if_improved=False` or fidelity
measurement disabled. `optimizer_result` includes the raw initial estimate
and `termination_reason="invalid_initial_loss"`. The failed cleanup adds no
postcheck or normalization contraction. This is a safeguard, not an automatic
boundary-accuracy correction during the local fit: the already selected caps,
FIT iterations, and target-norm value remain unchanged until the next fit's
calibration.

In `mode="global"`, the default optimizer is NLopt `LD_VAR2` with an evaluation
budget of 1200, using Torch autodiff through Quimb MPS contractions. Override
it via `global_optimize_kwargs` or the shared optimizer controls.
Global norm/overlap evaluation uses Quimb MPS `contract_boundary` by default.
With Torch autodiff, its global contraction cutoff is `1e-10`, independently
of the gate/warm-start cutoff. Pepsy registers its stabilized Torch SVD/QR
policy before optimization. Explicit global `norm_kwargs`, `normalize_kwargs`,
`loss_kwargs`, or `loss_opt` cutoffs override these defaults. Outer acceptance
and driver normalization retain their own configured contraction policies.

Selecting JAX supplies the JIT-compatible global loss defaults:

```python
optimizer = PepsOptimizer(
    state, gates, chi=chi, mode="global",
    global_optimize_kwargs={"autodiff_backend": "jax"},
)
```

This selects `jit_fn=True`, global loss `cutoff=0.0`, and
`strip_exponent=False`. Zero cutoff keeps SVD ranks independent of traced
singular values while still applying the requested boundary bond caps.
Exponent stripping is disabled because its current scalar conversion loses
JAX gradients and cannot be traced. This also applies when selecting JAX
through `optimizer_options` or per-run global options. Explicit
`global_kwargs["loss_kwargs"]` / `loss_opt` values and `jit_fn` take
precedence; positive cutoff with JIT or exponent stripping with JAX remain
unsupported combinations. These defaults apply to the optimization loss;
outer normalization and fidelity measurements retain their own settings.
Without stripping, very large contractions can overflow or underflow.
For complex128 calculations, enable JAX x64 before creating JAX arrays.

Before global optimization constructs or traces its loss, the driver calls
`register_jax_linalg(stabilized=True)` for JAX. This installs Pepsy's
truncation-safe SVD backward rule in Autoray; QR retains native JAX autodiff.
The SVD rule restores truncated cotangent shapes and delegates derivatives
to JAX, so it does not provide Torch's relative regularization at degenerate
singular values. Registration also runs for JAX with JIT disabled and for a
JAX-selected fallback. These Autoray registrations affect the current process.
Torch continues to use its existing `TorchLinalgConfig` SVD/QR policy.

Returned candidates retain the input backend, dtype, and device. The PEPS
driver owns final normalization: delegated global normalization is disabled,
so `normalize_final=True` normalizes the candidate once and `False` skips it.
Use the driver's `normalize_kwargs` and `normalize_chi` for this operation;
delegated `global_kwargs["normalize_kwargs"]` does not control driver output
normalization.

Global NLopt cleanup restores its best finite evaluated vector. An early
NLopt stop is reported in `optimizer_result`, including `stopped_early`,
`best_restored`, and `returned_loss`. The raw evaluation history's endpoint
remains in `loss_final`; `optimizer_infidelity` uses the returned state's
score instead. If recovery has no valid iterate, the driver retains the warm
start with `reason="optimizer_failed"`, including when acceptance checks are
disabled. Outer postchecks continue to decide whether a recovered candidate
improves on the warm start.

The FIT controls can be supplied directly to `PepsOptimizer`, matching the
`SweepOptimizer` names, for example `fit_mode`, `fit_layer_mode`,
`fit_layer_order`,
`fit_init_strategy`, `fit_sweep_sequence`, `fit_rtol`, and `fit_timing`.
Direct values override matching entries in `boundary_kwargs`; the mapping
form remains supported for compatibility and for other boundary metric
options.

## Boundary DMRG modes and SRC guesses

Dense PEPS boundaries can select the one-site/effective DMRG path with
`fit_mode="dmrg"` or `"dmrg1"`, mixed two-site/one-site DMRG with
`fit_mode="dmrg2"`, fixed two-site FIT with `fit_mode="two-site"`, or Quimb
compression with `fit_mode="direct"`, `"src"`/`"src-mps"`, `"zipup"`,
`"sdc"`/`"sdcr"`, or `"dm"`, including their supported `*-first` and
`*-oversample` variants, through `boundary_kwargs`:

```python
optimizer = pepsy.PepsOptimizer(
    state,
    gates,
    chi=32,
    boundary_chi=(64, 96),
    boundary_engine="dmrg",
    boundary_kwargs={
        "fit_mode": "dmrg2",
        "fit_layer_mode": "joint",
        "fit_init_strategy": "guess-src",
        "fit_init_seed": 7,
        "fit_sweep_sequence": "LR",
        "fit_rtol": 1e-8,
        "fit_min_iter": 2,
        "fit_patience": 2,
        "cutoff": 1e-12,
    },
)
```

`fit_init_strategy` selects only the disposable FIT initial guess:
`PepsOptimizer` defaults to `"guess-src"` (SRC initialization), with a boundary
FIT iteration budget of `n_iter=10`. The spelling `"src"` belongs to
`fit_mode`; use `"guess-src"` for initialization followed by FIT refinement.
Explicit `fit_init_strategy="direct"`, `"guess-direct"`, or `"guess-sdc"`
overrides this default, as does an entry in `boundary_kwargs` when the direct
argument is omitted. Standalone boundary helpers and `SweepOptimizer` retain
their own defaults. The exact boundary target, live state,
and reusable boundary handles remain authoritative. Symmray boundaries safely
fall back to direct initialization with a warning for dense Quimb guesses.

`fit_layer_mode` and `layer_tags` use the same semantics as the lower-level
boundary APIs. Keep the default `"joint"` for the ordinary PEPS BRA--KET
double layer. Direct Quimb modes may use `"sequential"` with explicit tags;
`fit_layer_order="input"` preserves those tags, while
`fit_layer_order="auto"` estimates dense intermediate sizes and is allowed
only with explicitly tagged, mathematically interchangeable layers;
This requires `boundary_engine="dmrg"` when using `PepsOptimizer`, because the
native Quimb MPS sweep provider handles layers jointly.
The shared `boundary_kwargs` mapping forwards this policy to normalization,
infidelity checks, and the delegated `SweepOptimizer`. `fit_timing` and
`fit_timing_sync_device` are forwarded the same way. Timing records can be
retrieved with `optimizer.get_fit_diagnostics()`.

For sweep cleanup, `normalize_boundaries=False` is now the default: the
boundary-store normalization pass is a separate opt-in safety operation and
does not replace physical-state `renormalize`. The dense local sweep path also
defaults to `simplify=False`; use `simplify=True` for the full-simplification
debug/correctness route.

`boundary_kwargs` is the shared FIT policy. Metric-only options such as
`method`, `mode_`, `sequence`, and `equalize_norms` may be placed in
`normalize_kwargs` or `infidelity_kwargs`; they are not passed to the delegated
sweep constructor. `balance_bonds` is normalization-only and belongs in
`normalize_kwargs`. `boundary_options` remains reserved for the reusable Quimb
MPS environment store used by sweep cleanup.

For sweep cleanup, tuple `boundary_chi` values cap the norm and overlap
boundaries independently. Normalization and diagnostic contractions receive
the corresponding scalar `chi`. DMRG two-site boundaries start at bond 1 and
grow through local SVDs instead of global padding. `fit_mode="eff"` remains
the compatibility default while two-site accuracy and wall time are
workload-dependent. Use `cutoff="auto"` and
`fit_cutoff_mode="auto"` for dtype-aware cutoff selection; the latter
resolves to the standard `"rsum2"` policy.

For the full-chain `eff` solver, `boundary_kwargs` can instead select native
block growth followed by one-site refinement:

```python
boundary_kwargs={
    "fit_mode": "eff",
    "fit_block_size": 2,
    "fit_adaptive_sweeps": 2,
    "fit_sweep_sequence": "RL",
}
```

This schedule is passed consistently to normalization, infidelity, and sweep
boundary initialization. `fit_rtol` is optional and adds one terminal norm
check per completed sweep only when enabled.

## One Torch SVD/QR policy

Torch autodiff through PEPS cleanup uses both SVD and QR. Configure both with
one `TorchLinalgConfig` object instead of registering the individual legacy
helpers separately:

```python
import pepsy

torch_linalg = pepsy.TorchLinalgConfig(
    mode="complex",          # selects complex-safe SVD/QR rules
    stabilized=True,          # finite SVD/QR VJPs for autodiff
    svd_driver="gesvd",      # CUDA: Pepsy's exact fast default
    cpu_svd="torch",         # CPU: native Torch LAPACK
    qr_rank_policy="warn",   # warn if a real QR block is rank deficient
    quimb_split_drivers=True, # required for raw Symmray blocks
)

optimizer = pepsy.PepsOptimizer(
    state,
    gates,
    chi=64,
    mode="global",
    torch_linalg_config=torch_linalg,
)
```

`stabilized=True` changes the reverse-mode rule, not the exact forward
factorization: it regularizes singular-gap and QR-pivot terms only where the
ordinary derivative is undefined or ill-conditioned. Use `stabilized=False`
for the fastest native forward/backward path when those gradients are not
needed. Pepsy defaults to the exact CUDA `svd_driver="gesvd"` route.
Benchmark `svd_driver="gesvdj"` on your hardware if desired, or use CPU
`cpu_svd="scipy_gesdd"` for an explicit LAPACK experiment; `gesvda` is approximate and requires an
explicit `allow_approximate=True` acknowledgement.

For dense states, `quimb_split_drivers` can remain `False`. For Symmray PEPS,
set it to `True` because Quimb receives raw Torch charge blocks that do not
pass through ordinary Autoray dispatch. `PepsOptimizer` enables this flag
automatically when it detects Symmray blocks, while preserving the other
settings in a supplied policy. `register_torch_svd=False` is retained only as
a compatibility escape hatch for disabling automatic registration.

The lower-level `reg_*_torch` functions and `register_torch_linalg(...)`
remain compatibility APIs; new optimizer code should pass or construct
`TorchLinalgConfig` so SVD and QR cannot silently diverge.

`SimpleUpdateGen` preserves quimb's arbitrary-geometry simple-update sweep and
energy bookkeeping, but routes every gate through `pepsy.gate_simple(...)`.
This lets sequential simple update handle long-range PEPS terms via Pepsy's
SWAP routing. Use `route_opts` for routing controls such as `sequence`,
`path_canonize`, and `path_compress`.

## Important cautions

- A second `run()` call applies the queued gates again to the current state.
  The default `reset_traces=True` resets diagnostics only; use `set_state(...)`
  when you want to replay from a fresh input state.
- If `normalize_chi` or `evaluation_chi` is left unset, standalone
  normalization and infidelity diagnostics use `(4*D, 5*D)`, independently
  of the optimizer environment cap. Increase these explicitly for stricter
  metric contractions, at extra computational cost.
- `accept_if_improved=True` is most consistent with
  `measure_final_infidelity=True`. If final measurement is disabled, the
  fallback optimizer loss can come from the coarser `boundary_chi` environment
  while the pre-check used `evaluation_chi`.
- Sweep mode defaults to optional NLopt `LD_LBFGS` for dense and Torch-backed
  Symmray states. Pass an explicit `optimizer` /
  `sweep_optimize_kwargs` value to override this choice.
- Unitary targets are not rescaled by default. `normalize_target=None` follows
  `non_unitary=False`; use `non_unitary=True` to normalize nonunitary targets,
  or override `normalize_target` explicitly. This does not implement the
  interval scheduling or norm-proxy machinery available in `MpsOptimizer`.
- `run(mode=...)` changes the backend for that call only; use `set_mode(...)`
  to change the configured default.
- Step records and fidelity traces are per measured two-site gate batch. One-site
  gates applied outside a two-site batch are not recorded as separate steps.
  For PEPS lattice gates, prefer coordinate-tuple sites such as
  `((x0, y0), (x1, y1))` over flat integer pairs.
- `SimpleUpdateGen(update="parallel")` currently supports only direct-neighbor
  terms. Long-range routed terms need route-aware layer scheduling and should
  use `update="sequential"` for now.
