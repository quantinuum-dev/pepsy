# Belief propagation and loop expansions

The BP namespace contains message-passing, loop-expansion, and partitioned
norm-estimation helpers. It is an advanced extension and is loaded lazily by
the top-level package.


> API details are maintained as handwritten Markdown in this page.

## Native fermionic PEPO norm workflow

Native fermionic PEPOs can be evolved with `gate_simple`, including its
simple-update gauge dictionary, and their Frobenius norm can be measured with
`PEPO.norm()`. The D2BP loop-cluster route accepts the evolved PEPO directly:

```python
from pepsy import Fermion, OneDMap
from pepsy.operators.gates import gate_simple
from pepsy.bp import loop_cluster_expand

pepo = fermion.to_pepo(
    terms,
    Lx=2,
    Ly=2,
    mapper=OneDMap(2, 2, mode="snake-row-major"),
    fermionic=True,
)
gauges = {}
pepo.gauge_all_simple_(gauges=gauges, progbar=False)
pepo = gate_simple(
    pepo,
    fermion.hopping_gate(0.001, t=1.0).H,
    where=((0, 0), (0, 1)),
    gauges=gauges,
    inplace=False,
)
norm = pepo.norm()
correction = loop_cluster_expand(pepo, gloops=2, norm="2norm")
```

This path preserves native `U1`, `U1U1`, and `Z2` tensors. Simple-update
gauges are used by `gate_simple`; D2BP recomputes its own norm messages. The
`norm="1norm"` gauge route requires a closed scalar tensor network, so it is
not the direct route for an open PEPO with ket/bra physical indices.

## Reduced loop-cluster compression

For the first-pass local full-update style compression, use
`compress_reduced_loop_cluster`. It selects the active bond, fills a finite
cluster by `max_distance`, closes its cut bonds with either SU vectors or
directed D2BP matrices, and solves the reduced `L/R` problem. This works
for ordinary Quimb PEPS and PEPOs backed by NumPy, Torch, JAX, or CuPy arrays
where the selected Quimb contraction and linalg operations support that
backend. Tensor-shaped intermediates remain native; no compression step
converts the network to NumPy. For a PEPO, Quimb's lower and upper physical
legs are fused only on the private reduced-update copy and restored before the
result is returned. Pass messages from a separately solved `two_norm_bp` run
without first changing the tensor network into an SU gauge:

```python
from pepsy.bp import compress_reduced_loop_cluster, two_norm_bp

bp = two_norm_bp(peps, max_iterations=1000, tol=1e-10)
result = compress_reduced_loop_cluster(
    peps,
    where=((0, 0), (1, 0)),
    boundary_messages=bp.messages,
    max_bond=chi,
    max_distance=1,
    inplace=False,
)
```

If neither boundary data source is supplied, the helper runs a fresh D2BP
solve automatically. Use `run_bp=False` only for a system-covering cluster or
when an unresolved-boundary error is intentional. With `input_mode="auto"`, a
supplied SU gauge mapping means that `tn` is an SU core; set
`input_mode="physical"` when `tn` already contains the gauge factors and the
vectors should be used only as diagonal closures.

Set `regauge=True` to refresh SU gauges after truncation; the default returns
the compressed physical network without refreshing them. Set
`max_loop_size>0` to add the open-leg loop-cluster correction on top of the
finite base cluster. For the paper's total-region convention, use
`max_cluster_size=...`; it includes the active/base support and is mutually
exclusive with a nonzero `max_loop_size`. `tree_reduction=True` (the default)
prunes tree-like intersection appendages to the protected active/base support,
matching the fixed-point tree reduction in Algorithm 1. Set it to `False` only
when auditing the un-reduced inclusion-exclusion regions. Supplied D2BP
boundary messages are Hermitian/PSD projected by default
(`message_psd_project=True`); set `message_psd_project=False` only for
diagnostic raw-message experiments.

When a fresh D2BP solve is requested, `bp_convergence` controls an unfinished
solve: `"ignore"` keeps the historical behavior, `"warn"` emits a
`RuntimeWarning`, and `"raise"` stops before the reduced pair is built. The
selected policy and BP diagnostics are available as `pair.bp_info`.

For a cutoff sweep, reuse the topology-only loop geometry. A prepared
`ReducedBondPair` owns a cache automatically, or an explicit
`ReducedLoopClusterCache` can be precomputed at the largest desired cutoff:

```python
from pepsy.bp import ContractionPlanCache, ReducedLoopClusterCache

cache = ReducedLoopClusterCache()
plans = ContractionPlanCache()
cache.precompute(pair, max_loop_size=10)
for gloops in (0, 4, 6, 8, 10):
    problem = loop_cluster_reduced_update_problem(
        pair,
        identity,
        max_loop_size=gloops,
        loop_cache=cache,
        plan_cache=plans,
    )
```

This reuses Quimb's `gen_gloops` results for all lower cutoffs and caches the
`gen_region_counts` inclusion-exclusion regions. For repeated contractions of
the same topology, pass a `ContractionPlanCache` as `plan_cache` to
`loop_cluster_reduced_update_problem`, `compress_reduced_loop_cluster`,
`apply_reduced_loop_cluster_gate`, or `gate_loop_cluster`; it reuses the
topology-only Cotengra trees while still contracting current tensor values and
boundary messages. Inspect `cache.snapshot()` for `plans`, `hits`, and
`misses`. Large cutoffs can still have combinatorially many regions.

The current reduced open metric uses the additive operator-valued ansatz
`N_red ~= sum_C c_C N_C`. The paper's product combination is reserved for
scalar cluster observables; it is not silently applied to a matrix-valued
reduced update. This additive choice is an implementation approximation, not
a theorem obtained by differentiating the paper's scalar product formula.
The returned loop problem records the choice as
`problem.combination == "additive"`; derive and validate a parameter-dependent
objective separately if a Hessian-level justification is required.

The solver options accept `solver="auto"` (the default), `solver="quimb"`,
`solver="autodiff"`, `solver="qr"`, or `solver="normal"`, plus a relative
`regularization` value. The autodiff route uses Quimb's public autodiff
fitter after factorizing the reduced open environment, so it retains the same
weighted objective without converting tensors to NumPy.
The automatic and explicit Quimb modes use Quimb's public
`tensor_network_fit_als` with prebuilt open-leg `tnAA`/`tnAB` overlap networks;
QR mode is the regularized dense fallback, and automatic mode falls back to
normal equations for an indefinite diagnostic metric. For repeated bond
dimensions, build one reduced problem and reuse it with `solve_reduced_als`
rather than rebuilding the environment. Reduced problem builders retain the
open Quimb environment directly, so native ALS does not first form the full
`N_red` matrix. The `.metric` and `.linear_term` attributes are lazy dense
compatibility views; accessing them, or selecting a dense solver, materializes
the physical-identity-expanded metric. Native Symmray compression stays
block-sparse through the graded SVD adapter; use `solver="auto"` or
`solver="quimb"` there. Dense QR/autodiff refinement is intentionally not
available for native arrays because it would flatten charge sectors.

For explicit dense diagnostics, use `problem.dense_metric()` and
`problem.dense_linear_term()`. Direct problem builders also accept
`materialize_metric=True`, but this is unnecessary for the native Quimb path.
All reduced problem builders Hermitianize and PSD-project the smaller open
environment by default. `problem.raw_min_eigenvalue` and
`problem.clipped_eigenvalues` expose the projection diagnostics.
The three builder return types share the `ReducedUpdateProblem` type alias for
annotations. The full-system `max_cluster_size` cutoff is an exact small-system
oracle (up to the requested Hermitian/PSD projection), so a practical
convergence study compares a ladder of total cutoffs against
`exact_reduced_update_problem(..., psd_project=False)` and reports the open
environment norm error before solving ALS.

The reduced-update message keys are `(bond_index, destination_tid)`, matching
Quimb's D2BP layout. Messages are copied at preparation time and each directed
pair is normalized with Quimb's message convention before use. They are not
refreshed after a gate; rerun D2BP before another update if the fixed-point
environment needs to track the changed state.

For compatibility with gate streams, `apply_reduced_loop_cluster_gate` and
`gate_loop_cluster` accept the same closure semantics for adjacent two-site
updates: `input_mode`, fresh-BP controls, PSD message handling, and contraction
cost budgets are forwarded through the wrapper.

Pass the same `ReducedLoopClusterCache` through repeated updates of one
active bond with `loop_cache=cache` to reuse generalized-loop geometry. The
cache is topology-only; each update still rebuilds contractions and uses the
current tensor values and boundary messages.

Pass `cost_check=True` to expose Cotengra's `flops_log10` and
`peak_memory_log2` estimates. Supplying either `max_flops_log10` or
`max_peak_memory_log2` enables preflight automatically; an over-budget
contraction raises `CompressionBudgetError` before tensor values are
contracted. `on_budget="warn"` additionally emits a warning, while
`on_budget="ignore"` reports and continues.

## Selected-bond PEPS/PEPO compression

When there is no gate and one particular virtual bond should be reduced, use
`compress_bond_cluster`. It fills a cluster around that bond, contracts the
cluster to a four-leg `B_reduce` environment, and optimizes only the two maps
on the selected bond:

```python
from pepsy.bp import compress_bond_cluster, two_norm_bp

bp = two_norm_bp(peps, max_iterations=1000, tol=1e-10)
result = compress_bond_cluster(
    peps,
    where=((0, 0), (1, 0)),
    boundary_messages=bp.messages,
    max_distance=1,
    max_bond=chi,
    steps=20,
)
compressed = result.compressed
```

The same helper can run fresh D2BP when `boundary_messages` and `gauges` are
omitted:

```python
result = compress_bond_cluster(
    peps,
    where=((0, 0), (1, 0)),
    max_bond=chi,
    bp_opts={"max_iterations": 1000, "tol": 1e-10},
    cost_check=True,
)
```

`result.bond_maps[result.bond_ind]` contains the fitted `(L, R)` pair with
shapes `(D, chi)` and `(chi, D)`. These are ordinary rectangular variational
tensors; no isometry, orthogonality, or adjoint constraint is imposed. The
endpoint and spectator site tensors are fixed, and no QR/LQ split or gate is
used. All other virtual bonds are unchanged.

The default `init="b_reduce"` uses Quimb tensor contractions to form the two
bond marginals of `B_reduce` and takes their dominant `chi` subspaces as the
ALS starting maps. Use `init="projector"` or `init="random"` for simpler
initializations.

The selected-bond path uses the same native Quimb/Autoray backend policy as the
reduced path: NumPy, Torch, JAX, and CuPy tensor data remain in place when the
backend supports the requested contraction and ALS operations. Native
block-sparse Symmray tensors still require their separate graded compression
adapter.

For a radius-zero cluster, every cut bond is closed by fresh D2BP messages,
explicit directed D2BP messages, or SU vectors. SU vectors are interpreted as
`diag(lambda)` boundary closures. Set `b_reduce=True` (the default) to
Hermitian/PSD-project the
contracted `B_reduce` before ALS; `result.B_reduce` exposes the retained
four-leg environment in `(left_ket, right_ket, left_bra, right_bra)` order.
This makes the local normal equations positive while avoiding the physical-leg
fusion that produces `d^4` PEPS or `d^8` PEPO dense spaces. PEPO lower and upper
legs remain separate. The cluster contraction can still be expensive as
`max_distance` grows, so pass an appropriate Cotengra optimizer and use
`max_flops_log10` / `max_peak_memory_log2` to preflight it. The estimate is
returned in `result.contraction_cost`.

## Cut-edge loop-series compression

`compress_bond_loop_series` implements the cut-edge construction from
Evenbly et al., arXiv:2409.03108. It cuts the selected virtual bond, leaves
its four norm legs open, and resolves every other internal edge as
`I = P + Q` in the D2BP basis. `edge_cutoff` counts excited Q edges:

```python
from pepsy.bp import compress_bond_loop_series

result = compress_bond_loop_series(
    peps,
    where=((0, 0), (1, 0)),
    max_bond=chi,
    edge_cutoff=6,
    bp_opts={"max_iterations": 1000, "tol": 1e-10},
)
```

Use `cut_edge_loop_series_expand` when only the environment is needed. The
returned `terms` and `term_count_by_degree` make the convergence ladder
explicit. `edge_cutoff=0` is the BP vacuum; when the cutoff reaches all
non-cut internal edges, `complete=True` and the finite-network sum is an
exact `P + Q` identity expansion at a converged BP fixed point, up to
numerical contraction error. Partial sums need not be monotone or PSD, so
`b_reduce=True` projects the environment before ALS. This finite route
retains disconnected Q configurations explicitly; it does not apply the
paper's infinite-lattice free-energy suppression factor.

This is separate from `compress_reduced_loop_cluster`. The latter combines
operator-valued regional environments additively for the reduced ALS metric.
The scalar loop-cluster product formula cannot be promoted to
`N_red ~= sum_C c_C N_C` without deriving the parameter-dependent scalar
objective and its Hessian. For example, if
`F(theta) = sum_C c_C log Z_C(theta)`, then

```text
H_C = c_C * (Z_C**-1 * d2Z_C
             - Z_C**-2 * outer(dZ_C, dZ_C))
```

so the Hessian contains derivative and cross-gradient terms. The explicit
cut-edge series avoids that assumption by approximating the matrix
environment itself term by term.

The two cutoff names are intentionally different. In
`compress_bond_loop_series`, `edge_cutoff` is the maximum number of *other*
virtual bonds carrying `Q` in one admissible configuration. The selected
`A--B` bond is excluded from this count and is returned as the four open
environment legs. A term may be an `A--B` excitation path, a closed loop, or
an admissible path together with disconnected closed loops. `term_count`
counts these configurations plus the degree-zero BP vacuum; it is not the
number of geometric loops. For a 4x4 PEPS with one bond cut there are 23
non-cut virtual bonds, so `edge_cutoff=16` is not the complete expansion;
`edge_cutoff=None` (or a value at least 23) requests the finite complete sum.

By contrast, `max_cluster_size` in
`loop_cluster_reduced_update_problem` counts PEPS site tensors in the
augmented inclusion-exclusion region. On a 4x4 lattice, the `16` row in a
total-region benchmark means that the full 16-site region is included. The
single surviving `term` there is the full-system contraction after counting
number cancellations, not one physical loop and not an edge cutoff of 16.

When comparing a partial loop-cluster metric with an exact finite-network
metric, compare a common normalization (for example, Frobenius-normalized
matrices or trace-normalized scalar diagnostics). Partial cluster closures
use normalized BP/SU boundary objects, so their raw global scale is not
necessarily the same as the raw finite-network contraction even when the
matrix shape is converging.

## Long-range PEPS expectations

Use `compute_boundary_expectation` for batched one- and two-site operators,
including separated supports. It delegates to Quimb's PEPS boundary
environment and preserves native fermionic Symmray operators.

```python
from pepsy.bp import compute_boundary_expectation

value = compute_boundary_expectation(
    peps.tn,
    terms,                    # {(site_a, site_b): native two-site operator}
    max_bond=chi,
    normalized=True,
)
```

For a controlled finite-region estimate, use
`compute_path_cluster_expectation`. For two-site support, Quimb connects the
sites by a graph path, expands it by `max_distance`, and optionally fills
lattice corners. Supply only compatible simple-update/SU bond vectors through
`gauges`; D2BP matrix messages are not SU gauges.

```python
from pepsy.bp import compute_path_cluster_expectation

value = compute_path_cluster_expectation(
    peps.tn,
    terms,
    max_distance=1,
    fillin=True,
    gauges=su_gauges,
    max_bond=chi,
    optimize="auto-hq",
)
```

`compute_bp_path_expectation` runs Pepsy's native fermionic D2BP, converts its
messages through the tested BP-to-SU bridge, and then evaluates the connected
path cluster. Native Symmray path clusters also accept `max_bond=chi`: Pepsy
keeps the local RDM physical legs unfused, pads zero-weight charge sectors on
the private cluster copy, and lets Quimb perform its usual QR/SVD compressed
contraction. Pass `optimize="auto-hq"`, another Quimb optimizer string, or a
standard Cotengra path optimizer through either path-cluster helper.

## Local reduced density matrices

For a single route-independent entry point, use `rho_expand`. Its
`expansion` value makes the cutoff semantics explicit: `"local"` means
tensor-region loop series, `"edge"` means explicit Q-edge degree,
`"open"` means paths plus closed loops, and `"cluster"` means generalized
loop-cluster regions:

```python
from pepsy.bp import rho_expand

rho = rho_expand(
    peps.tn,
    where=((0, 0), (0, 3)),
    cutoff=6,
    expansion="open",
    normalized=True,
)
```

The dispatcher delegates to the existing tested implementations and never
converts one route's cutoff into another route's convention. Use the direct
functions below when you need their route-specific caches, diagnostics, or
corridor controls.

For a D2BP loop-series estimate of a local reduced density matrix, keep the
requested physical sites open with `partial_trace_loop_series_expand`:

```python
from pepsy.bp import partial_trace_loop_series_expand

rho = partial_trace_loop_series_expand(
    peps.tn,
    where=((0, 0), (1, 1)),
    gloops=2,
    normalized=True,
)
```

The matrix is ordered as the selected sites on the ket side followed by the
same sites on the bra side. The D2BP virtual messages and `P`/`Q` projectors
remain native for fermionic Symmray PEPS; use `rho.to_dense()` only when a
dense physical matrix is needed for inspection or an external observable.

For a mapping of Hamiltonian terms, use the scalar companion and take `.real`
when the Hamiltonian is Hermitian:

```python
from pepsy.bp import compute_local_expectation_loop_series

energy = compute_local_expectation_loop_series(
    peps.tn,
    mag_terms,
    gloops=2,
    normalized="prod",  # compatibility spelling for a normalized local rho
).real
```

Native fermionic operators are preferred for Symmray PEPS: their charge
sectors and graded swaps are handled by inserting the gate into the ket before
forming the BP double layer. Do not reconstruct a fermionic expectation as a
dense `trace(rho @ operator)`: the returned rho is useful for diagnostics, but
that contraction misses the physical graded ordering.

### Explicit edge-subset loop series

The APIs above follow Quimb's *local-region* convention: their integer
`gloops` is a region cutoff. For a brute-force-compatible expansion over the
canonical virtual Q-edge sets used by `loop_series_expand`, use the explicit
edge path instead:

```python
from pepsy.bp import compute_local_expectation_edge_loop_series

energy = compute_local_expectation_edge_loop_series(
    peps.tn,
    mag_terms,
    gloops=4,  # maximum number of explicitly excited Q edges
    normalized="prod",
).real
```

`partial_trace_edge_loop_series_expand` supplies the matching diagnostic RDM.
The scalar function is the fermion-safe choice for its supported cases.
Explicit terms can be passed as `LoopSeriesTerm` objects (or virtual-edge sets); at present they cannot put
Q on a bond wholly internal to the selected observable support. A nonzero-Q
fermionic scalar correction is currently restricted to one-site gates; the
multi-site graded-Q contraction is rejected explicitly while its block routing
is completed.

For separated sites, use `partial_trace_open_loop_series_expand` when the
explicit configuration family should include Q paths between the retained
sites. Use `edge_cutoff` to set the maximum number of excited virtual edges.
A configuration is retained when degree-one Q vertices occur only at the
selected rho sites, so the sum contains open paths, closed loops, and
path-plus-loop combinations:

```python
from pepsy.bp import (
    OpenLoopSeriesCache,
    partial_trace_open_loop_series_expand,
    two_norm_bp,
)

bp = two_norm_bp(peps.tn, max_iterations=1000, tol=1e-10)
rho = partial_trace_open_loop_series_expand(
    peps.tn,
    where=((0, 0), (0, 7)),
    edge_cutoff=8,
    messages=bp.messages,
    run_bp=False,
)
```

The edge geometry is generated lazily, with shortest support-connecting paths
yielded first. Set `max_terms`, `max_enumeration_time`, or
`max_enumeration_memory` to fail before an uncontrolled geometry expansion;
limits raise `OpenLoopEnumerationLimitError` rather than returning a partial
sum. `gloops` remains a compatibility alias, but new code should use the
route-specific names.

Use `max_loop_terms` for a separate budget on closed-loop and path-plus-loop
corrections. This lets nearby supports retain a richer loop tail while keeping
long-range measurements focused on their shortest connecting paths.

For very distant supports, set `corridor_width` to use the bounded corridor
route. It retains a small weighted-shortest-path beam, inflates those paths
by the requested graph width, and adds connected loop decorations only near
sampled corridor segments:

```python
rho = partial_trace_open_loop_series_expand(
    peps.tn,
    where=((0, 0), (999, 999)),
    corridor_width=4,
    max_path_candidates=8,
    loop_decoration_size=6,
    corridor_segment_length=32,
    max_loop_clusters_per_segment=8,
    corridor_max_bond=128,
    max_corridor_edges=100_000,
)
```

This is an explicitly controlled approximation: disconnected products of
far-separated loop clusters are omitted. Increase the corridor width,
candidate count, decoration size, or boundary bond dimension and compare the
incremental correction using the `open_rho_corridor` and
`open_scalar_corridor` diagnostics. `path_edge_weights` can supply positive
edge costs for ranking routes; unspecified edges have unit cost.

For a measurement workflow that must inspect the geometry before doing any
numerical contractions, use `diagnose_open_loop_series`. It accepts native
operators made by `Fermion`, records the selected route, paths, loop terms,
Cotengra FLOP estimates, and peak-memory estimates, and can be reused during
measurement:

```python
from pepsy.bp import (
    OpenLoopObservableTerm,
    OpenLoopSeriesDiagnosticCache,
    compute_local_expectation_open_loop_series,
    diagnose_open_loop_series,
)

term = OpenLoopObservableTerm(
    ((0, 0), (999, 999)),
    fermion.hopping_operator(),
)
diagnostic = diagnose_open_loop_series(
    peps.tn,
    term,
    mode="auto",
    edge_cutoff=2_000,
    max_terms=10_000,
    diagnostic_cache=OpenLoopSeriesDiagnosticCache(),
)
value = compute_local_expectation_open_loop_series(
    peps.tn,
    term,
    mode="auto",
    diagnostic=diagnostic,
)
```

The diagnostic phase builds contraction trees but does not contract tensor
values. `mode="auto"` selects the graded cluster-compatible route first for
cyclic native fermionic supports, and selects a corridor when the support
distance exceeds `auto_corridor_distance`. Reusing the same
`OpenLoopSeriesDiagnosticCache` lets later measurements reuse the geometry and
cost report; distinct operator values with the same support and shape do not
share numerical results. If no `diagnostic` is supplied, scalar measurement
with `mode="auto"` performs this diagnostic pass internally before starting
the numerical contractions.

For production measurements, use `on_budget="raise"` and request the
auditable result record:

```python
from pepsy.bp import OpenLoopMeasurementResult

result = compute_local_expectation_open_loop_series(
    peps.tn,
    {support: gate},
    mode="auto",
    on_budget="raise",
    measure_resources=True,
    return_result=True,
)
assert isinstance(result, OpenLoopMeasurementResult)
assert result.complete and result.bp_converged
```

The default `on_budget="report"` preserves historical partial-sum behavior
but records `complete=False` and omitted terms in `info`. `on_budget="skip"`
is available for exploratory workflows. Cotengra estimates remain preflight
guards; `measure_resources=True` additionally records observed Python and
host-RSS high-water marks. The same flags are available on
`partial_trace_open_loop_series_expand`; its result record stores the rho in
`result.value` and its trace in `result.normalization`.

For controlled convergence, use an adaptive corridor or cluster ladder:

```python
from pepsy.bp import adaptive_open_loop_series

ladder = adaptive_open_loop_series(
    peps.tn,
    {support: gate},
    corridor_widths=(0, 1, 2, 4),
    on_budget="raise",
)
value = ladder.value
```

The ladder tests numerical stabilization, not a rigorous truncation bound.
For cyclic native fermions, pass `cluster_sizes=(...)` instead. The
`diagnose_open_rho_series` helper adds physical output shape and output-memory
estimates. Rectangular PBC corridor discovery treats the virtual graph as a
multigraph, so period-two seam bonds remain distinct paths and loop edges.

This path performs an explicit configuration sum and normalizes only after
the sum; it does not apply the scalar disconnected-loop resummation used by
`partial_trace_edge_loop_series_expand`.

For a cutoff sweep, reuse both the converged messages and the two caches. The
same `info` dictionary keeps already-contracted rho terms, regional
contraction paths, and physical output labels, while the `OpenLoopSeriesCache`
keeps the eligible edge configurations:

```python
cache = OpenLoopSeriesCache()
info = {}
for cutoff in (2, 4, 6, 8):
    rho = partial_trace_open_loop_series_expand(
        peps.tn,
        where=((0, 0), (0, 7)),
        edge_cutoff=cutoff,
        messages=bp.messages,
        run_bp=False,
        cache=cache,
        info=info,
    )
```

For convergence diagnostics, inspect `info["open_rho_family_counts"]` and
`info["open_rho_family_weights"]`. The families are `open_path`,
`closed_loop`, and `path_plus_loop`; `open_rho_base_weight` is the unexcited
BP contribution. For native fermionic PEPS, treat this rho as a diagnostic
(trace and charge-block check), and evaluate fermionic operators through the
graded scalar APIs such as `compute_local_expectation_open_loop_series`.
On cyclic native fermionic graphs, those scalar and rho APIs use the
equivalent graded loop-cluster contraction internally because Symmray cannot
currently contract arbitrary mixed open ``P/Q`` configurations; the returned
rho remains native and the gate is still inserted in the ket/bra network.
For that route, pass `cluster_size`, not `edge_cutoff`:

```python
rho = partial_trace_open_loop_series_expand(
    peps.tn,
    where=((0, 0), (0, 7)),
    cluster_size=8,
    messages=bp.messages,
    run_bp=False,
)
```

The cyclic native route is selected before explicit edge discovery, so its
cluster regions do not pay the open-edge enumeration cost. Inspect
`info["open_rho_cluster_region_costs"]` or
`info["open_scalar_cluster_region_costs"]` for its contraction decisions.
The same one-BP/many-support workflow is runnable in the downstream example
`../pepsy_examples/symmetric_tensors/peps/bp_open_rho_series.py`; the
long-range native doublon comparison is in
`../pepsy_examples/symmetric_tensors/peps/bp_long_doublon_4x4.py`.

To measure a long-range operator without materializing the diagnostic rho, use
the scalar companion. It keeps the same open-path, closed-loop, and
path-plus-loop family bookkeeping, inserts the observable into the physical
native contraction, and normalizes the accumulated numerator by the
accumulated denominator:

```python
from pepsy import build_contraction
from pepsy.bp import compute_local_expectation_open_loop_series

contraction_opt = build_contraction(
    max_time=2.0,
    max_repeats=8,
    parallel=False,
)
value = compute_local_expectation_open_loop_series(
    peps.tn,
    {((0, 0), (0, 7)): fermion.hopping_operator()},
    gloops=8,
    normalized=True,
    optimize=contraction_opt,
)
```

The `optimize` object is forwarded to every Quimb contraction, so a reusable
`build_contraction` optimizer can cache Cotengra path searches across the
explicit loop terms.

For a finite contraction budget, pass `max_flops_log10` and
`max_peak_memory_log2`. Each explicit term is contracted only when both
Cotengra tree diagnostics pass; on cyclic native fermionic graphs the same
limits are applied to the equivalent graded cluster contractions.
`info["open_scalar_term_costs"]` and `info["open_scalar_skipped_terms"]`
record the decision. The corresponding open-rho API exposes the same controls
and diagnostics. For a route-independent schema, use
`open_scalar_edge_term_costs` / `open_scalar_edge_skipped_terms` and
`open_scalar_cluster_region_costs` /
`open_scalar_cluster_region_skipped_terms`; the rho API provides the analogous
`open_rho_*` fields. The older `open_*_term_costs` names remain as
route-specific compatibility aliases.

For native Symmray fermions, the scalar route keeps the physical gate in the
native ket/bra contraction and uses the graded open-bond Q projector when the
gate has local off-diagonal fermion action (for example hopping or pairing).
Diagonal density operators use the unphased open-bond projector. This
preserves the gate's graded ordering; a dense `trace(rho @ gate)` is not an
equivalent fermionic observable contraction.

For reusable application code, `partial_trace_open_loop_series_sweep` wraps
the same pattern and accepts one-site, two-site, or larger retained supports:

```python
from pepsy.bp import partial_trace_open_loop_series_sweep

result = partial_trace_open_loop_series_sweep(
    peps.tn,
    supports=(((0, 0), (0, 7)), ((0, 0), (0, 1), (0, 7))),
    cutoffs=(2, 4, 6),
)
rho = result.get_rho(((0, 0), (0, 7)), 4)
families = result.diagnostics[((0, 0), (0, 7))][4]["family_counts"]
```

`partial_trace_loop_cluster_expand` and
`compute_local_expectation_loop_cluster` provide the parallel D2BP
generalized-loop-cluster route. Its default `combine="sum"` uses the usual
inclusion--exclusion region counts; reserve `combine="prod"` for compatibility
experiments with Quimb's elementwise product convention.
