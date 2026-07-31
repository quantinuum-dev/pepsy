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
