# Belief propagation and loop expansions

The BP namespace contains message-passing, loop-expansion, and partitioned
norm-estimation helpers. It is an advanced extension and is loaded lazily by
the top-level package.


> API details are maintained as handwritten Markdown in this page.

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
sites. Its integer `gloops` is a maximum number of excited virtual edges. A
configuration is retained when degree-one Q vertices occur only at the
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
    gloops=8,
    messages=bp.messages,
    run_bp=False,
)
```

This path performs an explicit configuration sum and normalizes only after
the sum; it does not apply the scalar disconnected-loop resummation used by
`partial_trace_edge_loop_series_expand`.

For a cutoff sweep, reuse both the converged messages and the two caches. The
same `info` dictionary keeps already-contracted rho terms, while the
`OpenLoopSeriesCache` keeps the eligible edge configurations:

```python
cache = OpenLoopSeriesCache()
info = {}
for cutoff in (2, 4, 6, 8):
    rho = partial_trace_open_loop_series_expand(
        peps.tn,
        where=((0, 0), (0, 7)),
        gloops=cutoff,
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
