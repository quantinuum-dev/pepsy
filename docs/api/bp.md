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

`partial_trace_loop_cluster_expand` and
`compute_local_expectation_loop_cluster` provide the parallel D2BP
generalized-loop-cluster route. Its default `combine="sum"` uses the usual
inclusion--exclusion region counts; reserve `combine="prod"` for compatibility
experiments with Quimb's elementwise product convention.
