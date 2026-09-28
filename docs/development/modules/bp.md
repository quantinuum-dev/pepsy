# Belief-propagation implementation

`pepsy.bp` exposes BP, loop-series, loop-cluster, and partitioned corrections
through lazy public imports. Their cutoffs and numerical meanings differ;
see the [BP API guide](../../api/bp.md).

| Module | Responsibility |
| --- | --- |
| `relay.py` | Plain and relay BP solves and convergence records |
| `cluster.py` | Tensor-region loop-cluster expansion |
| `_series_geometry.py` | Observable support parsing, edge terms, graph/corridor discovery, enumeration limits |
| `series.py` | Loop-series caches, route selection, projectors, contractions, diagnostics |
| `pne.py` | Partitioned network expansion |
| `compression.py` | Selected-bond and loop-series compression |
| `reduced_update.py` | Reduced local environments, ALS, and local gate/compression updates |
| `gauges.py` | BP/simple-update gauge conversion |
| `observables.py` | Boundary and path-cluster observables |
| `_symmray.py` | Native array/charge/fermionic compatibility |
| `_compression_utils.py`, `_backend.py` | Shared cost policy and array adapters |

Geometry discovery does not solve BP or contract tensors. The public
`LoopSeriesTerm` and enumeration error remain accessible through `bp.series`
and `pepsy.bp`; historical serialized globals still resolve. Cache objects
remain in `series.py` and retain topology validation and all-or-nothing cache
installation after successful enumeration.

The open-series entry points validate options and prepare BP normalization
before dispatching to `_contract_open_rho_cluster` / `_contract_open_rho_edges`
or their scalar-expectation counterparts. Each execution helper owns its
route's caches, budget decisions, and diagnostics. The native cyclic route
retains graded cluster contractions; the explicit-edge route retains its
projector terms. `observables.py` and `series.py` share `_term_sites` through
the geometry module, which does not import either contraction owner.

Native cyclic fermion route selection remains with the contraction layer.
Moving graph algorithms must not change that route, collapse parallel seam
bonds, confuse edge degree with region size, or return partial sums when
enumeration limits are exceeded.
