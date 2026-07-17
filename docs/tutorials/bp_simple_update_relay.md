# Simple-update initialization and Relay-BP

For a nonnegative, closed scalar tensor network, Pepsy can run D1BP from
either its ordinary initialization or Quimb simple-update bond gauges.
Relay-BP adds per-source-node memory over several warm-started legs to make
finding a fixed point more robust. It does not, by itself, make a loopy BP
contraction exact; use an exact small network to measure approximation error.

The runnable example below uses a 3×3 classical Ising factor network. It
compares all three paths against the exact contraction and records the absolute
message convergence and relative contraction error:

```{literalinclude} ../../examples/RelayBP/simple_update_relay_comparison.py
:language: python
```

The flow is:

```text
closed TN ── Quimb gauge_all_simple_ ──> (SU core, external gauges)
   │                                             │
   ├────────────── plain D1BP ───────────────────┤
   └──── D1BP from SU gauges / Relay D1BP ───────┘
```

On this weakly correlated loopy network, the three runs reach the same D1BP
fixed point and have a small, nonzero error relative to exact contraction.
That agreement checks the SU-to-BP initializer and that Relay-BP preserves an
easy fixed point; the reported residual separately verifies convergence.

## Relay stress cases

The following runnable exact-reference stress cases begin from deliberately
polarized messages on a near-deterministic odd cycle. Parallel D1BP stalls,
whereas Relay's per-source memory reaches a strict residual fixed point. The
reported exact-reference error is intentionally separate: convergence does not
turn a strongly loopy BP approximation into an exact contraction.

```{literalinclude} ../../examples/RelayBP/odd_cycle_stress.py
:language: python
```

For the reverse direction, use
`simple_update_core_and_gauges_from_messages(result.bp)`. For strictly
positive D1BP message products it returns a lossless `(core, gauges)` pair;
if a real SU run has singular products, pass an explicit small `smudge` to
obtain a regularized but still representation-preserving SU initializer.

## Unified SU ↔ D1BP workflow

`pepsy.gauge_all` is the high-level bridge. It keeps the SU and D1BP numerical
updates separate, but handles their warm starts and lossless conversions in one
place. For example, first find SU gauges, then use them to initialize D1BP:

```python
import pepsy as py

result = py.gauge_all(
    tn,
    start="su",
    target="bp",
    su_options={"max_iterations": 50},
    bp_options={"run_opts": {"tol": 1e-10}},
)

su_core = result.core
su_gauges = result.su_gauges
bp = result.bp
```

The reverse direction runs D1BP and returns its lossless SU-core split:

```python
result = py.gauge_all(
    tn,
    start="bp",
    target="su",
    bp_options={"run_opts": {"tol": 1e-10}},
)
```

This *vector-product* BP-to-SU route is deliberately D1BP-only. It requires
strictly positive, real products of opposite directed messages; pass
`conversion_options={"smudge": 1e-10}` for a regularized SU initializer when
a bond product is singular.

For plain 1-norm BP without an SU bridge, use `pepsy.one_norm_bp`. It supports
L1BP, HV1BP, and D1BP; select D1BP for the SU-compatible directed-message
representation:

```python
bp_result = py.one_norm_bp(tn, method="d1bp", tol=1e-10)
```

## 2-norm: PEPS SU ↔ D2BP

For a physical PEPS/state, use the distinct dense-2-norm bridge. Pass the
single-layer state—not the explicitly doubled ``peps.H & peps`` norm network.
The SU gauge ``lambda`` is inserted symmetrically as ``sqrt(lambda)`` on each
endpoint, and D2BP is initialized with the positive semidefinite matrix
``diag(lambda)`` in both directions of the virtual bond:

```python
result = py.gauge_all(
    peps,
    start="su",
    target="bp",
    norm="2norm",
    su_options={"max_iterations": 50},
    bp_options={"run_opts": {"tol": 1e-10}},
)
```

The reverse ``start="bp", target="su", norm="2norm"`` route uses both
D2BP density messages on each bond. It takes their PSD square roots and a
metric SVD to form the Vidal/SU spectrum, then absorbs the corresponding
matrix gauges into the returned core. The returned ``(core, gauges)`` exactly
reconstructs the input state, while its Vidal/isometry quality is controlled
by the D2BP residual on a loopy PEPS.

## Simple-update gauges and Relay

`gauge_all_simple` is the single entry point for real Quimb simple-update
gauges. With its default `relay=None`, it runs ordinary sequential SU. Passing
`RelayGaugeOptions` adds per-bond relay memory; every memory leg mixes old and
new nonnegative singular-value gauges and compensates the core so the returned
`(core, gauges)` represents exactly the original TN. Use Relay when
`peps.gauge_all_simple_` stalls, not as a claim that every PEPS will converge
faster:

```python
from pepsy.bp import RelayGaugeOptions, gauge_all_simple

core, gauges, info = gauge_all_simple(
    peps,
    max_iterations=200,
    tol=1e-8,
    relay=RelayGaugeOptions(
        num_legs=3,
        memory_first_leg=True,
        gamma_range=(0.6, 0.9),
        seed=0,
    ),
    damping=0.1,
    diis={"max_history": 6, "beta": 0.5},
)
```

The wrapper forwards the compatible Quimb SU controls—`smudge`, `power`,
`equalize_norms`, `touched_tids`, `reduce_opts`, and `compress_opts`—while
keeping `fuse_multibonds=False` so the external gauges, DIIS history, and warm
starts retain a stable topology. `damping` is applied directly to the external
gauge update, so it composes safely with per-bond Relay memory.

Set `bp_check_every` (and optionally `bp_tol`) to record the D1BP residual of
the current SU gauges. For CPU NumPy tensors, `schedule="parallel"` uses
edge-coloured batches: bonds that share no tensor endpoint update concurrently.
This changes the sweep order, so benchmark it for the target lattice; it is
restricted to stable pairwise topologies (`fuse_multibonds=False`) and preserves
the full represented tensor network exactly.

## Connected-loop corrections and local warm updates

`loop_cluster_expand` is a region/NLCE-style correction. For a closed pairwise
scalar network, `linked_cluster_expand` implements the different free-energy
cluster expansion of Midha and Zhang: it resolves the BP vacuum on every bond,
contracts connected excited loops, and sums their connected (Ursell) multisets
in `log(Z)`. This removes disconnected-loop proliferation rather than replacing
the existing region-cluster method.

```python
from pepsy.bp import LinkedClusterCache, linked_cluster_expand

cache = LinkedClusterCache()  # reuse while tensor ids and bonds are unchanged
corrected = linked_cluster_expand(
    tn,
    max_loop_weight=8,       # all individual loops through weight 8
    max_cluster_weight=8,    # total weight, including repeated loops
    tol=1e-10,
    cache=cache,
)

z_bp = corrected.bp_estimate
z_corrected = corrected.estimate
tail_weight = max(corrected.tail_by_weight, default=None)
tail = 0.0 if tail_weight is None else corrected.tail_by_weight[tail_weight]
```

The BP messages must be at a D1BP fixed point: unlike a system-covering region
cluster, this `I - |m><m|` construction relies on BP-vacuum cancellations.
For a systematic order `K`, use `max_loop_weight=max_cluster_weight=K` (or a
larger loop cutoff): otherwise single loops that should enter at order `K` are
missing. Pepsy rejects that incomplete setup by default; it is available only
as an explicitly labelled exploratory mode. Increase the complete cutoff and
inspect the highest-order `tail_by_weight` term; it is a convergence diagnostic,
not a certified error bar outside the loop-decay regime. The enumeration grows
exponentially, so keep the first cutoffs small and reuse `LinkedClusterCache`
for time steps and multi-start candidates.

For a value-only perturbation of a fixed-topology TN, cache the D1BP messages
and seed Quimb's local scheduler only at the changed tensors:

```python
from pepsy.bp import BPState

initial = py.one_norm_bp(tn, method="d1bp", tol=1e-10)
state = BPState.from_result(initial)

# `tn_next` has the same tensor ids/bonds; only these tensor data changed.
update = state.update_local(tn_next, changed_tids={17, 18}, tol=1e-10)
assert update.fully_converged
next_bp = update.result
```

Passing `radius=0` updates only the changed tensors' outgoing messages;
`radius=r` permits `r` propagation hops beyond them. Such a bounded result is
explicitly marked `fully_converged=False` and exposes `boundary_tids`; use it
only for a deliberately local calculation. The default `radius=None`
propagates to the new D1BP fixed point. Every changed tensor must be listed.

When several random/Relay BP starts have all converged, rank them by the
highest retained connected-cluster tail rather than by residual alone:

```python
from pepsy.bp import select_bp_candidate

selection = select_bp_candidate(
    tn,
    d1bp_candidates,
    max_loop_weight=4,
    max_cluster_weight=8,
    cache=cache,
)
chosen = selection.selected
```
