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

This BP-to-SU route is deliberately D1BP-only. It requires strictly positive,
real products of opposite directed messages; pass
`conversion_options={"smudge": 1e-10}` for a regularized SU initializer when
a bond product is singular.

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
