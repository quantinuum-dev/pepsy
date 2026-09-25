# Tensor contractions

## Reusable path optimization

`pepsy.tensors.build_optimizer(...)` returns a Cotengra
`ReusableHyperOptimizer`. Reuse the same instance across contractions with
the same topology to amortize path-search costs:

```python
from pepsy.tensors import build_optimizer

optimizer = build_optimizer(parallel=False)
value = tn.contract(all, optimize=optimizer)
```

Pepsy requires **Cotengra 0.8.0 or newer**. The default search uses CMA-ES
when installed; otherwise Pepsy warns and selects Cotengra's built-in
`sbplx` optimizer, introduced in 0.8.0. Cotengrust is optional: without it,
Cotengra uses Python pathfinders. Install `pepsy[contraction]` for both
acceleration dependencies.

The builder forwards `reconf_opts`, `slicing_opts`, and
`slicing_reconf_opts` when explicitly supplied. Without these overrides,
Cotengra 0.8's automatic subtree reconfiguration remains active. Persistent
tree caching is available through `directory=...`; the default cache is
in memory.

## Memory diagnostics

Exact contraction trees expose `peak_size()` (modeled concurrently live
tensor elements) and `max_contraction_size()` (the largest sum of two inputs
and their output). Multiply by the dtype's item size for modeled tensor
bytes, not process RSS or an allocator/autograd memory guarantee.

Pepsy does not automatically apply `reorder_for_peak_size()`. A bounded
exact-PEPS benchmark found no peak-memory reduction for its hyper-optimized
trees, and both increases and decreases for random-greedy trees. See the
[Cotengra audit](../../development/notes/cotengra_2026_09.md) for measurements
and execution-cache precautions.
