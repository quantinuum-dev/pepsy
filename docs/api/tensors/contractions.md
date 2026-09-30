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
Cotengra uses Python pathfinders. From a Pepsy checkout, run
`python -m pip install ".[contraction]"` for both acceleration dependencies.

The builder forwards `reconf_opts`, `slicing_opts`, and
`slicing_reconf_opts` when explicitly supplied. Without these overrides,
Cotengra 0.8's automatic subtree reconfiguration remains active. Persistent
tree caching is available through `directory=...`; the default cache is
in memory.

### Controlling search workers

Use `build_optimizer(parallel=False)` for small contractions or when another
layer of your application already runs tasks concurrently. Pass that instance
through the operation's `optimize` or `contraction_opt` argument. The builder's
default `parallel="auto"` delegates worker selection to Cotengra; presets such
as `"auto-hq"` can also create a process pool. Reusing a contraction optimizer
does not imply that its search runs in the calling process.

Loky's "worker stopped" warning means a worker exited while work remained.
It can result from an idle timeout or a memory-related restart; the warning
alone does not identify which. Enable standard multiprocessing logging in an
isolated reproduction to inspect worker exits:

```python
import logging
import multiprocessing.util

multiprocessing.util.log_to_stderr(logging.INFO)
```

Keep the warning visible and check that the computation completed. Select
`parallel=False` explicitly when process search is unnecessary; changing
global pool settings can affect other libraries using the same executor.

## Compressed contraction

`contract_hypercompressed_tn(..., cutoff_mode="abs")` forwards the selected
singular-value truncation policy to Quimb's compressed contraction. Omitting
`cutoff_mode` preserves Quimb's default. This option controls compression;
it does not change contraction-path optimization.

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
