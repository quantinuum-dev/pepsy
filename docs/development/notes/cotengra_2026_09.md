# Cotengra compatibility and exact-PEPS memory audit

Audit date: 2026-09-24. Scope: dependency metadata, the existing optimizer
fallback, and an experiment with exact contraction traversal. No runtime
contraction defaults or public API were changed.

## Upstream and installed stack

Reviewed the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[source changelog](https://github.com/jcmgray/cotengra/blob/main/docs/changelog.md),
and [Symmray repository](https://github.com/jcmgray/symmray).
The Symmray Abelian-array documentation URL failed to load on both attempts;
no Symmray behavior was changed. The served Cotengra changelog was stale;
the source changelog and installed code were used for the 0.8 features.

The device-local `genpy` environment reported:

| Package | Version |
| --- | --- |
| Pepsy | 0.4.1 |
| Quimb | 1.15.1.dev55+gd0591eb70 |
| Cotengra | 0.8.3.dev7+g1d7fd333f |
| Cotengrust | 0.2.1 |
| Autoray | 0.11.1.dev3+g1b476b305 |
| Symmray | 0.3.2.dev8+g6c6dd34b5 |
| NumPy | 2.5.2 |
| Torch | 2.9.1 |
| CMA-ES | 0.13.1 |

Installed API/source probes confirmed:

- `HyperOptimizer(..., parallel='auto', reconf_opts='auto', ...)` enables
  subtree reconfiguration when all refinement options are automatic.
- `ContractionTree.reorder_for_peak_size(self)` mutates child ordering and
  returns a Boolean, not a new tree.
- `get_peak_size(self, node)`, `peak_size(self, order=None, log=None)`,
  `max_contraction_size(self, log=None)`, and
  `reset_contraction_indices(self)` are available.
- `TensorNetwork.contraction_tree(self, optimize=None, output_inds=None,
  **kwargs)` and `TensorNetwork.contract(..., optimize=None, backend=None,
  ...)` accept the explicit tree used in the experiment.
- Autoray resolves `tensordot`/`transpose` to NumPy, Torch, and Symmray
  implementations. This was a dispatch probe, not a numerical backend test.

## Decisions

- **Adopt:** require Cotengra >=0.8.0. The existing
  `src/pepsy/tensors/contractions.py::_resolve_optlib` selects `sbplx` when
  CMA-ES is unavailable, but `sbplx` was introduced in 0.8.0. The previous
  >=0.6 requirement admitted incompatible base installations. Raising the
  floor preserves the existing fallback without a new compatibility shim.
- **Adopt existing behavior:** retain upstream automatic subtree
  reconfiguration, reusable caches, and available dimension-one pathfinding
  improvements. No separate Pepsy implementation or claimed speedup.
- **Prototype, then defer integration:** memory-aware traversal. The bounded
  experiment below does not justify adding a Pepsy option or changing its
  defaults. Reconsider for larger, irregular exact networks.
- **Defer:** a new random-greedy builder mode. Random-greedy was a benchmark
  comparator; the existing `build_optimizer` remains unchanged. Planning and
  execution costs need workload-specific amortization before selecting it.
- **Defer:** alternative contraction implementations such as `pytblis`,
  changes to compressed contraction, and new slicing/exponent handling.

## Exact-PEPS experiment

Six networks, two planners each: complex128 NumPy PEPS, open boundaries,
physical dimension 2, tensor seed 42. Norms retain separate bra/ket layers;
the amplitude selects a checkerboard basis configuration. Random-greedy uses
32 trials, seed 42, and `parallel=False`. Pepsy's builder uses 16 trials,
`max_time=None`, `parallel=False`, its default combo-64 objective and CMA-ES.
Its search is stochastic; the recorded paths belong to this run.

Each planned tree is copied before execution, reordered, and passed through
the public `reset_contraction_indices()` method. The original and reordered
trees are contracted through Quimb with one BLAS thread. After warmup, six
executions per variant are timed in alternating order. All twelve pairs
preserved FLOP counts and matched values at `rtol=atol=1e-11`.

| Network | Planner | Peak elements before → after | Median execution ms before → after |
| --- | --- | --- | --- |
| 3x3 norm, D=2 | random-greedy | 944 → 944 | 0.199 → 0.188 |
| 3x3 norm, D=2 | Pepsy hyper | 880 → 880 | 0.195 → 0.202 |
| 4x4 norm, D=2 | random-greedy | 2672 → 2704 | 0.398 → 0.396 |
| 4x4 norm, D=2 | Pepsy hyper | 1312 → 1312 | 0.375 → 0.381 |
| 3x3 norm, D=3 | random-greedy | 3734 → 3734 | 0.250 → 0.252 |
| 3x3 norm, D=3 | Pepsy hyper | 3734 → 3734 | 0.249 → 0.252 |
| 6x6 amplitude, D=2 | random-greedy | 492 → 492 | 0.366 → 0.370 |
| 6x6 amplitude, D=2 | Pepsy hyper | 444 → 444 | 0.359 → 0.367 |
| 5x5 norm, D=2 | random-greedy | 13040 → 13040 | 1.551 → 1.558 |
| 5x5 norm, D=2 | Pepsy hyper | 9072 → 9072 | 0.683 → 0.673 |
| 4x4 norm, D=3 | random-greedy | 102501 → 98775 | 0.932 → 1.039 |
| 4x4 norm, D=3 | Pepsy hyper | 20835 → 20835 | 0.584 → 0.586 |

No Pepsy hyper tree had a peak-size reduction. The largest random-greedy
case saved 3.6% of modeled peak elements (about 58 KiB at 16 bytes/element),
while the 4x4 D=2 case increased 1.2%. Cotengra's recursive
`get_peak_size(root)` and `peak_size()` model different live-input behavior;
minimizing the former does not guarantee improvement in the latter. All
single-contraction maxima were unchanged. Sub-millisecond timings are noisy
and provide no evidence of a consistent speedup.

Memory numbers are modeled tensor sizes, not measured process/GPU peaks.
This small CPU experiment does not establish behavior for large PEPS,
autograd, native Symmray, or compressed contraction.

Two implementation precautions matter if this is revisited:

1. `copy()` also copies cached contractors and index-order information.
   Reordering an already used tree requires public cache invalidation on
   the copy before execution.
2. Rebuilding a tree from `get_path()` resets children to Cotengra's
   heavier-subtree-first convention; it can undo the intended traversal.
   The reported experiment executes the reordered tree directly.

The temporary harness and JSON results (including paths) are
`/tmp/pepsy_cotengra_memory_benchmark.py` and
`/tmp/pepsy_cotengra_memory_results.json`. They are local artifacts, not
package files; the method and results above are the persistent record.

## Validation

- `tests/test_contraction_dependencies.py`, `tests/test_public_api.py`, and
  `tests/test_package_layout.py`: **53 passed**, with existing alias warnings.
- The isolated dependency regression now blocks CMA-ES/Cotengrust imports,
  performs an actual fallback search, checks an exact contraction against
  NumPy, and repeats the contraction with the reusable optimizer.
- Downloaded Cotengra 0.8.0 with `--no-deps` to
  `/tmp/pepsy-cotengra-080` and ran that regression body with this directory
  first on `PYTHONPATH`: **passed**. The active environment was not changed.
  This verifies the corrected Cotengra floor in the current stack, not all
  minimum versions of Pepsy's other dependencies.
- `python -m ruff check src tests`: **passed**.
- Full suite not run: production numerical code and backend routes are
  unchanged. No installed package edits, commits, or pushes.
