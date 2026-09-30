# TreeSampler native Symmray path — 2026-09-16

## Scope

`TreeSampler(backend="symmray")` now retains native Abelian and fermionic
Symmray tree tensors for sampling, amplitudes, probabilities, and edge
entropies. The default `backend="auto"` remains the existing dense batched
compatibility path so large sampling jobs do not change performance silently.

The native path uses a canonical private TTN copy, Quimb's public `isel_()`
projection, and complete native norm contractions for conditional branch
weights. Source physical `chargemap` metadata is retained separately from the
canonical private physical indices. No installed package or upstream source is
modified.

## Dependency audit

The active Pepsy environment contains:

| dependency | version |
| --- | --- |
| Autoray | `0.11.1.dev3+g1b476b305` |
| Cotengra | `0.8.3.dev7+g1d7fd333f` |
| Quimb | `1.15.1.dev51+g2e99c793e` |
| Symmray | `0.3.2.dev8+g6c6dd34b5` |

The installed API probes were:

- `symmray.AbelianArray.to_dense(self, index_maps=None)` — retained only for
  the explicit dense compatibility route, not the native route.
- `symmray.AbelianArray.tensordot(self, other, axes=2, mode="auto",
  preserve_array=False)` — native contraction remains available through the
  Quimb/Symmray network contraction path.
- `symmray.AbelianArray.reshape(self, newshape, inplace=False)` — native
  array shape metadata remains available without materialization.
- `quimb.tensor.TensorNetwork.isel_(self, selectors, *, inplace=True)` and
  `TensorNetwork.contract(..., optimize=None, ...)` — used as the public
  projection and contraction boundary.

The upstream [Symmray documentation and API overview](https://github.com/jcmgray/symmray)
describes native `tensordot`, `reshape`, `transpose`, and blockwise/fused
contraction modes; the implementation adopts native array preservation and
does not add a compatibility shim.

## Validation

- `pytest -q -o addopts='' tests/test_tree_sampler.py`: 38 passed.
- `python -m ruff check src tests`: passed.
- `pytest -q` (configured smoke suite): 131 passed, 19 expected warnings.

The native branch sampler is intentionally a general correctness fallback: it
projects one physical code at a time, so it is not yet the throughput path for
large shot batches. A future optimization can add tree-specific block-sparse
message caching without changing the public code-map contract.
