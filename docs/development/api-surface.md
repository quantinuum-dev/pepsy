# Public API surface

Baseline recorded: 2026-08-19

This document separates Pepsy's canonical API from its compatibility facade.
The machine-readable root compatibility manifest is
[`api-manifest.txt`](api-manifest.txt).

## Baseline inventory

The current package exposes:

- 271 lazy symbols from the top-level `pepsy` compatibility facade;
- 284 entries in `pepsy.__all__`, including version and namespace exports;
- canonical responsibility-based namespaces for those symbols;
- lazy advanced-domain discovery through `pepsy.experimental`.

Every current root symbol already has an owning namespace. The root facade is
therefore compatibility convenience, not a second canonical API. The manifest
test verifies that the owner map, each owner's `__all__`, and root `__all__`
remain synchronized. New symbols must be added to the owning namespace and to
the manifest only when an intentional compatibility decision approves a new
root alias.

The owner map is implemented in the private lazy-safe `pepsy._api` module, so
the top-level initializer does not need to carry the full compatibility
registry alongside its namespace-loading logic.

## Redundancy identified

These backend helpers belong only to `pepsy.backends`:

`backend_cupy`, `backend_jax`, `backend_numpy`, `backend_torch`,
`build_backend`, `get_default_array_backend`, `get_default_grad_backend`,
`get_torch_linalg_config`, `register_jax_linalg`, `register_torch_linalg`,
`reset_default_backends`, `reset_linalg_registrations`,
`set_default_array_backend`, `set_default_grad_backend`, and
`TorchLinalgConfig`.

Duplicate exports from `pepsy.tensors` and its `core` aggregator have been
removed. Tensor modules no longer re-export backend configuration helpers.

`pepsy.optimizers.mera` remains as a lazy compatibility path because a current
Pepsy example imports it. The unused `pepsy.experimental.mera` discovery alias
was removed; use `pepsy.experimental.qmera`.

## Cleanup policy

1. Keep the manifest guard so accidental root-surface growth is rejected.
2. Keep one canonical namespace for each responsibility.
3. Retain compatibility aliases only when a current consumer needs them.
4. Remove unused aliases and update the migration guide when removing them.

Canonical examples:

```python
from pepsy.backends import TorchLinalgConfig
from pepsy.boundary import contract_boundary
from pepsy.operators import gate
from pepsy.optimizers import MpsOptimizer
from pepsy.tensors import ps_to_peps
```
