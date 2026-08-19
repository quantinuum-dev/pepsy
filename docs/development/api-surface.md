# Public API surface

Baseline recorded: 2026-08-18

This document separates Pepsy's canonical API from its compatibility facade.
The machine-readable root compatibility manifest is
[`api-manifest.txt`](api-manifest.txt).

## Baseline inventory

The current package exposes:

- 260 lazy symbols from the top-level `pepsy` compatibility facade;
- 273 entries in `pepsy.__all__`, including version and namespace exports;
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

These backend helpers currently appear in both `pepsy.backends` and
`pepsy.tensors`:

`backend_cupy`, `backend_jax`, `backend_numpy`, `backend_torch`,
`build_backend`, `get_default_array_backend`, `get_default_grad_backend`,
`get_torch_linalg_config`, `register_jax_linalg`, `register_torch_linalg`,
`reset_default_backends`, `reset_linalg_registrations`,
`set_default_array_backend`, `set_default_grad_backend`, and
`TorchLinalgConfig`.

The canonical home for all of them is `pepsy.backends`. The
`pepsy.tensors` exports remain functional as deprecated compatibility aliases
for the current 0.x line and are protected by the existing compatibility
tests.

`pepsy.optimizers.mera`, `pepsy.experimental.mera`, and their `qmera`
counterparts point to the same QMERA implementation. `qmera` is the preferred
spelling; the two `mera` paths remain transitional compatibility aliases and
emit deprecation warnings when accessed.

## Cleanup policy

1. Keep the manifest guard so accidental root-surface growth is rejected.
2. Document and deprecate redundant namespace aliases before removal.
3. Preserve lazy compatibility imports during the current 0.x line.
4. Remove approved redundant aliases only in a planned breaking release.

Canonical examples:

```python
from pepsy.backends import TorchLinalgConfig
from pepsy.boundary import contract_boundary
from pepsy.operators import gate
from pepsy.optimizers import MpsOptimizer
from pepsy.tensors import ps_to_peps
```
