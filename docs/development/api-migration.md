# API migration guide

Pepsy is keeping the current public names during the 0.x compatibility
window. The entries below are deprecated compatibility paths, not removals.
New code and documentation should use the canonical import on the right.

## Deprecated aliases

| Deprecated import | Canonical import |
| --- | --- |
| `pepsy.tensors.backend_cupy` | `pepsy.backends.backend_cupy` |
| `pepsy.tensors.backend_jax` | `pepsy.backends.backend_jax` |
| `pepsy.tensors.backend_numpy` | `pepsy.backends.backend_numpy` |
| `pepsy.tensors.backend_torch` | `pepsy.backends.backend_torch` |
| `pepsy.tensors.build_backend` | `pepsy.backends.build_backend` |
| `pepsy.tensors.get_default_array_backend` | `pepsy.backends.get_default_array_backend` |
| `pepsy.tensors.get_default_grad_backend` | `pepsy.backends.get_default_grad_backend` |
| `pepsy.tensors.get_torch_linalg_config` | `pepsy.backends.get_torch_linalg_config` |
| `pepsy.tensors.register_jax_linalg` | `pepsy.backends.register_jax_linalg` |
| `pepsy.tensors.register_torch_linalg` | `pepsy.backends.register_torch_linalg` |
| `pepsy.tensors.reset_default_backends` | `pepsy.backends.reset_default_backends` |
| `pepsy.tensors.reset_linalg_registrations` | `pepsy.backends.reset_linalg_registrations` |
| `pepsy.tensors.set_default_array_backend` | `pepsy.backends.set_default_array_backend` |
| `pepsy.tensors.set_default_grad_backend` | `pepsy.backends.set_default_grad_backend` |
| `pepsy.tensors.TorchLinalgConfig` | `pepsy.backends.TorchLinalgConfig` |
| `pepsy.experimental.mera` | `pepsy.experimental.qmera` |
| `pepsy.optimizers.mera` | `pepsy.optimizers.qmera` |

These aliases emit `DeprecationWarning` when resolved. Applications can make
the transition visible in CI with:

```bash
python -W error::DeprecationWarning -m pytest
```

## Removal policy

No alias is scheduled for removal from the current 0.x line. Before a planned
breaking release, maintainers should review warning usage, publish release
notes with the table above, and remove only aliases that have completed the
deprecation window. The root-level compatibility facade is governed by
[`api-manifest.txt`](api-manifest.txt) and remains unchanged until that
review.
