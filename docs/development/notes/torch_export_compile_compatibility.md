# Torch export/compile compatibility audit

Audit date: 2026-09-01

## Installed development stack

The audit was run in the shared Python 3.12 environment:

| package | version |
| --- | --- |
| PyTorch | `2.6.0+cu124` |
| Quimb | `1.15.1.dev37+gdf03dbe79` |
| Autoray | `0.11.1.dev1+gc56f64427` |
| Cotengra | `0.8.3.dev6+g08fe1a3a1` |
| Symmray | `0.3.1` |

Relevant callable probes:

- `torch.export.export(mod, args, *, dynamic_shapes=None, strict=True, ...)`
- `torch.compile(model, *, fullgraph=False, dynamic=None, backend='inductor', mode=None, ...)`
- `torch.vmap(func, in_dims=0, out_dims=0, randomness='error', *, chunk_size=None)`
- `quimb.tensor.pack(obj)` and `quimb.tensor.unpack(params, skeleton)`
- `PEPS.contract(..., strip_exponent=False, ...)`
- `autoray.do(fn, *args, like=None, **kwargs)`
- `cotengra.array_contract(arrays, inputs, output=None, optimize='auto', strip_exponent=False, ...)`
- `symmray.AbelianArray.tensordot(self, other, axes=2, mode='auto', ...)`

## Finding and disposition

PyTorch 2.6 can return `FakeTensor` values when `torch.compile(...,
backend="eager")` wraps the exported/vmapped stable-log graph. The previous
log wrapper also entered the batched `forward_log` method during export,
introducing a nested `vmap` and causing internal tensors to be lifted as
constants. The affected path is the exact dense/flat-PEPS opt-in compiler
route; ordinary eager/vmap and approximate contractions are independent.

Disposition: **compatibility shim**. The scalar log contraction is exported
directly, and `backend="eager"` uses the resulting exported/vmapped fixed-
shape graph without an additional Dynamo wrapper. Other backends continue to
use `torch.compile`. This keeps parameter leaves as explicit graph inputs and
keeps the existing fixed-batch proposal padding contract.

The upstream audit found no required Pepsy changes for the probed Quimb,
Autoray, Cotengra, or Symmray signatures. Their exact-contraction APIs remain
behind the existing Pepsy wrappers. Revisit this shim when the minimum Torch
version or the development Torch stack changes.

Focused validation:

- `python -m pytest -q -o addopts='' tests/test_vmc_torch_compile.py`
- `python -m pytest -q -o addopts='' tests/test_vmc_api.py`
- `python -m ruff check src tests`
