# Alternating x/y tree hierarchy

## Contract and implementation

Added 2026-09-23 in `pepsy.optimizers.tree.layout`:

- `TreePlan.from_alternating_lattice((Lx, Ly), site=...)` constructs a
  bottom-up binary hierarchy. It pairs adjacent x blocks, then y blocks,
  repeating until one root remains. Odd edge blocks carry forward without
  unary nodes; axes of size one are skipped. A one-site lattice is one leaf.
- `TreeLayoutFinder(order="alternating-xy")`, with the equivalent
  `map_mode` spelling, selects that topology directly. No search stage,
  balanced repartition, or `coarse_grain` parameter changes it. The root is
  binary, unlike the ordinary finder's default ternary root. Incompatible
  explicit arities and physical root qubits are rejected.
- All physical sites remain separate leaves with their original logical
  labels. Preorder node numbering ensures the public `mpo_order()` keeps
  each subtree contiguous. A leaf order alone cannot reconstruct this
  hierarchy through balanced bisection on general odd-sized lattices.
- Existing `coarse-*` options retain their traversal-only semantics.
  Pepsy's library defaults do not change. The roughening frontend in
  `pepsy_examples` opts into this preset by default only for tree runs.

For 9x10 the block grids are 9x10, 5x10, 5x5, 3x5, 3x3, 2x3, 2x2, 1x2,
and 1x1. There are 90 physical leaves and 89 binary merge nodes.

## Upstream compatibility audit — 2026-09-23

The new builder performs only integer-coordinate grouping and uses the
existing `TreePlan.from_children` validator. It introduces no upstream
contraction, split, QR, backend conversion, or dispatch calls. Existing tree
state/operator paths consume the resulting plan unchanged.

Reviewed sources and decisions:

- [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html):
  **defer** numerical-policy changes. Upstream's default relative cutoff,
  randomized-SVD cutoff restrictions, and SDC developments do not require
  changes to a geometry-only constructor. Preserve Pepsy's numerical defaults.
- [Autoray repository](https://github.com/jcmgray/autoray): **defer** dispatch
  changes; existing backend routes suffice. No new registrations or shims.
- [Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
  [changelog](https://cotengra.readthedocs.io/en/latest/changelog.html):
  **defer** contraction-search changes. The served changelog stopped at
  0.7.5, older than the installed development checkout; local probes below
  are authoritative for this environment.
- [Symmray Abelian-array documentation](https://symmray.readthedocs.io/en/latest/abelian_arrays.html)
  could not be fetched (repeated fetch errors). Reviewed the
  [Symmray repository](https://github.com/jcmgray/symmray) and installed
  signatures instead: **defer** array/QR policy changes. This feature neither
  manipulates native arrays nor alters the centralized native QR safeguard.

No upstream algorithm adoption, compatibility shim, or prototype is needed.

Installed packages in the shared `~/envs/py312` environment:

| Package | Version |
| --- | --- |
| Quimb | 1.15.1.dev51+g2e99c793e |
| Autoray | 0.11.1.dev3+g1b476b305 |
| Cotengra | 0.8.3.dev7+g1d7fd333f |
| Cotengrust | 0.2.1 |
| Symmray | 0.3.2.dev8+g6c6dd34b5 |
| Torch | 2.6.0+cu124 |

Inspected actual callable signatures for the unchanged numerical consumers:

```text
quimb.tensor.tensor_contract(*tensors, output_inds=None, optimize=None,
    get=None, backend=None, preserve_tensor=False, drop_tags=False,
    strip_exponent=False, exponent=None, **contract_opts)
quimb.tensor.Tensor.split(T, left_inds, *, method='auto', absorb='auto',
    max_bond=None, cutoff=1e-10, cutoff_mode='rel', renorm=None, get=None,
    ltags=None, rtags=None, stags=None, bond_ind=None, right_inds=None,
    matrix_svals=False, info=None, **kwargs)
cotengra.array_contract(arrays, inputs, output=None, optimize='auto',
    strip_exponent=False, cache_expression=True, backend=None, **kwargs)
symmray.AbelianArray(indices, charge=None, blocks=(), symmetry=None, label=None)
```

Resolved Autoray `get_lib_fn` entries for `reshape`, `tensordot`, `linalg.qr`,
and `linalg.svd` on NumPy, Torch, and Symmray. NumPy resolves to NumPy/linalg;
Torch to its reshape, functional tensordot, and native linalg builtins;
Symmray to its interface/linalg functions. No registrations were changed.

## Validation

Run with the shared Python environment active and `pytest -q -o addopts=''`:

- `tests/test_tree_alternating_layout.py`: exact dyadic-rectangle subtree
  masks, odd/degenerate shapes, strict binary structure, traversal intervals,
  custom labels, finder overrides, invalid configurations, and tree-native
  operator measurement.
- `tests/test_public_api.py tests/test_package_layout.py`: public imports.
- `tests/test_optimize_tree.py`: existing geometry, dense/backend/native
  evolution, compression, and layout regressions.
- `tests/test_contraction_dependencies.py`: existing contraction integration.
- Examples' `experiments/mps_magnetization/benchmark/tests/test_roughening.py`:
  defaults and metadata for both layouts; exact-state, sampling, entropy,
  NumPy/CuPy, and all advertised tree replay-mode checks.

Repository lint: `python -m ruff check src tests`. No notebooks, stored
experiment outputs, running jobs, or installed dependencies are modified.

Results: 438 tree-optimizer tests, 137 roughening tests, 65 new-layout/public
API/package-layout tests, and 5 contraction-dependency tests passed (645
distinct tests). Pepsy's repository lint and lint of the changed example
Python files passed. Existing deprecation and diagnostic warnings remain;
there were no test failures after correcting the new measurement test to
use the public `TreeMPO.expectation(state)` signature.
