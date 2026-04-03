# How-To: Troubleshooting

## `ValueError: Input network must contain X* and Y* lattice tags`

Cause:

- `build_bra_ket` expects lattice tags for shape inference.

Fix:

- Ensure the network is a tagged PEPS-like object with `X{i}` and `Y{j}` tags.

## `Provided bra must have internal index names disjoint from ket`

Cause:

- You passed a custom `bra` whose internal indices collide with `ket`.

Fix:

- Reindex bra internal indices before calling `build_bra_ket`.

## Slow runtime

Common causes:

- large `chi`
- high `n_iter`
- difficult contraction geometry

Practical actions:

1. Reduce `chi` and `n_iter` first to baseline runtime.
2. Use `track_boundary_fidelity=True` to see where quality drops.
3. Increase only the parameter that improves that bottleneck.

## Fidelity list is empty

Cause:

- `track_boundary_fidelity=False` during `contract_boundary` call.

Fix:

- Set `track_boundary_fidelity=True`.
