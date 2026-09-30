# Tensor maps

`OneDMap` assigns one-dimensional site labels to a regular 2D or 3D lattice.
Use the same map when constructing states, Hamiltonians, and gate supports.

```python
from pepsy.tensors import OneDMap

mapper = OneDMap(3, 4, mode="snake")
site_to_coordinate, coordinate_to_site = mapper.build()
assert len(site_to_coordinate) == 12
assert all(
    coordinate_to_site[coordinate] == site
    for site, coordinate in site_to_coordinate.items()
)
```

Common modes include `row-major`, `col-major`, `snake`, `alternate-x`,
`alternate-y`, and `hilbert`. Pass a third lattice dimension for 3D mapping.
The `finder` mode uses a circuit's gate supports to choose an MPS layout;
see the [mapping implementation guide](../../development/modules/tensors.md#main-responsibilities)
for its handoff and label requirements.

For tree geometry, use [TreePlan and TreeLayoutFinder](../optimizers/tree_layout.md).
