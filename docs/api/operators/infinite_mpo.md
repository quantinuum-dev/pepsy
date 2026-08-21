# Infinite and unit-cell MPOs

`pepsy.operators.InfiniteMPO` stores a periodic unit cell separately from the
finite, open-boundary `FirstDegreeMPO` used by the higher-order Taylor engine.
Every tensor has `(left, right, upper, lower)` order and the final right bond
must match the first left bond.

```python
from pepsy.operators import InfiniteMPO

cell = InfiniteMPO(unit_cell_arrays)
window = cell.finite_window(
    cells=4,
    left_boundary=left_vector,
    right_boundary=right_vector,
)
```

Boundary vectors are mandatory when the selected unit-cell seam has dimension
greater than one. This keeps three distinct objects explicit:

- an infinite periodic transfer object;
- a finite window with chosen open boundaries;
- a finite periodic operator formed by tracing the virtual seam, which is not
  inferred by `finite_window`.

`shift(offset)` rotates the unit-cell origin and `repeat_cell(cells)` creates
an equivalent enlarged unit cell. `InfiniteMPO.from_finite_cell(mpo)` is a
literal periodic repetition of an existing finite cell; a singleton open seam
therefore describes independent repeated cells, not an automatically inferred
translation-invariant Hamiltonian.

Higher-order Algorithms 1--4 remain finite-boundary transformations. Apply
them to a deliberately bounded `finite_window`; no finite-as-infinite rewrite
is performed internally.
