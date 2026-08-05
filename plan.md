# Native fermionic PEPO plan

Status: deferred design item. No implementation is planned until this work is
prioritized.

## Objective

Add a direct two-dimensional native fermionic PEPO builder for sums of local
and two-body fermionic operator terms. The builder must use Symmray graded
tensors and must not introduce Jordan--Wigner strings or dense intermediate
operators.

## Planned approach

1. Accept one- and two-site terms as native Symmray fermionic arrays with
   `U1`, `U1U1`, or `Z2` symmetry.
2. Factor each one- or two-site term exactly using graded operator-Schmidt
   decomposition. Keep every channel; do not truncate or numerically
   compress.
3. Build local PEPO tensors from a shared finite-state automaton. Virtual
   states represent idle propagation, opened/closed fermionic operators,
   charge and spin sectors, and coefficient insertion.
4. Construct tensors directly from Symmray block structure. Charge flow and
   graded contraction supply the fermionic signs natively.
5. Use a small constant-channel automaton for local or finite-range terms.
6. Use a Crosswhite--Bacon / Fröwis--Nebendahl--Dür style 2D automaton for
   arbitrary long-range two-body coefficients. Share horizontal and vertical
   propagation rather than routing one independent path per term.

## Expected scaling

- Local and finite-range interactions: constant bond dimension, up to factors
  from the operator-Schmidt rank and symmetry channels.
- Generic arbitrary two-body coefficients on an `L x L` lattice: linear
  virtual-bond scaling in `L` for the exact 2D automaton, with additional
  constant factors for spin and charge sectors.
- Special structured kernels, such as separable or distance-dependent
  interactions, can use smaller specialized automata or sums of auxiliary
  PEPOs.

The construction is exact and has optimal scaling for the generic pairwise
family treated by Fröwis et al.; it is not a universal proof of globally
minimal PEPO bond dimension for every Hamiltonian.

## Implementation sequence

1. Implement and test native local one-/two-site PEPO terms.
2. Add nearest-neighbor and finite-range term sums with shared channels.
3. Add the exact long-range 2D automaton.
4. Validate `U1U1` hopping, onsite, density, and long-range terms on small
   lattices against the existing native snake-MPO PEPO route.
5. Add structured-kernel strategies only after the generic exact builder is
   stable.

## Possible API

```python
fermion.to_pepo(
    terms,
    shape=(Lx, Ly),
    strategy="native_2d",
    long_range="automaton",
)
```

## References

- Crosswhite and Bacon, *Finite automata for caching in matrix product
  algorithms*, [arXiv:0708.1221](https://arxiv.org/abs/0708.1221).
- Fröwis, Nebendahl, and Dür, *Tensor operators: constructions and
  applications for long-range interaction systems*,
  [arXiv:1003.1047](https://arxiv.org/abs/1003.1047).
- O'Rourke, Li, and Chan, *Efficient representation of long-range
  interactions in tensor network algorithms*,
  [arXiv:1807.08378](https://arxiv.org/abs/1807.08378).
