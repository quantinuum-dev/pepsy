# `pepsy.operators.hamiltonians`

`ham_tn.build_mpo` retains its original explicit local-operator form. It also
accepts a `Fermion` model for symmetry-aware MPO construction:

```python
fermion = pepsy.Fermion(spinful=True, symmetry="U1U1")
builder = pepsy.ham_tn(Lx=3, Ly=1)
mpo = builder.build_mpo(
    fermion=fermion,
    edges=[(0, 1), (1, 2)],
    t=1.0,
    U=2.0,
    mu=0.1,
)
```

The native model-facing shorthand is
`fermion.build_mpo(edges, L=3, t=..., U=..., mu=...)`. Couplings remain
explicit; they are not stored on the `Fermion` object. Pass `fermionic=False`
to the same builder when the Jordan-Wigner-compatible MPO convention is
wanted.

`Fermion.build_mpo(...)` and `SymHamiltonian.to_mpo(..., fermionic=True)` return
native graded `FermionicArray` MPO tensors. Explicit mappings can contain
arbitrary homogeneous-charge multi-site terms; non-contiguous supports are
represented by charged virtual channels, and the open boundary carries the
operator charge when it is nonzero. `ham_tn.build_mpo(..., fermionic=True)`
selects the same native path. `Fermion.to_mpo(...)` remains a compatibility
alias. Pass `to_backend=...` to map the stored Symmray blocks to a selected
array backend.

For a mixed-charge operator, request an explicit charge-sector decomposition:

```python
sectors = fermion.build_mpo(
    mixed_terms,
    L=4,
    fermionic=True,
    charge_sectors=True,
)
# sectors[charge] is one homogeneous native MPO.
```

The same `charge_sectors=True` option is available on
`SymHamiltonian.to_mpo`, `Fermion.to_pepo`, `SymHamiltonian.to_pepo`,
`ham_tn.build_mpo`, and `ham_tn.build_pepo`; those methods return
`{charge: MPO}` or `{charge: PEPO}`. This keeps each block-sparse tensor
network within one charge sector while preserving the exact sum decomposition.

The corresponding 2D entry points all use the same `OneDMap` ordering:

```python
mapper = pepsy.OneDMap(3, 2, mode="snake-row-major")

pepo = hamiltonian.to_pepo(
    Lx=3,
    Ly=2,
    mapper=mapper,
    fermionic=True,
)
pepo = fermion.build_pepo(
    {(left, right): native_term},
    Lx=3,
    Ly=2,
    mapper=mapper,
    fermionic=True,
)
pepo = builder.build_pepo(
    {(left, right): native_term},
    fermion=fermion,
    mapper=mapper,
    fermionic=True,
)
```

Use a coordinate-keyed mapping for native terms, with one-site support written
as `((x, y),)`. PEPO embedding currently requires `snake` or
`snake-row-major` ordering; transverse lattice bonds are rank one unless
periodic PEPO bonds are requested with `cycle_peps=True`.

Native MPO assembly, replay, and exact energy measurement are supported. The
native energy path applies the MPO sitewise as a factorized graded MPO-MPS
contraction, so it does not materialize the global physical operator. Its cost
is controlled by the MPS and MPO bond dimensions.

> API details are maintained as handwritten Markdown in this page.
