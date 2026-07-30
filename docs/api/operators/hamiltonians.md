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
`fermion.to_mpo(edges, L=3, t=..., U=..., mu=...)`. Couplings remain explicit;
they are not stored on the `Fermion` object. Use `fermion.build_mpo(...)` when
the Jordan-Wigner-compatible MPO convention is wanted.

`Fermion.to_mpo(...)` and `SymHamiltonian.to_mpo(..., fermionic=True)` return
native graded `FermionicArray` MPO tensors. Explicit mappings can contain
arbitrary neutral multi-site terms; non-contiguous supports are represented by
charged virtual channels. `ham_tn.build_mpo(..., fermionic=True)` selects the
same native path. Pass `to_backend=...` to map the stored Symmray blocks to a
selected array backend.

Native MPO assembly, replay, and exact energy measurement are supported. The
native energy path applies the MPO sitewise as a factorized graded MPO-MPS
contraction, so it does not materialize the global physical operator. Its cost
is controlled by the MPS and MPO bond dimensions.

> API details are maintained as handwritten Markdown in this page.
