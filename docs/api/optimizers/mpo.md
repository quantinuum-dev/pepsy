# `pepsy.optimizers.mpo.optimizer`

`MpoOptimizer` accepts ordinary Quimb MPOs and Symmray block-sparse MPOs. For
a native graded fermion workflow, use `Fermion.to_mpo(...)` and replay the
matching native gate stream:

```python
fermion = pepsy.Fermion(spinful=True, symmetry="U1U1")
edges = [(0, 1), (1, 2)]
hamiltonian = fermion.hamiltonian(edges, t=1.0, U=2.0, mu=0.1)
mpo = fermion.to_mpo(hamiltonian=hamiltonian, L=3)

opt = pepsy.MpoOptimizer(
    mpo,
    hamiltonian.trotter_gates(0.01),
    chi=16,
    mode="svd",
)
mpo = opt.run(progbar=False)
```

For an explicit neutral term collection, arbitrary one- or multi-site support
is accepted:

```python
term = fermion.operator_term(
    [(1.0, ((0, "create_up"), (2, "annihilate_up")))],
    sites=(0, 2),
    add_hc=True,
)
mpo = fermion.to_mpo({(0, 2): term}, L=3)
```

The Jordan-Wigner compatibility path remains available through
`Fermion.build_mpo(...)` and the matching
`SymHamiltonian.jw_trotter_gates(...)` stream:

```python
fermion = pepsy.Fermion(spinful=True, symmetry="U1U1")
edges = [(0, 1), (1, 2)]
hamiltonian = fermion.hamiltonian(edges, t=1.0, U=2.0, mu=0.1)
mpo = fermion.build_mpo(edges, L=3, t=1.0, U=2.0, mu=0.1)

opt = pepsy.MpoOptimizer(
    mpo,
    hamiltonian.jw_trotter_gates(0.01),
    chi=16,
    mode="svd",
)
mpo = opt.run(progbar=False)
```

Symmray gates keep their charge and dual metadata and are not coerced to dense
arrays. Native graded gates from `Fermion.strang_gate_stream(...)` are accepted
as well; even gates are adapted explicitly to the MPO's Jordan-Wigner
convention. `mode="svd"` is the symmetry-aware compression path. For a
Symmray MPO, `mode="mpo"` and `mode="dmrg"` use that same block-aware path
because generic dense auxiliary MPO compression and bond padding do not support
multi-sector Symmray bonds reliably.

Native MPO tensors retain their Symmray graded metadata throughout replay and
compression. `mode="svd"`, `mode="mpo"`, and `mode="dmrg"` use the same
block-aware path for native Symmray MPOs; the optimizer does not require a
dense conversion of the input MPO.
`ham_tn.build_mpo(..., fermionic=True)` is also routed to the native
`Fermion.to_mpo(...)` entry point. Use `to_backend=...` on the model-facing
builder when the stored blocks must be moved to Torch or another supported
backend.

Native MPO assembly/replay is also measurable with a native fermionic MPS.
`MpsEnergyOptimizer` applies the native MPO sitewise as a factorized graded
MPO-MPS network, preserving Symmray ordering while retaining MPO bond scaling.
Repeated evaluations reuse the optimizer's contraction paths. Optional
controlled truncation is available through
``native_mpo_compression={"max_bond": ..., "cutoff": ..., "method": "svd"}``;
the default remains exact and uncompressed.

To compress an existing MPO without replaying gates, use
`MpoOptimizer(mpo, gates=[], chi=...).compress()`. Symmray may retain a small
sector-multiplicity overshoot above the requested numeric bond cap.

> API details are maintained as handwritten Markdown in this page.
