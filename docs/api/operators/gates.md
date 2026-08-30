# `pepsy.operators.gates`

The gate-to-operator builders accept native Symmray fermionic gates directly;
they do not convert them through dense arrays. Charge-neutral gates work by
default, such as `Fermion.hopping_gate(...)`:

```python
fermion = pepsy.Fermion(spinful=True, symmetry="U1U1")
gate = fermion.hopping_gate(0.01, t=1.0)

mpo = pepsy.build_mpo_from_gates(gate, where=(0, 1), max_bond=16)
pepo = pepsy.build_pepo_from_gates(
    gate,
    where=((0, 0), (0, 1)),
    mapper=pepsy.OneDMap(2, 2, mode="snake-row-major"),
    max_bond=16,
)
```

The resulting tensors remain `U1U1FermionicArray` (or the corresponding
`U1`/`Z2` native type). `build_pepo_from_gates` uses `OneDMap` to choose the
MPO ordering before embedding it on the 2D lattice. A definite-charge native
operator can also be used explicitly:

```python
charged_mpo = pepsy.build_mpo_from_gates(
    charged_gate,
    where=(0, 1),
    allow_charged=True,
)
charged_pepo = pepsy.build_pepo_from_gates(
    charged_gate,
    where=((0, 0), (0, 1)),
    mapper=pepsy.OneDMap(2, 2, mode="snake-row-major"),
    allow_charged=True,
)
```

`allow_charged=True` means the returned operator carries the accumulated
charge of the sequential gate product. It is opt-in because charged gates
change the symmetry sector; ordinary charge-preserving evolution should leave
it disabled.

> API details are maintained as handwritten Markdown in this page.

## Gate transforms

`pepsy.operators.gates.gate` and `gate_simple` accept `dagger=True` or
`transpose=True`. The option is capability-checked against the installed
Quimb gate API and applied to the requested user gate. If a non-local gate is
routed through internal SWAPs, those SWAPs are always applied normally; only
the final requested gate is transformed.
