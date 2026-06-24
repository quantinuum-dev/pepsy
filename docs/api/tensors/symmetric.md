# Symmetric Tensor States

## Choosing sectors and charges

Physical sectors are charge maps: ``{charge: sector_size}``. For example,
``{0: 1, 1: 1}`` is a two-state U(1) or Z2 local space, while
``{0: 1, 1: 2, 2: 1}`` is the spinful Fermi-Hubbard local space
``|0>, |up>, |down>, |up down>``.

```python
import pepsy as py

spinless = py.default_physical_sectors("U1", 2)
spinful = py.default_physical_sectors(model="fermi_hubbard")

psi = py.SymMPS.random(
    6,
    symmetry="U1",
    phys_dim=spinless,
    bond_dim={0: 2, 1: 2},
    site_charge=py.site_charge_from_occupations([1, 0, 1, 0, 1, 0]),
)
```

The local ``site_charge`` pattern fixes the global sector represented by the
state. For U(1), ``psi.overall_charge()`` is the sum of local charges. For Z2,
``psi.overall_parity()`` is the same sum modulo two.

```python
even = py.site_charge_uniform(0)
checkerboard = py.site_charge_alternating(even=0, odd=1)
custom = py.site_charge_from_map({(0, 0): 1, (0, 1): 0}, default=0)

peps = py.SymPEPS.random(
    2,
    2,
    symmetry="Z2",
    phys_dim={0: 1, 1: 1},
    site_charge=custom,
)

peps.site_charges()
peps.overall_parity()
```

## Time evolution

Hamiltonians produce a canonical bundled gate stream, so the same stream can be
used by the PEPSY gate wrappers and MPS optimizer.

```python
psi = py.SymMPS.for_model("heisenberg", 8, bond_dim=4)
ham = psi.build_hamiltonian()
gates = ham.gate_stream(0.01)

psi.time_evolve_mps_optimizer(0.01, hamiltonian=ham, chi=16, mode="mpo")
```

For Symmray-backed MPS gate streams, ``mode="swap"`` and ``mode="svd"`` use
quimb's block-aware auto-swap split path for nonlocal 1D gate streams such as a
row-major square lattice. ``mode="mpo"`` uses its usual sub-MPO compression for
nearest-neighbor gates and falls back to the same Symmray auto-swap path for
nonlocal gates, because the current quimb/Symmray sub-MPO path mixes in dense
helper tensors. ``mode="exact"`` is useful as a small-system reference, and
``mode="dmrg"`` works when the initial symmetric MPS was built with enough
block-sparse bond capacity, for example ``bond_dim >= chi``.

```python
peps = py.SymPEPS.for_model("itf", 4, 4, bond_dim=2)
ham = peps.build_hamiltonian(jx=-1.0, hz=-0.5)
gauges = {}

peps.time_evolve(
    0.005,
    hamiltonian=ham,
    method="simple",
    gauges=gauges,
    max_bond=8,
)
```

## Measuring observables

Use ``state.measure(obs, where=...)`` for local one- or two-site observables on
both MPS and PEPS. Dense operators are converted to Symmray arrays using the
state's physical sectors. Use ``charge=0`` for symmetry-preserving observables
such as number or ``Z``; use nonzero operator charge for charge-changing
operators.

MPS measurements use direct tensor-network contraction. PEPS measurements use
quimb's PEPS plaquette-environment boundary contraction, so pass ``chi`` or
reusable plaquette-environment holders. Dense operators are converted to
Symmray arrays before calling quimb because quimb's Symmray PEPS gate path
expects block-sparse operators.

```python
import numpy as np

n_op = np.diag([0.0, 1.0])
zz_op = np.diag([1.0, -1.0, -1.0, 1.0])

psi.measure(n_op, where=3)
psi.measure(zz_op, where=(2, 3))

bdy_obs = {}
peps.measure(
    n_op,
    where=(1, 2),
    chi=32,
    bdy=bdy_obs,
    mode="mps",  # or "projector"
)
```

```{eval-rst}
.. automodule:: pepsy.tensors.symmetric
   :members:
   :show-inheritance:
```
