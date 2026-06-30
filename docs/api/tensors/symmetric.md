# Symmetric Tensor States

## Choosing sectors and charges

Physical sectors are charge maps: ``{charge: sector_size}``. For example,
``{0: 1, 1: 1}`` is a two-state U(1) or Z2 local space, while
``{0: 1, 1: 2, 2: 1}`` is the spinful Fermi-Hubbard local space with total
particle-number U(1). For spin-resolved particle-number sectors use
``model="fermi_hubbard_u1u1"``, whose local charges are ``(n_up, n_down)``.

```python
import pepsy as py

spinless = py.default_physical_sectors("U1", 2)
spinful = py.default_physical_sectors(model="fermi_hubbard")
spinful_spin_resolved = py.default_physical_sectors(model="fermi_hubbard_u1u1")

psi = py.SymMPS.random(
    6,
    symmetry="U1",
    phys_dim=spinless,
    bond_dim={0: 2, 1: 2},
    site_charge=py.site_charge_from_occupations([1, 0, 1, 0, 1, 0]),
)
```

The local ``site_charge`` pattern fixes the global sector represented by the
state. For U(1), ``psi.overall_charge()`` is the sum of local charges. For
``U1U1``, it is a pair such as ``(N_up, N_down)``. For Z2,
``psi.overall_parity()`` is the charge sum modulo two.

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

```python
half_filled_4x4 = py.site_charge_from_occupations(
    [(1, 0), (0, 1)] * 8,
)

fh = py.SymMPS.for_model(
    "fermi_hubbard_u1u1",
    16,
    bond_dim=4,
    site_charge=half_filled_4x4,
)

fh.overall_charge()  # (8, 8)
```

For direct fermionic Fermi-Hubbard examples, use Gao et al., "Fermionic tensor
network contraction for arbitrary geometries", Phys. Rev. Research 7, 023193
(2025), https://doi.org/10.1103/PhysRevResearch.7.023193 as the primary
methods reference. Pepsy should keep fermionic parity, additional Abelian
symmetries such as ``U1`` or ``U1U1``, and leg-order metadata in Symmray-backed
arrays, while relying on quimb/cotengra graph optimizers for contraction
ordering.

Use ``state.fermionic_ordering()`` when a workflow needs the package-level
record of the graph and local order data carried by a symmetric state:

```python
ordering = fh.fermionic_ordering()

ordering["enabled"]       # True for direct fermionic states
ordering["site_order"]    # the site labels in tensor-network order
ordering["edge_order"]    # the stored graph edge order
ordering["edges"][0]["index_directions"]
```

The same record is available as ``summary["fermionic_ordering"]`` from
``symmray_mps_summary`` and ``symmray_peps_summary``.

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

## Inspecting Symmray blocks

Use ``symmray_block_summary`` to inspect the charge sectors and stored block
dimensions of any Symmray-backed local operator or tensor. Use
``draw_symmray_blocks`` for a lightweight ``quimb.schematic`` drawing of the
same information, including the stored-vs-dense entry count.

For whole MPS states, use ``symmray_mps_summary`` and ``draw_symmray_mps``. For
PEPS, use ``symmray_peps_summary`` and ``draw_symmray_peps``. These expose the
scientific structure that is usually hidden in a dense drawing: each site
tensor's block count, physical charge sectors, virtual-bond sector maps, and
aggregate block-sparse storage density. The default schematics follow the
compact quimb style with tensor nodes, physical legs, and bond arrows; extended
bond/physical labels and diagnostics are opt-in.

In the detailed drawing mode, ``T_i`` is the site tensor, ``B`` is the number of
stored block sectors, ``e_i`` is a virtual bond, and ``q_e``/``q_p`` are the
virtual/physical charge-sector maps. PEPS node circles show compact white charge
labels by default: spin-resolved two-component charges use total charge ``Q``
and spin projection ``S_z=(N_up-N_down)/2``, while other charges use the raw
Symmray charge ``q`` and total particle number ``N`` where available. Set
``charge_in_node=False`` to move the raw charge back outside the node with the
tensor label. Bond labels include the two local index orientations, for example
``out->in``, so the charge-flow convention is visible on the same line as the
bond dimension. Diagnostics include both ``charge_total`` and ``Q_total``; for
``Z2`` states ``Q_total`` is reduced modulo two. Colored block tiles are
available with ``show_blocks=True`` for a focused block-sector view, but the
overview diagrams leave them off by default and show ``B`` instead.
The MPS/PEPS summary dictionaries also expose ``charge_total`` and ``Q_total``
alongside the legacy ``total_charge`` key.

```python
psi = py.SymMPS.for_model(
    "itf",
    4,
    bond_dim=2,
    site_charge=py.site_charge_from_occupations([0] * 4),
)
rz_gate = psi.operator_from_dense(py.rz(0.1), charge=0, sites=1)

summary = py.symmray_block_summary(rz_gate)
summary["blocks"]

gate_drawing = py.draw_symmray_blocks(rz_gate, title="Z2 RZ gate")
display(gate_drawing.fig)

mps_summary = py.symmray_mps_summary(psi.tn)
mps_summary["bonds"]

mps_drawing = py.draw_symmray_mps(psi.tn, title="Symmray ITF MPS")
display(mps_drawing.fig)

detailed_mps_drawing = py.draw_symmray_mps(
    psi.tn,
    title="Symmray ITF MPS with dimensions",
    show_bond_labels=True,
    show_phys_labels=True,
    show_diagnostics=True,
)
display(detailed_mps_drawing.fig)

peps = py.SymPEPS.for_model("itf", 3, 3, bond_dim=2)
peps_summary = py.symmray_peps_summary(peps)
peps_summary["bonds"]

peps_drawing = py.draw_symmray_peps(
    peps,
    title="Symmray ITF PEPS",
    show_bond_labels=True,
    show_phys_labels=True,
    show_diagnostics=True,
)
display(peps_drawing.fig)
```

At the end of a Symmray ``MpsOptimizer`` notebook, pass the optimized chain
directly:

```python
opt = py.MpsOptimizer(psi.tn.copy(), gates, chi=8, mode="mpo")
opt.run(progbar=False)

py.draw_symmray_mps(
    opt.p,
    title="Final Symmray ITF MPS",
    center="middle",
    show_bond_labels=True,
    show_diagnostics=True,
)
```

```{eval-rst}
.. automodule:: pepsy.tensors.symmetric
   :members:
   :show-inheritance:
```
