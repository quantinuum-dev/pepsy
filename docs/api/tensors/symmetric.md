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

## Backend conversion

Pass ``to_backend=`` to build Symmray blocks directly on a chosen array
backend. The callable is applied to each stored dense block, preserving the
Symmray charge maps and sparse block structure.

```python
import torch

to_backend = py.backend_torch(dtype=torch.complex128)
peps_site_charge = py.site_charge_from_occupations(
    {
        (i, j): (1, 0) if (i + j) % 2 == 0 else (0, 1)
        for i in range(3)
        for j in range(3)
    }
)

peps = py.SymPEPS.random(
    3,
    3,
    symmetry="U1U1",
    fermionic=True,
    phys_dim=py.default_physical_sectors(model="fermi_hubbard_u1u1"),
    site_charge=peps_site_charge,
    bond_dim=4,
    to_backend=to_backend,
)

ham = py.SymHamiltonian.from_edges(
    "fermi_hubbard_u1u1",
    "U1U1",
    peps.edges,
    t=1.0,
    U=8.0,
    mu=0.0,
    to_backend=to_backend,
)

# Existing wrappers can be converted too.
peps_torch = peps.to_backend(to_backend, inplace=False)
ham_torch = ham.to_backend(to_backend, inplace=False)
```

## Symmetric MPO mapping

Symmetric Hamiltonians can be flattened into an MPS-chain MPO with
``SymHamiltonian.to_mpo(...)``. Coordinate edges from a 2D or 3D lattice can be
mapped with an explicit ``OneDMap`` path. The mapping is required for
coordinate edges because it defines where nonlocal chain channels, including
fermionic parity strings, sit.

Current support is:

- generic charge-neutral rank-4 two-site terms, including ``tfim``/``Z2`` and
  ``heisenberg``/``U1``;
- spinless Fermi-Hubbard ``model="fermi_hubbard_spinless"`` with ``U1`` or
  ``Z2`` symmetry, hopping, density interaction, and chemical-potential terms
  (pairing ``delta != 0`` is not implemented yet);
- spinful Fermi-Hubbard ``model="fermi_hubbard_u1u1"`` with
  ``symmetry="U1U1"``, hopping, onsite interaction, nearest-neighbor density
  interaction, and chemical-potential terms.

Spinful total-particle-number ``model="fermi_hubbard"`` with ``symmetry="U1"``
still raises ``NotImplementedError``; use ``model="fermi_hubbard_u1u1"`` when
an MPO is required for spinful Fermi-Hubbard.

For spinful ``U1U1`` Fermi-Hubbard, the local physical space has four spinful
Hubbard states with charges ``(n_up, n_down)``. Onsite terms use
``U * n_up * n_down - mu_up * n_up - mu_down * n_down``. ``mu`` may be a scalar
or ``(mu_up, mu_down)``. Hopping terms use
``-t_sigma c^dagger_i_sigma c_j_sigma`` plus the reverse direction, and ``t``
may be a scalar or ``(t_up, t_down)``. Nearest-neighbor density terms use
``V * (n_up + n_down)_i * (n_up + n_down)_j`` on each supplied edge; ``V`` may
be a scalar, edge mapping, or edge callable.

For fermionic models, non-adjacent mapped hopping edges insert the dense
fermionic parity operator on every intermediate chain site, so a 2D
nearest-neighbor edge can become a nonlocal MPS-chain term without dropping
fermionic signs.

- The returned object is a ``quimb.tensor.MatrixProductOperator`` whose tensor
  data are Symmray block-sparse arrays. Its physical index families default to
  ``k{}`` and ``b{}``, matching Pepsy's MPS/MPO conventions.
- ``to_mpo`` assembles the symmetry-preserving MPO first. The default
  ``compress=True`` then calls quimb MPO compression; pass ``compress=False``
  when you want to inspect or compare the uncompressed assembled MPO exactly.

```python
mapper = py.OneDMap(Lx=4, Ly=4, mode="snake")
idx2coo, coo2idx = mapper.build()

ham = py.SymHamiltonian.from_edges(
    "fermi_hubbard_u1u1",
    "U1U1",
    square_lattice_edges,
    t=1.0,
    U=8.0,
    V=0.25,
    mu=0.0,
)

mpo = ham.to_mpo(mapper=mapper)

# Equivalent when a workflow already stores the maps explicitly:
mpo = ham.to_mpo(idx2coo=idx2coo, coo2idx=coo2idx)
```

For a Hamiltonian whose edges are already integer MPS-chain sites, pass
``L=...`` instead of a coordinate mapper:

```python
ham = py.SymHamiltonian.from_edges(
    "fermi_hubbard_u1u1",
    "U1U1",
    [(0, 2)],
    t=(1.0, 0.8),
    U=4.0,
    mu=(0.1, 0.2),
)

mpo = ham.to_mpo(L=3, compress=True, max_bond=16, cutoff=1e-12)
```

``to_mpo`` also accepts ``to_backend=`` to map each stored Symmray block to an
array backend, and ``dtype=`` to choose the dense local operator dtype used
before conversion to block-sparse arrays. The implementation is validated by
checking that supported adjacent MPO energies agree with the local two-site
``SymHamiltonian`` energy path, that compressed long-range MPOs preserve the
uncompressed energy, and that ``OneDMap`` coordinate edges match equivalent
flat integer edges.

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
MPOs, use ``symmray_mpo_summary`` and ``draw_symmray_mpo``. For PEPS, use
``symmray_peps_summary`` and ``draw_symmray_peps``. These expose the scientific
structure that is usually hidden in a dense drawing: each site tensor's block
count, physical charge sectors, virtual-bond sector maps, and aggregate
block-sparse storage density. The default schematics follow the compact quimb
style with tensor nodes, physical legs, and bond arrows; extended bond/physical
labels and diagnostics are opt-in.
MPS and MPO drawings also accept ``mapper=OneDMap(...)`` to place the 1D chain
on its 2D lattice path. The mapped view keeps the charge-sector labels but uses
site-colored nodes and quieter gray bonds instead of left/right region shading.
For backwards-compatible notebook use, ``draw_symmray_peps`` also dispatches to
the MPS/MPO drawers when the input is a 1D MPS or MPO object.

In the detailed drawing mode, ``T_i`` is the site tensor, ``B`` is the number of
stored block sectors, and ``e_i`` is a virtual bond. PEPS node circles show
compact white charge labels by default: spin-resolved two-component charges use
total charge ``Q`` and spin projection ``S_z=(N_up-N_down)/2``, while other
charges use the raw Symmray charge ``q`` and total particle number ``N`` where
available. Set ``charge_in_node=False`` to move the raw charge back outside the
node with the tensor label. Bond labels include the two local index
orientations, for example ``out->in``, so the charge-flow convention is visible
on the same line as the bond dimension. Use ``show_bond_sectors=True`` to add
compact ``q_e`` virtual-bond sector maps, and ``show_phys_labels=True`` with
``show_leg_chargemaps=True`` to show ``q_p`` physical-sector maps. For PEPS
wrappers, the overview draws one primary shared index per configured
``SymPEPS.edges`` entry; set ``show_extra_bonds=True`` to debug all non-lattice
or multibond shared indices introduced by routing/gauges. Diagnostics include
both ``charge_total`` and ``Q_total``; for ``Z2`` states ``Q_total`` is reduced
modulo two. Colored block tiles are available with ``show_blocks=True`` for a
focused block-sector view, but the overview diagrams leave them off by default
and show ``B`` instead.
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

ham = py.SymHamiltonian.from_edges("tfim", "Z2", [(0, 1), (1, 2), (2, 3)])
mpo = ham.to_mpo(L=4)
mpo_summary = py.symmray_mpo_summary(mpo)
mpo_summary["bonds"]

mpo_drawing = py.draw_symmray_mpo(
    mpo,
    title="Symmray ITF MPO",
    show_bond_labels=True,
    show_phys_labels=True,
    show_diagnostics=True,
)
display(mpo_drawing.fig)

mapper = py.OneDMap(Lx=2, Ly=2, mode="snake")
mapped_mps_drawing = py.draw_symmray_mps(
    psi.tn,
    mapper=mapper,
    title="Symmray ITF MPS on OneDMap",
    show_bond_labels=True,
    show_phys_labels=True,
    show_diagnostics=True,
)
display(mapped_mps_drawing.fig)

mapped_mpo_drawing = py.draw_symmray_mpo(
    mpo,
    mapper=mapper,
    title="Symmray ITF MPO on OneDMap",
    show_bond_labels=True,
    show_phys_labels=True,
    show_diagnostics=True,
)
display(mapped_mpo_drawing.fig)

peps = py.SymPEPS.for_model("itf", 3, 3, bond_dim=2)
peps_summary = py.symmray_peps_summary(peps)
peps_summary["bonds"]

peps_drawing = py.draw_symmray_peps(
    peps,
    title="Symmray ITF PEPS",
    show_bond_labels=True,
    show_diagnostics=True,
)
display(peps_drawing.fig)

sector_debug = py.draw_symmray_peps(
    peps,
    title="Symmray ITF PEPS bond sectors",
    show_bond_labels=True,
    show_bond_sectors=True,
    show_extra_bonds=True,
    show_diagnostics=True,
)
display(sector_debug.fig)
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
