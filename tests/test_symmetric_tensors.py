"""Tests for Symmray-backed symmetric MPS/PEPS helpers."""

import numpy as np
import pytest

import pepsy
from pepsy.operators import gate, gate_simple
from pepsy.tensors import symmetric as symmetric_mod
from pepsy.tensors import (
    OneDMap,
    SymGateStream,
    SymHamiltonian,
    SymMPS,
    SymPEPS,
    default_physical_sectors,
    draw_symmray_blocks,
    draw_symmray_mps,
    draw_symmray_mpo,
    draw_symmray_peps,
    sector_index_map,
    site_charge_alternating,
    site_charge_from_map,
    site_charge_from_occupations,
    site_charge_uniform,
    symmray_block_summary,
    symmray_mps_summary,
    symmray_mpo_summary,
    symmray_peps_summary,
    symm_operator_from_dense,
)


sr = pytest.importorskip("symmray")


def test_symmray_block_summary_and_schematic_for_z2_gate():
    """Symmray gate helpers should expose and draw block-sector structure."""
    state = SymMPS.random(
        4,
        symmetry="Z2",
        phys_dim={0: 1, 1: 1},
        site_charge=site_charge_from_occupations([0] * 4),
        bond_dim=2,
        seed=1,
        dtype="complex128",
    )
    rz_gate = state.operator_from_dense(pepsy.rz(0.1), charge=0, sites=1)

    summary = symmray_block_summary(rz_gate)

    assert summary["shape"] == (2, 2)
    assert summary["num_blocks"] == 2
    assert summary["stored_size"] == 2
    assert summary["dense_size"] == 4
    assert [block["sector"] for block in summary["blocks"]] == [(0, 0), (1, 1)]
    assert [block["shape"] for block in summary["blocks"]] == [(1, 1), (1, 1)]
    assert summary["indices"][0]["direction"] == "out"
    assert summary["indices"][1]["direction"] == "in"

    pytest.importorskip("matplotlib")
    drawing, drawn_summary = draw_symmray_blocks(
        rz_gate,
        title="Z2 RZ gate",
        return_summary=True,
    )
    assert drawing is not None
    assert drawn_summary["blocks"] == summary["blocks"]


def test_symmray_mps_summary_and_schematic_for_z2_chain():
    """Symmray MPS drawings should expose site blocks and bond-sector metadata."""
    state = SymMPS.random(
        4,
        symmetry="Z2",
        phys_dim={0: 1, 1: 1},
        site_charge=site_charge_from_occupations([0] * 4),
        bond_dim=2,
        seed=2,
        dtype="complex128",
    )

    summary = symmray_mps_summary(state.tn)

    assert summary["num_sites"] == 4
    assert summary["max_bond_dim"] == 2
    assert summary["max_bond_sectors"] == 2
    assert summary["total_stored_size"] == 12
    assert summary["total_dense_size"] == 24
    assert summary["charge_total"] == summary["total_charge"]
    assert summary["Q_total"] == summary["total_parity"]
    assert summary["tensors"][0]["site"] == 0
    assert summary["tensors"][0]["physical"]["chargemap"] == {0: 1, 1: 1}
    assert summary["tensors"][0]["physical"]["direction"] in {"in", "out"}
    assert summary["tensors"][1]["left_bond"]["between"] == (0, 1)
    assert summary["bonds"][0]["chargemap"] == {0: 1, 1: 1}
    assert summary["bonds"][0]["left_direction"] in {"in", "out"}
    assert summary["bonds"][0]["right_direction"] in {"in", "out"}

    pytest.importorskip("matplotlib")
    drawing, drawn_summary = draw_symmray_mps(
        state.tn,
        title="Z2 MPS",
        return_summary=True,
    )
    assert hasattr(drawing, "fig")
    assert hasattr(drawing, "ax")
    assert drawn_summary["bonds"] == summary["bonds"]

    compat_drawing, compat_summary = draw_symmray_peps(
        state.tn,
        return_summary=True,
    )
    assert hasattr(compat_drawing, "fig")
    assert compat_summary["bonds"] == summary["bonds"]


def test_symmray_mpo_summary_and_schematic_for_z2_operator():
    """Symmray MPO drawings should expose upper/lower physical legs."""
    ham = SymHamiltonian.from_edges(
        "tfim",
        "Z2",
        [(0, 1)],
        jx=-1.0,
        hz=-0.5,
    )
    mpo = ham.to_mpo(L=2, compress=False)

    summary = symmray_mpo_summary(mpo)

    assert summary["num_sites"] == 2
    assert summary["symmetry"] == "Z2"
    assert summary["fermionic_ordering"]["network_kind"] == "mpo"
    assert summary["max_bond_dim"] == 5
    assert summary["max_bond_sectors"] == 2
    assert summary["tensors"][0]["upper_ind"] == mpo.upper_ind(0)
    assert summary["tensors"][0]["lower_ind"] == mpo.lower_ind(0)
    assert summary["tensors"][0]["upper_physical"]["chargemap"] == {0: 1, 1: 1}
    assert summary["tensors"][0]["lower_physical"]["chargemap"] == {0: 1, 1: 1}
    assert summary["tensors"][0]["lower_physical"]["direction"] in {"in", "out"}
    assert summary["bonds"][0]["between"] == (0, 1)
    assert summary["bonds"][0]["left_direction"] in {"in", "out"}
    assert summary["bonds"][0]["right_direction"] in {"in", "out"}

    pytest.importorskip("matplotlib")
    drawing, drawn_summary = draw_symmray_mpo(
        mpo,
        title="Z2 MPO",
        show_phys_labels=True,
        return_summary=True,
    )
    assert hasattr(drawing, "fig")
    assert hasattr(drawing, "ax")
    assert drawn_summary["bonds"] == summary["bonds"]

    compat_drawing, compat_summary = draw_symmray_peps(mpo, return_summary=True)
    assert hasattr(compat_drawing, "fig")
    assert compat_summary["bonds"] == summary["bonds"]


def test_symmray_mps_mpo_schematics_accept_onedmap_layout():
    """1D Symmray chain drawings should optionally render on a OneDMap grid."""
    pytest.importorskip("matplotlib")
    mapper = OneDMap(2, 2, mode="snake")
    state = SymMPS.random(
        4,
        symmetry="Z2",
        phys_dim={0: 1, 1: 1},
        site_charge=site_charge_from_occupations([0] * 4),
        bond_dim=2,
        seed=20,
        dtype="complex128",
    )

    mps_drawing, mps_summary = draw_symmray_mps(
        state.tn,
        mapper=mapper,
        max_sites=3,
        show_bond_labels=True,
        show_phys_labels=True,
        show_diagnostics=True,
        return_summary=True,
    )

    assert hasattr(mps_drawing, "fig")
    assert mps_summary["num_sites"] == 4
    assert mps_drawing.ax.get_aspect() == 1.0
    assert any("+1 sites hidden" in text.get_text() for text in mps_drawing.ax.texts)

    ham = SymHamiltonian.from_edges(
        "tfim",
        "Z2",
        [(0, 1), (1, 2), (2, 3)],
        jx=-1.0,
        hz=-0.5,
    )
    mpo = ham.to_mpo(L=4, compress=False)

    mpo_drawing, mpo_summary = draw_symmray_mpo(
        mpo,
        mapper=mapper,
        show_bond_labels=True,
        show_phys_labels=True,
        show_diagnostics=True,
        return_summary=True,
    )
    compat_drawing = draw_symmray_peps(mpo, mapper=mapper)

    assert hasattr(mpo_drawing, "fig")
    assert hasattr(compat_drawing, "fig")
    assert mpo_summary["num_sites"] == 4
    assert mpo_drawing.ax.get_aspect() == 1.0

    with pytest.raises(ValueError, match="does not match network length"):
        draw_symmray_mps(state.tn, mapper=OneDMap(3, 1))


def test_symmray_peps_summary_and_schematic_for_z2_grid():
    """Symmray PEPS drawings should expose grid bonds and block-sector metadata."""
    state = SymPEPS.random(
        2,
        2,
        symmetry="Z2",
        phys_dim={0: 1, 1: 1},
        site_charge=site_charge_from_occupations({(i, j): 0 for i in range(2) for j in range(2)}),
        bond_dim=2,
        seed=3,
        dtype="complex128",
    )

    summary = symmray_peps_summary(state)

    assert summary["Lx"] == 2
    assert summary["Ly"] == 2
    assert summary["num_sites"] == 4
    assert len(summary["bonds"]) == 4
    assert summary["max_bond_dim"] == 2
    assert summary["max_bond_sectors"] == 2
    assert summary["total_stored_size"] == 16
    assert summary["total_dense_size"] == 32
    assert summary["charge_total"] == summary["total_charge"]
    assert summary["Q_total"] == summary["total_parity"]
    assert summary["tensors"][0]["site"] == (0, 0)
    assert summary["tensors"][0]["physical"]["chargemap"] == {0: 1, 1: 1}
    assert summary["tensors"][0]["physical"]["direction"] in {"in", "out"}
    assert summary["tensors"][0]["bonds"]["right"]["between"] == ((0, 0), (0, 1))
    assert summary["tensors"][0]["bonds"]["down"]["between"] == ((0, 0), (1, 0))
    assert summary["bonds"][0]["site_a_direction"] in {"in", "out"}
    assert summary["bonds"][0]["site_b_direction"] in {"in", "out"}

    pytest.importorskip("matplotlib")
    drawing, drawn_summary = draw_symmray_peps(
        state,
        title="Z2 PEPS",
        return_summary=True,
    )
    assert hasattr(drawing, "fig")
    assert hasattr(drawing, "ax")
    assert drawn_summary["bonds"] == summary["bonds"]

    node_drawing = draw_symmray_peps(state, charge_in_node=True)
    assert hasattr(node_drawing, "fig")


def test_symmray_peps_schematic_hides_auxiliary_bonds_and_sectors_by_default():
    """PEPS drawings should keep routed/multibond debug structure opt-in."""
    state = SymPEPS.random(
        2,
        2,
        symmetry="Z2",
        phys_dim={0: 1, 1: 1},
        site_charge=site_charge_from_occupations({(i, j): 0 for i in range(2) for j in range(2)}),
        bond_dim=2,
        seed=31,
        dtype="complex128",
    )

    tn = state.peps.copy()
    top_ind = next(iter(tn[(0, 0)].bonds(tn[(0, 1)])))
    bottom_ind = next(iter(tn[(1, 0)].bonds(tn[(1, 1)])))
    tn.reindex_({bottom_ind: top_ind})
    state.peps = tn

    summary = symmray_peps_summary(state)
    assert len(summary["bonds"]) > len(state.edges)
    assert summary["num_extra_bonds"] > 0

    pytest.importorskip("matplotlib")
    drawing = draw_symmray_peps(state, show_bond_labels=True, show_tensor_labels=False)
    labels = [text.get_text() for text in drawing.ax.texts]
    bond_labels = [label for label in labels if label.startswith("$e_{")]

    assert len(bond_labels) == len(state.edges)
    assert not any("$q_e:$" in label for label in labels)

    debug_drawing = draw_symmray_peps(
        state,
        show_bond_labels=True,
        show_bond_sectors=True,
        show_extra_bonds=True,
        show_tensor_labels=False,
    )
    debug_labels = [text.get_text() for text in debug_drawing.ax.texts]
    debug_bond_labels = [label for label in debug_labels if label.startswith("$e_{")]

    assert len(debug_bond_labels) == len(summary["bonds"])
    assert any("$q_e:$" in label for label in debug_labels)


def test_symmray_peps_schematic_shows_spinful_charge_labels_inside_nodes():
    """Spin-resolved PEPS charges should render as white Q/Sz node labels."""
    pytest.importorskip("matplotlib")
    state = SymPEPS.random(
        2,
        2,
        symmetry="U1U1",
        phys_dim=default_physical_sectors(model="fermi_hubbard_u1u1"),
        fermionic=True,
        site_charge=site_charge_from_occupations(
            {
                (0, 0): (1, 0),
                (0, 1): (0, 1),
                (1, 0): (1, 0),
                (1, 1): (0, 1),
            }
        ),
        bond_dim=2,
        seed=30,
        dtype="complex128",
    )

    drawing = draw_symmray_peps(state, show_tensor_labels=False)
    labels = [text.get_text() for text in drawing.ax.texts]
    node_texts = [text for text in drawing.ax.texts if "$Q=" in text.get_text()]

    assert any("$Q=1$" in label and "$S_z=+1/2$" in label for label in labels)
    assert any("$Q=1$" in label and "$S_z=-1/2$" in label for label in labels)
    assert node_texts
    assert all(text.get_color() == (1.0, 1.0, 1.0, 1.0) for text in node_texts)


def test_symmetric_constructors_apply_to_backend_to_symmray_blocks():
    """Symmetric state/Hamiltonian constructors should backend-map stored blocks."""
    torch = pytest.importorskip("torch")
    to_backend = pepsy.backend_torch(dtype=torch.complex128)
    site_charge = site_charge_from_occupations(
        {
            (0, 0): (1, 0),
            (0, 1): (0, 1),
            (1, 0): (1, 0),
            (1, 1): (0, 1),
        }
    )

    state = SymPEPS.random(
        2,
        2,
        symmetry="U1U1",
        phys_dim=default_physical_sectors(model="fermi_hubbard_u1u1"),
        fermionic=True,
        site_charge=site_charge,
        bond_dim=2,
        seed=34,
        dtype="complex128",
        to_backend=to_backend,
    )

    tensor = next(iter(state.peps.tensor_map.values()))
    block = next(iter(tensor.data.blocks.values()))
    assert tensor.data.backend == "torch"
    assert isinstance(block, torch.Tensor)
    assert block.dtype == torch.complex128
    summary = symmray_peps_summary(state)
    assert summary["total_stored_size"] > 0

    ham = SymHamiltonian.from_edges(
        "fermi_hubbard_u1u1",
        "U1U1",
        state.edges,
        t=1.0,
        U=4.0,
        mu=0.0,
        to_backend=to_backend,
    )
    term = next(iter(ham.terms.values()))
    term_block = next(iter(term.blocks.values()))
    assert term.backend == "torch"
    assert isinstance(term_block, torch.Tensor)
    assert term_block.dtype == torch.complex128


def test_symmetric_as_scalar_handles_backend_scalars_before_numpy_conversion():
    """Backend scalar conversion should not require NumPy array coercion."""

    class BackendScalar:
        shape = ()

        def detach(self):
            return self

        def cpu(self):
            return self

        def item(self):
            return 1.25

        def __array__(self, *_args, **_kwargs):
            raise AssertionError("NumPy conversion should not be used.")

    class BackendVector:
        shape = (2,)

        def __array__(self, *_args, **_kwargs):
            raise AssertionError("Non-scalar backend arrays should be returned.")

    vector = BackendVector()

    assert symmetric_mod._as_scalar(BackendScalar()) == pytest.approx(1.25)
    assert symmetric_mod._as_scalar(vector) is vector


def test_symmetric_to_backend_copy_preserves_original_blocks():
    """to_backend(..., inplace=False) should convert a copied wrapper only."""
    torch = pytest.importorskip("torch")
    state = SymMPS.random(
        4,
        symmetry="Z2",
        phys_dim={0: 1, 1: 1},
        site_charge=site_charge_from_occupations([0, 0, 0, 0]),
        bond_dim=2,
        seed=35,
        dtype="complex128",
    )

    converted = state.to_backend(
        pepsy.backend_torch(dtype=torch.complex128),
        inplace=False,
    )
    original_tensor = next(iter(state.mps.tensor_map.values()))
    converted_tensor = next(iter(converted.mps.tensor_map.values()))
    original_block = next(iter(original_tensor.data.blocks.values()))
    converted_block = next(iter(converted_tensor.data.blocks.values()))

    assert converted is not state
    assert not isinstance(original_block, torch.Tensor)
    assert isinstance(converted_block, torch.Tensor)


def test_symmetric_state_uses_psi_with_network_compatibility_alias():
    """SymMPS should prefer psi/mps naming while keeping network compatibility."""
    state = SymMPS.random(
        4,
        symmetry="Z2",
        phys_dim={0: 1, 1: 1},
        site_charge=site_charge_from_occupations([0] * 4),
        bond_dim=2,
        seed=4,
        dtype="complex128",
    )

    assert state.psi is state.tn
    assert state.network is state.psi
    assert state.mps is state.psi

    wrapped_from_psi = SymMPS(
        psi=state.psi.copy(),
        symmetry=state.symmetry,
        edges=state.edges,
        site_ind_id=state.site_ind_id,
        phys_sectors=state.phys_sectors,
        site_charge=state.site_charge,
    )
    wrapped_from_network = SymMPS(
        network=state.psi.copy(),
        symmetry=state.symmetry,
        edges=state.edges,
        site_ind_id=state.site_ind_id,
        phys_sectors=state.phys_sectors,
        site_charge=state.site_charge,
    )
    wrapped_from_mps = SymMPS(
        mps=state.psi.copy(),
        symmetry=state.symmetry,
        edges=state.edges,
        site_ind_id=state.site_ind_id,
        phys_sectors=state.phys_sectors,
        site_charge=state.site_charge,
    )

    assert wrapped_from_psi.tn is wrapped_from_psi.psi
    assert wrapped_from_network.network is wrapped_from_network.psi
    assert wrapped_from_mps.mps is wrapped_from_mps.psi

    replacement = state.psi.copy()
    wrapped_from_psi.network = replacement
    assert wrapped_from_psi.psi is replacement

    with pytest.raises(TypeError, match="exactly one"):
        SymMPS(
            psi=state.psi,
            network=state.psi,
            symmetry=state.symmetry,
            edges=state.edges,
        )

    with pytest.raises(TypeError, match="only valid"):
        SymMPS(
            peps=state.psi,
            symmetry=state.symmetry,
            edges=state.edges,
        )


def test_sympeps_accepts_peps_constructor_alias():
    """SymPEPS should expose peps as the shape-specific wrapped state name."""
    state = SymPEPS.random(
        2,
        2,
        symmetry="Z2",
        phys_dim={0: 1, 1: 1},
        site_charge=site_charge_from_occupations({(i, j): 0 for i in range(2) for j in range(2)}),
        bond_dim=2,
        seed=5,
        dtype="complex128",
    )

    assert state.peps is state.psi
    assert state.network is state.psi

    wrapped = SymPEPS(
        peps=state.peps.copy(),
        symmetry=state.symmetry,
        edges=state.edges,
        site_ind_id=state.site_ind_id,
        phys_sectors=state.phys_sectors,
        site_charge=state.site_charge,
    )

    assert wrapped.peps is wrapped.psi
    assert wrapped.tn is wrapped.psi

    replacement = state.peps.copy()
    wrapped.peps = replacement
    assert wrapped.psi is replacement

    with pytest.raises(TypeError, match="only valid"):
        SymPEPS(
            mps=state.peps,
            symmetry=state.symmetry,
            edges=state.edges,
        )


def _square_lattice_edges(Lx, Ly):
    """Return nearest-neighbor square-lattice edges in row-major MPS order."""
    edges = []

    def site(x, y):
        return x * Ly + y

    for x in range(Lx):
        for y in range(Ly):
            if x + 1 < Lx:
                edges.append((site(x, y), site(x + 1, y)))
            if y + 1 < Ly:
                edges.append((site(x, y), site(x, y + 1)))
    return tuple(edges)


def _square_lattice_coordinate_edges(Lx, Ly):
    """Return nearest-neighbor square-lattice edges as PEPS coordinates."""
    edges = []
    for x in range(Lx):
        for y in range(Ly):
            if y + 1 < Ly:
                edges.append(((x, y), (x, y + 1)))
            if x + 1 < Lx:
                edges.append(((x, y), (x + 1, y)))
    return tuple(edges)


def _xy_u1_hamiltonian(edges):
    """Build the U(1)-symmetric XY Hamiltonian from an explicit dense term."""
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sy = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    xy_term_dense = 0.5 * (np.kron(sx, sx) + np.kron(sy, sy))
    xy_term = symm_operator_from_dense(
        xy_term_dense,
        {0: 1, 1: 1},
        symmetry="U1",
        charge=0,
        sites=2,
    )
    return SymHamiltonian(
        model="xy",
        symmetry="U1",
        edges=tuple(edges),
        terms={edge: xy_term for edge in edges},
        parameters={"J": 1.0},
    )


def _all_tensor_data_symmray(tn):
    """Return whether every tensor stores Symmray block-sparse data."""
    return all(
        hasattr(tensor.data, "blocks") and hasattr(tensor.data, "indices")
        for tensor in tn.tensors
    )


def _finite_double_layer_norm(tn):
    """Return whether the direct double-layer norm contraction is finite."""
    norm = (tn.H & tn).contract(all, optimize="auto-hq")
    return np.isfinite(np.real(norm))


def _raw_mps_norm(mps):
    """Return an MPS norm with PEPSY's stored exponent removed."""
    raw = mps.copy()
    if hasattr(raw, "exponent"):
        raw.exponent = 0.0
    return raw.norm()


def _build_3x3_symmray_mps_case(name):
    """Build an explicit 3x3 state, Hamiltonian, and gate stream."""
    edges = _square_lattice_edges(3, 3)

    if name == "itf_z2":
        state = SymMPS.random(
            9,
            symmetry="Z2",
            phys_dim={0: 1, 1: 1},
            site_charge=site_charge_from_occupations([0] * 9),
            bond_dim=4,
            seed=41,
            dtype="complex128",
        )
        hamiltonian = SymHamiltonian.from_edges(
            "itf",
            "Z2",
            edges,
            jx=-1.0,
            hz=-0.5,
        )
        gates = hamiltonian.gate_stream(0.001, imaginary=False)
        return state, hamiltonian, gates, False, 0

    if name == "xy_u1":
        occupations = [1, 0, 1, 0, 1, 0, 1, 0, 1]
        state = SymMPS.random(
            9,
            symmetry="U1",
            phys_dim={0: 1, 1: 1},
            site_charge=site_charge_from_occupations(occupations),
            bond_dim=4,
            seed=42,
            dtype="complex128",
        )
        hamiltonian = _xy_u1_hamiltonian(edges)
        gates = hamiltonian.gate_stream(0.001, imaginary=False)
        return state, hamiltonian, gates, False, sum(occupations)

    if name == "fermi_hubbard_u1":
        occupations = [1] * 9
        state = SymMPS.random(
            9,
            symmetry="U1",
            phys_dim={0: 1, 1: 2, 2: 1},
            fermionic=True,
            site_charge=site_charge_from_occupations(occupations),
            bond_dim=4,
            seed=43,
            dtype="complex128",
        )
        hamiltonian = SymHamiltonian.from_edges(
            "fermi_hubbard",
            "U1",
            edges,
            t=1.0,
            U=2.0,
            mu=0.1,
        )
        gates = hamiltonian.gate_stream(0.0005, imaginary=True)
        return state, hamiltonian, gates, True, sum(occupations)

    raise ValueError(f"Unknown symmetric MPS case {name!r}.")


def _build_3x3_symmray_peps_case(name, *, edges=None):
    """Build an explicit 3x3 PEPS state, Hamiltonian, and gate stream."""
    edges = _square_lattice_coordinate_edges(3, 3) if edges is None else tuple(edges)

    if name == "itf_z2":
        charges = {(i, j): 0 for i in range(3) for j in range(3)}
        state = SymPEPS.random(
            3,
            3,
            symmetry="Z2",
            phys_dim={0: 1, 1: 1},
            site_charge=site_charge_from_occupations(charges),
            bond_dim=2,
            seed=51,
            dtype="complex128",
        )
        hamiltonian = SymHamiltonian.from_edges(
            "itf",
            "Z2",
            edges,
            jx=-1.0,
            hz=-0.5,
        )
        gates = hamiltonian.gate_stream(0.001, imaginary=False)
        return state, hamiltonian, gates, 0

    if name == "xy_u1":
        charges = {(i, j): (i + j) % 2 for i in range(3) for j in range(3)}
        state = SymPEPS.random(
            3,
            3,
            symmetry="U1",
            phys_dim={0: 1, 1: 1},
            site_charge=site_charge_from_occupations(charges),
            bond_dim=2,
            seed=52,
            dtype="complex128",
        )
        hamiltonian = _xy_u1_hamiltonian(edges)
        gates = hamiltonian.gate_stream(0.001, imaginary=False)
        return state, hamiltonian, gates, sum(charges.values())

    if name == "fermi_hubbard_u1":
        charges = {(i, j): 1 for i in range(3) for j in range(3)}
        state = SymPEPS.random(
            3,
            3,
            symmetry="U1",
            phys_dim={0: 1, 1: 2, 2: 1},
            fermionic=True,
            site_charge=site_charge_from_occupations(charges),
            bond_dim=2,
            seed=53,
            dtype="complex128",
        )
        hamiltonian = SymHamiltonian.from_edges(
            "fermi_hubbard",
            "U1",
            edges,
            t=1.0,
            U=2.0,
            mu=0.1,
        )
        gates = hamiltonian.gate_stream(0.0005, imaginary=True)
        return state, hamiltonian, gates, sum(charges.values())

    raise ValueError(f"Unknown symmetric PEPS case {name!r}.")


def test_sector_and_charge_helpers_make_total_charge_explicit():
    """Physical sectors and local charges should be easy to inspect."""
    assert default_physical_sectors("U1", 4) == {0: 1, 1: 2, 2: 1}
    assert default_physical_sectors(model="fermi_hubbard") == {0: 1, 1: 2, 2: 1}
    assert default_physical_sectors(model="fermi_hubbard_u1u1") == {
        (0, 0): 1,
        (0, 1): 1,
        (1, 0): 1,
        (1, 1): 1,
    }
    assert sector_index_map({0: 1, 1: 2, 2: 1}) == {0: 0, 1: 1, 2: 1, 3: 2}

    occupations = [1, 0, 1, 0]
    state = SymMPS.random(
        4,
        symmetry="U1",
        bond_dim=2,
        phys_dim={0: 1, 1: 1},
        site_charge=site_charge_from_occupations(occupations),
        seed=31,
        dtype="complex128",
    )

    assert state.phys_sectors == {0: 1, 1: 1}
    assert state.site_charges() == {0: 1, 1: 0, 2: 1, 3: 0}
    assert state.overall_charge() == 2
    assert state.overall_parity() == 0
    assert site_charge_uniform(1)("anything") == 1
    assert site_charge_alternating(even=0, odd=1)((1, 2)) == 1
    assert site_charge_from_map({(0, 0): 1}, default=0)((1, 1)) == 0


def test_symmps_heisenberg_builds_energy_and_imaginary_step():
    """SymMPS should build U(1) Heisenberg terms and evolve in place."""
    state = SymMPS.for_model(
        "heisenberg",
        4,
        bond_dim=2,
        seed=1,
        dtype="complex128",
    )

    ham = state.build_hamiltonian()

    assert isinstance(ham, SymHamiltonian)
    assert state.symmetry == "U1"
    assert not state.fermionic
    assert len(ham.terms) == 3
    assert all(term.shape == (2, 2, 2, 2) for term in ham.terms.values())
    assert state.tn.L == 4

    energy_before = state.energy(ham)
    state.ground_state(dt=0.01, steps=1, hamiltonian=ham, max_bond=4)
    energy_after = state.energy(ham)

    assert np.isfinite(np.real(energy_before))
    assert np.isfinite(np.real(energy_after))
    assert state.tn.max_bond() <= 4
    assert state.norm() == pytest.approx(1.0)


def test_symmps_measures_dense_generic_observables():
    """SymMPS.measure should convert dense local operators to Symmray arrays."""
    state = SymMPS.for_model(
        "heisenberg",
        4,
        bond_dim=2,
        seed=32,
        dtype="complex128",
    )
    z_op = np.diag([1.0, -1.0])
    zz_op = np.diag([1.0, -1.0, -1.0, 1.0])
    z_sym = symm_operator_from_dense(
        z_op,
        state.phys_sectors,
        symmetry=state.symmetry,
        charge=0,
    )

    measured_dense = state.measure(z_op, where=1, contraction_opt="auto-hq")
    measured_sym = state.measure(z_sym, where=1, contraction_opt="auto-hq")
    measured_zz = state.measure(zz_op, where=(1, 2), contraction_opt="auto-hq")

    assert measured_dense == pytest.approx(measured_sym)
    assert np.isfinite(np.real(measured_zz))


def test_symmps_fermi_hubbard_defaults_to_fermionic_u1():
    """Fermi-Hubbard convenience defaults should use U(1) fermionic tensors."""
    state = SymMPS.for_model(
        "fermi_hubbard",
        3,
        bond_dim=2,
        seed=2,
        dtype="complex128",
    )

    ham = state.build_hamiltonian(t=1.0, U=4.0, mu=0.5)
    gates = state.trotter_gates(0.01, hamiltonian=ham, imaginary=True)

    assert state.symmetry == "U1"
    assert state.fermionic
    assert len(ham.terms) == 2
    assert all(term.shape == (4, 4, 4, 4) for term in ham.terms.values())
    assert isinstance(gates, SymGateStream)
    assert len(gates) == 2

    evolved = state.time_evolve(
        0.01,
        steps=1,
        hamiltonian=ham,
        imaginary=True,
        max_bond=4,
        inplace=False,
    )

    assert evolved is not state
    assert evolved.tn.max_bond() <= 4
    assert evolved.norm() == pytest.approx(1.0)


def test_fermi_hubbard_u1u1_preset_uses_spin_resolved_fermionic_tensors():
    """U1U1 preset should expose spin-resolved spinful Hubbard sectors."""
    sectors = default_physical_sectors(model="fermi_hubbard_u1u1")
    mps = SymMPS.for_model(
        "fermi_hubbard_u1u1",
        4,
        bond_dim=2,
        site_charge=site_charge_from_occupations([(1, 0), (0, 1), (1, 0), (0, 1)]),
        seed=3,
        dtype="complex128",
    )
    ham_mps = mps.build_hamiltonian(t=1.0, U=4.0, mu=0.0)
    evolved_mps = mps.time_evolve(
        0.001,
        steps=1,
        hamiltonian=ham_mps,
        imaginary=True,
        max_bond=4,
        inplace=False,
    )

    assert mps.symmetry == "U1U1"
    assert mps.fermionic
    assert mps.phys_sectors == sectors
    assert mps.overall_charge() == (2, 2)
    assert all(type(term).__name__ == "U1U1FermionicArray" for term in ham_mps.terms.values())
    mps_ordering = mps.fermionic_ordering()
    assert mps_ordering["enabled"] is True
    assert mps_ordering["network_kind"] == "mps"
    assert mps_ordering["methods_reference"]["doi"] == "10.1103/PhysRevResearch.7.023193"
    assert mps_ordering["site_order"] == (0, 1, 2, 3)
    assert mps_ordering["edge_order"] == mps.edges
    assert mps_ordering["edges"][0]["edge"] == (0, 1)
    assert mps_ordering["edges"][0]["edge_order"] == 0
    assert mps_ordering["edges"][0]["index_directions"][0]["site"] == 0
    assert mps_ordering["edges"][0]["index_directions"][0]["direction"] in {"in", "out"}
    assert evolved_mps.overall_charge() == (2, 2)
    assert evolved_mps.tn.max_bond() <= 4
    assert evolved_mps.norm() == pytest.approx(1.0)

    peps_charges = {
        (0, 0): (1, 0),
        (0, 1): (0, 1),
        (1, 0): (1, 0),
        (1, 1): (0, 1),
    }
    peps = SymPEPS.for_model(
        "fermi_hubbard_u1u1",
        2,
        2,
        bond_dim=2,
        site_charge=site_charge_from_occupations(peps_charges),
        seed=4,
        dtype="complex128",
    )
    ham_peps = peps.build_hamiltonian(t=1.0, U=4.0, mu=0.0)
    evolved_peps = peps.time_evolve(
        0.001,
        steps=1,
        hamiltonian=ham_peps,
        imaginary=True,
        max_bond=4,
        method="gate",
        inplace=False,
    )

    assert peps.symmetry == "U1U1"
    assert peps.fermionic
    assert peps.phys_sectors == sectors
    assert peps.overall_charge() == (2, 2)
    assert all(type(term).__name__ == "U1U1FermionicArray" for term in ham_peps.terms.values())
    peps_ordering = peps.fermionic_ordering()
    assert peps_ordering["enabled"] is True
    assert peps_ordering["network_kind"] == "peps"
    assert peps_ordering["methods_reference"]["doi"] == "10.1103/PhysRevResearch.7.023193"
    assert peps_ordering["site_order"] == ((0, 0), (0, 1), (1, 0), (1, 1))
    assert peps_ordering["edge_order"] == peps.edges
    peps_first_edge = peps.edges[0]
    peps_first_record = next(
        record for record in peps_ordering["edges"]
        if record["edge"] == peps_first_edge
    )
    assert peps_first_record["edge_order"] == 0
    assert tuple(item["site"] for item in peps_first_record["index_directions"]) == peps_first_edge
    assert peps_first_record["index_directions"][0]["direction"] in {"in", "out"}
    assert evolved_peps.overall_charge() == (2, 2)
    assert evolved_peps.tn.max_bond() <= 4


def test_fermi_hubbard_u1u1_hamiltonian_builds_mpo_energy_path():
    """The FH MPO should match adjacent two-site Symmray term energy."""
    state = SymMPS.for_model(
        "fermi_hubbard_u1u1",
        2,
        bond_dim=3,
        site_charge=site_charge_from_occupations([(1, 0), (0, 1)]),
        seed=11,
        dtype="complex128",
    )
    ham = SymHamiltonian.from_edges(
        "fermi_hubbard_u1u1",
        "U1U1",
        [(0, 1)],
        t=1.0,
        U=8.0,
        mu=0.2,
    )
    mpo = ham.to_mpo(L=2, compress=False)

    mpo_energy = pepsy.MpsEnergyOptimizer(
        state,
        mpo,
        energy_per_site=False,
        real=False,
    ).energy().energy
    term_energy = state.energy(
        hamiltonian=ham,
        normalized=True,
        contraction_opt="auto-hq",
    )

    assert complex(mpo_energy) == pytest.approx(complex(term_energy))


@pytest.mark.parametrize(
    "model,symmetry,site_charges,params",
    [
        ("tfim", "Z2", [0, 0], {"jx": -1.0, "hz": -0.5}),
        ("heisenberg", "U1", [1, 0], {}),
    ],
)
def test_symmetric_hamiltonian_to_mpo_supports_spin_models(
    model,
    symmetry,
    site_charges,
    params,
):
    """Generic SymHamiltonian MPOs should support non-fermionic Z2/U1 terms."""
    state = SymMPS.for_model(
        model,
        2,
        bond_dim=3,
        site_charge=site_charge_from_occupations(site_charges),
        seed=21,
        dtype="complex128",
    )
    ham = SymHamiltonian.from_edges(model, symmetry, [(0, 1)], **params)
    mpo = ham.to_mpo(L=2, compress=False)

    mpo_energy = pepsy.MpsEnergyOptimizer(
        state,
        mpo,
        energy_per_site=False,
        real=False,
    ).energy().energy
    term_energy = state.energy(
        hamiltonian=ham,
        normalized=True,
        contraction_opt="auto-hq",
    )

    assert mpo.L == 2
    assert type(mpo[0].data).__name__.startswith(symmetry)
    assert complex(mpo_energy) == pytest.approx(complex(term_energy))


def test_spinless_fermi_hubbard_u1_hamiltonian_builds_mpo_energy_path():
    """Spinless FH U1 MPOs should preserve the fermionic contraction signs."""
    state = SymMPS.for_model(
        "fermi_hubbard_spinless",
        2,
        bond_dim=3,
        site_charge=site_charge_from_occupations([1, 0]),
        seed=23,
        dtype="complex128",
    )
    ham = SymHamiltonian.from_edges(
        "fermi_hubbard_spinless",
        "U1",
        [(0, 1)],
        t=1.0,
        V=0.5,
        mu=0.1,
    )
    mpo = ham.to_mpo(L=2, compress=False)

    mpo_energy = pepsy.MpsEnergyOptimizer(
        state,
        mpo,
        energy_per_site=False,
        real=False,
    ).energy().energy
    term_energy = state.energy(
        hamiltonian=ham,
        normalized=True,
        contraction_opt="auto-hq",
    )

    assert type(mpo[0].data).__name__ == "U1Array"
    assert complex(mpo_energy) == pytest.approx(complex(term_energy))


def test_spinless_fermi_hubbard_u1_hamiltonian_mpo_compresses_long_range():
    """Spinless FH long-range MPOs should insert parity strings and compress."""
    state = SymMPS.for_model(
        "fermi_hubbard_spinless",
        3,
        bond_dim=3,
        site_charge=site_charge_from_occupations([1, 0, 0]),
        seed=25,
        dtype="complex128",
    )
    ham = SymHamiltonian.from_edges(
        "fermi_hubbard_spinless",
        "U1",
        [(0, 2)],
        t=1.0,
        V=0.0,
        mu=0.0,
    )
    mpo = ham.to_mpo(L=3, compress=False)
    mpo_compressed = ham.to_mpo(L=3, compress=True, max_bond=16, cutoff=1e-12)

    energy = pepsy.MpsEnergyOptimizer(
        state,
        mpo,
        energy_per_site=False,
        real=False,
    ).energy().energy
    energy_compressed = pepsy.MpsEnergyOptimizer(
        state,
        mpo_compressed,
        energy_per_site=False,
        real=False,
    ).energy().energy

    assert mpo.max_bond() >= 3
    assert mpo_compressed.max_bond() <= mpo.max_bond()
    assert complex(energy_compressed) == pytest.approx(complex(energy))


def test_spinless_fermi_hubbard_u1_hamiltonian_mpo_maps_2d_long_range_edge():
    """Spinless FH coordinate edges should match their mapped flat edge."""
    mapper = OneDMap(2, 2, mode="snake")
    idx2coo, coo2idx = mapper.build()
    edge_2d = ((0, 0), (1, 0))
    occupations = {
        (0, 0): 1,
        (0, 1): 0,
        (1, 0): 0,
        (1, 1): 0,
    }
    state = SymMPS.for_model(
        "fermi_hubbard_spinless",
        4,
        bond_dim=3,
        site_charge=site_charge_from_occupations(
            [occupations[idx2coo[i]] for i in range(4)]
        ),
        seed=26,
        dtype="complex128",
    )
    params = {"t": 1.0, "V": 0.0, "mu": 0.0}
    ham_2d = SymHamiltonian.from_edges(
        "fermi_hubbard_spinless",
        "U1",
        [edge_2d],
        **params,
    )
    flat_edge = tuple(coo2idx[site] for site in edge_2d)
    ham_flat = SymHamiltonian.from_edges(
        "fermi_hubbard_spinless",
        "U1",
        [flat_edge],
        **params,
    )

    mpo_from_mapper = ham_2d.to_mpo(mapper=mapper, compress=False)
    mpo_from_flat = ham_flat.to_mpo(L=4, compress=False)

    def energy(mpo):
        return pepsy.MpsEnergyOptimizer(
            state,
            mpo,
            energy_per_site=False,
            real=False,
        ).energy().energy

    assert abs(coo2idx[edge_2d[0]] - coo2idx[edge_2d[1]]) > 1
    assert complex(energy(mpo_from_mapper)) == pytest.approx(complex(energy(mpo_from_flat)))


def test_spinful_fermi_hubbard_total_u1_mpo_fails_clearly():
    """Spinful FH total-U1 MPOs are intentionally not routed through U1U1."""
    ham = SymHamiltonian.from_edges(
        "fermi_hubbard",
        "U1",
        [(0, 1)],
        t=1.0,
        U=4.0,
        mu=0.1,
    )

    with pytest.raises(NotImplementedError, match="total-U1"):
        ham.to_mpo(L=2)


def test_fermi_hubbard_u1u1_hamiltonian_mpo_handles_long_range_string():
    """Long-range mapped FH terms should include a parity string and compress."""
    state = SymMPS.for_model(
        "fermi_hubbard_u1u1",
        3,
        bond_dim=3,
        site_charge=site_charge_from_occupations([(1, 0), (0, 0), (0, 1)]),
        seed=12,
        dtype="complex128",
    )
    ham = SymHamiltonian.from_edges(
        "fermi_hubbard_u1u1",
        "U1U1",
        [(0, 2)],
        t=1.0,
        U=0.0,
        mu=0.0,
    )
    mpo = ham.to_mpo(L=3, compress=False)
    mpo_compressed = ham.to_mpo(L=3, compress=True, max_bond=16, cutoff=1e-12)

    energy = pepsy.MpsEnergyOptimizer(
        state,
        mpo,
        energy_per_site=False,
        real=False,
    ).energy().energy
    energy_compressed = pepsy.MpsEnergyOptimizer(
        state,
        mpo_compressed,
        energy_per_site=False,
        real=False,
    ).energy().energy

    assert mpo.max_bond() >= 3
    assert mpo_compressed.max_bond() <= mpo.max_bond()
    assert complex(energy_compressed) == pytest.approx(complex(energy))


def test_fermi_hubbard_u1u1_hamiltonian_mpo_maps_2d_edges_with_onedmap():
    """Coordinate FH edges should align with an explicit OneDMap chain path."""
    mapper = OneDMap(2, 2, mode="snake")
    idx2coo, coo2idx = mapper.build()
    edges_2d = (
        ((0, 0), (1, 0)),
        ((0, 1), (1, 1)),
    )
    occupations = {
        (0, 0): (1, 0),
        (0, 1): (0, 1),
        (1, 0): (0, 1),
        (1, 1): (1, 0),
    }
    state = SymMPS.for_model(
        "fermi_hubbard_u1u1",
        4,
        bond_dim=3,
        site_charge=site_charge_from_occupations(
            [occupations[idx2coo[i]] for i in range(4)]
        ),
        seed=13,
        dtype="complex128",
    )
    params = {"t": 1.0, "U": 4.0, "mu": 0.25}
    ham_2d = SymHamiltonian.from_edges(
        "fermi_hubbard_u1u1",
        "U1U1",
        edges_2d,
        **params,
    )
    flat_edges = tuple((coo2idx[left], coo2idx[right]) for left, right in edges_2d)
    ham_flat = SymHamiltonian.from_edges(
        "fermi_hubbard_u1u1",
        "U1U1",
        flat_edges,
        **params,
    )

    mpo_from_mapper = ham_2d.to_mpo(mapper=mapper, compress=False)
    mpo_from_maps = ham_2d.to_mpo(idx2coo=idx2coo, coo2idx=coo2idx, compress=False)
    mpo_from_flat = ham_flat.to_mpo(L=4, compress=False)

    def energy(mpo):
        return pepsy.MpsEnergyOptimizer(
            state,
            mpo,
            energy_per_site=False,
            real=False,
        ).energy().energy

    energy_flat = energy(mpo_from_flat)
    assert mpo_from_mapper.L == 4
    assert mpo_from_maps.L == 4
    assert complex(energy(mpo_from_mapper)) == pytest.approx(complex(energy_flat))
    assert complex(energy(mpo_from_maps)) == pytest.approx(complex(energy_flat))


def test_fermi_hubbard_u1u1_hamiltonian_mpo_requires_mapper_for_2d_edges():
    """Coordinate FH edges need an explicit chain path for fermionic strings."""
    ham = SymHamiltonian.from_edges(
        "fermi_hubbard_u1u1",
        "U1U1",
        [((0, 0), (1, 0))],
        t=1.0,
        U=0.0,
        mu=0.0,
    )

    with pytest.raises(ValueError, match="requires mapper=OneDMap"):
        ham.to_mpo()


def test_symmps_gate_stream_runs_mps_optimizer_mpo_heisenberg():
    """Symmray U(1) gates should run through MpsOptimizer(mode='mpo')."""
    state = SymMPS.for_model(
        "heisenberg",
        4,
        bond_dim=2,
        seed=7,
        dtype="complex128",
    )
    ham = state.build_hamiltonian()
    gates = ham.gate_stream(0.01)

    opt = pepsy.MpsOptimizer(state.tn.copy(), gates, chi=4, mode="mpo")
    out = opt.run(progbar=False, cutoff=1e-10, fidelity_samples=0)

    assert out.L == 4
    assert out.max_bond() <= 4
    assert np.isfinite(np.real((out.H & out).contract(all, optimize="auto-hq")))


def test_symmps_mps_optimizer_handles_spinful_fermi_hubbard_dims():
    """MpsOptimizer MPO mode should accept 4-state Fermi-Hubbard gates."""
    state = SymMPS.for_model(
        "fermi_hubbard",
        3,
        bond_dim=2,
        seed=8,
        dtype="complex128",
    )
    ham = state.build_hamiltonian(t=1.0, U=2.0, mu=0.1)

    evolved = state.time_evolve_mps_optimizer(
        0.005,
        steps=1,
        hamiltonian=ham,
        imaginary=True,
        chi=4,
        inplace=False,
    )

    assert evolved is not state
    assert evolved.tn.L == 3
    assert evolved.tn.max_bond() <= 4
    assert np.isfinite(np.real(evolved.norm()))
    raw = evolved.tn.copy()
    raw.exponent = 0
    raw_norm = (raw.H & raw).contract(all, optimize="auto-hq")
    assert np.isfinite(np.real(raw_norm))
    assert np.real(raw_norm) > 0.0


def test_symmps_mps_optimizer_coerces_dense_hamiltonian_terms():
    """Dense custom Hamiltonian terms should not mix NumPy gates into Symmray MPS."""
    state = SymMPS.for_model(
        "heisenberg",
        4,
        bond_dim=2,
        seed=12,
        dtype="complex128",
    )
    zz_term = np.diag([1.0, -1.0, -1.0, 1.0]).reshape(2, 2, 2, 2)
    dense_hamiltonian = {(i, i + 1): zz_term for i in range(3)}

    ham = state.require_hamiltonian(hamiltonian=dense_hamiltonian)
    assert all(type(term).__module__.split(".")[0] == "symmray" for term in ham.terms.values())

    evolved = state.time_evolve_mps_optimizer(
        0.01,
        steps=1,
        hamiltonian=dense_hamiltonian,
        chi=4,
        mode="mpo",
        run_kwargs={"progbar": False, "fidelity_samples": 5},
        inplace=False,
    )

    assert evolved is not state
    assert evolved.tn.L == 4
    assert evolved.tn.max_bond() <= 4
    assert np.isfinite(np.real(evolved.norm()))


@pytest.mark.parametrize("case_name", ["itf_z2", "xy_u1", "fermi_hubbard_u1"])
@pytest.mark.parametrize("mode", ["dmrg", "mpo", "swap", "svd", "exact"])
def test_symmps_mps_optimizer_3x3_streams_cover_supported_modes(case_name, mode):
    """Explicit 3x3 Symmray streams should run through supported MPS modes."""
    state, hamiltonian, gates, non_unitary, expected_charge = _build_3x3_symmray_mps_case(
        case_name
    )

    assert len(hamiltonian.edges) == 12
    assert len(gates) == 12
    assert state.L == 9
    assert state.overall_charge() == expected_charge
    valid_gate_shapes = {(2, 2, 2, 2), (4, 4, 4, 4)}
    assert all(term.shape in valid_gate_shapes for term in hamiltonian.terms.values())
    assert all(gate.shape in valid_gate_shapes for gate, _ in gates)

    opt = pepsy.MpsOptimizer(state.tn.copy(), gates, chi=4, mode=mode)
    run_kwargs = {
        "progbar": False,
        "cutoff": 1.0e-10,
        "fidelity_samples": 0,
        "n_iter": 4,
    }
    if non_unitary and mode != "exact":
        run_kwargs.update(
            {
                "non_unitary": True,
                "normalize_every": 1,
                "normalize_final": True,
            }
        )

    out = opt.run(**run_kwargs)

    assert _all_tensor_data_symmray(out)
    assert _finite_double_layer_norm(out)

    if mode == "exact":
        assert len(out.tensors) == 1
    else:
        assert out.L == 9
        assert out.max_bond() <= 4

    if non_unitary and mode != "exact":
        events = opt.get_normalizations()
        raw_norm = _raw_mps_norm(out)
        assert len(events) == len(gates)
        assert all(event["method"] == "local_tensors" for event in events)
        assert all(event["reason"] == "compression" for event in events)
        assert all(event["sites"] for event in events)
        assert all(np.isfinite(event["log10_scale"]) for event in events)
        assert np.isfinite(np.real(raw_norm))
        assert np.real(raw_norm) > 0.0
    else:
        assert opt.get_normalizations() == []


def test_symmps_mps_optimizer_symmray_dmrg_expansion_caveat_is_explicit():
    """DMRG should fail clearly when Symmray bond expansion would be required."""
    state = SymMPS.random(
        4,
        symmetry="Z2",
        phys_dim={0: 1, 1: 1},
        site_charge=site_charge_from_occupations([0] * 4),
        bond_dim=2,
        seed=46,
        dtype="complex128",
    )
    nearest_ham = SymHamiltonian.from_edges(
        "itf",
        "Z2",
        [(0, 1)],
        jx=-1.0,
        hz=-0.5,
    )
    nonlocal_ham = SymHamiltonian.from_edges(
        "itf",
        "Z2",
        [(0, 2)],
        jx=-1.0,
        hz=-0.5,
    )

    nearest_gates = nearest_ham.gate_stream(0.001)
    nonlocal_gates = nonlocal_ham.gate_stream(0.001)

    for mode, gates in [
        ("mpo", nearest_gates),
        ("mpo", nonlocal_gates),
        ("svd", nonlocal_gates),
    ]:
        out = pepsy.MpsOptimizer(
            state.tn.copy(),
            gates,
            chi=2,
            mode=mode,
        ).run(progbar=False, cutoff=1.0e-10, fidelity_samples=0)
        assert out.L == 4
        assert out.max_bond() <= 2

    with pytest.raises(ValueError, match="bond_dim >= chi"):
        pepsy.MpsOptimizer(state.tn.copy(), nearest_gates, chi=4, mode="dmrg").run(
            progbar=False
        )


@pytest.mark.parametrize("mode", ["mpo", "svd"])
def test_symmps_mps_optimizer_symmray_auto_swap_tracks_infidelity(mode):
    """Symmray auto-swap fallbacks should still support true infidelity samples."""
    state = SymMPS.random(
        4,
        symmetry="Z2",
        phys_dim={0: 1, 1: 1},
        site_charge=site_charge_from_occupations([0] * 4),
        bond_dim=2,
        seed=47,
        dtype="complex128",
    )
    hamiltonian = SymHamiltonian.from_edges(
        "itf",
        "Z2",
        [(0, 2)],
        jx=-1.0,
        hz=-0.5,
    )

    opt = pepsy.MpsOptimizer(
        state.tn.copy(),
        hamiltonian.gate_stream(0.001),
        chi=2,
        mode=mode,
    )
    out = opt.run(
        progbar=False,
        cutoff=1.0e-10,
        fidelity_samples=0,
        track_infidelity=True,
    )

    samples = opt.get_infidelity_samples()
    assert out.L == 4
    assert out.max_bond() <= 2
    assert len(samples) == 1
    assert 0.0 <= samples[0]["fidelity"] <= 1.0
    assert 0.0 <= opt.get_infidelities()[-1] <= 1.0


def test_sympeps_tfim_builds_z2_terms_and_step():
    """SymPEPS should build Z2 TFIM terms on a square grid."""
    state = SymPEPS.for_model(
        "itf",
        2,
        2,
        bond_dim=2,
        seed=3,
        dtype="complex128",
    )

    ham = state.build_hamiltonian(jx=-1.0, hz=-0.5)

    assert state.symmetry == "Z2"
    assert not state.fermionic
    assert state.Lx == 2
    assert state.Ly == 2
    assert len(ham.terms) == 4
    assert all(term.shape == (2, 2, 2, 2) for term in ham.terms.values())

    state.time_evolve(
        0.005,
        steps=1,
        hamiltonian=ham,
        imaginary=False,
        max_bond=4,
        normalize=False,
    )

    assert state.tn.max_bond() <= 4
    assert np.isfinite(np.real(state.norm()))


def test_sympeps_measures_dense_generic_observables_and_parity():
    """SymPEPS.measure should use quimb PEPS boundary contraction."""
    charges = {(0, 0): 1, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    state = SymPEPS.random(
        2,
        2,
        symmetry="Z2",
        bond_dim=2,
        phys_dim={0: 1, 1: 1},
        site_charge=site_charge_from_occupations(charges),
        seed=33,
        dtype="complex128",
    )
    z_op = np.diag([1.0, -1.0])

    bdy = {}
    measured = state.measure(
        z_op,
        where=(0, 0),
        contraction_opt="auto-hq",
        chi=4,
        bdy=bdy,
        progress=False,
    )
    exact = pepsy.measure_obs(
        state.tn,
        state.operator_from_dense(z_op),
        where=(0, 0),
        ind_id=state.site_ind_id,
        contraction_opt="auto-hq",
    )

    assert state.site_charges() == charges
    assert state.overall_parity() == 1
    assert "plaquette_envs" in bdy
    assert "plaquette_map" in bdy
    assert measured == pytest.approx(exact)


def test_sympeps_measure_requires_chi_without_boundary_holder():
    """Quimb PEPS boundary measurement should make chi selection explicit."""
    state = SymPEPS.for_model("itf", 2, 2, bond_dim=2, seed=34, dtype="complex128")
    z_op = np.diag([1.0, -1.0])

    with pytest.raises(ValueError, match="Provide chi"):
        state.measure(z_op, where=(0, 0), progress=False)


def test_sympeps_measure_delegates_to_quimb_boundary_modes():
    """Quimb MPS/projector boundary modes and CTMRG should accept Symmray data."""
    state = SymPEPS.for_model("itf", 3, 3, bond_dim=2, seed=44, dtype="complex128")
    z_op = np.diag([1.0, -1.0])
    z_sym = state.operator_from_dense(z_op)

    direct_quimb = state.tn.compute_local_expectation(
        {(1, 1): z_sym},
        max_bond=8,
        normalized=True,
        mode="mps",
        contract_optimize="auto-hq",
    )
    wrapped_mps = state.measure(
        z_op,
        where=(1, 1),
        chi=8,
        mode="mps",
        contraction_opt="auto-hq",
        progress=False,
    )
    wrapped_projector = state.measure(
        z_op,
        where=(1, 1),
        chi=8,
        mode="projector",
        contraction_opt="auto-hq",
        progress=False,
    )
    wrapped_ctmrg_alias = state.measure(
        z_op,
        where=(1, 1),
        chi=8,
        mode="ctmrg",
        contraction_opt="auto-hq",
        progress=False,
    )

    norm = state.tn.make_norm()
    exact_norm = norm.contract(all, optimize="auto-hq")
    ctmrg_norm = norm.contract_ctmrg(
        max_bond=8,
        mode="projector",
        final_contract=True,
        final_contract_opts={"optimize": "auto-hq"},
        progbar=False,
    )

    assert wrapped_mps == pytest.approx(direct_quimb)
    assert wrapped_projector == pytest.approx(direct_quimb)
    assert wrapped_ctmrg_alias == pytest.approx(direct_quimb)
    assert ctmrg_norm == pytest.approx(exact_norm)


def test_sympeps_gate_stream_runs_pepsy_gate_and_gate_simple():
    """SymPEPS gate streams should work with PEPSY gate wrappers."""
    state = SymPEPS.for_model(
        "heisenberg",
        2,
        2,
        bond_dim=2,
        seed=9,
        dtype="complex128",
    )
    ham = state.build_hamiltonian()
    gates = ham.gate_stream(0.005)

    out_gate = state.copy().apply_gates(
        gates,
        method="gate",
        max_bond=4,
        cutoff=1e-10,
    )
    gauges = {}
    out_simple = state.copy().apply_gates(
        gates,
        method="simple",
        gauges=gauges,
        max_bond=4,
        cutoff=1e-10,
    )

    assert out_gate.tn.max_bond() <= 4
    assert out_simple.tn.max_bond() <= 4
    assert len(gauges) > 0
    assert np.isfinite(np.real(out_gate.norm()))
    assert np.isfinite(np.real(out_simple.norm()))


@pytest.mark.parametrize("case_name", ["itf_z2", "xy_u1", "fermi_hubbard_u1"])
@pytest.mark.parametrize("method", ["gate", "simple"])
def test_sympeps_gate_wrappers_3x3_streams_cover_symmetries(case_name, method):
    """PEPSY gate wrappers should handle 3x3 Symmray PEPS gate streams."""
    state, hamiltonian, gates, expected_charge = _build_3x3_symmray_peps_case(
        case_name
    )

    assert len(hamiltonian.edges) == 12
    assert len(gates) == 12
    assert state.Lx == 3
    assert state.Ly == 3
    assert state.overall_charge() == expected_charge

    if method == "gate":
        out = gate(
            state.tn.copy(),
            gates,
            max_bond=4,
            cutoff=1.0e-10,
            inplace=False,
        )
    else:
        gauges = {}
        out = gate_simple(
            state.tn.copy(),
            gates,
            gauges=gauges,
            max_bond=4,
            cutoff=1.0e-10,
            inplace=False,
        )
        assert len(gauges) == len(gates)

    assert out.Lx == 3
    assert out.Ly == 3
    assert out.max_bond() <= 4
    assert _all_tensor_data_symmray(out)


@pytest.mark.parametrize("case_name", ["itf_z2", "xy_u1", "fermi_hubbard_u1"])
@pytest.mark.parametrize("method", ["gate", "simple"])
def test_sympeps_gate_wrappers_route_nonlocal_symmray_swaps(case_name, method):
    """Internal routed SWAPs should be Symmray arrays for nonlocal PEPS gates."""
    nonlocal_edge = (((0, 0), (2, 2)),)
    state, _, gates, _ = _build_3x3_symmray_peps_case(
        case_name,
        edges=nonlocal_edge,
    )

    if method == "gate":
        out = gate(
            state.tn.copy(),
            gates,
            max_bond=4,
            cutoff=1.0e-10,
            inplace=False,
        )
    else:
        gauges = {}
        out = gate_simple(
            state.tn.copy(),
            gates,
            gauges=gauges,
            max_bond=4,
            cutoff=1.0e-10,
            inplace=False,
        )
        assert len(gauges) > 0

    assert out.max_bond() <= 4
    assert _all_tensor_data_symmray(out)


def test_sympeps_gate_method_preserves_pepsy_gate_contract_default(monkeypatch):
    """SymPEPS method='gate' should not override pepsy.gate's default."""
    state = SymPEPS.for_model(
        "heisenberg",
        2,
        2,
        bond_dim=2,
        seed=12,
        dtype="complex128",
    )
    calls = []

    def _fake_gate(tn, gates, **kwargs):
        calls.append((gates, kwargs.copy()))
        return tn

    monkeypatch.setattr("pepsy.operators.gate", _fake_gate)

    out = state.copy().apply_gates(
        ((np.eye(2, dtype=np.complex128), ((0, 0),)),),
        method="gate",
    )

    assert out.tn is not None
    assert "contract" not in calls[0][1]


def test_sympeps_raw_pepsy_gate_functions_accept_symmray_streams():
    """The plain gate functions should accept a SymGateStream directly."""
    state = SymPEPS.for_model("itf", 2, 2, bond_dim=2, seed=10, dtype="complex128")
    gates = state.build_hamiltonian().gate_stream(0.005)
    gauges = {}

    out_gate = gate(state.tn.copy(), gates, max_bond=4, cutoff=1e-10, inplace=False)
    out_simple = gate_simple(
        state.tn.copy(),
        gates,
        gauges=gauges,
        max_bond=4,
        cutoff=1e-10,
        inplace=False,
    )

    assert out_gate.max_bond() <= 4
    assert out_simple.max_bond() <= 4
    assert len(gauges) > 0


def test_symmetric_classes_are_top_level_lazy_exports():
    """Top-level pepsy exports should resolve to the tensor namespace classes."""
    assert pepsy.SymHamiltonian is SymHamiltonian
    assert pepsy.SymGateStream is SymGateStream
    assert pepsy.SymMPS is SymMPS
    assert pepsy.SymPEPS is SymPEPS
    assert pepsy.default_physical_sectors is default_physical_sectors
    assert pepsy.symm_operator_from_dense is symm_operator_from_dense
