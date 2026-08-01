"""Small native-Symmray U1/U1U1 PEPO checks on a 2x3 lattice."""

import numpy as np
import pytest
import quimb.tensor as qtn

import pepsy
from pepsy.tensors import OneDMap


def _expectation(state, operator):
    acted = operator.apply(state, contract=True, compress=False)
    numerator = complex(
        np.asarray((state.H & acted).contract(all, optimize="auto-hq")).item()
    )
    denominator = complex(
        np.asarray((state.H & state).contract(all, optimize="auto-hq")).item()
    )
    return numerator / denominator


def _state(fermion, symmetry):
    lattice = (2, 3)
    sites = {
        (x, y): (
            1
            if symmetry == "U1"
            else ((1, 0) if (x + y) % 2 == 0 else (0, 1))
        )
        for x in range(lattice[0])
        for y in range(lattice[1])
    }
    state = pepsy.ps_to_peps(
        lattice,
        fermion=fermion,
        occupations=sites,
        seed=3,
        dtype="complex128",
        cyclic=False,
    )
    if symmetry == "U1":
        # Use a nontrivial charge-one local superposition, as in the Etienne
        # Neel-X state, while preserving total-U1 charge at every site.
        for x, y in sites:
            sign = -1.0 if (x + y) % 2 == 0 else 1.0
            tensor = state[x, y]
            (sector, block), = tensor.data.blocks.items()
            tensor.data.blocks[sector] = (
                np.asarray([1.0, sign], dtype=np.complex128)
                .reshape(block.shape)
                / np.sqrt(2.0)
            )
    return state


@pytest.mark.parametrize("symmetry", ["U1", "U1U1"])
def test_native_fermion_pepo_2x3_gate_simple_matches_state(symmetry):
    """Native U1 and U1U1 PEPO projection agrees with direct evolution."""
    pytest.importorskip("symmray")

    fermion = pepsy.Fermion(
        spinful=True,
        symmetry=symmetry,
        dtype="complex128",
    )
    state = _state(fermion, symmetry)
    mapper = OneDMap(2, 3, mode="snake")
    where = ((0, 0),)
    operator = fermion.to_pepo(
        {where: fermion.observable("number_up")},
        Lx=2,
        Ly=3,
        mapper=mapper,
        max_bond=64,
        cutoff=0.0,
        compress=False,
        cyclic=False,
        fermionic=True,
    )
    gates = (
        (
            fermion.hopping_gate(0.11, t=0.7),
            ((0, 0), (1, 0)),
        ),
        (
            fermion.heisenberg_gate(0.07),
            ((0, 0), (0, 1)),
        ),
    )

    direct = state.copy()
    backward = operator.copy()
    gauges = {}
    backward.gauge_all_simple_(gauges=gauges, progbar=False)
    for gate, gate_where in gates:
        direct = pepsy.gate(
            direct,
            gate,
            where=gate_where,
            contract="split",
            max_bond=64,
            cutoff=0.0,
            inplace=True,
        )

    # Heisenberg replay is reverse-order, while the state stream is forward.
    for gate, gate_where in reversed(gates):
        pepsy.gate_simple(
            backward,
            gate.H,
            where=gate_where,
            gauges=gauges,
            renorm=False,
            max_bond=64,
            cutoff=0.0,
            contract="split",
            inplace=True,
        )
    measured = backward.copy()
    measured.gauge_simple_insert(gauges)

    direct_value = _expectation(direct, operator)
    backward_value = _expectation(state, measured)
    assert abs(backward_value - direct_value) < 1.0e-8
    assert all(
        type(tensor.data).__name__.endswith("FermionicArray")
        for tensor in operator
    )
    assert all(
        type(tensor.data).__name__.endswith("FermionicArray")
        for tensor in measured
    )


@pytest.mark.parametrize("symmetry", ["U1", "U1U1"])
def test_native_fermion_2x3_hopping_plus_u_sandwich_projection(symmetry):
    """Project ``U.H @ (T + U n_up n_down) @ U`` natively on 2x3."""
    pytest.importorskip("symmray")

    fermion = pepsy.Fermion(
        spinful=True,
        symmetry=symmetry,
        dtype="complex128",
    )
    state = _state(fermion, symmetry)
    mapper = OneDMap(2, 3, mode="snake")
    hopping_support = ((0, 0), (1, 0))
    interaction_support = ((0, 0),)
    terms = {
        hopping_support: -0.7 * fermion.hopping_operator(),
        interaction_support: fermion.onsite_term((0, 0), U=8.0),
    }
    operator = fermion.to_pepo(
        terms,
        Lx=2,
        Ly=3,
        mapper=mapper,
        max_bond=64,
        cutoff=0.0,
        compress=False,
        cyclic=False,
        fermionic=True,
    )
    unitary = fermion.hopping_gate(0.11, t=0.7)

    direct = pepsy.gate(
        state,
        unitary,
        where=hopping_support,
        contract="split",
        max_bond=64,
        cutoff=0.0,
        inplace=False,
    )
    backward = operator.copy()
    gauges = {}
    backward.gauge_all_simple_(gauges=gauges, progbar=False)
    pepsy.gate_simple(
        backward,
        unitary.H,
        where=hopping_support,
        gauges=gauges,
        renorm=False,
        max_bond=64,
        cutoff=0.0,
        contract="split",
        inplace=True,
    )
    measured = backward.copy()
    measured.gauge_simple_insert(gauges)

    direct_value = _expectation(direct, operator)
    projected_value = _expectation(state, measured)
    assert projected_value == pytest.approx(direct_value, abs=1.0e-8)

    # The projected sum must also agree with independently projected native
    # term PEPOs, which catches a gauge/projection error hidden by cancellation.
    separate_projected = 0.0j
    for support, term in terms.items():
        term_operator = fermion.to_pepo(
            {support: term},
            Lx=2,
            Ly=3,
            mapper=mapper,
            max_bond=64,
            cutoff=0.0,
            compress=False,
            cyclic=False,
            fermionic=True,
        )
        term_gauges = {}
        term_operator.gauge_all_simple_(gauges=term_gauges, progbar=False)
        pepsy.gate_simple(
            term_operator,
            unitary.H,
            where=hopping_support,
            gauges=term_gauges,
            renorm=False,
            max_bond=64,
            cutoff=0.0,
            contract="split",
            inplace=True,
        )
        term_operator.gauge_simple_insert(term_gauges)
        separate_projected += _expectation(state, term_operator)

    assert projected_value == pytest.approx(separate_projected, abs=1.0e-8)


def test_native_fermion_pepo_nonchain_edge_identity_sandwich_4x2():
    """Dimension-one lattice bonds preserve a native fermionic sandwich."""
    pytest.importorskip("symmray")

    fermion = pepsy.Fermion(
        spinful=True,
        symmetry="U1",
        dtype="complex128",
    )
    state = pepsy.ps_to_peps(
        (4, 2),
        fermion=fermion,
        occupations={(x, y): 1 for x in range(4) for y in range(2)},
        seed=17,
        dtype="complex128",
        cyclic=False,
    )
    operator = fermion.to_pepo(
        {((0, 0),): fermion.observable("identity")},
        Lx=4,
        Ly=2,
        mapper=OneDMap(4, 2, mode="snake"),
        max_bond=64,
        cutoff=0.0,
        compress=False,
        cyclic=True,
        fermionic=True,
    )
    left = operator["I2,0"]
    right = operator["I3,0"]
    bond = next(iter(qtn.bonds(left, right)))
    left_axis = left.inds.index(bond)
    right_axis = right.inds.index(bond)
    assert left.data.indices[left_axis].dual != right.data.indices[right_axis].dual

    hopping = fermion.hopping_gate(0.17, t=0.73)
    projected = operator.copy()
    pepsy.gate(
        projected,
        hopping.H,
        where=((3, 0), (2, 0)),
        which="upper",
        contract=True,
        inplace=True,
    )
    pepsy.gate(
        projected,
        hopping.T,
        where=((3, 0), (2, 0)),
        which="lower",
        contract=True,
        inplace=True,
    )

    assert _expectation(state, projected) == pytest.approx(1.0, abs=1.0e-10)
