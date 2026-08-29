"""Symmray fermionic coverage for Pepsy's 2-norm BP corrections."""

from __future__ import annotations

import numpy as np
import pytest
import pepsy.bp.series as bp_series

sr = pytest.importorskip("symmray")
qtn = pytest.importorskip("quimb.tensor")
ctg = pytest.importorskip("cotengra")

from pepsy.bp import (  # noqa: E402
    OpenLoopSeriesCache,
    compute_boundary_expectation,
    compute_bp_path_expectation,
    compute_local_expectation_edge_loop_series,
    compute_local_expectation_open_loop_series,
    compute_local_expectation_loop_cluster,
    compute_local_expectation_loop_series,
    compute_path_cluster_expectation,
    compress_reduced_loop_cluster,
    gauge_all,
    loop_cluster_expand,
    loop_series_expand,
    one_norm_bp,
    partial_trace_edge_loop_series_expand,
    partial_trace_open_loop_series_expand,
    partial_trace_open_loop_series_sweep,
    partial_trace_loop_series_expand,
    partial_trace_loop_cluster_expand,
    partitioned_expand,
    relay_bp,
    two_norm_bp,
    weight_pass,
)
from pepsy.bp._symmray import (  # noqa: E402
    install_quimb_symmray_compat,
)
from pepsy.operators.gates import gate_loop_cluster, gate_simple  # noqa: E402
from pepsy.tensors import (  # noqa: E402
    Fermion,
    OneDMap,
    SymPEPS,
    ps_to_peps,
    site_charge_alternating,
    site_charge_from_map,
)


def test_quimb_safe_inverse_compat_handles_symmray_block_vector():
    """The old Quimb axis reduction must not break projector compression."""
    install_quimb_symmray_compat()
    from quimb.tensor.decomp import safe_inverse

    values = sr.BlockVector(
        {
            0: np.asarray([1.0, 2.0]),
            1: np.asarray([3.0]),
        }
    )
    result = safe_inverse(values, power=0.5).to_dense()

    np.testing.assert_allclose(result, 1.0 / np.sqrt([1.0, 2.0, 3.0]))


def _long_range_density_term(where, *, symmetry="U1"):
    """Return a neutral native density-density observable at ``where``."""
    fermion = Fermion(spinful=True, symmetry=symmetry)
    return {where: fermion.density_operator()}


def _jw_annihilator(num_modes, mode):
    """Return a spinless Jordan--Wigner annihilator in site-major order."""
    identity = np.eye(2, dtype="complex128")
    parity = np.diag([1.0, -1.0]).astype("complex128")
    annihilate = np.array([[0.0, 1.0], [0.0, 0.0]], dtype="complex128")
    factors = [
        parity if site < mode else annihilate if site == mode else identity
        for site in range(num_modes)
    ]
    out = factors[0]
    for factor in factors[1:]:
        out = np.kron(out, factor)
    return out


def _jw_hopping_operator(num_modes, left, right):
    """Return ``c_left^dag c_right + h.c.`` with its JW parity string."""
    left_annihilate = _jw_annihilator(num_modes, left)
    right_annihilate = _jw_annihilator(num_modes, right)
    return (
        left_annihilate.conj().T @ right_annihilate
        + right_annihilate.conj().T @ left_annihilate
    )


def _bosonic_hopping_operator(num_modes, left, right):
    """Deliberately omit the parity string for the negative-control oracle."""
    identity = np.eye(2, dtype="complex128")
    annihilate = np.array([[0.0, 1.0], [0.0, 0.0]], dtype="complex128")

    def local_annihilator(mode):
        factors = [annihilate if site == mode else identity for site in range(num_modes)]
        out = factors[0]
        for factor in factors[1:]:
            out = np.kron(out, factor)
        return out

    left_annihilate = local_annihilator(left)
    right_annihilate = local_annihilator(right)
    return (
        left_annihilate.conj().T @ right_annihilate
        + right_annihilate.conj().T @ left_annihilate
    )


def _unitary_from_hermitian(hamiltonian, dt):
    """Exponentiate a small dense Hermitian Hamiltonian without SciPy."""
    values, vectors = np.linalg.eigh(hamiltonian)
    return (vectors * np.exp(-1j * dt * values)) @ vectors.conj().T


def _imaginary_time_from_hermitian(hamiltonian, dt):
    """Apply ``exp(-dt * hamiltonian)`` for a small dense reference state."""
    values, vectors = np.linalg.eigh(hamiltonian)
    return (vectors * np.exp(-dt * values)) @ vectors.conj().T


def _jw_eta_pair_operator(left, right):
    """Return ``Delta_left^dag Delta_right + h.c.`` in JW mode order."""
    num_modes = 8

    def pair_create(site):
        up = _jw_annihilator(num_modes, 2 * site).conj().T
        down = _jw_annihilator(num_modes, 2 * site + 1).conj().T
        return up @ down

    def pair_annihilate(site):
        up = _jw_annihilator(num_modes, 2 * site)
        down = _jw_annihilator(num_modes, 2 * site + 1)
        return down @ up

    return (
        pair_create(left) @ pair_annihilate(right)
        + pair_create(right) @ pair_annihilate(left)
    )


def _spinless_sign_sensitive_peps():
    """Prepare a 2x2 U1 PEPS whose diagonal hop needs a JW minus sign."""
    fermion = Fermion(spinful=False, symmetry="U1")
    occupations = {
        (0, 0): 1,
        (0, 1): 1,
        (1, 0): 0,
        (1, 1): 0,
    }
    peps = ps_to_peps(
        2,
        2,
        fermion=fermion,
        occupations=occupations,
        dtype="complex128",
    )
    state = SymPEPS(
        peps=peps,
        symmetry="U1",
        edges=tuple(qtn.edges_2d_square(2, 2)),
        fermionic=True,
        phys_sectors=fermion.physical_sectors,
        site_charge=occupations,
        site_ind_id="k{},{}",
    )
    dt = 0.37
    gate = fermion.hopping_gate(dt, t=1.0)
    state.apply_gates(
        (
            (gate, ((0, 0), (1, 0))),
            (gate, ((1, 0), (1, 1))),
        ),
        method="direct",
        contract="split",
        max_bond=4,
        cutoff=0.0,
    )

    initial = np.zeros(16, dtype="complex128")
    initial[np.ravel_multi_index((1, 1, 0, 0), (2,) * 4)] = 1.0
    hop_02 = _jw_hopping_operator(4, 0, 2)
    hop_23 = _jw_hopping_operator(4, 2, 3)
    dense_state = _unitary_from_hermitian(-hop_23, dt) @ (
        _unitary_from_hermitian(-hop_02, dt) @ initial
    )
    return state, fermion, dense_state


def test_native_reduced_loop_compression_uses_graded_svd_adapter():
    """Native U1 tensors stay block-sparse through reduced compression."""
    state, fermion, _ = _spinless_sign_sensitive_peps()
    result = compress_reduced_loop_cluster(
        state.tn,
        where=((0, 0), (1, 0)),
        run_bp=False,
        max_distance=1,
        max_bond=1,
        als_opts={"solver": "quimb"},
    )

    assert result.solution.costs[0] == pytest.approx(result.solution.costs[-1])
    assert result.physical_tn.ind_size(result.pair.bond_ind) == 1
    assert all(
        type(tensor.data).__module__.startswith("symmray")
        for tensor in result.physical_tn.tensors
    )
    with pytest.raises(NotImplementedError, match="blockwise autodiff/ALS"):
        compress_reduced_loop_cluster(
            state.tn,
            where=((0, 0), (1, 0)),
            run_bp=False,
            max_distance=1,
            max_bond=1,
            als_opts={"solver": "qr"},
        )

    updated, results = gate_loop_cluster(
        state.tn,
        ((fermion.hopping_gate(0.01, t=1.0), ((0, 0), (1, 0))),),
        gauges={},
        run_bp=False,
        base_radius=1,
        max_bond=1,
        als_opts={"solver": "quimb"},
        regauge_opts={"max_iterations": 1},
        inplace=False,
        return_results=True,
    )
    assert len(results) == 1
    assert all(
        type(tensor.data).__module__.startswith("symmray")
        for tensor in updated.tensors
    )


def test_fermionic_long_range_hopping_sign_survives_su_and_bp_gauges():
    """A JW sign-sensitive long-range hop is invariant under fermionic gauges."""
    state, fermion, dense_state = _spinless_sign_sensitive_peps()
    where = ((0, 0), (1, 1))
    terms = {where: fermion.hopping_operator()}

    jw_operator = _jw_hopping_operator(4, 0, 3)
    bosonic_operator = _bosonic_hopping_operator(4, 0, 3)
    jw_value = dense_state.conj() @ jw_operator @ dense_state
    bosonic_value = dense_state.conj() @ bosonic_operator @ dense_state
    assert abs(jw_value) > 1.0e-6
    assert jw_value == pytest.approx(-bosonic_value, abs=1e-12)

    exact_auto = state.tn.compute_local_expectation_exact(
        terms,
        normalized=True,
        optimize="auto-hq",
    )
    exact_greedy = state.tn.compute_local_expectation_exact(
        terms,
        normalized=True,
        optimize="greedy",
    )
    assert exact_auto == pytest.approx(jw_value, rel=1e-10, abs=1e-10)
    assert exact_greedy == pytest.approx(exact_auto, rel=1e-10, abs=1e-10)

    su = gauge_all(
        state.tn,
        start="su",
        target="su",
        norm="2norm",
        su_options={"max_iterations": 8, "tol": 0.0},
    )
    assert all(
        type(gauge).__module__.startswith("symmray")
        for gauge in su.gauges.values()
    )
    su_reconstructed = su.core.copy()
    su_reconstructed.gauge_simple_insert(su.gauges)
    su_exact = su_reconstructed.compute_local_expectation_exact(
        terms,
        normalized=True,
        optimize="auto-hq",
    )
    assert su_exact == pytest.approx(exact_auto, rel=1e-10, abs=1e-10)

    bp_options = {
        "run_opts": {
            "max_iterations": 150,
            "tol": 1e-10,
            "diis": False,
        }
    }
    bridge = gauge_all(
        state.tn,
        start="bp",
        target="su",
        norm="2norm",
        bp_options=bp_options,
        conversion_options={"smudge": 1e-12},
    )
    assert bridge.bp_result.converged
    assert all(
        type(message).__name__ == "U1FermionicArray"
        for message in bridge.messages.values()
    )
    bp_reconstructed = bridge.core.copy()
    bp_reconstructed.gauge_simple_insert(bridge.gauges)
    bp_exact = bp_reconstructed.compute_local_expectation_exact(
        terms,
        normalized=True,
        optimize="auto-hq",
    )
    assert bp_exact == pytest.approx(exact_auto, rel=1e-10, abs=1e-10)

    cluster = compute_path_cluster_expectation(
        bridge.core,
        terms,
        gauges=bridge.gauges,
        max_distance=1,
        fillin=True,
        max_bond=None,
        normalized=True,
        optimize="auto-hq",
    )
    assert cluster == pytest.approx(exact_auto, rel=1e-10, abs=1e-10)

    with pytest.warns(UserWarning, match="not a compressed one"):
        compressed_cluster = compute_path_cluster_expectation(
            bridge.core,
            terms,
            gauges=bridge.gauges,
            max_distance=1,
            fillin=True,
            max_bond=2,
            normalized=True,
            optimize="auto-hq",
        )
    assert compressed_cluster == pytest.approx(
        jw_value,
        rel=1e-10,
        abs=1e-10,
    )

    bp_helper = compute_bp_path_expectation(
        state.tn,
        terms,
        max_distance=1,
        fillin=True,
        max_bond=None,
        normalized=True,
        optimize="auto-hq",
        bp_options=bp_options,
        conversion_options={"smudge": 1e-12},
    )
    assert bp_helper == pytest.approx(exact_auto, rel=1e-10, abs=1e-10)


def _spinful_sign_sensitive_peps(symmetry):
    """Prepare a controlled spinful state with an occupied JW string."""
    fermion = Fermion(spinful=True, symmetry=symmetry)
    doublon = 2 if symmetry == "U1" else (1, 1)
    empty = 0 if symmetry == "U1" else (0, 0)
    occupations = {
        (0, 0): doublon,
        (0, 1): doublon,
        (1, 0): empty,
        (1, 1): empty,
    }
    peps = ps_to_peps(
        2,
        2,
        fermion=fermion,
        occupations=occupations,
        dtype="complex128",
    )
    state = SymPEPS(
        peps=peps,
        symmetry=symmetry,
        edges=tuple(qtn.edges_2d_square(2, 2)),
        fermionic=True,
        phys_sectors=fermion.physical_sectors,
        site_charge=occupations,
        site_ind_id="k{},{}",
    )
    dt = 0.29
    gate = fermion.hopping_gate(dt, t=(1.0, 0.0))
    state.apply_gates(
        (
            (gate, ((0, 0), (1, 0))),
            (gate, ((1, 0), (1, 1))),
        ),
        method="direct",
        contract="split",
        max_bond=4,
        cutoff=0.0,
    )

    initial = np.zeros(256, dtype="complex128")
    initial[np.ravel_multi_index((1, 1, 1, 1, 0, 0, 0, 0), (2,) * 8)] = 1.0
    hop_04 = _jw_hopping_operator(8, 0, 4)
    hop_46 = _jw_hopping_operator(8, 4, 6)
    dense_state = _unitary_from_hermitian(-hop_46, dt) @ (
        _unitary_from_hermitian(-hop_04, dt) @ initial
    )
    return state, fermion, dense_state


@pytest.mark.parametrize("symmetry", ("U1", "U1U1"))
def test_spinful_long_range_hopping_sign_survives_su_and_bp_gauges(symmetry):
    """Spinful U1 gauges preserve a JW-sensitive up-fermion correlator."""
    state, fermion, dense_state = _spinful_sign_sensitive_peps(symmetry)
    where = ((0, 0), (1, 1))
    terms = {where: fermion.hopping_operator(spin="up")}

    jw_operator = _jw_hopping_operator(8, 0, 6)
    bosonic_operator = _bosonic_hopping_operator(8, 0, 6)
    jw_value = dense_state.conj() @ jw_operator @ dense_state
    bosonic_value = dense_state.conj() @ bosonic_operator @ dense_state
    assert abs(jw_value) > 1.0e-6
    assert jw_value == pytest.approx(-bosonic_value, abs=1e-12)

    exact_auto = state.tn.compute_local_expectation_exact(
        terms,
        normalized=True,
        optimize="auto-hq",
    )
    exact_greedy = state.tn.compute_local_expectation_exact(
        terms,
        normalized=True,
        optimize="greedy",
    )
    assert exact_auto == pytest.approx(jw_value, rel=1e-10, abs=1e-10)
    assert exact_greedy == pytest.approx(exact_auto, rel=1e-10, abs=1e-10)

    su = gauge_all(
        state.tn,
        start="su",
        target="su",
        norm="2norm",
        su_options={"max_iterations": 8, "tol": 0.0},
    )
    assert all(
        type(gauge).__module__.startswith("symmray")
        for gauge in su.gauges.values()
    )
    su_reconstructed = su.core.copy()
    su_reconstructed.gauge_simple_insert(su.gauges)
    su_exact = su_reconstructed.compute_local_expectation_exact(
        terms,
        normalized=True,
        optimize="auto-hq",
    )
    assert su_exact == pytest.approx(exact_auto, rel=1e-10, abs=1e-10)

    bp_options = {
        "run_opts": {
            "max_iterations": 150,
            "tol": 1e-10,
            "diis": False,
        }
    }
    bridge = gauge_all(
        state.tn,
        start="bp",
        target="su",
        norm="2norm",
        bp_options=bp_options,
        conversion_options={"smudge": 1e-12},
    )
    assert bridge.bp_result.converged
    expected_message_type = (
        "U1FermionicArray" if symmetry == "U1" else "U1U1FermionicArray"
    )
    assert all(
        type(message).__name__ == expected_message_type
        for message in bridge.messages.values()
    )
    bp_reconstructed = bridge.core.copy()
    bp_reconstructed.gauge_simple_insert(bridge.gauges)
    bp_exact = bp_reconstructed.compute_local_expectation_exact(
        terms,
        normalized=True,
        optimize="auto-hq",
    )
    assert bp_exact == pytest.approx(exact_auto, rel=1e-10, abs=1e-10)

    cluster = compute_path_cluster_expectation(
        bridge.core,
        terms,
        gauges=bridge.gauges,
        max_distance=1,
        fillin=True,
        max_bond=None,
        normalized=True,
        optimize="auto-hq",
    )
    bp_helper = compute_bp_path_expectation(
        state.tn,
        terms,
        max_distance=1,
        fillin=True,
        max_bond=None,
        normalized=True,
        optimize="auto-hq",
        bp_options=bp_options,
        conversion_options={"smudge": 1e-12},
    )
    assert cluster == pytest.approx(exact_auto, rel=1e-10, abs=1e-10)
    assert bp_helper == pytest.approx(exact_auto, rel=1e-10, abs=1e-10)


@pytest.mark.parametrize("symmetry", ("U1", "U1U1"))
def test_spinful_eta_pair_measurement_survives_su_and_bp_gauges(symmetry):
    """Native eta-pair measurements agree with JW through both gauge routes."""
    fermion = Fermion(spinful=True, symmetry=symmetry)
    doublon = 2 if symmetry == "U1" else (1, 1)
    empty = 0 if symmetry == "U1" else (0, 0)
    occupations = {
        (0, 0): doublon,
        (0, 1): doublon,
        (1, 0): empty,
        (1, 1): empty,
    }
    peps = ps_to_peps(
        2,
        2,
        fermion=fermion,
        occupations=occupations,
        dtype="complex128",
    )
    state = SymPEPS(
        peps=peps,
        symmetry=symmetry,
        edges=tuple(qtn.edges_2d_square(2, 2)),
        fermionic=True,
        phys_sectors=fermion.physical_sectors,
        site_charge=occupations,
        site_ind_id="k{},{}",
    )
    where = ((0, 0), (1, 1))
    operator = fermion.eta_pair_operator()
    terms = {where: operator}
    dt = 0.17
    state.apply_gates(
        ((fermion.operator_gate(operator, dt, imaginary=True), where),),
        method="gate",
        max_bond=16,
        cutoff=0.0,
    )

    initial = np.zeros(256, dtype="complex128")
    initial[np.ravel_multi_index((1, 1, 1, 1, 0, 0, 0, 0), (2,) * 8)] = 1.0
    jw_operator = _jw_eta_pair_operator(0, 3)
    dense_state = _imaginary_time_from_hermitian(jw_operator, dt) @ initial
    jw_value = (
        dense_state.conj() @ jw_operator @ dense_state
    ) / (dense_state.conj() @ dense_state)
    assert abs(jw_value) > 1.0e-6

    exact_auto = state.tn.compute_local_expectation_exact(
        terms,
        normalized=True,
        optimize="auto-hq",
    )
    exact_greedy = state.tn.compute_local_expectation_exact(
        terms,
        normalized=True,
        optimize="greedy",
    )
    assert exact_auto == pytest.approx(jw_value, rel=1e-10, abs=1e-10)
    assert exact_greedy == pytest.approx(exact_auto, rel=1e-10, abs=1e-10)

    open_info = {}
    open_value = compute_local_expectation_open_loop_series(
        state.tn,
        terms,
        gloops=3,
        max_iterations=200,
        tol=1e-10,
        diis=False,
        info=open_info,
    )
    assert open_value == pytest.approx(exact_auto, rel=1e-10, abs=1e-10)
    assert open_info["open_scalar_fermionic_q_phase"] is True

    su = gauge_all(
        state.tn,
        start="su",
        target="su",
        norm="2norm",
        su_options={"max_iterations": 8, "tol": 0.0},
    )
    su_reconstructed = su.core.copy()
    su_reconstructed.gauge_simple_insert(su.gauges)
    su_exact = su_reconstructed.compute_local_expectation_exact(
        terms,
        normalized=True,
        optimize="auto-hq",
    )
    assert su_exact == pytest.approx(exact_auto, rel=1e-10, abs=1e-10)

    bp_options = {
        "run_opts": {
            "max_iterations": 150,
            "tol": 1e-10,
            "diis": False,
        }
    }
    bridge = gauge_all(
        state.tn,
        start="bp",
        target="su",
        norm="2norm",
        bp_options=bp_options,
        conversion_options={"smudge": 1e-12},
    )
    assert bridge.bp_result.converged
    bp_reconstructed = bridge.core.copy()
    bp_reconstructed.gauge_simple_insert(bridge.gauges)
    bp_exact = bp_reconstructed.compute_local_expectation_exact(
        terms,
        normalized=True,
        optimize="auto-hq",
    )
    assert bp_exact == pytest.approx(exact_auto, rel=1e-10, abs=1e-10)

    cluster = compute_path_cluster_expectation(
        bridge.core,
        terms,
        gauges=bridge.gauges,
        max_distance=1,
        fillin=True,
        max_bond=None,
        normalized=True,
        optimize="auto-hq",
    )
    bp_helper = compute_bp_path_expectation(
        state.tn,
        terms,
        max_distance=1,
        fillin=True,
        max_bond=None,
        normalized=True,
        optimize="auto-hq",
        bp_options=bp_options,
        conversion_options={"smudge": 1e-12},
    )
    assert cluster == pytest.approx(exact_auto, rel=1e-10, abs=1e-10)
    assert bp_helper == pytest.approx(exact_auto, rel=1e-10, abs=1e-10)


@pytest.mark.parametrize("symmetry", ("U1", "U1U1"))
def test_fermionic_long_range_doublon_product_survives_cluster_gauges(symmetry):
    """A nontrivial long-range doublon correlator survives native clusters."""
    fermion = Fermion(spinful=True, symmetry=symmetry)
    doublon = 2 if symmetry == "U1" else (1, 1)
    empty = 0 if symmetry == "U1" else (0, 0)
    occupations = {
        (0, 0): doublon,
        (1, 1): doublon,
        (0, 1): empty,
        (1, 0): empty,
    }
    peps = ps_to_peps(
        2,
        2,
        fermion=fermion,
        occupations=occupations,
        dtype="complex128",
    )
    state = SymPEPS(
        peps=peps,
        symmetry=symmetry,
        edges=tuple(qtn.edges_2d_square(2, 2)),
        fermionic=True,
        phys_sectors=fermion.physical_sectors,
        site_charge=occupations,
        site_ind_id="k{},{}",
    )

    gate = fermion.hopping_gate(0.31, t=(1.0, 0.0))
    state.apply_gates(
        (
            (gate, ((0, 0), (1, 0))),
            (gate, ((1, 1), (0, 1))),
        ),
        method="direct",
        contract="split",
        max_bond=4,
        cutoff=0.0,
    )

    operator = fermion.operator_term(
        [(1.0, ((0, "double"), (1, "double")))],
        sites=(0, 1),
    )
    where = ((0, 0), (1, 1))
    terms = {where: operator}
    norm_before = complex(state.tn.norm())

    exact = state.tn.compute_local_expectation_exact(
        terms,
        normalized=True,
        optimize="auto-hq",
    )

    su = gauge_all(
        state.tn,
        start="su",
        target="su",
        norm="2norm",
        su_options={"max_iterations": 8, "tol": 0.0},
    )
    cluster = compute_path_cluster_expectation(
        su.core,
        terms,
        gauges=su.gauges,
        max_distance=1,
        fillin=True,
        max_bond=None,
        normalized=True,
        optimize="auto-hq",
    )

    bp_helper = compute_bp_path_expectation(
        state.tn,
        terms,
        max_distance=1,
        fillin=True,
        max_bond=None,
        normalized=True,
        optimize="auto-hq",
        bp_options={
            "run_opts": {
                "max_iterations": 150,
                "tol": 1e-10,
                "diis": False,
            }
        },
        conversion_options={"smudge": 1e-12},
    )

    with pytest.warns(UserWarning, match="not a compressed one"):
        compressed = compute_path_cluster_expectation(
            su.core,
            terms,
            gauges=su.gauges,
            max_distance=1,
            fillin=True,
            max_bond=8,
            normalized=True,
            optimize="auto-hq",
        )

    expected_type = (
        "U1FermionicArray" if symmetry == "U1" else "U1U1FermionicArray"
    )
    assert exact.real > 0.1
    assert cluster == pytest.approx(exact, rel=1e-10, abs=1e-10)
    assert bp_helper == pytest.approx(exact, rel=1e-10, abs=1e-10)
    assert compressed == pytest.approx(exact, rel=1e-10, abs=1e-10)
    assert complex(state.tn.norm()) == pytest.approx(norm_before)
    assert all(
        type(tensor.data).__name__ == expected_type
        for tensor in state.tn.tensors
    )


def test_fermionic_long_range_boundary_expectation_matches_exact():
    """The boundary route preserves a distant native density operator."""
    state = SymPEPS.random(
        2,
        2,
        symmetry="U1",
        bond_dim=3,
        phys_dim=4,
        fermionic=True,
        seed=1500,
        dtype="complex128",
    )
    terms = _long_range_density_term(((0, 1), (1, 0)))

    exact = state.tn.compute_local_expectation_exact(
        terms,
        normalized=True,
        optimize="auto-hq",
    )
    measured = compute_boundary_expectation(
        state.tn,
        terms,
        max_bond=8,
        mode="mps",
        normalized=True,
        contract_optimize="auto-hq",
    )

    assert measured == pytest.approx(exact, rel=1e-10, abs=1e-10)


def test_fermionic_path_cluster_accepts_native_su_gauges():
    """A distant fermionic density cluster accepts native SU bond vectors."""
    state = SymPEPS.random(
        3,
        3,
        symmetry="U1",
        bond_dim=2,
        phys_dim=4,
        fermionic=True,
        seed=1501,
        dtype="complex128",
    )
    where = ((0, 1), (2, 1))
    terms = _long_range_density_term(where)
    exact = state.tn.compute_local_expectation_exact(
        terms,
        normalized=True,
        optimize="auto-hq",
    )
    norm_before = complex(state.tn.norm())

    su = gauge_all(
        state.tn,
        start="su",
        target="su",
        norm="2norm",
        su_options={"max_iterations": 8, "tol": 0.0},
    )
    assert su.gauges
    assert all(
        type(gauge).__module__.startswith("symmray")
        for gauge in su.gauges.values()
    )

    ungauged = compute_path_cluster_expectation(
        su.core,
        terms,
        max_distance=0,
        fillin=False,
        max_bond=None,
        normalized=True,
        optimize="auto-hq",
    )
    measured = compute_path_cluster_expectation(
        su.core,
        terms,
        max_distance=0,
        fillin=False,
        gauges=su.gauges,
        max_bond=None,
        normalized=True,
        optimize="auto-hq",
    )
    assert np.isfinite(measured)
    assert measured != pytest.approx(ungauged, rel=1e-6, abs=1e-10)

    full_cluster = compute_path_cluster_expectation(
        su.core,
        terms,
        max_distance=1,
        fillin=True,
        gauges=su.gauges,
        max_bond=None,
        normalized=True,
        optimize="auto-hq",
    )

    assert full_cluster == pytest.approx(exact, rel=1e-10, abs=1e-10)
    assert complex(state.tn.norm()) == pytest.approx(norm_before)


@pytest.mark.parametrize("symmetry", ("U1", "U1U1"))
def test_fermionic_path_cluster_compression_preserves_native_symmetry(symmetry):
    """Native path clusters support Symmray-aware QR/SVD compression."""
    site_charge = None
    if symmetry == "U1U1":
        site_charge = site_charge_alternating((1, 0), (0, 1))
    state = SymPEPS.random(
        3,
        3,
        symmetry=symmetry,
        bond_dim=2,
        phys_dim=4,
        fermionic=True,
        site_charge=site_charge,
        seed=1620,
        dtype="complex128",
    )
    where = ((0, 1), (2, 1))
    terms = _long_range_density_term(where, symmetry=symmetry)
    su = gauge_all(
        state.tn,
        start="su",
        target="su",
        norm="2norm",
        su_options={"max_iterations": 8, "tol": 0.0},
    )

    exact_cluster = compute_path_cluster_expectation(
        su.core,
        terms,
        gauges=su.gauges,
        max_distance=1,
        fillin=True,
        max_bond=None,
        optimize="auto-hq",
    )
    with pytest.warns(UserWarning, match="not a compressed one"):
        high_chi = compute_path_cluster_expectation(
            su.core,
            terms,
            gauges=su.gauges,
            max_distance=1,
            fillin=True,
            max_bond=8,
            optimize="auto-hq",
        )
    with pytest.warns(UserWarning, match="not a compressed one"):
        low_chi = compute_path_cluster_expectation(
            su.core,
            terms,
            gauges=su.gauges,
            max_distance=1,
            fillin=True,
            max_bond=2,
            optimize="auto-hq",
        )

    assert high_chi == pytest.approx(exact_cluster, rel=1e-10, abs=1e-10)
    assert np.isfinite(low_chi)
    assert low_chi != pytest.approx(high_chi, rel=1e-6, abs=1e-10)
    assert all(
        type(tensor.data).__name__
        == ("U1FermionicArray" if symmetry == "U1" else "U1U1FermionicArray")
        for tensor in su.core.tensors
    )

    cotengra_optimizer = ctg.HyperOptimizer(
        max_repeats=2,
        parallel=False,
        progbar=False,
    )
    with pytest.warns(UserWarning, match="not a compressed one"):
        via_cotengra = compute_path_cluster_expectation(
            su.core,
            terms,
            gauges=su.gauges,
            max_distance=1,
            fillin=True,
            max_bond=2,
            optimize=cotengra_optimizer,
        )
    assert np.isfinite(via_cotengra)


def test_fermionic_bp_path_expectation_uses_native_bp_to_su_bridge():
    """BP-derived SU gauges close a distant fermionic density cluster."""
    state = SymPEPS.random(
        3,
        3,
        symmetry="U1",
        bond_dim=2,
        phys_dim=4,
        fermionic=True,
        seed=1502,
        dtype="complex128",
    )
    where = ((0, 1), (2, 1))
    terms = _long_range_density_term(where)
    norm_before = complex(state.tn.norm())
    bp_options = {
        "run_opts": {
            "max_iterations": 150,
            "tol": 1e-10,
            "diis": False,
        }
    }
    conversion_options = {"smudge": 1e-12}

    bridge = gauge_all(
        state.tn,
        start="bp",
        target="su",
        norm="2norm",
        bp_options=bp_options,
        conversion_options=conversion_options,
    )
    assert bridge.bp_result.converged
    assert all(
        type(message).__name__ == "U1FermionicArray"
        for message in bridge.messages.values()
    )
    via_bridge = compute_path_cluster_expectation(
        bridge.core,
        terms,
        max_distance=0,
        fillin=False,
        gauges=bridge.gauges,
        max_bond=None,
        normalized=True,
        optimize="auto-hq",
    )

    measured = compute_bp_path_expectation(
        state.tn,
        terms,
        max_distance=0,
        fillin=False,
        max_bond=None,
        normalized=True,
        optimize="auto-hq",
        bp_options=bp_options,
        conversion_options=conversion_options,
    )

    assert np.isfinite(via_bridge)
    assert measured == pytest.approx(via_bridge, rel=1e-10, abs=1e-10)

    with pytest.warns(UserWarning, match="not a compressed one"):
        compressed_via_bridge = compute_path_cluster_expectation(
            bridge.core,
            terms,
            max_distance=1,
            fillin=True,
            gauges=bridge.gauges,
            max_bond=2,
            normalized=True,
            optimize="auto-hq",
        )
    with pytest.warns(UserWarning, match="not a compressed one"):
        compressed_measured = compute_bp_path_expectation(
            state.tn,
            terms,
            max_distance=1,
            fillin=True,
            max_bond=2,
            normalized=True,
            optimize="auto-hq",
            bp_options=bp_options,
            conversion_options=conversion_options,
        )
    assert np.isfinite(compressed_via_bridge)
    assert compressed_measured == pytest.approx(
        compressed_via_bridge,
        rel=1e-10,
        abs=1e-10,
    )
    assert complex(state.tn.norm()) == pytest.approx(norm_before)
    assert all(
        type(tensor.data).__name__ == "U1FermionicArray"
        for tensor in state.tn.tensors
    )


@pytest.mark.parametrize("bond_dim", (2, 3))
def test_fermionic_u1_two_norm_bp_is_exact_on_a_tree(bond_dim):
    """D2BP preserves the native graded contraction on a PEPS tree."""
    state = SymPEPS.random(
        1,
        4,
        symmetry="U1",
        bond_dim=bond_dim,
        phys_dim=2,
        fermionic=True,
        seed=700 + bond_dim,
        dtype="complex128",
    )

    exact = complex(state.norm())
    bp = two_norm_bp(
        state.tn,
        max_iterations=100,
        tol=1e-10,
        diis=False,
    )

    assert bp.converged
    assert abs(complex(bp.contract()) - exact) <= 1e-10 * max(1.0, abs(exact))

    corrected = loop_cluster_expand(
        state.tn,
        gloops=0,
        norm="2norm",
        max_iterations=100,
        tol=1e-10,
        diis=False,
    )
    assert corrected.bp_converged
    assert abs(complex(corrected.estimate) - exact) <= 1e-10 * max(
        1.0, abs(exact)
    )


def test_d2bp_repairs_labelled_odd_arrays_after_new_with():
    """D2BP repairs the phase metadata dropped by Symmray ``new_with``."""
    state = SymPEPS.random(
        2,
        2,
        symmetry="U1",
        bond_dim=2,
        phys_dim=2,
        fermionic=True,
        seed=750,
        dtype="complex128",
    )
    broken = state.tn.copy()
    for tensor in broken.tensors:
        data = tensor.data
        if not getattr(data, "parity", 0):
            continue
        tensor.modify(
            data=type(data).from_blocks(
                data.blocks,
                duals=data.indices,
                charge=data.charge,
                symmetry=data.symmetry,
                phases=data.phases,
                label=data.label,
                dummy_modes=(),
            )
        )

    assert all(
        not tensor.data.dummy_modes
        for tensor in broken.tensors
        if getattr(tensor.data, "parity", 0)
    )

    result = two_norm_bp(
        broken,
        max_iterations=100,
        tol=1e-10,
        diis=False,
    )
    assert result.converged
    assert all(
        tensor.data.dummy_modes
        for tensor in result.bp.tn.tensors
        if getattr(tensor.data, "parity", 0)
    )

    cluster = loop_cluster_expand(
        broken,
        gloops=0,
        norm="2norm",
        max_iterations=100,
        tol=1e-10,
        diis=False,
    )
    assert cluster.bp_converged
    assert all(
        tensor.data.dummy_modes
        for tensor in cluster.bp.tn.tensors
        if getattr(tensor.data, "parity", 0)
    )


@pytest.mark.parametrize("bond_dim", (2, 3))
def test_fermionic_u1_loop_cluster_runs_on_3x4_peps(bond_dim):
    """D2BP loop clusters pad sparse charge blocks without densifying."""
    state = SymPEPS.random(
        3,
        4,
        symmetry="U1",
        bond_dim=bond_dim,
        phys_dim=2,
        fermionic=True,
        seed=800 + bond_dim,
        dtype="complex128",
    )

    result = loop_cluster_expand(
        state.tn,
        gloops=4,
        norm="2norm",
        max_iterations=300,
        tol=1e-10,
        diis=False,
    )

    assert result.bp_converged
    assert result.bp.__class__.__name__ == "D2BP"
    assert np.isfinite(float(np.real(result.estimate)))
    assert all(
        type(message).__name__ == "U1FermionicArray"
        for message in result.messages.values()
    )
    assert all(
        type(tensor.data).__name__ == "U1FermionicArray"
        for tensor in state.tn.tensors
    )


@pytest.mark.parametrize("symmetry", ["U1U1", "U1", "Z2"])
def test_native_fermionic_pepo_simple_update_norm_and_loop_cluster(symmetry):
    """Native PEPO evolution and D2BP norm correction compose cleanly."""
    fermion = Fermion(spinful=True, symmetry=symmetry)
    left, right = (0, 0), (0, 1)
    term = fermion.operator_term(
        [(1.0, ((left, "create_up"), (right, "annihilate_up")))],
        sites=(left, right),
        add_hc=True,
    )
    pepo = fermion.to_pepo(
        {(left, right): term},
        Lx=2,
        Ly=2,
        mapper=OneDMap(2, 2, mode="snake-row-major"),
        max_bond=16,
        compress=False,
    )
    gauges = {}
    pepo.gauge_all_simple_(gauges=gauges, progbar=False)
    norm_before = complex(pepo.norm())

    evolved = gate_simple(
        pepo,
        fermion.hopping_gate(0.001, t=1.0).H,
        where=(left, right),
        gauges=gauges,
        max_bond=16,
        cutoff=1e-10,
        contract="split",
        renorm=False,
        inplace=False,
    )
    norm_after = complex(evolved.norm())
    correction = loop_cluster_expand(
        evolved,
        gloops=2,
        norm="2norm",
        max_iterations=100,
        tol=1e-8,
        diis=False,
        progbar=False,
    )

    assert len(gauges) > 0
    assert np.isfinite(norm_before.real)
    assert np.isfinite(norm_after.real)
    assert np.isfinite(float(np.real(correction.estimate)))
    assert all(
        type(tensor.data).__name__.endswith("FermionicArray")
        for tensor in evolved
    )


def _fermionic_symmetry_cases():
    return (
        (
            "U1",
            "fermi_hubbard",
            {"symmetry": "U1", "phys_dim": 4},
        ),
        (
            "U1U1",
            "fermi_hubbard_u1u1",
            {
                "symmetry": "U1U1",
                "phys_dim": 4,
                "site_charge": site_charge_alternating(
                    (1, 0), (0, 1)
                ),
            },
        ),
        (
            "Z2",
            "fermi_hubbard_spinless",
            {"symmetry": "Z2", "phys_dim": 2},
        ),
    )


@pytest.mark.parametrize(
    ("label", "model", "state_options"),
    _fermionic_symmetry_cases(),
)
def test_fermionic_sympeps_su_d2bp_bridge_preserves_symmetry_blocks(
    label, model, state_options
):
    """SU gauges and BP loop expansion preserve native fermionic blocks."""
    state = SymPEPS.for_model(
        model,
        3,
        4,
        bond_dim=2,
        seed=1200,
        dtype="complex128",
        **state_options,
    )
    exact_norm = state.tn.norm()

    forward = gauge_all(
        state.tn,
        start="su",
        target="bp",
        norm="2norm",
        su_options={"max_iterations": 8, "tol": 0.0},
        bp_options={
            "run_opts": {
                "max_iterations": 300,
                "tol": 1e-10,
                "diis": False,
            }
        },
    )
    assert forward.bp.converged, label
    reconstructed = forward.core.copy()
    reconstructed.gauge_simple_insert(forward.gauges)
    np.testing.assert_allclose(reconstructed.norm(), exact_norm, rtol=1e-10)
    assert all(
        type(message).__name__ == type(state.tn.tensors[0].data).__name__
        for message in forward.messages.values()
    )

    cluster = loop_cluster_expand(
        forward.bp.tn,
        gloops=4,
        norm="2norm",
        messages=forward.messages,
        run_bp=False,
    )
    assert np.isfinite(float(np.real(cluster.estimate))), label

    reverse = gauge_all(
        state.tn,
        start="bp",
        target="su",
        norm="2norm",
        bp_options={
            "run_opts": {
                "max_iterations": 300,
                "tol": 1e-10,
                "diis": False,
            }
        },
        conversion_options={"smudge": 1e-12},
    )
    assert reverse.bp.converged, label
    reverse_reconstructed = reverse.core.copy()
    reverse_reconstructed.gauge_simple_insert(reverse.gauges)
    np.testing.assert_allclose(
        reverse_reconstructed.norm(), exact_norm, rtol=1e-10
    )


def test_fermionic_u1_d2bp_corrections_and_relay_use_native_messages():
    """The D2 relay, series, and PNE paths share native Symmray handling."""
    state = SymPEPS.random(
        2,
        2,
        symmetry="U1",
        bond_dim=2,
        phys_dim=2,
        fermionic=True,
        seed=1300,
        dtype="complex128",
    )
    bond = next(iter(state.tn.inner_inds()))

    relay = relay_bp(
        state.tn,
        method="d2bp",
        num_relays=2,
        max_iterations=200,
        tol=1e-10,
    )
    assert relay.converged
    assert all(
        type(message).__name__ == "U1FermionicArray"
        for message in relay.messages.values()
    )

    series = loop_series_expand(
        state.tn,
        2,
        norm="2norm",
        max_iterations=200,
        tol=1e-10,
    )
    assert series.bp_converged
    assert np.isfinite(float(np.real(series.estimate)))

    pne = partitioned_expand(
        state.tn,
        partition_inds=(bond,),
        norm="2norm",
        max_iterations=200,
        tol=1e-10,
    )
    assert pne.bp_converged
    assert np.isfinite(float(np.real(pne.estimate)))
    assert all(
        type(message).__name__ == "U1FermionicArray"
        for message in pne.messages.values()
    )


def test_partial_trace_loop_series_matches_quimb_on_dense_peps():
    """The Pepsy local rho wrapper matches Quimb's dense D2BP reference."""
    state = qtn.PEPS.rand(
        2,
        2,
        bond_dim=2,
        seed=1901,
        dtype="complex128",
    )
    where = ((0, 0), (1, 1))

    rho = partial_trace_loop_series_expand(
        state,
        where,
        gloops=2,
        max_iterations=200,
        tol=1e-10,
        diis=False,
    )
    reference_bp = two_norm_bp(
        state,
        max_iterations=200,
        tol=1e-10,
        diis=False,
    )
    reference = reference_bp.bp.partial_trace_loop_series_expansion(
        where=where,
        gloops=2,
        normalized=True,
    )

    np.testing.assert_allclose(rho, reference, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.trace(rho), 1.0, rtol=1e-10, atol=1e-12)

    gate = np.diag([1.0, -1.0, -1.0, 1.0])
    value = compute_local_expectation_loop_series(
        state,
        {where: gate},
        gloops=2,
        normalized="prod",
        max_iterations=200,
        tol=1e-10,
        diis=False,
    )
    np.testing.assert_allclose(value, np.trace(reference @ gate))

    cluster_rho = partial_trace_loop_cluster_expand(
        state,
        where,
        gloops=2,
        max_iterations=200,
        tol=1e-10,
        diis=False,
    )
    cluster_reference = reference_bp.bp.partial_trace_gloop_expand(
        where,
        gloops=2,
        combine="sum",
        normalized=True,
    )
    np.testing.assert_allclose(
        cluster_rho,
        cluster_reference,
        rtol=1e-10,
        atol=1e-12,
    )
    cluster_value = compute_local_expectation_loop_cluster(
        state,
        {where: gate},
        gloops=2,
        max_iterations=200,
        tol=1e-10,
        diis=False,
    )
    np.testing.assert_allclose(cluster_value, np.trace(cluster_reference @ gate))


def test_fermionic_partial_trace_loop_series_keeps_native_rho():
    """Local D2 loop-series rho keeps fermionic Symmray block structure."""
    state = SymPEPS.random(
        2,
        2,
        symmetry="U1",
        bond_dim=3,
        phys_dim=2,
        fermionic=True,
        seed=1903,
        dtype="complex128",
    )
    where = ((0, 0), (1, 1))

    rho = partial_trace_loop_series_expand(
        state.tn,
        where,
        gloops=2,
        max_iterations=200,
        tol=1e-10,
        diis=False,
    )

    assert type(rho).__name__ == "U1FermionicArray"
    assert rho.ndim == 2
    np.testing.assert_allclose(
        np.trace(rho.to_dense()),
        1.0,
        rtol=1e-10,
        atol=1e-12,
    )

    cluster_rho = partial_trace_loop_cluster_expand(
        state.tn,
        where,
        gloops=2,
        max_iterations=200,
        tol=1e-10,
        diis=False,
    )
    assert type(cluster_rho).__name__ == "U1FermionicArray"
    np.testing.assert_allclose(
        np.trace(cluster_rho.to_dense()),
        1.0,
        rtol=1e-10,
        atol=1e-12,
    )


def test_fermionic_open_rho_is_native_diagnostic_and_scalar_path_is_graded():
    """Long-range open rho stays native while observables use graded gates."""
    state = SymPEPS.random(
        1,
        4,
        symmetry="U1",
        bond_dim=3,
        phys_dim=2,
        fermionic=True,
        seed=1912,
        dtype="complex128",
    )
    where = ((0, 0), (0, 3))
    bp = two_norm_bp(
        state.tn,
        max_iterations=200,
        tol=1e-10,
        diis=False,
    )
    info = {}
    rho = partial_trace_open_loop_series_expand(
        state.tn,
        where,
        gloops=3,
        messages=bp.messages,
        run_bp=False,
        cache=OpenLoopSeriesCache(),
        info=info,
    )

    assert type(rho).__name__ == "U1FermionicArray"
    assert info["open_rho_family_counts"] == {"open_path": 1}
    np.testing.assert_allclose(
        np.trace(rho.to_dense()),
        1.0,
        rtol=1e-10,
        atol=1e-12,
    )

    fermion = Fermion(spinful=False, symmetry="U1")
    local_where = ((0, 1),)
    local_gate = fermion.chemical_potential_operator()
    exact = state.tn.compute_local_expectation_exact(
        {local_where: local_gate},
        normalized=True,
        optimize="auto-hq",
    )
    graded = compute_local_expectation_loop_series(
        state.tn,
        {local_where: local_gate},
        gloops=0,
        messages=bp.messages,
        run_bp=False,
        max_iterations=200,
        tol=1e-10,
        diis=False,
    )
    np.testing.assert_allclose(graded, exact, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize(
    "symmetry,phys_dim,site_charge",
    (
        ("U1", 2, None),
        ("U1U1", 4, site_charge_alternating((1, 0), (0, 1))),
        ("Z2", 2, None),
    ),
)
def test_fermionic_open_rho_sweep_supports_native_symmetries(
    symmetry,
    phys_dim,
    site_charge,
):
    """One-BP open-rho sweeps preserve native U1, U1U1, and Z2 arrays."""
    state = SymPEPS.random(
        1,
        4,
        symmetry=symmetry,
        bond_dim=2,
        phys_dim=phys_dim,
        fermionic=True,
        site_charge=site_charge,
        seed=1920 + len(symmetry),
        dtype="complex128",
    )
    supports = (((0, 0), (0, 3)), ((0, 0), (0, 1), (0, 3)))
    result = partial_trace_open_loop_series_sweep(
        state.tn,
        supports,
        (0, 2, 3),
        max_iterations=200,
        tol=1e-10,
        diis=False,
    )

    assert result.bp_converged
    for support in supports:
        for cutoff in (0, 2, 3):
            rho = result.get_rho(support, cutoff)
            assert type(rho).__module__.startswith("symmray")
            np.testing.assert_allclose(
                np.trace(rho.to_dense()),
                1.0,
                rtol=1e-10,
                atol=1e-12,
            )


def test_fermionic_local_expectation_loop_series_aligns_charge_support():
    """Graded gate insertion is exact on a fermionic D2BP tree."""
    state = SymPEPS.random(
        1,
        4,
        symmetry="U1",
        bond_dim=3,
        phys_dim=4,
        fermionic=True,
        seed=1904,
        dtype="complex128",
    )
    where = ((0, 1), (0, 2))
    gate = Fermion(spinful=True, symmetry="U1").eta_pair_operator()
    exact = state.tn.compute_local_expectation_exact(
        {where: gate},
        normalized=True,
        optimize="auto-hq",
    )
    series_value = compute_local_expectation_loop_series(
        state.tn,
        {where: gate},
        gloops=0,
        max_iterations=200,
        tol=1e-10,
        diis=False,
    )
    cluster_value = compute_local_expectation_loop_cluster(
        state.tn,
        {where: gate},
        gloops=0,
        max_iterations=200,
        tol=1e-10,
        diis=False,
    )

    np.testing.assert_allclose(series_value, exact, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(cluster_value, exact, rtol=1e-10, atol=1e-12)

    reverse_where = where[::-1]
    reverse_gate = gate.transpose((1, 0, 3, 2))
    reverse_exact = state.tn.compute_local_expectation_exact(
        {reverse_where: reverse_gate},
        normalized=True,
        optimize="auto-hq",
    )
    reverse_series = compute_local_expectation_loop_series(
        state.tn,
        {reverse_where: reverse_gate},
        gloops=0,
        max_iterations=200,
        tol=1e-10,
        diis=False,
    )
    np.testing.assert_allclose(
        reverse_series,
        reverse_exact,
        rtol=1e-10,
        atol=1e-12,
    )


def test_explicit_edge_loop_series_uses_fermion_safe_gate_path():
    """The explicit-edge scalar API is exact on a fermionic tree."""
    state = SymPEPS.random(
        1,
        4,
        symmetry="U1",
        bond_dim=3,
        phys_dim=4,
        fermionic=True,
        seed=1905,
        dtype="complex128",
    )
    where = ((0, 2),)
    gate = Fermion(spinful=True, symmetry="U1").chemical_potential_operator()
    exact = state.tn.compute_local_expectation_exact(
        {where: gate}, normalized=True, optimize="auto-hq"
    )
    value = compute_local_expectation_edge_loop_series(
        state.tn,
        {where: gate},
        gloops=0,
        max_iterations=200,
        tol=1e-10,
        diis=False,
    )
    rho = partial_trace_edge_loop_series_expand(
        state.tn,
        where,
        gloops=0,
        max_iterations=200,
        tol=1e-10,
        diis=False,
    )

    np.testing.assert_allclose(value, exact, rtol=1e-10, atol=1e-12)
    assert type(rho).__name__ == "U1FermionicArray"
    np.testing.assert_allclose(np.trace(rho.to_dense()), 1.0)


def test_open_scalar_series_uses_native_fermion_projectors_for_long_range_hopping():
    """Long-range native hopping stays graded while using one BP solve."""
    state = SymPEPS.random(
        1,
        4,
        symmetry="U1",
        bond_dim=3,
        phys_dim=2,
        fermionic=True,
        seed=1931,
        dtype="complex128",
    )
    where = ((0, 0), (0, 3))
    gate = Fermion(spinful=False, symmetry="U1").hopping_operator()
    exact = state.tn.compute_local_expectation_exact(
        {where: gate}, normalized=True, optimize="auto-hq"
    )
    bp = two_norm_bp(
        state.tn,
        max_iterations=200,
        tol=1e-10,
        diis=False,
    )
    info = {}
    value = compute_local_expectation_open_loop_series(
        state.tn,
        {where: gate},
        gloops=3,
        messages=bp.messages,
        run_bp=False,
        max_iterations=200,
        tol=1e-10,
        diis=False,
        info=info,
    )

    np.testing.assert_allclose(value, exact, rtol=1e-10, atol=1e-12)
    assert info["open_scalar_native_route"] == "graded_open_projectors"
    assert info["open_scalar_fermionic_q_phase"] is True
    assert info["open_scalar_family_counts"] == {"open_path": 1}

    reverse_where = where[::-1]
    reverse_gate = gate.transpose((1, 0, 3, 2))
    reverse_exact = state.tn.compute_local_expectation_exact(
        {reverse_where: reverse_gate},
        normalized=True,
        optimize="auto-hq",
    )
    reverse_value = compute_local_expectation_open_loop_series(
        state.tn,
        {reverse_where: reverse_gate},
        gloops=3,
        messages=bp.messages,
        run_bp=False,
        max_iterations=200,
        tol=1e-10,
        diis=False,
    )
    np.testing.assert_allclose(
        reverse_value,
        reverse_exact,
        rtol=1e-10,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "symmetry,phys_dim,site_charge,gate_name,phase",
    (
        ("U1", 4, None, "density", False),
        ("U1U1", 4, site_charge_alternating((1, 0), (0, 1)), "hopping", True),
        ("Z2", 2, None, "density", False),
    ),
)
def test_open_scalar_series_native_gate_phase_matches_symmetry(
    symmetry,
    phys_dim,
    site_charge,
    gate_name,
    phase,
):
    """Native open projectors handle diagonal and off-diagonal gates."""
    state = SymPEPS.random(
        1,
        4,
        symmetry=symmetry,
        bond_dim=3,
        phys_dim=phys_dim,
        fermionic=True,
        site_charge=site_charge,
        seed=1960 + len(symmetry),
        dtype="complex128",
    )
    fermion = Fermion(
        spinful=phys_dim == 4,
        symmetry=symmetry,
    )
    gate = getattr(fermion, f"{gate_name}_operator")()
    where = ((0, 0), (0, 3))
    exact = state.tn.compute_local_expectation_exact(
        {where: gate},
        normalized=True,
        optimize="auto-hq",
    )
    info = {}
    value = compute_local_expectation_open_loop_series(
        state.tn,
        {where: gate},
        gloops=3,
        max_iterations=200,
        tol=1e-10,
        diis=False,
        info=info,
    )

    np.testing.assert_allclose(value, exact, rtol=1e-10, atol=1e-12)
    assert info["open_scalar_fermionic_q_phase"] is phase


def test_open_scalar_series_cyclic_native_u1_matches_exact_d3():
    """Cyclic native U1 open corrections remain exact at full cutoff."""
    site_charge = site_charge_from_map(
        {(0, 0): 2, (1, 2): 2},
        default=0,
    )
    state = SymPEPS.random(
        2,
        3,
        symmetry="U1",
        bond_dim=3,
        phys_dim=4,
        fermionic=True,
        site_charge=site_charge,
        seed=2053,
        dtype="complex128",
    )
    fermion = Fermion(spinful=True, symmetry="U1")
    where = ((0, 0), (1, 2))
    gate = fermion.operator_term(
        [(1.0, ((0, "double"), (1, "double")))],
        sites=(0, 1),
    )
    exact = state.tn.compute_local_expectation_exact(
        {where: gate},
        normalized=True,
        optimize="auto-hq",
    )
    bp = two_norm_bp(
        state.tn,
        max_iterations=200,
        tol=1e-9,
        diis=False,
    )
    info = {}
    value = compute_local_expectation_open_loop_series(
        state.tn,
        {where: gate},
        gloops=7,
        messages=bp.messages,
        run_bp=False,
        max_iterations=200,
        tol=1e-9,
        diis=False,
        info=info,
    )
    np.testing.assert_allclose(value, exact, rtol=1e-10, atol=1e-12)
    assert info["open_scalar_native_route"] == "graded_cluster_compatible"

    rho_info = {}
    rho = partial_trace_open_loop_series_expand(
        state.tn,
        where,
        gloops=7,
        messages=bp.messages,
        run_bp=False,
        max_iterations=200,
        tol=1e-9,
        diis=False,
        info=rho_info,
    )
    assert type(rho).__name__ == "U1FermionicArray"
    np.testing.assert_allclose(np.trace(rho.to_dense()), 1.0, atol=1e-12)
    assert rho_info["open_rho_native_route"] == "graded_cluster_compatible"


@pytest.mark.parametrize(
    "symmetry,site_charge,seed",
    (
        (
            "U1",
            site_charge_from_map({(0, 0): 2, (1, 2): 2}, default=0),
            2083,
        ),
        (
            "U1",
            site_charge_from_map({(0, 0): 2, (1, 2): 2}, default=0),
            2085,
        ),
        (
            "U1U1",
            site_charge_alternating((1, 0), (0, 1)),
            2084,
        ),
        (
            "U1U1",
            site_charge_alternating((1, 0), (0, 1)),
            2086,
        ),
    ),
)
def test_cyclic_native_open_series_gate_probes_and_rho_match_cluster(
    symmetry,
    site_charge,
    seed,
):
    """Cyclic native rho is cluster-consistent and gate probes are exact."""
    state = SymPEPS.random(
        2,
        3,
        symmetry=symmetry,
        bond_dim=3,
        phys_dim=4,
        fermionic=True,
        site_charge=site_charge,
        seed=seed,
        dtype="complex128",
    )
    where = ((0, 0), (1, 2))
    fermion = Fermion(spinful=True, symmetry=symmetry)
    bp = two_norm_bp(
        state.tn,
        max_iterations=200,
        tol=1e-9,
        diis=False,
    )

    open_rho = partial_trace_open_loop_series_expand(
        state.tn,
        where,
        gloops=7,
        messages=bp.messages,
        run_bp=False,
        max_iterations=200,
        tol=1e-9,
        diis=False,
    )
    cluster_rho = partial_trace_loop_cluster_expand(
        state.tn,
        where,
        gloops=7,
        messages=bp.messages,
        run_bp=False,
        max_iterations=200,
        tol=1e-9,
        diis=False,
    )
    assert type(open_rho).__name__ == f"{symmetry}FermionicArray"
    assert type(cluster_rho).__name__ == f"{symmetry}FermionicArray"
    np.testing.assert_allclose(
        open_rho.to_dense(),
        cluster_rho.to_dense(),
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.trace(open_rho.to_dense()),
        1.0,
        rtol=1e-10,
        atol=1e-12,
    )

    for gate in (
        fermion.density_operator(),
        fermion.hopping_operator(),
        fermion.eta_pair_operator(),
    ):
        exact = state.tn.compute_local_expectation_exact(
            {where: gate},
            normalized=True,
            optimize="auto-hq",
        )
        value = compute_local_expectation_open_loop_series(
            state.tn,
            {where: gate},
            gloops=7,
            messages=bp.messages,
            run_bp=False,
            max_iterations=200,
            tol=1e-9,
            diis=False,
        )
        np.testing.assert_allclose(value, exact, rtol=1e-10, atol=1e-12)

        reverse_where = where[::-1]
        reverse_gate = gate.transpose((1, 0, 3, 2))
        reverse_exact = state.tn.compute_local_expectation_exact(
            {reverse_where: reverse_gate},
            normalized=True,
            optimize="auto-hq",
        )
        reverse_value = compute_local_expectation_open_loop_series(
            state.tn,
            {reverse_where: reverse_gate},
            gloops=7,
            messages=bp.messages,
            run_bp=False,
            max_iterations=200,
            tol=1e-9,
            diis=False,
        )
        np.testing.assert_allclose(
            reverse_value,
            reverse_exact,
            rtol=1e-10,
            atol=1e-12,
        )


def test_cyclic_native_open_series_honors_contraction_cost_limits():
    """Safe cyclic fallback still records and applies Cotengra budgets."""
    site_charge = site_charge_from_map(
        {(0, 0): 2, (1, 2): 2},
        default=0,
    )
    state = SymPEPS.random(
        2,
        3,
        symmetry="U1",
        bond_dim=3,
        phys_dim=4,
        fermionic=True,
        site_charge=site_charge,
        seed=2093,
        dtype="complex128",
    )
    where = ((0, 0), (1, 2))
    gate = Fermion(spinful=True, symmetry="U1").operator_term(
        [(1.0, ((0, "double"), (1, "double")))],
        sites=(0, 1),
    )
    exact = state.tn.compute_local_expectation_exact(
        {where: gate},
        normalized=True,
        optimize="auto-hq",
    )
    info = {}
    value = compute_local_expectation_open_loop_series(
        state.tn,
        {where: gate},
        gloops=7,
        max_iterations=200,
        tol=1e-9,
        diis=False,
        max_flops_log10=11.0,
        max_peak_memory_log2=30.0,
        info=info,
    )
    np.testing.assert_allclose(value, exact, rtol=1e-10, atol=1e-12)
    assert info["open_scalar_native_route"] == "graded_cluster_compatible"
    assert not info["open_scalar_edge_term_costs"]
    assert not info["open_scalar_edge_skipped_terms"]
    assert info["open_scalar_cluster_region_costs"]
    assert info["open_scalar_cluster_region_skipped_terms"] == {}
    assert info["open_scalar_term_costs"]
    for costs in info["open_scalar_term_costs"].values():
        for cost in costs.values():
            if cost is not None:
                assert cost["flops_log10"] <= 11.0
                assert cost["peak_memory_log2"] <= 30.0

    rho_info = {}
    rho = partial_trace_open_loop_series_expand(
        state.tn,
        where,
        gloops=7,
        max_iterations=200,
        tol=1e-9,
        diis=False,
        max_flops_log10=11.0,
        max_peak_memory_log2=30.0,
        info=rho_info,
    )
    assert type(rho).__name__ == "U1FermionicArray"
    assert rho_info["open_rho_native_route"] == "graded_cluster_compatible"
    assert not rho_info["open_rho_edge_term_costs"]
    assert not rho_info["open_rho_edge_skipped_terms"]
    assert rho_info["open_rho_cluster_region_costs"]
    assert rho_info["open_rho_cluster_region_skipped_terms"] == {}
    assert rho_info["open_rho_term_costs"]
    np.testing.assert_allclose(np.trace(rho.to_dense()), 1.0, atol=1e-12)
    for cost in rho_info["open_rho_term_costs"].values():
        assert cost["flops_log10"] <= 11.0
        assert cost["peak_memory_log2"] <= 30.0


def test_cyclic_native_route_skips_open_edge_enumeration(monkeypatch):
    """Cluster fallback is selected before combinatorial edge discovery."""
    state = SymPEPS.random(
        2,
        3,
        symmetry="U1",
        bond_dim=2,
        phys_dim=2,
        fermionic=True,
        cyclic=(True, True),
        seed=2201,
        dtype="complex128",
    )
    where = ((0, 0), (1, 2))
    bp = two_norm_bp(
        state.tn,
        max_iterations=100,
        tol=1e-9,
        diis=False,
    )

    def fail_if_enumerated(*args, **kwargs):
        raise AssertionError("cyclic native route enumerated edge terms")

    monkeypatch.setattr(
        bp_series,
        "_iter_open_edge_loops",
        fail_if_enumerated,
    )
    rho_info = {}
    rho = partial_trace_open_loop_series_expand(
        state.tn,
        where,
        cluster_size=3,
        max_terms=0,
        messages=bp.messages,
        run_bp=False,
        info=rho_info,
    )
    assert rho_info["open_rho_native_route"] == "graded_cluster_compatible"
    np.testing.assert_allclose(np.trace(rho.to_dense()), 1.0, atol=1e-12)

    scalar_info = {}
    value = compute_local_expectation_open_loop_series(
        state.tn,
        {where: Fermion(spinful=False, symmetry="U1").density_operator()},
        cluster_size=3,
        max_terms=0,
        messages=bp.messages,
        run_bp=False,
        info=scalar_info,
    )
    assert scalar_info["open_scalar_native_route"] == (
        "graded_cluster_compatible"
    )
    assert np.isfinite(value)


def test_explicit_edge_loop_series_preserves_dense_edge_degree_terms():
    """Edge-degree terms are distinct from the local-region cutoff API."""
    state = qtn.PEPS.rand(2, 2, bond_dim=2, seed=1906, dtype="complex128")
    where = ((0, 0),)
    gate = np.diag([1.0, -1.0])
    info = {}
    rho = partial_trace_edge_loop_series_expand(
        state,
        where,
        gloops=4,
        max_iterations=200,
        tol=1e-10,
        diis=False,
        info=info,
    )
    value = compute_local_expectation_edge_loop_series(
        state,
        {where: gate},
        gloops=4,
        max_iterations=200,
        tol=1e-10,
        diis=False,
    )

    assert info["edge_rho_terms"]
    assert all(len(term.edges) <= 4 for term in info["edge_rho_terms"])
    np.testing.assert_allclose(value, np.trace(rho @ gate))


def test_explicit_edge_loop_series_rejects_multisite_fermionic_q_terms():
    """Unsupported graded multi-site Q terms fail before contraction."""
    state = SymPEPS.random(
        2,
        2,
        symmetry="U1",
        bond_dim=2,
        phys_dim=4,
        fermionic=True,
        seed=1907,
        dtype="complex128",
    )
    fermion = Fermion(spinful=True, symmetry="U1")
    where = ((0, 0), (1, 1))
    gate = fermion.eta_pair_operator()
    with pytest.raises(NotImplementedError, match="multi-site gates"):
        compute_local_expectation_edge_loop_series(
            state.tn,
            {where: gate},
            gloops=4,
            max_iterations=200,
            tol=1e-10,
            diis=False,
        )


def _closed_u1_scalar_network():
    """Build a small native Symmray scalar network for the D1 compatibility path."""
    maps = ({0: 0, 1: 1},) * 2
    left = sr.U1Array.from_dense(
        np.diag([2.0, 3.0]),
        index_maps=maps,
        duals=(False, True),
        charge=0,
    )
    right = sr.U1Array.from_dense(
        np.diag([5.0, 7.0]),
        index_maps=maps,
        duals=(True, False),
        charge=0,
    )
    return qtn.TensorNetwork(
        [
            qtn.Tensor(left, inds=("x", "y")),
            qtn.Tensor(right, inds=("x", "y")),
        ]
    )


def test_native_symmray_closed_scalar_network_works_through_one_norm_apis():
    """Valid closed 1-norm inputs use the dense-compatible BP shadow."""
    tn = _closed_u1_scalar_network()

    result = one_norm_bp(
        tn,
        method="d1bp",
        max_iterations=100,
        tol=1e-10,
    )
    assert result.converged
    assert all(type(tensor.data).__name__ == "ndarray" for tensor in result.bp.tn)

    cluster = loop_cluster_expand(
        tn,
        0,
        norm="1norm",
        max_iterations=100,
        tol=1e-10,
    )
    assert cluster.bp_converged
    assert np.isfinite(float(np.real(cluster.estimate)))

    series = loop_series_expand(
        tn,
        0,
        norm="1norm",
        max_iterations=100,
        tol=1e-10,
    )
    assert series.bp_converged
    assert np.isfinite(float(np.real(series.estimate)))

    pne = partitioned_expand(
        tn,
        partition_inds=("x",),
        norm="1norm",
        max_iterations=100,
        tol=1e-10,
    )
    assert pne.bp_converged
    assert np.isfinite(float(np.real(pne.estimate)))

    weights = weight_pass(tn, max_iterations=2)
    assert all(
        type(tensor.data).__name__ == "ndarray" for tensor in weights.network
    )
