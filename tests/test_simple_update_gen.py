"""Tests for Pepsy's routed SimpleUpdateGen."""

import numpy as np
import pytest
import quimb.tensor as qtn

import pepsy
from pepsy.operators import gate_simple
from pepsy.optimizers import SimpleUpdateGen


def _zz_term():
    z_op = np.diag([1.0, -1.0])
    return np.kron(z_op, z_op)


def _dense_peps(seed=1, dtype="complex128"):
    return qtn.PEPS.rand(
        Lx=2,
        Ly=3,
        bond_dim=2,
        phys_dim=2,
        seed=seed,
        dtype=dtype,
    )


def _ham_for_edge(edge, term=None):
    return qtn.LocalHamGen({edge: _zz_term() if term is None else term})


def _state_vector(tn):
    return np.asarray(tn.to_dense(tn.outer_inds(), optimize="auto-hq"))


def test_simple_update_gen_routes_long_range_gate_where_quimb_fails():
    """Pepsy's driver should route non-adjacent terms through SWAPs."""
    where = ((0, 0), (0, 2))
    ham = _ham_for_edge(where)
    raw = qtn.SimpleUpdateGen(
        _dense_peps(seed=2),
        ham,
        tau=0.01,
        D=3,
        cutoff=0.0,
        gate_opts={"cutoff_mode": "rsum2"},
        ordering=(where,),
        equilibrate_start=False,
        compute_energy_final=False,
        progbar=False,
    )

    with pytest.raises(ValueError, match="not enough values to unpack"):
        raw.sweep(0.01)

    routed = SimpleUpdateGen(
        _dense_peps(seed=2),
        ham,
        tau=0.01,
        D=3,
        cutoff=0.0,
        route_opts={"sequence": "x_then_y"},
        ordering=(where,),
        equilibrate_start=False,
        compute_energy_final=False,
        progbar=False,
    )

    routed.sweep(0.01)

    assert routed.state.max_bond() <= 3


def test_simple_update_gen_calls_pepsy_gate_simple_with_route_options(monkeypatch):
    """The overridden gate hook should forward Pepsy route options."""
    calls = []
    where = ((0, 0), (0, 2))

    def _fake_gate_simple(tn, gate, **kwargs):
        calls.append((tn, gate, kwargs))
        return tn

    monkeypatch.setattr(
        "pepsy.optimizers.peps.simple_update.pepsy_gate_simple",
        _fake_gate_simple,
    )
    su = SimpleUpdateGen(
        _dense_peps(seed=3),
        _ham_for_edge(where),
        D=5,
        cutoff=1.0e-9,
        route_opts={"sequence": "auto"},
        ordering=(where,),
        equilibrate_start=False,
        compute_energy_final=False,
        progbar=False,
    )
    gate = np.eye(4, dtype=np.complex128).reshape(2, 2, 2, 2)

    su.gate(gate, where)

    assert len(calls) == 1
    _, called_gate, kwargs = calls[0]
    assert called_gate is gate
    assert kwargs["where"] == where
    assert kwargs["gauges"] is su.gauges
    assert kwargs["inplace"] is True
    assert kwargs["max_bond"] == 5
    assert kwargs["cutoff"] == pytest.approx(1.0e-9)
    assert kwargs["cutoff_mode"] == "rsum2"
    assert kwargs["sequence"] == "auto"
    assert kwargs["path_canonize"] is True
    assert kwargs["path_compress"] is False


def test_simple_update_gen_route_opts_override_gate_opts():
    """route_opts should make Pepsy routing intent explicit and authoritative."""
    where = ((0, 0), (0, 2))
    su = SimpleUpdateGen(
        _dense_peps(seed=33),
        _ham_for_edge(where),
        D=5,
        gate_opts={"sequence": "x_then_y", "path_canonize": False},
        route_opts={"sequence": "y_then_x"},
        ordering=(where,),
        equilibrate_start=False,
        compute_energy_final=False,
        progbar=False,
    )

    assert su.gate_opts["sequence"] == "y_then_x"
    assert su.gate_opts["path_canonize"] is False
    assert su.gate_opts["path_compress"] is False
    assert su.gate_opts["cutoff_mode"] == "rsum2"


def test_simple_update_gen_rejects_parallel_long_range_terms():
    """Parallel quimb bookkeeping is not route-aware, so fail clearly."""
    where = ((0, 0), (0, 2))
    su = SimpleUpdateGen(
        _dense_peps(seed=4),
        _ham_for_edge(where),
        D=3,
        cutoff=0.0,
        ordering=(where,),
        update="parallel",
        equilibrate_start=False,
        compute_energy_final=False,
        progbar=False,
    )

    with pytest.raises(ValueError, match="route-aware layer scheduling"):
        su.sweep(0.01)


def test_simple_update_gen_rejects_unsupported_quimb_gate_options():
    """Quimb-only gate_simple_ kwargs should not be silently ignored."""
    where = ((0, 0), (0, 1))
    su = SimpleUpdateGen(
        _dense_peps(seed=5),
        _ham_for_edge(where),
        D=3,
        cutoff=0.0,
        gate_opts={"contract": "reduce-split"},
        ordering=(where,),
        equilibrate_start=False,
        compute_energy_final=False,
        progbar=False,
    )
    gate = np.eye(4, dtype=np.complex128).reshape(2, 2, 2, 2)

    with pytest.raises(TypeError, match="unsupported option"):
        su.gate(gate, where)


def test_simple_update_gen_sweep_matches_manual_pepsy_gate_simple():
    """One routed sweep should match manually applying the same routed gate."""
    where = ((0, 0), (0, 2))
    tau = 0.01
    psi0 = _dense_peps(seed=7)
    ham = _ham_for_edge(where)
    route_opts = {"sequence": "x_then_y"}

    su = SimpleUpdateGen(
        psi0.copy(),
        ham,
        tau=tau,
        D=3,
        cutoff=0.0,
        route_opts=route_opts,
        ordering=(where,),
        equilibrate_start=False,
        compute_energy_final=False,
        progbar=False,
    )
    su.sweep(tau)

    manual = psi0.copy()
    gauges = {}
    manual.gauge_all_simple_(
        max_iterations=1,
        gauges=gauges,
        fuse_multibonds=False,
    )
    gate = ham.get_gate_expm(where, -tau)
    gate_simple(
        manual,
        gate,
        where=where,
        gauges=gauges,
        max_bond=3,
        cutoff=0.0,
        cutoff_mode="rsum2",
        sequence="x_then_y",
        smudge=1.0e-6,
        path_canonize=True,
        path_compress=False,
    )
    manual_with_gauges = manual.copy()
    manual_with_gauges.gauge_simple_insert(gauges)

    assert np.allclose(
        _state_vector(su.state),
        _state_vector(manual_with_gauges),
        atol=1.0e-10,
        rtol=1.0e-10,
    )


def test_simple_update_gen_records_finite_cluster_energy():
    """Energy checks should remain quimb's cluster-energy machinery."""
    where = ((0, 0), (0, 2))
    su = SimpleUpdateGen(
        _dense_peps(seed=8, dtype="float64"),
        _ham_for_edge(where),
        tau=0.01,
        D=3,
        cutoff=0.0,
        ordering=(where,),
        equilibrate_start=False,
        compute_energy_every=1,
        compute_energy_final=True,
        compute_energy_per_site=True,
        progbar=False,
    )

    su.evolve(1, progbar=False)

    assert tuple(su.energy_ns) == (0, 1)
    assert len(su.energies) == 2
    assert all(np.isfinite(su.energies))


def test_simple_update_gen_nearest_neighbor_matches_quimb_simple_update():
    """Direct-neighbor terms should stay equivalent to raw quimb simple update."""
    where = ((0, 0), (0, 1))
    tau = 0.01
    psi0 = _dense_peps(seed=9)
    ham = _ham_for_edge(where)

    raw = qtn.SimpleUpdateGen(
        psi0.copy(),
        ham,
        tau=tau,
        D=3,
        cutoff=0.0,
        gate_opts={"cutoff_mode": "rsum2"},
        ordering=(where,),
        equilibrate_start=False,
        compute_energy_final=False,
        progbar=False,
    )
    routed = SimpleUpdateGen(
        psi0.copy(),
        ham,
        tau=tau,
        D=3,
        cutoff=0.0,
        route_opts={"path_canonize": False},
        ordering=(where,),
        equilibrate_start=False,
        compute_energy_final=False,
        progbar=False,
    )

    raw.sweep(tau)
    routed.sweep(tau)

    assert np.allclose(
        _state_vector(routed.state),
        _state_vector(raw.state),
        atol=1.0e-10,
        rtol=1.0e-10,
    )


def test_simple_update_gen_public_exports():
    """Routed simple update should resolve from public namespaces."""
    assert pepsy.SimpleUpdateGen is SimpleUpdateGen
    assert pepsy.optimizers.SimpleUpdateGen is SimpleUpdateGen
    assert pepsy.optimizers.peps.SimpleUpdateGen is SimpleUpdateGen


def test_simple_update_gen_routes_long_range_symmray_peps():
    """Symmray gates should run through the routed update hook."""
    pytest.importorskip("symmray")
    from pepsy.tensors import SymHamiltonian, SymPEPS, site_charge_from_occupations

    where = ((0, 0), (2, 2))
    state = SymPEPS.random(
        3,
        3,
        symmetry="Z2",
        phys_dim={0: 1, 1: 1},
        site_charge=site_charge_from_occupations({
            (i, j): 0
            for i in range(3)
            for j in range(3)
        }),
        bond_dim=2,
        seed=6,
        dtype="complex128",
    )
    hamiltonian = SymHamiltonian.from_edges(
        "itf",
        "Z2",
        (where,),
        jx=-1.0,
        hz=-0.5,
    )
    gate, _ = hamiltonian.gate_stream(0.001)[0]

    class _GateHam:
        nsites = 9
        terms = hamiltonian.terms

        def get_gate_expm(self, where_arg, scale):
            _ = (where_arg, scale)
            return gate

    su = SimpleUpdateGen(
        state.tn.copy(),
        _GateHam(),
        tau=0.001,
        D=4,
        cutoff=1.0e-10,
        ordering=(where,),
        equilibrate_start=False,
        compute_energy_final=False,
        progbar=False,
    )

    su.sweep(0.001)

    out = su.state
    assert out.max_bond() <= 4
    assert all(hasattr(tensor.data, "blocks") for tensor in out)
