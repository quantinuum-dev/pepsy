"""Measurement dispatch and independent intermediate/final compression policy."""

from copy import deepcopy
import warnings

import numpy as np
import pytest
import quimb as qu
import quimb.tensor as qtn

import pepsy
from pepsy._internal.quimb import (
    quimb_2d_options,
    quimb_compression_options,
    quimb_1d_compression_function,
    quimb_callable_option_supported,
)
from pepsy.bp import compute_boundary_expectation
from pepsy.optimizers.energy.peps import PepsEnergyOptimizer


def _advanced_options(method):
    if not quimb_callable_option_supported(
        quimb_1d_compression_function(method), "compress_opts_final"
    ):
        pytest.skip("Installed Quimb lacks independent final compression options")
    return dict(
        max_bond_oversample=16,
        cutoff_oversample=0.0,
        cutoff_mode_oversample="rel",
        compress_opts_final=dict(method="svd", cutoff=0.0, cutoff_mode="rsum2"),
    )


def test_boundary_keyword_capabilities_and_conflicts():
    def modern(*, method=None, route=None, **kwargs):
        pass

    def legacy(*, mode="mps", **kwargs):
        pass

    options = dict(mode="mps", route="envs", cutoff=1e-7)
    assert quimb_2d_options(modern, options) == dict(method="mps", route="envs", cutoff=1e-7)
    assert options["mode"] == "mps"
    assert quimb_2d_options(legacy, dict(method="mps", route="boundary")) == dict(mode="mps")
    with pytest.raises(NotImplementedError, match="route"):
        quimb_2d_options(legacy, dict(route="envs"))
    with pytest.raises(ValueError, match="Conflicting"):
        quimb_2d_options(modern, dict(mode="mps", method="projector"))
    assert quimb_2d_options(modern, dict(mode="full-bond", method="eigh")) == dict(
        method="full-bond", similarity_method="eigh"
    )
    assert quimb_2d_options(legacy, dict(mode="full-bond", method="eigh")) == dict(
        mode="full-bond", method="eigh"
    )


def test_explicit_environment_route_keeps_measurement_options_out_of_svd():
    if not quimb_callable_option_supported(qtn.PEPS.compute_local_expectation, "route"):
        pytest.skip("Quimb does not expose the environment route")
    state = qtn.PEPS.rand(2, 2, 2, seed=62, dtype="complex128")
    terms = {(0, 0): qu.pauli("Z")}
    expected = state.compute_local_expectation_exact({((0, 0),): qu.pauli("Z")}, normalized=True)
    actual = compute_boundary_expectation(state, terms, max_bond=32, cutoff=0.0, route="envs")
    np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_legacy_measurement_dictionary_normalization():
    from pepsy.boundary._measurements import compute_peps_local_expectation

    class LegacyPEPS:
        Lx = Ly = 2

        def compute_local_expectation(
            self, terms, *, mode="mps", normalized=False, return_all=False
        ):
            assert mode == "mps"
            assert return_all
            return {where: (6.0, 2.0 if normalized else None) for where in terms}

    for normalize, expected in [(True, 3.0), (False, 6.0), ("return", (6.0, 2.0))]:
        assert compute_peps_local_expectation(
            LegacyPEPS(),
            {(0, 0): None},
            method="mps",
            route="boundary",
            normalized=normalize,
            return_all=True,
        ) == {(0, 0): expected}


@pytest.mark.parametrize("shape", [(1, 1), (1, 4), (4, 1), (2, 2)])
@pytest.mark.parametrize("normalized", [False, True, "return"])
def test_boundary_measurement_matches_exact_values_and_norms(shape, normalized):
    state = qtn.PEPS.rand(*shape, bond_dim=2, seed=61, dtype="complex128")
    state.multiply_(2.3)
    last = (shape[0] - 1, shape[1] - 1)
    terms = {(0, 0): qu.pauli("Z")}
    if last != (0, 0):
        terms[((0, 0), last)] = np.kron(qu.pauli("X"), qu.pauli("X"))
    norm = (state.H & state).contract(all)
    expected = {
        where: state.local_expectation_exact(
            gate, (where,) if state.has_site(where) else where, normalized=False
        )
        for where, gate in terms.items()
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        values = compute_boundary_expectation(
            state,
            terms,
            max_bond=64,
            cutoff=0.0,
            method="mps",
            route="boundary",
            normalized=normalized,
            return_all=True,
        )
    assert not any("deprecated" in str(w.message) and "mode" in str(w.message) for w in caught)
    for where, value in values.items():
        if normalized == "return":
            np.testing.assert_allclose(value, (expected[where], norm), atol=1e-10)
        else:
            np.testing.assert_allclose(
                value, expected[where] / norm if normalized else expected[where], atol=1e-10
            )
    total = compute_boundary_expectation(
        state,
        terms,
        max_bond=64,
        cutoff=0.0,
        normalized=normalized,
    )
    np.testing.assert_allclose(
        total, sum(expected.values()) / norm if normalized else sum(expected.values()), atol=1e-10
    )


@pytest.mark.parametrize("shape", [(1, 4), (4, 1)])
def test_single_line_energy_torch_gradients_match_exact(shape):
    torch = pytest.importorskip("torch")
    state = qtn.PEPS.rand(*shape, bond_dim=2, seed=21, dtype="float64")
    state.apply_to_arrays(lambda x: torch.tensor(x, requires_grad=True))
    terms = {((0, 0),): torch.diag(torch.tensor([1.0, -1.0], dtype=torch.float64))}
    kwargs = dict(terms=terms, chi=8, cutoff=0.0, energy_per_site=False)
    value = PepsEnergyOptimizer._loss_state(state, **kwargs)
    reference = PepsEnergyOptimizer._loss_state(state, boundary_mode="exact", **kwargs)
    params = tuple(t.data for t in state.tensors)
    grads = torch.autograd.grad(value, params, retain_graph=True)
    expected = torch.autograd.grad(reference, params)
    torch.testing.assert_close(value, reference)
    for grad, target in zip(grads, expected):
        torch.testing.assert_close(grad, target, atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize("shape", [(1, 4), (4, 1)])
def test_single_line_native_fermion_measurement(shape):
    pytest.importorskip("symmray")
    state = pepsy.SymPEPS.random(
        *shape,
        symmetry="Z2",
        fermionic=True,
        phys_dim={0: 1, 1: 1},
        bond_dim=2,
        seed=55,
        dtype="complex128",
    )
    operator = state.operator_from_dense(np.diag([1.0, -1.0]))
    reference = pepsy.measure_obs(state.tn, operator, where=(0, 0), ind_id=state.site_ind_id)
    result = state.measure(operator, where=(0, 0))
    assert result == pytest.approx(reference, abs=1e-10)


@pytest.mark.parametrize("method", ["sdc-oversample", "sdcr-oversample", "zipup-oversample"])
@pytest.mark.parametrize("kind", ["mps", "submpo", "mpo"])
def test_independent_compression_reconstructs_interior_gate_and_preserves_inputs(method, kind):
    options = _advanced_options(method)
    saved = deepcopy(options)
    gate, where = qu.CNOT(), (1, 4)
    if kind == "mpo":
        state = qtn.MPO_rand(6, 2, seed=2, dtype="complex128")
        cls = pepsy.MpoOptimizer
        ref_opt = cls(state, [(gate, where)], chi=64, mode="direct")
        expected = ref_opt.run(cutoff=0.0, fidelity_samples=0).to_dense()
    else:
        state = qtn.MPS_rand_state(6, 2, seed=2, dtype="complex128")
        cls = pepsy.MpsOptimizer
        expected = state.gate_nonlocal(gate, where, max_bond=64, cutoff=0.0).to_dense()
    before = state.to_dense().copy()
    if kind == "submpo":
        operator = qtn.MatrixProductOperator.from_dense(gate, dims=(2, 2), sites=where, L=6)
        event = cls.submpo_event(operator, where)
    else:
        event = (gate, where)
    optimizer = cls(state, [event], chi=8, mode=method)
    result = optimizer.run(cutoff=0.0, compression_opts=options)
    np.testing.assert_allclose(result.to_dense(), expected, atol=1e-9, rtol=1e-9)
    np.testing.assert_array_equal(state.to_dense(), before)
    assert result.max_bond() <= 8
    assert options == saved
    assert all(t.data.dtype == np.complex128 for t in result.tensors)


@pytest.mark.parametrize(
    "cls,constructor",
    [(pepsy.MpsOptimizer, qtn.MPS_rand_state), (pepsy.MpoOptimizer, qtn.MPO_rand)],
)
def test_unsupported_compression_controls_fail_before_layout(cls, constructor):
    state = constructor(4, 2, seed=2, dtype="complex128")
    optimizer = cls(state, [(qu.CNOT(), (0, 3))], chi=2, mode="direct")
    before = optimizer.p.to_dense().copy()
    with pytest.raises(NotImplementedError, match="does not explicitly support"):
        optimizer.run(compression_opts={"cutoff_oversample": 0.0}, layout=(3, 2, 1, 0))
    np.testing.assert_array_equal(optimizer.p.to_dense(), before)


def test_compression_rejects_hidden_final_cap_and_sdcr_cumulative_intermediate():
    options = _advanced_options("sdcr-oversample")
    options["compress_opts_final"]["max_bond"] = 99
    with pytest.raises(ValueError, match="bond cap"):
        quimb_compression_options("sdcr-oversample", options)
    options.pop("compress_opts_final")
    options["cutoff_mode_oversample"] = "rsum2"
    with pytest.raises(ValueError, match="abs or rel"):
        quimb_compression_options("sdcr-oversample", options)


def test_final_compression_cannot_bypass_truncation_with_qr():
    options = _advanced_options("sdc-oversample")
    options["compress_opts_final"]["method"] = "qr"
    with pytest.raises(ValueError, match="Final compression method"):
        quimb_compression_options("sdc-oversample", options)


@pytest.mark.parametrize("method", ["sdc-oversample", "sdcr-oversample", "zipup-oversample"])
@pytest.mark.parametrize("final_method", ["svd", "svd:eig", "rsvd"])
def test_final_cumulative_cutoff_is_independent_of_intermediate_cutoff(method, final_method):
    options = _advanced_options(method)
    options["compress_opts_final"]["cutoff"] = 0.15
    options["compress_opts_final"]["method"] = final_method
    if final_method == "rsvd":
        options["compress_opts_final"].update(cutoff=0.4, cutoff_mode="rel")
    vector = np.array([np.sqrt(0.9), 0.0, 0.0, np.sqrt(0.1)])
    state = qtn.MatrixProductState.from_dense(vector, dims=[2, 2])
    optimizer = pepsy.MpsOptimizer(state, [(np.eye(4), (0, 1))], chi=2, mode=method)
    out = optimizer.run(cutoff=0.0, normalize_final=False, compression_opts=options)
    assert out.max_bond() == 1
    np.testing.assert_allclose(out.to_dense().ravel(), [np.sqrt(0.9), 0.0, 0.0, 0.0], atol=1e-12)
    assert np.linalg.norm(out.to_dense().ravel() - vector) ** 2 == pytest.approx(0.1)


@pytest.mark.parametrize("method", ["sdc-oversample", "zipup-oversample", "sdcr-oversample"])
def test_peps_boundary_independent_compression_matches_exact_norm(method):
    options = _advanced_options(method)
    state = qtn.PEPS.rand(3, 3, bond_dim=2, seed=71, dtype="complex128")
    expected = (state.H & state).contract(all, optimize="greedy")
    value = pepsy.peps_norm(
        state,
        chi=64,
        fit_mode=method,
        fit_max_bond=64,
        fit_compression_opts=options,
        cutoff=0.0,
        max_separation=0,
        contraction_opt="greedy",
        progress=False,
    )
    np.testing.assert_allclose(value, expected, atol=1e-7, rtol=1e-9)


def test_shot_replay_preserves_explicit_final_compression_policy():
    options = _advanced_options("sdc-oversample")
    options["compress_opts_final"]["cutoff"] = 0.15
    vector = np.array([np.sqrt(0.9), 0.0, 0.0, np.sqrt(0.1)])
    state = qtn.MatrixProductState.from_dense(vector, dims=[2, 2])
    optimizer = pepsy.MpsOptimizer(state, [(np.eye(4), (0, 1))], chi=2, mode="sdc-oversample")
    result = optimizer.run(
        shots=2, strategy="independent", workers=1, cutoff=0.0, compression_opts=options
    )
    for shot in result.optimizers:
        assert shot.p.max_bond() == 1
        np.testing.assert_allclose(
            shot.p.to_dense().ravel(), [np.sqrt(0.9), 0.0, 0.0, 0.0], atol=1e-12
        )


@pytest.mark.parametrize("kind", ["mps", "mpo"])
def test_seeded_intermediate_compression_remains_reproducible(kind):
    options = _advanced_options("src-oversample")
    options.pop("cutoff_mode_oversample")  # SRC is rank controlled.
    options["max_bond_oversample"] = 4
    cls = pepsy.MpsOptimizer if kind == "mps" else pepsy.MpoOptimizer
    constructor = qtn.MPS_rand_state if kind == "mps" else qtn.MPO_rand
    state = constructor(6, 3, seed=11, dtype="complex128")
    values = []
    for _ in range(2):
        optimizer = cls(state, [(qu.CNOT(), (0, 5))], chi=2, mode="src-oversample")
        values.append(optimizer.run(compression_seed=81, compression_opts=options).to_dense())
        assert optimizer.p.max_bond() <= 2
    np.testing.assert_allclose(*values, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("shape", [(1, 4), (4, 1)])
def test_single_line_fermionic_hopping_retains_long_range_sign(shape):
    pytest.importorskip("symmray")
    fermion = pepsy.Fermion(spinful=False, symmetry="U1")
    state = pepsy.SymPEPS.random(
        *shape,
        symmetry="U1",
        fermionic=True,
        phys_dim={0: 1, 1: 1},
        site_charge=lambda site: int(sum(site) % 2 == 0),
        bond_dim=4,
        seed=42,
        dtype="complex128",
    )
    where = ((0, 0), (shape[0] - 1, shape[1] - 1))
    operator = fermion.hopping_operator()
    expected = pepsy.measure_obs(state.tn, operator, where=where, ind_id=state.site_ind_id)
    actual = compute_boundary_expectation(state.tn, {where: operator}, max_bond=8)
    assert abs(expected) > 1e-5
    assert actual == pytest.approx(expected, abs=1e-10)
