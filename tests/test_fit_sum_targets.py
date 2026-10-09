"""Variational sums use termwise environments and one combined local update."""

import numpy as np
import pytest
import quimb.tensor as qtn

import pepsy as py
from pepsy.fitting import FIT


pytestmark = [pytest.mark.core, pytest.mark.mps]


@pytest.mark.parametrize("operator", [False, True])
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("method,block", [
    ("run", 1), ("run_eff", 1), ("run_eff", 2), ("run_eff", 3),
    ("run_gate", 1), ("run_gate", 2), ("run_gate", 3),
])
def test_sum_matches_materialized_reference(operator, dtype, method, block):
    constructor = qtn.MPO_rand if operator else qtn.MPS_rand_state
    a = constructor(4, 2, dtype=dtype, seed=81)
    b = constructor(4, 3, dtype=dtype, seed=82)
    guess = constructor(4, 2, dtype=dtype, seed=83)
    terms = [a, (0.2 + 0.4j) * b, -0.6 * a]
    target = terms[0] + terms[1] + terms[2]
    snapshots = [t.to_dense().copy() for t in terms]
    fit = FIT(terms, p=guess, range_int=(0, 3))
    reference = FIT(target, p=guess, range_int=(0, 3))
    options = dict(n_iter=4, verbose=True)
    if method != "run":
        options.update(block_size=block, max_bond=16, cutoff=0, rtol=None,
                       sweep_sequence="LR")
    getattr(fit, method)(**options)
    getattr(reference, method)(**options)
    tol = 3e-5 if dtype == "complex64" else 2e-11
    np.testing.assert_allclose(fit.p.to_dense(), reference.p.to_dense(), atol=tol)
    if block > 1:
        np.testing.assert_allclose(fit.p.to_dense(), target.to_dense(), atol=tol)
    np.testing.assert_allclose(fit.fidelity_trace, reference.fidelity_trace, atol=tol)
    for term, before in zip(terms, snapshots):
        np.testing.assert_array_equal(term.to_dense(), before)
    assert isinstance(fit, FIT)
    assert len(fit.prepared_target) == 3


@pytest.mark.parametrize("direction", ["RL", "LR"])
def test_layered_mpo_cancellation_without_materializing_sum(monkeypatch, direction):
    a, b, p = [qtn.MPO_rand(5, 2, dtype="complex128", seed=s) for s in (1, 2, 3)]
    product = a.apply(b, contract=False)
    identity = qtn.MPO_identity(5, dtype="complex128")
    reference = identity.to_dense()

    def forbidden(*args, **kwargs):
        raise AssertionError("Fitting must not materialize or densify a sum target")

    with monkeypatch.context() as patch:
        patch.setattr(qtn.MatrixProductOperator, "add_MPO", forbidden)
        patch.setattr(qtn.TensorNetwork, "to_dense", forbidden)
        fit = FIT([product, -product, identity], p=p, range_int=(0, 4))
        fit.run_gate(n_iter=5, block_size=3, max_bond=2, cutoff=1e-12,
                     rtol=None, sweep_sequence=direction)
    np.testing.assert_allclose(fit.p.to_dense(), reference, atol=1e-11)
    assert fit.p.max_bond() <= 2
    assert fit._sweep_environment_reuse_count == 4


@pytest.mark.parametrize("direction", ["RL", "LR"])
def test_window_contracts_outside_coefficients_and_preserves_outside(direction):
    p = qtn.MPS_rand_state(6, 2, dtype="complex128", seed=7)
    p.canonize(2)
    a = p.gate(np.diag([1, 0.2j]), 2, contract=True)
    b = p.gate(np.array([[0, 1], [1, 0]]), 3, contract=True)
    # Put the weight outside the active window; dropping outside environments
    # would fit a+b instead of 2a+0.3j*b.
    a[0].modify(data=2 * a[0].data)
    b[5].modify(data=0.3j * b[5].data)
    expected = a.to_dense() + b.to_dense()
    fit = FIT([a, b], p=p, range_int=(2, 3))
    outside = {i: fit.p[i].data for i in (0, 1, 4, 5)}
    for _ in range(2):
        fit.run_gate(n_iter=2, block_size=2, max_bond=4, cutoff=0,
                     rtol=None, sweep_sequence=direction)
        np.testing.assert_allclose(fit.p.to_dense(), expected, atol=1e-11)
        for i, data in outside.items():
            assert fit.p[i].data is data
    fit.run_eff(n_iter=2, block_size=2, max_bond=4, cutoff=0)
    np.testing.assert_allclose(fit.p.to_dense(), expected, atol=1e-11)


def test_sum_cache_matches_rebuild():
    a, b, p = [qtn.MPS_rand_state(6, 2, dtype="complex128", seed=s) for s in (1, 2, 3)]
    cached = FIT([a, b], p=p, range_int=(0, 5))
    rebuilt = FIT([a, b], p=p, range_int=(0, 5))
    rebuilt._allow_sweep_environment_reuse = False
    options = dict(n_iter=5, block_size=3, max_bond=3, cutoff=1e-12, rtol=None,
                   two_site_transition_sweeps=1, final_one_site_sweeps=1)
    cached.run_gate(**options)
    rebuilt.run_gate(**options)
    np.testing.assert_allclose(cached.p.to_dense(), rebuilt.p.to_dense(), atol=1e-11)
    # Five main sweeps plus the requested final one-site polish sweep.
    assert cached._sweep_environment_reuse_count == 5
    assert rebuilt._sweep_environment_reuse_count == 0


def test_sum_torch_gradient_matches_dense_objective():
    torch = pytest.importorskip("torch")
    a, b, p = [qtn.MPS_rand_state(3, 2, dtype="complex128", seed=s) for s in (1, 2, 3)]
    for state in (a, b, p):
        state.apply_to_arrays(torch.as_tensor)
    theta = torch.tensor(0.31, dtype=torch.float64, requires_grad=True)
    weighted = b.copy()
    weighted[0].modify(data=theta * weighted[0].data)
    fit = FIT([a, weighted], p=p, range_int=(0, 2))
    fit.run_gate(n_iter=2, block_size=2, max_bond=2, cutoff=0, rtol=None)
    value = (fit.p.to_dense().abs()**2).sum()
    expected = ((a.to_dense() + theta * b.to_dense()).abs()**2).sum()
    actual_grad, = torch.autograd.grad(value, theta, retain_graph=True)
    expected_grad, = torch.autograd.grad(expected, theta)
    assert torch.isfinite(actual_grad)
    torch.testing.assert_close(value, expected, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("method", ["run_eff", "run_gate"])
@pytest.mark.parametrize("block", [1, 2, 3])
def test_native_u1u1_sum_stays_native(monkeypatch, method, block):
    pytest.importorskip("symmray")
    fermion = py.Fermion(spinful=True, symmetry="U1U1", dtype="complex128")
    p = py.hrs_to_mps(4, fermion=fermion, occupations=((1, 0), (0, 1)) * 2,
                      chi=2, random_rounds=2, seed=61, dtype="complex128")
    gate = fermion.onsite_gate(0.3, site=1, U=1.7, mu=0.2)
    b = p.gate(gate, 1, contract=True)
    reference = p.gate(fermion.operator("identity") + 0.3j * gate, 1, contract=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("Native sum FIT must not densify")

    with monkeypatch.context() as patch:
        for cls in {type(t.data) for t in p.tensors}:
            patch.setattr(cls, "to_dense", forbidden)
        fit = FIT([p, 0.3j * b], p=p, range_int=(0, 3))
        getattr(fit, method)(n_iter=3, block_size=block, max_bond=8,
                             cutoff=1e-12, rtol=None, verbose=True)
        assert float(py.tn_fidelity(fit.p, reference)) == pytest.approx(1, abs=1e-11)
        actual_norm = (fit.p.H & fit.p).contract(all)
        expected_norm = (reference.H & reference).contract(all)
        np.testing.assert_allclose(actual_norm, expected_norm, atol=1e-11)
        np.testing.assert_allclose(fit.fidelity_trace, 1, atol=1e-11)
        assert all(type(t.data).__name__ == "U1U1FermionicArray" for t in fit.p.tensors)


def test_sum_validation_and_ownership():
    p = qtn.MPS_rand_state(3, 2, seed=5)
    with pytest.raises(ValueError, match="non-empty"):
        FIT([], p=p)
    with pytest.raises(TypeError, match="TensorNetwork"):
        FIT([p, [p]], p=p)
    with pytest.raises(TypeError, match="Initial `p`"):
        FIT([p], p=object())
    untagged = p.copy()
    untagged.drop_tags()
    with pytest.raises(ValueError, match="exactly one site tag"):
        FIT([untagged], p=p)
    wrong = qtn.MPS_rand_state(3, 2, phys_dim=3, seed=6)
    with pytest.raises(ValueError, match="physical dimensions"):
        FIT([p, wrong], p=p)
    wrong = p.reindex({"k0": "other"})
    with pytest.raises(ValueError, match="outer indices"):
        FIT([p, wrong], p=p)
    fit = FIT((p, p), p=p)
    assert fit.p is not p
    fit.run_eff(n_iter=2)
    np.testing.assert_allclose(fit.p.to_dense(), 2 * p.to_dense(), atol=1e-11)
    target = p.copy()
    fit = FIT([target], p=p, inplace=True, copy_target=False)
    assert fit.p is p
    assert fit.tn[0] is target


def test_sum_retains_network_exponents():
    a, b, p = [qtn.MPS_rand_state(3, 2, seed=s) for s in (1, 2, 3)]
    a.exponent = 1.0
    b.exponent = -1.0
    expected = a.to_dense() + b.to_dense()
    fit = FIT([a, b], p=p)
    fit.run_eff(n_iter=2, block_size=2, max_bond=4, cutoff=0)
    np.testing.assert_allclose(fit.p.to_dense(), expected, atol=1e-11)
    assert a.exponent == 1.0
    assert b.exponent == -1.0


@pytest.mark.parametrize("block", [1, 2, 3])
def test_native_sum_against_independently_built_mpo(monkeypatch, block):
    pytest.importorskip("symmray")
    fermion = py.Fermion(spinful=True, symmetry="U1U1", dtype="complex128")
    t1 = fermion.onsite_term(1, U=1.0, mu=0.2)
    t2 = fermion.onsite_term(2, U=0.4, mu=0.1)
    a = fermion.build_mpo({1: t1}, L=4, compress=False)
    b = fermion.build_mpo({2: t2}, L=4, compress=False)
    reference = fermion.build_mpo({1: t1, 2: t2}, L=4, compress=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("Native MPO sums must not densify")

    for cls in {type(t.data) for t in a.tensors}:
        monkeypatch.setattr(cls, "to_dense", forbidden)
    fit = FIT([a, b], p=a, range_int=(0, 3))
    fit.run_gate(n_iter=3, block_size=block, max_bond=8, cutoff=1e-12, rtol=None)
    assert float(py.tn_fidelity(fit.p, reference)) == pytest.approx(1, abs=1e-11)
    overlap = (reference.H & fit.p).contract(all)
    norm = (reference.H & reference).contract(all)
    np.testing.assert_allclose(overlap, norm, atol=1e-11)


@pytest.mark.parametrize("method", ["run", "run_eff", "run_gate"])
def test_single_site_and_exact_zero_sum(method):
    p = qtn.MPS_rand_state(1, 1, dtype="complex128", seed=5)
    fit = FIT([p, -p], p=p)
    getattr(fit, method)(n_iter=1)
    np.testing.assert_allclose(fit.p.to_dense(), 0, atol=1e-14)


def test_repeated_owned_term_and_retag():
    p = qtn.MPS_rand_state(4, 2, dtype="complex128", seed=1)
    target = p.copy()
    target.drop_tags()
    fit = FIT([target, target], p=p, retag=True, copy_target=False)
    fit.run_eff(n_iter=2)
    np.testing.assert_allclose(fit.p.to_dense(), 2 * p.to_dense(), atol=1e-11)


@pytest.mark.parametrize("block", [1, 2, 3])
def test_native_bosonic_sum(monkeypatch, block):
    pytest.importorskip("symmray")
    state = py.SymMPS.random(
        4, symmetry="Z2", phys_dim={0: 1, 1: 1},
        site_charge=py.site_charge_from_occupations([0] * 4),
        bond_dim=2, seed=1, dtype="complex128",
    )
    gate = state.operator_from_dense(py.rz(0.3), charge=0, sites=1)
    identity = state.operator_from_dense(np.eye(2), charge=0, sites=1)
    p = state.tn
    b = p.gate(gate, 1, contract=True)
    reference = p.gate(identity + 0.3j * gate, 1, contract=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("Native bosonic sum FIT must not densify")

    for cls in {type(t.data) for t in p.tensors}:
        monkeypatch.setattr(cls, "to_dense", forbidden)
    fit = FIT([p, 0.3j * b], p=p, range_int=(0, 3))
    fit.run_gate(n_iter=3, block_size=block, max_bond=4, cutoff=1e-12, rtol=None)
    assert float(py.tn_fidelity(fit.p, reference)) == pytest.approx(1, abs=1e-11)
    overlap = (reference.H & fit.p).contract(all)
    norm = (reference.H & reference).contract(all)
    np.testing.assert_allclose(overlap, norm, atol=1e-11)


def test_sum_reuses_layered_selections_and_fixed_environments(monkeypatch):
    from collections import Counter

    a, b, p = [qtn.MPO_rand(6, 2, dtype="complex128", seed=s) for s in (1, 2, 3)]
    fit = FIT([a.apply(b, contract=False), b.apply(a, contract=False)],
              p=p, range_int=(0, 5))
    selections = Counter()
    builds = []
    select = FIT._target_components
    build = FIT._build_active_environments

    def counted_select(self, sites, **kwargs):
        selections[id(self), tuple(sites)] += 1
        return select(self, sites, **kwargs)

    def counted_build(self, *args, **kwargs):
        builds.append(kwargs["block_size"])
        return build(self, *args, **kwargs)

    monkeypatch.setattr(FIT, "_target_components", counted_select)
    monkeypatch.setattr(FIT, "_build_active_environments", counted_build)
    fit.run_gate(n_iter=6, block_size=3, max_bond=3, cutoff=0, rtol=None,
                 two_site_transition_sweeps=1)
    assert builds == [3]
    assert selections and set(selections.values()) == {1}
    assert fit._sweep_environment_reuse_count == 5
    before = selections.copy()
    # A subsequent run reuses only target routing, rebuilding fitted-state
    # environments after the previous run changed the guess.
    fit.run_gate(n_iter=6, block_size=3, max_bond=3, cutoff=0, rtol=None,
                 two_site_transition_sweeps=1)
    assert builds == [3, 3]
    assert selections == before


def test_sum_builds_one_shared_bra_per_environment_site(monkeypatch):
    targets = [qtn.MPS_rand_state(4, 2, dtype="complex128", seed=s) for s in (1, 2, 3)]
    fit = FIT(targets, p=targets[0])
    calls = []
    conj = qtn.Tensor.conj

    def counted_conj(self, *args, **kwargs):
        if self is fit.p[3]:
            calls.append(self)
        return conj(self, *args, **kwargs)

    monkeypatch.setattr(qtn.Tensor, "conj", counted_conj)
    environments = fit._overlap_environment_site(fit.p, 3, 0, 3)
    assert len(calls) == 1
    for term, environment in zip(fit._terms, environments):
        reference = term._overlap_environment_site(fit.p, 3, 0, 3)
        np.testing.assert_allclose(environment.data, reference.data, atol=1e-12)


def test_sum_verbose_norm_computed_once_per_run(monkeypatch):
    targets = [qtn.MPS_rand_state(4, 2, dtype="complex128", seed=s) for s in (1, 2, 3)]
    fit = FIT(targets, p=targets[0])
    contractions = []
    contract = qtn.TensorNetwork.contract

    def counted_contract(self, *args, **kwargs):
        contractions.append(self)
        return contract(self, *args, **kwargs)

    monkeypatch.setattr(qtn.TensorNetwork, "contract", counted_contract)
    # Dense cached sweeps use raw tensor contractions. Network contractions
    # here are only verbose diagnostics: 6 target pairs once + 4 per sweep.
    fit.run_eff(n_iter=3, verbose=True)
    assert len(contractions) == 6 + 3 * 4
    norm_before = fit._sum_target_norm
    fit.tn[0][0].modify(data=2 * fit.tn[0][0].data)
    contractions.clear()
    fit.run_eff(n_iter=2, verbose=True)
    assert len(contractions) == 6 + 2 * 4
    assert not np.isclose(fit._sum_target_norm, norm_before)
    expected = sum(target.to_dense() for target in fit.tn)
    np.testing.assert_allclose(fit._sum_target_norm, np.vdot(expected, expected), atol=1e-11)
