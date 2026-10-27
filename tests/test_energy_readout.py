"""Shared local readout routes, batching, and state-lifetime regressions."""

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.optimizers.energy import MpsEnergyOptimizer, TreeEnergyOptimizer
from pepsy.optimizers.tree import TreePlan, TreeTensorNetwork


X = np.array([[0., 1.], [1., 0.]])
Y = np.array([[0., -1j], [1j, 0.]])
Z = np.diag([1., -1.])
TERMS = {(0,): 0.3 * X, (0, 3): 0.7 * np.kron(Z, Z)}
OBS = {"y": {(0,): Y}, "correlation": {(0, 3): np.kron(X, Y)},
       "reverse": {(3, 0): np.kron(X, Z)}, "empty": {}}


def dense_expectation(vector, terms, normalized=True):
    vector = np.asarray(vector).reshape(-1)
    total = 0j
    for where, operator in terms.items():
        where = (where,) if isinstance(where, int) else where
        axes = tuple(where) + tuple(i for i in range(4) if i not in where)
        local = vector.reshape((2,) * 4).transpose(axes).reshape(2 ** len(where), -1)
        total += np.einsum("ai,ab,bi->", local.conj(), operator, local)
    return total / np.vdot(vector, vector) if normalized else total


@pytest.mark.parametrize("route", ["canonical", "envs", "exact"])
@pytest.mark.parametrize("normalized", [False, True])
def test_mps_named_readout_matches_dense_and_preserves_state(route, normalized):
    state = qtn.MPS_rand_state(4, 3, dtype="complex128", seed=11)
    state[0].modify(data=2.3 * state[0].data)
    state.exponent = 1.0
    before = [tensor.data.copy() for tensor in state.tensors]
    vector = state.to_dense()
    opt = MpsEnergyOptimizer(state, terms=TERMS, route=route,
                             contraction_opt="greedy", normalized=normalized)
    result = opt.measure(OBS, torch_batch=True)
    assert result["energy"] == pytest.approx(dense_expectation(vector, TERMS, normalized), abs=1e-11)
    assert opt.energy().energy == pytest.approx(result["energy"], abs=1e-11)
    for name, terms in OBS.items():
        assert result["observables"][name] == pytest.approx(
            dense_expectation(vector, terms, normalized), abs=1e-11,
        )
    assert result["metadata"]["num_supports"] == 3
    assert result["metadata"]["torch_batch"] == "loop"  # NumPy fallback
    for tensor, data in zip(state.tensors, before):
        np.testing.assert_array_equal(tensor.data, data)


@pytest.mark.smoke
def test_envs_shares_one_sweep_and_rebuilds_after_state_change(monkeypatch):
    state = qtn.MPS_rand_state(4, 3, dtype="complex128", seed=23)
    cls = type(state.make_norm())
    original = cls.gen_block_environments
    calls = []

    def tracked(network, blocks, **kwargs):
        calls.append(tuple(blocks))
        yield from original(network, blocks, **kwargs)

    monkeypatch.setattr(cls, "gen_block_environments", tracked)
    opt = MpsEnergyOptimizer(state, terms=TERMS, route="envs")
    first = opt.measure(OBS)
    path_optimizer = opt._local_path_optimizer
    assert len(calls) == 1
    # The normal and reverse pair share the same enclosing interval.
    assert len(calls[0]) == 2
    state.gate_(np.diag([0.4, 1.5]), (0,), contract=True)
    second = opt.measure(OBS)
    assert len(calls) == 2
    assert opt._local_path_optimizer is path_optimizer
    assert second["energy"] == pytest.approx(dense_expectation(state.to_dense(), TERMS), abs=1e-11)
    assert abs(first["energy"] - second["energy"]) > 1e-5


def test_mps_measure_warns_and_falls_back_without_upstream_batch_api(monkeypatch):
    state = qtn.MPS_rand_state(4, 2, dtype="complex128", seed=9)
    opt = MpsEnergyOptimizer(state, terms=TERMS, contraction_opt="greedy")
    monkeypatch.setattr(state, "compute_partial_traces", None)
    with pytest.warns(RuntimeWarning, match="using exact local readout"):
        result = opt.measure(OBS, route="envs")
    assert result["metadata"]["route"] == "exact"
    assert result["metadata"]["requested_route"] == "envs"
    assert result["energy"] == pytest.approx(dense_expectation(state.to_dense(), TERMS), abs=1e-11)


@pytest.mark.parametrize("route", ["canonical", "envs", "exact"])
def test_normalized_mps_readout_cancels_large_exponent(route):
    state = qtn.MPS_rand_state(4, 2, dtype="complex128", seed=2)
    expected = dense_expectation(state.to_dense(), TERMS)
    state.exponent = 400.
    result = MpsEnergyOptimizer(state, terms=TERMS, contraction_opt="greedy").measure(OBS, route=route)
    assert result["energy"] == pytest.approx(expected, abs=1e-11)
    assert state.exponent == 400.


@pytest.mark.parametrize("route", ["canonical", "envs", "exact"])
@pytest.mark.parametrize("batch", [False, True, "vmap"])
def test_torch_readout_batch_values_and_gradients(route, batch):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(91)
    base = torch.tensor(rng.normal(size=16) + 1j * rng.normal(size=16), dtype=torch.complex128)
    theta = torch.tensor(0.13, dtype=torch.float64, requires_grad=True)
    vector = base * torch.exp(theta * torch.linspace(-1., 1., 16))
    state = qtn.MatrixProductState.from_dense(vector, dims=[2] * 4, cutoff=0.0)
    terms = dict(TERMS)
    terms[(0,)] = (0.3 + theta) * torch.tensor(X, dtype=torch.complex128)
    opt = MpsEnergyOptimizer(state, terms=terms, route=route, contraction_opt="greedy")
    result = opt.measure(OBS, torch_batch=batch, chunk_size=1)
    loss = result["energy"] + 0.2 * result["observables"]["y"].real
    # Independent dense operator reference, including a complex observable.
    full = 0.3 * np.kron(X, np.eye(8))
    full += 0.7 * np.kron(np.kron(Z, np.eye(4)), Z)
    full = full.astype(complex) + 0.2 * np.kron(Y, np.eye(8))
    full = torch.tensor(full) + theta * torch.tensor(np.kron(X, np.eye(8)))
    reference = torch.vdot(vector, full @ vector).real / torch.vdot(vector, vector).real
    actual_grad, = torch.autograd.grad(loss, theta, retain_graph=True)
    expected_grad, = torch.autograd.grad(reference, theta)
    torch.testing.assert_close(loss, reference, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-8, atol=1e-9)
    assert result["metadata"]["torch_batch"] == ("einsum" if batch is True else batch or "loop")


def test_cyclic_mps_envs_readout_matches_full_reference():
    state = qtn.MPS_rand_state(4, 2, cyclic=True, dtype="complex128", seed=44)
    opt = MpsEnergyOptimizer(state, terms=TERMS, contraction_opt="greedy")
    result = opt.measure(OBS, route="envs")
    assert result["energy"] == pytest.approx(dense_expectation(state.to_dense(), TERMS), abs=1e-10)
    with pytest.raises((ValueError, NotImplementedError)):
        opt.measure(OBS, route="canonical")


def test_cuda_vmap_readout_stays_on_device():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    state = qtn.MPS_rand_state(4, 2, dtype="complex128", seed=34)
    expected = dense_expectation(state.to_dense(), TERMS)
    state.apply_to_arrays(lambda a: torch.tensor(a, dtype=torch.complex128, device="cuda"))
    result = MpsEnergyOptimizer(state, terms=TERMS, contraction_opt="greedy").measure(
        OBS, route="envs", torch_batch="vmap", chunk_size=2,
    )
    assert result["energy"].device.type == "cuda"
    assert result["energy"].item() == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("batch", [False, True, "vmap"])
def test_tree_named_readout_preserves_gauge_and_matches_dense(batch):
    state = TreeTensorNetwork.rand(TreePlan.from_order(range(4)), D=3, seed=5)
    before = [tensor.data.copy() for tensor in state.tensors]
    region = state.canonical_region
    vector = state.to_dense()
    opt = TreeEnergyOptimizer(state, terms=TERMS, contraction_opt="greedy")
    result = opt.measure(OBS, torch_batch=batch, chunk_size=2)
    assert result["energy"] == pytest.approx(dense_expectation(vector, TERMS), abs=1e-11)
    for name, terms in OBS.items():
        assert result["observables"][name] == pytest.approx(dense_expectation(vector, terms), abs=1e-11)
    assert state.canonical_region == region
    for tensor, data in zip(state.tensors, before):
        np.testing.assert_array_equal(tensor.data, data)


@pytest.mark.parametrize("batch", [True, "vmap"])
def test_tree_torch_named_readout_gradient_matches_full_contraction(batch):
    torch = pytest.importorskip("torch")
    state = TreeTensorNetwork.rand(TreePlan.from_order(range(4)), D=2, seed=8)
    state.apply_to_arrays(lambda a: torch.tensor(a, dtype=torch.complex128))
    theta = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    root = state.node_tensor(state.plan.root)
    weights = torch.linspace(-1., 1., root.data.numel()).reshape(root.shape)
    root.modify(data=root.data * torch.exp(theta * weights))
    result = TreeEnergyOptimizer(state, terms=TERMS, contraction_opt="greedy").measure(
        OBS, torch_batch=batch, chunk_size=2,
    )
    loss = result["energy"] + 0.2 * result["observables"]["y"].real
    vector = qtn.TensorNetwork(state.tensors).contract(
        all, output_inds=tuple(state.site_ind(i) for i in range(4)), optimize="greedy",
    ).data.reshape(-1)
    full = (0.3 * np.kron(X, np.eye(8)) + 0.7 * np.kron(np.kron(Z, np.eye(4)), Z)).astype(complex)
    full += 0.2 * np.kron(Y, np.eye(8))
    expected = torch.vdot(vector, torch.tensor(full) @ vector).real / torch.vdot(vector, vector).real
    actual_grad, = torch.autograd.grad(loss, theta, retain_graph=True)
    expected_grad, = torch.autograd.grad(expected, theta)
    torch.testing.assert_close(loss, expected, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("route", ["canonical", "envs", "exact"])
def test_native_mps_named_readout_retains_graded_fallback(route):
    pytest.importorskip("symmray")
    import pepsy
    fermion = pepsy.Fermion(spinful=False, symmetry="U1")
    state = pepsy.ps_to_mps(3, fermion=fermion, occupations=(1, 0, 1), dtype="complex128")
    n = fermion.observable("number")
    optimizer = MpsEnergyOptimizer(state, terms={(0,): n, (1,): n}, contraction_opt="greedy")
    result = optimizer.measure({"n2": {(2,): n}}, route=route, torch_batch="vmap")
    assert result["energy"] == pytest.approx(1.)
    assert result["observables"]["n2"] == pytest.approx(1.)
    assert result["metadata"]["route"] == "exact"
    assert result["metadata"]["torch_batch"] == "loop"


def test_native_tree_named_readout_retains_graded_fallback():
    pytest.importorskip("symmray")
    import pepsy

    fermion = pepsy.Fermion(spinful=False, symmetry="U1")
    state = pepsy.ps_to_ttn(3, tree=TreePlan.from_order(range(3)),
                           fermion=fermion, occupations=(1, 0, 1), dtype="complex128")
    n = fermion.observable("number")
    opt = TreeEnergyOptimizer(state, terms={(0,): n}, contraction_opt="greedy")
    result = opt.measure({"n0": {(0,): n}, "n2": {(2,): n}}, torch_batch="vmap")
    assert result["energy"] == pytest.approx(1.)
    assert result["observables"] == pytest.approx({"n0": 1., "n2": 1.})
    assert result["metadata"]["route"] == "native_exact"
    assert result["metadata"]["torch_batch"] == "loop"


def test_jax_envs_readout_preserves_gradient_and_uses_loop_fallback():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    def value(theta):
        state = qtn.MPS_product_state([jnp.stack([jnp.cos(theta), jnp.sin(theta)])] * 2)
        opt = MpsEnergyOptimizer(state, terms={(0,): Z}, contraction_opt="greedy", route="envs")
        result = opt.measure({"x": {(0,): X}}, torch_batch=True)
        assert result["metadata"]["torch_batch"] == "loop"
        return jnp.real(result["energy"] + result["observables"]["x"])

    theta = jnp.asarray(0.2)
    assert float(value(theta)) == pytest.approx(np.cos(.4) + np.sin(.4), abs=1e-6)
    assert float(jax.grad(value)(theta)) == pytest.approx(2 * (np.cos(.4) - np.sin(.4)), abs=2e-6)


@pytest.mark.parametrize("options", [
    {"route": "unknown"}, {"torch_batch": "bad"}, {"chunk_size": 0},
])
def test_mps_measure_rejects_invalid_options(options):
    state = qtn.MPS_computational_state("0000")
    opt = MpsEnergyOptimizer(state, terms=TERMS, contraction_opt="greedy")
    with pytest.raises(ValueError):
        opt.measure(OBS, **options)
