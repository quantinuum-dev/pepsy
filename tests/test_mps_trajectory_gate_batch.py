"""Batched gate/SVD replay preserves each parent's rank and quantum state."""

import autoray as ar
import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.optimizers import MpsOptimizer, noise
from pepsy.optimizers.mps import _trajectory_batch as batch

pytestmark = [pytest.mark.optional, pytest.mark.integration]


@pytest.fixture(params=["torch", "cupy"])
def convert(request):
    if request.param == "torch":
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        return lambda x: torch.as_tensor(np.array(x), device="cuda")
    cp = pytest.importorskip("cupy")
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("CUDA unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA unavailable")
    return cp.asarray


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("cutoff_mode", ["abs", "rel", "sum1", "sum2", "rsum1", "rsum2"])
def test_individual_truncation_dense_reference_and_metadata(convert, dtype, cutoff_mode):
    rng = np.random.default_rng(817)
    gate = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0].astype(dtype)
    nodes = []
    refs = []
    for seed in range(4):
        p = qtn.MPS_rand_state(6, 4, dtype=dtype, seed=100 + seed)
        p.apply_to_arrays(convert)
        opt = MpsOptimizer(p, chi=3, mode="swap")
        nodes.append(noise._CoalescedNode(opt, 1))
        refs.append(opt.copy())
    entries = [(convert(gate), (2, 3))]
    kwargs = dict(cutoff=.1, cutoff_mode=cutoff_mode, stabilize_unitary=True)
    for ref in refs:
        noise._run_trajectory_entries(ref, entries, kwargs)
    assert batch.try_run_gate_batch(nodes, entries, kwargs,
        lambda node: noise._run_trajectory_entries(node.optimizer, entries, kwargs))
    tol = 8e-6 if dtype == "complex64" else 2e-12
    for node, ref in zip(nodes, refs):
        opt = node.optimizer
        assert opt.p.bond_sizes() == ref.p.bond_sizes()
        np.testing.assert_allclose(ar.to_numpy(opt.to_dense()), ar.to_numpy(ref.to_dense()),
                                   atol=tol, rtol=tol)
        assert opt.info_c["cur_orthog"] == ref.info_c["cur_orthog"]
        assert [t.tags for t in opt.p.tensors] == [t.tags for t in ref.p.tensors]
        assert opt.norm_diagnostics()["infidelity"] == pytest.approx(
            ref.norm_diagnostics()["infidelity"], abs=tol)
        assert opt._trajectory_diagnostics["max_gate_parent_batch"] == 4
        assert not hasattr(opt, "_trajectory_prepared_gate")
        left = opt.p[2].data
        if ar.infer_backend(left) == "torch":
            assert left.untyped_storage().nbytes() == left.numel() * left.element_size()
        else:
            assert left.base is None or left.base.nbytes == left.nbytes


def test_different_ranks_are_not_padded(convert):
    spectra = convert(np.array([[.99, .1, .01, .001], [.8, .5, .3, .1]]))
    assert batch._retained_ranks(spectra, .02, "rsum2", 4) == [1, 3]
    assert batch._retained_ranks(spectra, 0, "rsum2", 2) == [2, 2]
    nodes = []
    for angle in (.01, .7):
        state = qtn.MPS_product_state([np.array([np.cos(angle), np.sin(angle)],
                                                dtype="complex128"),
                                       np.array([1., 0.], dtype="complex128")])
        state.apply_to_arrays(convert)
        nodes.append(noise._CoalescedNode(MpsOptimizer(state, chi=2, mode="swap"), 1))
    entries = [("cnot", 0, 1)]
    kwargs = dict(cutoff=.05)
    assert batch.try_run_gate_batch(nodes, entries, kwargs,
        lambda node: noise._run_trajectory_entries(node.optimizer, entries, kwargs))
    assert [node.optimizer.p.bond_size(0, 1) for node in nodes] == [1, 2]


@pytest.mark.parametrize("option", ["mode", "normalize_every", "finite_check", "nonlocal"])
def test_unsupported_configuration_returns_before_state_updates(convert, option):
    p = qtn.MPS_rand_state(4, 2, dtype="complex128", seed=84)
    p.apply_to_arrays(convert)
    nodes = [noise._CoalescedNode(MpsOptimizer(p.copy(), chi=2, mode="swap"), 1)
             for _ in range(2)]
    entries = [("cnot", 0, 2 if option == "nonlocal" else 1)]
    kwargs = {} if option == "nonlocal" else {option: "direct" if option == "mode" else True}
    before = [[t.data for t in node.optimizer.p.tensors] for node in nodes]
    assert not batch.try_run_gate_batch(nodes, entries, kwargs,
                                        lambda _: pytest.fail("executed unsupported batch"))
    for node, arrays in zip(nodes, before):
        assert all(t.data is a for t, a in zip(node.optimizer.p.tensors, arrays))


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_complete_trajectory_replay_matches_serial(convert, dtype, monkeypatch):
    p = qtn.MPS_rand_state(8, 4, dtype=dtype, seed=203)
    p.apply_to_arrays(convert)
    stream = [("x_error", .4, 0), ("cnot", 2, 3),
              ("amplitude_damping", .2, 3), ("cnot", 3, 4)]
    opt = MpsOptimizer(p, stream, chi=4, mode="swap")
    kwargs = dict(shots=64, seed=7, strategy="coalesced", progress=False)
    actual = opt.run(**kwargs)
    assert actual.diagnostics.max_gate_parent_batch >= 2
    with monkeypatch.context() as patch:
        patch.setattr(batch, "try_run_gate_batch", lambda *args: False)
        expected = opt.run(**kwargs)
    assert actual.counts == expected.counts
    tol = 5e-6 if dtype == "complex64" else 1e-12
    for a, b in zip(actual.optimizers, expected.optimizers):
        np.testing.assert_allclose(ar.to_numpy(a.to_dense()), ar.to_numpy(b.to_dense()),
                                   atol=tol, rtol=tol)


@pytest.mark.parametrize("chi", [2, 4])
@pytest.mark.parametrize("stabilized", [False, True])
def test_torch_gradient_matches_unbatched_and_dense(chi, stabilized):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from pepsy.backends import TorchLinalgConfig, get_torch_linalg_config

    p = qtn.MPS_rand_state(4, 2, dtype="complex128", seed=76)
    p.apply_to_arrays(lambda x: torch.tensor(x, device="cuda"))
    theta = torch.tensor(.27, device="cuda", dtype=torch.float64, requires_grad=True)
    rng = np.random.default_rng(213)
    matrix = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    z = torch.tensor(matrix + matrix.conj().T, device="cuda")
    gate = torch.matrix_exp(-1j * theta * z)
    entries = [(gate, (1, 2))]
    nodes = [noise._CoalescedNode(MpsOptimizer(p.copy(), chi=chi, mode="swap"), 1)
             for _ in range(3)]
    kwargs = dict(cutoff=0.)
    previous = get_torch_linalg_config()
    with TorchLinalgConfig(stabilized=stabilized).activated():
        assert batch.try_run_gate_batch(nodes, entries, kwargs,
            lambda node: noise._run_trajectory_entries(node.optimizer, entries, kwargs))
        actual = nodes[0].optimizer.to_dense().ravel()
        vector = p.to_dense().reshape(2, 4, 2)
        expected = torch.einsum("ij,ajb->aib", gate, vector).reshape(-1)
        if chi == 2:
            ref = MpsOptimizer(p.copy(), chi=chi, mode="swap")
            noise._run_trajectory_entries(ref, entries, kwargs)
            expected = ref.to_dense().ravel()
        weights = torch.arange(16, device="cuda", dtype=torch.float64)
        loss = (weights * actual.abs()**2).sum()
        reference = (weights * expected.abs()**2).sum()
        grad = torch.autograd.grad(loss, theta, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(reference, theta)[0]
        assert abs(expected_grad.item()) > .01
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(grad, expected_grad, atol=1e-10, rtol=1e-9)
    assert get_torch_linalg_config() == (previous or TorchLinalgConfig())


@pytest.mark.slow
def test_larger_circuit_matches_unbatched_mps(convert, monkeypatch):
    rng = np.random.default_rng(923)
    p = qtn.MPS_rand_state(18, 12, dtype="complex128", seed=294)
    p.apply_to_arrays(convert)
    stream = []
    for step in range(24):
        site = (7 * step) % 17
        matrix = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        gate = np.linalg.qr(matrix)[0]
        stream.extend([("x_error", .04, site), (convert(gate), (site, site + 1))])
    sim = MpsOptimizer(p, stream, chi=12, mode="swap")
    kwargs = dict(shots=32, seed=120, strategy="coalesced", progress=False,
                  cutoff=1e-12, stabilize_unitary=True)
    actual = sim.run(**kwargs)
    assert actual.diagnostics.max_gate_parent_batch > 1
    with monkeypatch.context() as patch:
        patch.setattr(batch, "try_run_gate_batch", lambda *args: False)
        expected = sim.run(**kwargs)
    assert actual.counts == expected.counts
    for a, b in zip(actual.optimizers, expected.optimizers):
        overlap = complex(ar.to_numpy(a.p.H @ b.p))
        assert abs(overlap - 1.) < 2e-10
        assert a.p.bond_sizes() == b.p.bond_sizes()
        assert a.norm_diagnostics()["infidelity"] == pytest.approx(
            b.norm_diagnostics()["infidelity"], abs=2e-11)
