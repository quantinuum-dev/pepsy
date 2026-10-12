"""Paired MPO replay derivatives against dense operators and finite differences."""

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.optimizers import MpoOptimizer

torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.optional, pytest.mark.integration]


def embed(gate, where):
    bits = [[(i >> (2 - s)) & 1 for s in range(3)] for i in range(8)]
    return torch.stack([torch.stack([
        gate[2 * row[where[0]] + row[where[1]], 2 * col[where[0]] + col[where[1]]]
        if all(row[s] == col[s] for s in range(3) if s not in where)
        else gate.new_zeros(()) for col in bits]) for row in bits])


@pytest.mark.parametrize("cap", [2, 4])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_layer_replay_preserves_dense_values_and_gradients(cap, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    rng = np.random.default_rng(537)
    convert = lambda x: torch.tensor(x, dtype=torch.complex128, device=device)
    dense = convert(rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8)))
    p = qtn.MatrixProductOperator.from_dense(dense, dims=[2] * 3, cutoff=0.)
    original = p.to_dense().detach().clone()
    h = convert(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    h = h + h.mH
    weight = convert(rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8)))
    x = torch.tensor(.21, dtype=torch.float64, device=device, requires_grad=True)

    def evaluate(t):
        g = torch.matrix_exp(-1j * t * h)
        opt = MpoOptimizer(p, [((g, None), (0, 1)), ((None, g), (1, 2)),
                               ((g, None), (0, 2))], chi=cap, mode="direct")
        result = opt.run(cutoff=0., mpo_factorization="projector", progbar=False)
        assert result.max_bond() <= cap
        value = (weight.conj() * result.to_dense()).real.sum()
        return value, result.to_dense(), g

    value, actual, g = evaluate(x)
    grad, = torch.autograd.grad(value, x, retain_graph=True)
    if cap == 4:
        expected = embed(g, (0, 2)).T @ embed(g, (0, 1)).T @ dense @ embed(g, (1, 2)).conj()
        torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-11)
        exact, = torch.autograd.grad((weight.conj() * expected).real.sum(), x)
        torch.testing.assert_close(grad, exact, atol=2e-9, rtol=2e-9)
    for eps in (1e-5, 1e-6):
        fd = (evaluate(x.detach() + eps)[0] - evaluate(x.detach() - eps)[0]) / (2 * eps)
        torch.testing.assert_close(grad, fd, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(p.to_dense(), original, atol=0, rtol=0)
    assert x.grad is None


def test_invalid_factorization_rejected_before_replay():
    p = qtn.MPO_identity(3)
    gate = np.eye(4)
    opt = MpoOptimizer(p, [(gate, (0, 1))], mode="direct", chi=4)
    before = opt.p.to_dense().copy()
    with pytest.raises(ValueError, match="mpo_factorization"):
        opt.run(mpo_factorization="unknown")
    with pytest.raises(NotImplementedError, match="direct"):
        opt.run(mpo_factorization="projector", submpo_method="dm")
    with pytest.raises(NotImplementedError, match="layout"):
        opt.run(mpo_factorization="projector", layout=True)
    np.testing.assert_array_equal(opt.p.to_dense(), before)


def test_bare_gate_uses_paired_layer_route(monkeypatch):
    from pepsy.operators.gates import gate_nonlocal_opt
    import pepsy.optimizers.mpo.optimizer as module

    calls = []
    def observe(*args, **kwargs):
        calls.append(kwargs["mpo_factorization"])
        return gate_nonlocal_opt(*args, **kwargs)
    monkeypatch.setattr(module, "gate_nonlocal_opt", observe)
    opt = MpoOptimizer(qtn.MPO_identity(2), [(np.eye(4), (0, 1))], chi=4)
    result = opt.run(cutoff=0., mpo_factorization="projector")
    assert calls == ["projector", "projector"]
    np.testing.assert_allclose(result.to_dense(), np.eye(4), atol=1e-13)


def test_chart_error_capture_is_explicit_scoped_and_nonfinite():
    from pepsy.backends import ProjectorDerivativeError, capture_projector_derivative_errors
    from pepsy.backends.projector_split import projector_split

    def fail():
        x = torch.eye(3, dtype=torch.float64, requires_grad=True)
        q, _, b = projector_split(x, max_bond=1, cutoff=0.)
        return torch.autograd.grad((q @ b).square().sum(), x)[0]
    before = torch.autograd.is_multithreading_enabled()
    with capture_projector_derivative_errors() as errors:
        assert torch.isnan(fail()).all()
    assert len(errors) == 1 and 'kept/discarded' in errors[0]
    assert torch.autograd.is_multithreading_enabled() == before
    with pytest.raises(ProjectorDerivativeError, match='kept/discarded'):
        fail()
