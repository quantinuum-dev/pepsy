"""Observable derivatives of isometric factors, including singular charts."""

import numpy as np
import pytest

from pepsy.backends.projector_split import projector_split

torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.optional, pytest.mark.integration]


def _loss(x, c):
    return (c.conj() * x).real.sum() + .1 * x.abs().square().sum()


@pytest.mark.parametrize("shape", [(6, 4), (4, 6), (5, 5)])
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
@pytest.mark.parametrize("absorb", ["left", "right"])
def test_projector_truncation_matches_independent_native_svd(shape, dtype, absorb):
    rng = torch.Generator().manual_seed(891)
    a = torch.randn(shape, dtype=dtype, generator=rng, requires_grad=True)
    c = torch.randn(shape, dtype=dtype, generator=rng)
    left, _, right = projector_split(a, max_bond=2, cutoff=0., absorb=absorb)
    actual = left @ right
    u, s, vh = torch.linalg.svd(a, full_matrices=False)
    expected = (u[:, :2] * s[:2]) @ vh[:2]
    torch.testing.assert_close(actual, expected, atol=2e-13, rtol=2e-13)
    got, = torch.autograd.grad(_loss(actual, c), a)
    want, = torch.autograd.grad(_loss(expected, c), a)
    torch.testing.assert_close(got, want, atol=3e-12, rtol=3e-12)


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
@pytest.mark.parametrize("absorb", ["left", "right"])
def test_rank_deficient_composition_preserves_constant_rank_tangents(dtype, absorb):
    rng = torch.Generator().manual_seed(36)
    l = torch.randn((7, 2), dtype=dtype, generator=rng, requires_grad=True)
    r = torch.randn((2, 5), dtype=dtype, generator=rng, requires_grad=True)
    c = torch.randn((7, 5), dtype=dtype, generator=rng)
    a = l @ r
    q, _, b = projector_split(a, cutoff=0., absorb=absorb)
    assert q.shape[1] == b.shape[0] == 2
    torch.testing.assert_close(q @ b, a, atol=2e-13, rtol=2e-13)
    got = torch.autograd.grad(_loss(q @ b, c), (l, r), retain_graph=True)
    want = torch.autograd.grad(_loss(a, c), (l, r))
    for actual, expected in zip(got, want):
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_repeated_retained_singular_values_have_a_smooth_projector(dtype):
    # Internal singular vectors are not unique; the retained subspace is.
    a = torch.diag(torch.tensor([3., 3., .7, 0.], dtype=dtype)).requires_grad_()

    def reconstruct(x):
        q, _, b = projector_split(x, max_bond=2, cutoff=0.)
        return q @ b

    assert torch.autograd.gradcheck(reconstruct, (a,), eps=1e-6, atol=2e-7, rtol=2e-6)


def test_closed_truncation_gap_rejects_an_undefined_backward():
    a = torch.eye(3, dtype=torch.float64, requires_grad=True)
    q, _, b = projector_split(a, max_bond=1, cutoff=0.)
    with pytest.raises(RuntimeError, match="kept/discarded"):
        (q @ b).square().sum().backward()


def test_noninvariant_singular_vector_observable_is_rejected():
    a = torch.diag(torch.tensor([3., 2., 1.], dtype=torch.float64)).requires_grad_()
    q, _, _ = projector_split(a, max_bond=2)
    with pytest.raises(RuntimeError, match="gauge-invariant"):
        q[0, 1].backward()


def test_zero_overlap_has_zero_gradient_but_nonzero_zero_matrix_cotangent_is_rejected():
    a = torch.zeros((4, 3), dtype=torch.complex128, requires_grad=True)
    q, _, b = projector_split(a, max_bond=2)
    (q @ b).abs().square().sum().backward()
    torch.testing.assert_close(a.grad, torch.zeros_like(a))
    q, _, b = projector_split(a, max_bond=2)
    with pytest.raises(RuntimeError, match="zero-matrix"):
        (q @ b).real.sum().backward()


@pytest.mark.parametrize("mode", ["abs", "rel", "sum2", "rsum2", "sum1", "rsum1"])
def test_cutoff_policy_and_numpy_torch_forward_agree(mode):
    import quimb.tensor.decomp as qd

    a = np.diag([4., 1., .2, .01])
    expected, _, _ = qd.array_split(a, method="svd", cutoff=.1, cutoff_mode=mode, absorb="right")
    qn, _, bn = projector_split(a, cutoff=.1, cutoff_mode=mode)
    qt, _, bt = projector_split(torch.tensor(a), cutoff=.1, cutoff_mode=mode)
    assert qn.shape[1] == qt.shape[1] == expected.shape[1]
    np.testing.assert_allclose(qn @ bn, (qt @ bt).numpy(), atol=1e-13)


def test_projector_honors_forward_policy_and_rejects_unsupported_absorption():
    from pepsy.backends import TorchLinalgConfig, get_torch_linalg_config

    a = torch.tensor([[2., .3], [.2, 1.]], dtype=torch.float64, requires_grad=True)
    with TorchLinalgConfig(stabilized=False).activated():
        before = get_torch_linalg_config()
        with TorchLinalgConfig(cpu_svd="scipy_gesvd", stabilized=True).activated():
            q, _, b = projector_split(a, cutoff=0.)
            got, = torch.autograd.grad((q @ b).square().sum(), a)
        assert get_torch_linalg_config() == before
    torch.testing.assert_close(got, 2*a)
    with pytest.raises(ValueError, match="absorb"):
        projector_split(a, absorb="both")
    with pytest.raises(ValueError, match="renorm"):
        projector_split(a, renorm=1)


def test_public_flat_projector_preserves_backend_exponent_inputs_and_dense_derivatives():
    import quimb.tensor as qtn

    from pepsy.boundary import contract_flat

    network = qtn.PEPS.rand(3, 3, 2, phys_dim=1, seed=45, dtype="complex128").squeeze()
    network.exponent = .25
    initial = [t.data.copy() for t in network]
    expected = contract_flat(network, method="exact", contraction_opt="greedy")
    value_np = contract_flat(network, method="mps", chi=16, cutoff=0.,
                             mps_factorization="projector", contraction_opt="greedy")
    np.testing.assert_allclose(value_np, expected, rtol=2e-12)
    for tensor, array in zip(network, initial):
        np.testing.assert_array_equal(tensor.data, array)
    network.apply_to_arrays(lambda x: torch.tensor(x, requires_grad=True))
    value = contract_flat(network, method="mps", chi=16, cutoff=0.,
                          mps_factorization="projector", contraction_opt="greedy", preserve_backend=True)
    reference = contract_flat(network, method="exact", contraction_opt="greedy", preserve_backend=True)
    got = torch.autograd.grad(value.real + .2*value.imag, network.arrays, retain_graph=True)
    want = torch.autograd.grad(reference.real + .2*reference.imag, network.arrays)
    for actual, target in zip(got, want):
        torch.testing.assert_close(actual, target, atol=2e-10, rtol=2e-10)
    with pytest.raises(ValueError, match="requires 2D"):
        contract_flat(network, method="exact", mps_factorization="projector")
    with pytest.raises(ValueError, match="direct boundary"):
        contract_flat(network, method="mps", chi=2, compression_mode="dm", mps_factorization="projector")
