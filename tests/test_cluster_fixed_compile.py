"""Compiler compatibility of exact fixed-index local MPO kernels."""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.operators]


def test_torch_fullgraph_fixed_kernel_values_and_gradients(monkeypatch):
    torch = pytest.importorskip("torch")
    from pepsy.operators import mpo_product

    def forbidden(*args, **kwargs):
        pytest.fail("fixed kernel must not invoke SVD")

    monkeypatch.setattr(mpo_product, "_fixed_rank_svd", forbidden)

    def kernel(matrix):
        cores = mpo_product._operator_schmidt(matrix, 2, 2, 0.0, factorization="fixed")
        return torch.einsum("aij,akl->ikjl", cores[0][0], cores[1][:, 0]).reshape(4, 4)

    # Resolve optional backend imports before entering a captured tensor graph.
    kernel(torch.zeros((4, 4), dtype=torch.float64))
    compiled = torch.compile(kernel, backend="aot_eager", fullgraph=True)
    for value in (0.0, 0.2):
        matrix = torch.full((4, 4), value, dtype=torch.float64, requires_grad=True)
        result = compiled(matrix)
        torch.testing.assert_close(result, matrix)
        (gradient,) = torch.autograd.grad(result.sum(), (matrix,))
        torch.testing.assert_close(gradient, torch.ones_like(matrix))


@pytest.mark.parametrize("step", [0.1, -0.1j])
@pytest.mark.parametrize("assembly", ["direct", "recursive"])
def test_fixed_bond_only_mpo_infers_backend_for_python_step(step, assembly):
    import numpy as np
    from pepsy.operators import MPOBasis, MPOParameter

    torch = pytest.importorskip("torch")
    z = np.diag([1.0, -1.0])
    basis = MPOBasis.from_local_terms(2, [((0, 1), (z, z), MPOParameter("j"))])
    compiled = basis.compile_graph_cluster_expansion(
        graph="chain", cluster_size=2, factorization="fixed", cutoff=0.0, assembly=assembly
    )
    for value in (0.0, 0.2):
        j = torch.tensor(value, dtype=torch.float64, requires_grad=True)
        actual = compiled(step, parameters={"j": j}).to_mpo().to_dense()
        expected = torch.matrix_exp(step * j * torch.as_tensor(np.kron(z, z)))
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        # Include imaginary values so the real-time derivative at j=0 is tested.
        loss = actual.real[0, 0] + (actual.imag[0, 0] if actual.is_complex() else 0.0)
        target = expected.real[0, 0] + (expected.imag[0, 0] if expected.is_complex() else 0.0)
        (a,) = torch.autograd.grad(loss, (j,), retain_graph=True)
        (b,) = torch.autograd.grad(target, (j,))
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
