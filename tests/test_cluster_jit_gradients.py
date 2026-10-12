"""Gradient checks at the fixed cluster JIT/graph-capture boundary."""

import numpy as np
import pytest

from pepsy.operators import (
    ClusterLattice, MPOBasis, MPOClusterFactor, MPOClusterProductExpansion,
    MPOParameter, MPOProductTerm, PEPOClusterProductExpansion,
    PauliPEPOBasis, PauliPEPOTerm,
)

pytestmark = [pytest.mark.integration, pytest.mark.operators]

X = np.array([[0.0, 1.0], [1.0, 0.0]])
Z = np.diag([1.0, -1.0])
I = np.eye(2)


@pytest.mark.parametrize("representation", ["mpo", "pepo"])
def test_complete_fixed_three_site_jax_jit_values_and_gradients(representation):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    hx = jnp.asarray(sum(
        (np.kron(np.kron(X if i == 0 else I, X if i == 1 else I),
                    X if i == 2 else I) for i in range(3))
    ))
    hz = jnp.asarray(np.kron(np.kron(Z, Z), I) + np.kron(np.kron(I, Z), Z))
    if representation == "mpo":
        terms = [((i,), (X,), MPOParameter("h")) for i in range(3)]
        terms += [((i, i + 1), (Z, Z), 0.3) for i in range(2)]
        compiled = MPOBasis.from_local_terms(3, terms).compile_graph_cluster_expansion(
            graph="chain", cluster_size=3, factorization="fixed", cutoff=0.0,
            assembly="recursive",
        )

        def dense(h, step):
            return compiled(step, parameters={"h": h}).to_mpo().to_dense()
    else:
        compiled = PauliPEPOBasis(
            1, 3, [("onsite", "X", MPOParameter("h")), ("edge", "ZZ", 0.3)],
            order=3, factorization="fixed",
        ).compile_exp()

        def dense(h, step):
            return compiled(step, parameters={"h": h}).to_pepo().to_dense()

    def objective(operator):
        # The off-diagonal entry has a nonzero field derivative at h=0.
        return jnp.real(operator[0, 0] + operator[0, 1])

    compiled_value_grad = jax.jit(jax.value_and_grad(
        lambda h, step: objective(dense(h, step)), argnums=(0, 1)
    ))
    reference_value_grad = jax.value_and_grad(
        lambda h, step: objective(jax.scipy.linalg.expm(step * (h * hx + 0.3 * hz))),
        argnums=(0, 1),
    )
    for h_value, step_value in ((0.0, 0.1), (0.2, 0.0), (0.2, 0.1)):
        inputs = jnp.asarray(h_value), jnp.asarray(step_value)
        actual, derivatives = compiled_value_grad(*inputs)
        expected, expected_derivatives = reference_value_grad(*inputs)
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
        for derivative, expected_derivative in zip(derivatives, expected_derivatives):
            np.testing.assert_allclose(
                derivative, expected_derivative, atol=2e-6, rtol=2e-6
            )


@pytest.mark.parametrize("representation", ["mpo", "pepo"])
def test_joint_fixed_two_site_jax_jit_values_and_gradients(representation):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    hx = jnp.asarray(np.kron(X, I) + np.kron(I, X))
    hzz = jnp.asarray(np.kron(Z, Z))
    if representation == "mpo":
        factors = (
            MPOClusterFactor([
                MPOProductTerm((i,), (X,), MPOParameter("h")) for i in range(2)
            ]),
            MPOClusterFactor([MPOProductTerm((0, 1), (Z, Z), 0.3)]),
        )
        compiled = MPOClusterProductExpansion(
            2, factors, graph=ClusterLattice.from_edges(range(2), ((0, 1),)),
            cluster_size=2, factorization="fixed", cutoff=0.0,
            assembly="recursive",
        ).compile_exp()

        def dense(h, step):
            return compiled(step, parameters={"h": h}).to_mpo().to_dense()
    else:
        sites = ((0, 0), (0, 1))
        bases = (
            PauliPEPOBasis(
                1, 2, [
                    PauliPEPOTerm("onsite", "X", MPOParameter("h"), where=site)
                    for site in sites
                ], order=2, factorization="fixed",
            ),
            PauliPEPOBasis(
                1, 2, [
                    PauliPEPOTerm("edge", "ZZ", 0.3, where=sites)
                ], order=2, factorization="fixed",
            ),
        )
        compiled = PEPOClusterProductExpansion.from_bases(bases).compile_exp()

        def dense(h, step):
            return compiled(step, parameters={"h": h}).to_dense()

    def objective(matrix):
        return jnp.real(matrix[0, 0] + matrix[0, 1])

    value_grad = jax.jit(jax.value_and_grad(
        lambda h, step: objective(dense(h, step)), argnums=(0, 1)
    ))
    reference = jax.value_and_grad(
        lambda h, step: objective(
            jax.scipy.linalg.expm(step * h * hx)
            @ jax.scipy.linalg.expm(step * 0.3 * hzz)
        ), argnums=(0, 1),
    )
    for h_value, step_value in ((0.0, 0.1), (0.2, 0.0), (0.2, 0.1)):
        inputs = jnp.asarray(h_value), jnp.asarray(step_value)
        actual, derivatives = value_grad(*inputs)
        expected, expected_derivatives = reference(*inputs)
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
        for derivative, expected_derivative in zip(derivatives, expected_derivatives):
            np.testing.assert_allclose(
                derivative, expected_derivative, atol=2e-6, rtol=2e-6
            )


@pytest.mark.parametrize("exponential", ["mpo", "pepo"])
def test_torch_fullgraph_fixed_local_exponential_and_split_gradients(exponential):
    torch = pytest.importorskip("torch")
    from pepsy.operators import mpo_product, pepo_dense, pepo_product

    z = torch.diag(torch.tensor([1.0, -1.0], dtype=torch.float64))
    zz = torch.kron(z, z)
    exp = (mpo_product._matrix_exponential if exponential == "mpo"
           else pepo_dense._backend_expm)

    def kernel(h, step):
        h = pepo_product._as_backend_dtype(h, like=step)
        local = exp(h * step * zz)
        residual = local - mpo_product._identity(4, like=local)
        cores = mpo_product._operator_schmidt(
            residual, 2, 2, 0.0, factorization="fixed"
        )
        return torch.einsum(
            "aij,akl->ikjl", cores[0][0], cores[1][:, 0]
        ).reshape(4, 4)[0, 0]

    kernel(torch.tensor(0.2, dtype=torch.float64),
           torch.tensor(0.1, dtype=torch.float64))
    compiled = torch.compile(kernel, backend="aot_eager", fullgraph=True)
    for h_value, step_value in ((0.0, 0.1), (0.2, 0.0), (0.2, 0.1)):
        h = torch.tensor(h_value, dtype=torch.float64, requires_grad=True)
        step = torch.tensor(step_value, dtype=torch.float64, requires_grad=True)
        actual = compiled(h, step)
        derivatives = torch.autograd.grad(actual, (h, step))
        expected = torch.matrix_exp(h * step * zz)[0, 0] - 1
        expected_derivatives = torch.autograd.grad(expected, (h, step))
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        for derivative, expected_derivative in zip(derivatives, expected_derivatives):
            torch.testing.assert_close(
                derivative, expected_derivative, atol=1e-12, rtol=1e-12
            )

@pytest.mark.parametrize("representation", ["mpo", "pepo"])
def test_complete_fixed_two_site_jax_jit_real_time_gradient(representation):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    if representation == "mpo":
        compiled = MPOBasis.from_local_terms(
            2, [((0, 1), (Z, Z), MPOParameter("j"))]
        ).compile_graph_cluster_expansion(
            graph="chain", cluster_size=2, factorization="fixed", cutoff=0.0,
            assembly="recursive",
        )

        def dense(j, time):
            return compiled(-1j * time, parameters={"j": j}).to_mpo().to_dense()
    else:
        compiled = PauliPEPOBasis(
            1, 2, [("edge", "ZZ", MPOParameter("j"))],
            order=2, factorization="fixed",
        ).compile_exp()

        def dense(j, time):
            return compiled(-1j * time, parameters={"j": j}).to_pepo().to_dense()

    value_grad = jax.jit(jax.value_and_grad(
        lambda j, time: jnp.imag(dense(j, time)[0, 0]), argnums=(0, 1)
    ))
    for j_value in (0.0, 0.2):
        value, (dj, dt) = value_grad(jnp.asarray(j_value), jnp.asarray(0.1))
        np.testing.assert_allclose(value, -np.sin(0.1 * j_value), atol=1e-6)
        np.testing.assert_allclose(dj, -0.1 * np.cos(0.1 * j_value), atol=1e-6)
        np.testing.assert_allclose(dt, -j_value * np.cos(0.1 * j_value), atol=1e-6)


def test_torch_fullgraph_native_scalar_promotion():
    torch = pytest.importorskip("torch")
    from pepsy.operators.pepo_product import _as_backend_dtype

    def convert(value, like):
        return _as_backend_dtype(value, like=like)

    compiled = torch.compile(convert, backend="aot_eager", fullgraph=True)
    value = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    like = torch.tensor(0.1, dtype=torch.float32)
    result = compiled(value, like)
    assert result.dtype == torch.float64
    torch.testing.assert_close(result, value)
    (derivative,) = torch.autograd.grad(result, (value,))
    torch.testing.assert_close(derivative, torch.ones_like(value))


@pytest.mark.parametrize("representation", ["mpo", "pepo"])
def test_complete_fixed_square_order_four_jax_jit_gradient(representation):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    edges = ((0, 1), (0, 2), (1, 3), (2, 3))
    if representation == "mpo":
        compiled = MPOBasis.from_local_terms(
            4, [(edge, (Z, Z), MPOParameter("j")) for edge in edges]
        ).compile_graph_cluster_expansion(
            graph=ClusterLattice.from_edges(range(4), edges),
            cluster_size=4, factorization="fixed", cutoff=0.0,
            assembly="recursive",
        )

        def dense(j, step):
            return compiled(step, parameters={"j": j}).to_mpo().to_dense()
    else:
        compiled = PauliPEPOBasis(
            2, 2, [("edge", "ZZ", MPOParameter("j"))],
            order=4, factorization="fixed",
        ).compile_exp()

        def dense(j, step):
            return compiled(step, parameters={"j": j}).to_pepo().to_dense()

    def embed_edge(edge):
        result = np.ones((1, 1))
        for site in range(4):
            result = np.kron(result, Z if site in edge else I)
        return result

    hamiltonian = jnp.asarray(sum(map(embed_edge, edges), start=np.zeros((16, 16))))
    value_grad = jax.jit(jax.value_and_grad(
        lambda j, step: jnp.real(dense(j, step)[0, 0]), argnums=(0, 1)
    ))
    reference = jax.value_and_grad(
        lambda j, step: jnp.real(
            jax.scipy.linalg.expm(step * j * hamiltonian)[0, 0]
        ),
        argnums=(0, 1),
    )
    for j_value in (0.0, 0.2):
        inputs = jnp.asarray(j_value), jnp.asarray(0.1)
        actual, derivatives = value_grad(*inputs)
        expected, expected_derivatives = reference(*inputs)
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
        for derivative, expected_derivative in zip(derivatives, expected_derivatives):
            np.testing.assert_allclose(
                derivative, expected_derivative, atol=2e-6, rtol=2e-6
            )


def test_torch_fullgraph_fixed_pepo_tree_values_and_gradients():
    torch = pytest.importorskip("torch")
    from pepsy.operators import pepo_dense

    z = torch.diag(torch.tensor([1.0, -1.0], dtype=torch.float64))
    zz = torch.kron(z, z)

    def kernel(j, step):
        residual = pepo_dense._backend_expm(j * step * zz) - torch.eye(
            4, dtype=zz.dtype
        )
        tensors, _, _, _, _ = pepo_dense._tree_factorize_operator_backend(
            residual, ((0, 1, "r"),), 2, 2, factorization="fixed"
        )
        root = tensors[0][1]
        leaf = tensors[1][1]
        paired = torch.einsum("rp,qr->pq", root, leaf).reshape(2, 2, 2, 2)
        return paired.permute(0, 2, 1, 3).reshape(4, 4)

    kernel(torch.tensor(0.2, dtype=torch.float64),
           torch.tensor(0.1, dtype=torch.float64))
    compiled = torch.compile(kernel, backend="aot_eager", fullgraph=True)
    for j_value, step_value in ((0.0, 0.1), (0.2, 0.0), (0.2, 0.1)):
        j = torch.tensor(j_value, dtype=torch.float64, requires_grad=True)
        step = torch.tensor(step_value, dtype=torch.float64, requires_grad=True)
        actual = compiled(j, step)
        expected = torch.matrix_exp(j * step * zz) - torch.eye(4, dtype=zz.dtype)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        derivatives = torch.autograd.grad(actual[0, 0], (j, step))
        expected_derivatives = torch.autograd.grad(expected[0, 0], (j, step))
        for derivative, expected_derivative in zip(derivatives, expected_derivatives):
            torch.testing.assert_close(
                derivative, expected_derivative, atol=1e-12, rtol=1e-12
            )


def test_complete_fixed_located_square_jax_jit_reused_lower_gradients():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    sites = tuple((i, j) for i in range(2) for j in range(2))
    edges = ((0, 1), (0, 2), (1, 3), (2, 3))
    located = [
        PauliPEPOTerm("onsite", "X", MPOParameter("h"), where=site)
        for site in sites
    ]
    located += [
        PauliPEPOTerm("edge", "ZZ", 0.3, where=(sites[a], sites[b]))
        for a, b in edges
    ]
    compiled = PauliPEPOBasis(
        2, 2, located, order=4, factorization="fixed", spatial_reuse=True
    ).compile_exp()

    def embed(operators):
        result = np.ones((1, 1))
        for index in range(4):
            result = np.kron(result, operators.get(index, I))
        return result

    hx = jnp.asarray(sum((embed({i: X}) for i in range(4)), start=np.zeros((16, 16))))
    hz = jnp.asarray(sum(
        (embed({a: Z, b: Z}) for a, b in edges), start=np.zeros((16, 16))
    ))

    def objective(matrix):
        return jnp.real(matrix[0, 0] + matrix[0, 1])

    value_grad = jax.jit(jax.value_and_grad(
        lambda h, step: objective(
            compiled(step, parameters={"h": h}).to_pepo().to_dense()
        ),
        argnums=(0, 1),
    ))
    reference = jax.value_and_grad(
        lambda h, step: objective(jax.scipy.linalg.expm(step * (h * hx + 0.3 * hz))),
        argnums=(0, 1),
    )
    for h_value in (0.0, 0.2):
        inputs = jnp.asarray(h_value), jnp.asarray(0.1)
        actual, derivatives = value_grad(*inputs)
        expected, expected_derivatives = reference(*inputs)
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
        for derivative, expected_derivative in zip(derivatives, expected_derivatives):
            np.testing.assert_allclose(
                derivative, expected_derivative, atol=2e-6, rtol=2e-6
            )
