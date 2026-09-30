"""Trace-only connected-cluster evaluation without MPO/PEPO construction."""

import numpy as np
import pytest
from scipy.linalg import expm

from pepsy.operators import (
    ClusterLattice, MPOClusterFactor, MPOClusterProductExpansion, MPOParameter,
    MPOProductTerm, PEPOClusterProductExpansion, PauliPEPOBasis, PauliPEPOTerm,
)

pytestmark = [pytest.mark.integration, pytest.mark.operators]
X = np.array([[0.0, 1.0], [1.0, 0.0]])
Z = np.diag([1.0, -1.0])
I = np.eye(2)
EDGES = ((0, 1), (0, 2), (1, 3), (2, 3))
SITES = ((0, 0), (0, 1), (1, 0), (1, 1))


def _square_products(order):
    mpo_factors = (
        MPOClusterFactor(
            [MPOProductTerm((site,), (X,), 0.2) for site in range(4)]
            + [MPOProductTerm(edge, (Z, Z), 0.3) for edge in EDGES]
        ),
        MPOClusterFactor(
            [MPOProductTerm((site,), (Z,), -0.13) for site in range(4)]
            + [MPOProductTerm(edge, (X, X), 0.17) for edge in EDGES]
        ),
    )
    pepo_factors = (
        PauliPEPOBasis.compile(
            2, 2,
            [PauliPEPOTerm("onsite", "X", 0.2, where=site) for site in SITES]
            + [PauliPEPOTerm("edge", "ZZ", 0.3,
                             where=tuple(SITES[i] for i in edge)) for edge in EDGES],
            order=order, factorization="fixed",
        ),
        PauliPEPOBasis.compile(
            2, 2,
            [PauliPEPOTerm("onsite", "Z", -0.13, where=site) for site in SITES]
            + [PauliPEPOTerm("edge", "XX", 0.17,
                             where=tuple(SITES[i] for i in edge)) for edge in EDGES],
            order=order, factorization="fixed",
        ),
    )
    mpo = MPOClusterProductExpansion(
        4, mpo_factors, graph=ClusterLattice(tuple(range(4)), EDGES),
        cluster_size=order, factorization="fixed", cutoff=0.,
        graph_assembly="exact", assembly="recursive",
    ).compile_exp()
    pepo = PEPOClusterProductExpansion.from_bases(pepo_factors).compile_exp()
    return mpo, pepo


@pytest.mark.parametrize("order", [2, 3, 4])
def test_joint_square_trace_matches_constructed_operators(order, monkeypatch):
    step = -0.041j
    mpo, pepo = _square_products(order)
    expected_mpo = np.trace(mpo.exp(step).to_mpo().to_dense())
    expected_pepo = np.trace(pepo.exp(step).to_dense())
    np.testing.assert_allclose(expected_mpo, expected_pepo, atol=2e-11)
    if order == 4:
        def pauli_product(operators):
            result = operators[0]
            for operator in operators[1:]:
                result = np.kron(result, operator)
            return result

        h_a = sum((.2 * pauli_product([X if i == site else I for i in range(4)])
                   for site in range(4)), start=np.zeros((16, 16)))
        h_a += sum((.3 * pauli_product([Z if i in edge else I for i in range(4)])
                    for edge in EDGES), start=np.zeros((16, 16)))
        h_b = sum((-.13 * pauli_product([Z if i == site else I for i in range(4)])
                   for site in range(4)), start=np.zeros((16, 16)))
        h_b += sum((.17 * pauli_product([X if i in edge else I for i in range(4)])
                    for edge in EDGES), start=np.zeros((16, 16)))
        exact_trace = np.trace(expm(step * h_a) @ expm(step * h_b))
        np.testing.assert_allclose(expected_mpo, exact_trace, atol=2e-11)

    # The trace path must not enter residual-matrix subtraction or PEPO assembly.
    monkeypatch.setattr(mpo.basis, "_graph_residuals",
                        lambda *_args: pytest.fail("MPO residual matrices built"))
    monkeypatch.setattr(pepo.expansion.factors[0].basis, "_build_inhomogeneous_active",
                        lambda *_args, **_kwargs: pytest.fail("PEPO blocks built"))
    for trace in (mpo.trace_exp, pepo.partition_trace_exp):
        np.testing.assert_allclose(trace(step), expected_mpo, atol=2e-11)
        np.testing.assert_allclose(
            trace(step, normalized=True), expected_mpo / 16, atol=2e-11,
        )
        np.testing.assert_allclose(trace(0), 16, atol=2e-11)




def test_single_basis_coefficient_overrides_are_fresh():
    compiled = PauliPEPOBasis.compile(
        1, 2, [("onsite", "X"), ("edge", "ZZ")],
        order=2, factorization="fixed",
    ).compile_exp()
    for coefficients in ((.2, .3), (-.1, .4)):
        expected = np.trace(compiled.exp(.07, coefficients=coefficients).to_pepo().to_dense())
        np.testing.assert_allclose(
            compiled.trace_exp(.07, coefficients=coefficients), expected, atol=1e-12,
        )


def test_periodic_square_partition_trace_keeps_parallel_bond_occurrences():
    basis = PauliPEPOBasis.compile(
        2, 2, [("onsite", "X", .2), ("edge", "ZZ", .3)],
        order=4, cyclic=True, symmetry="C4", factorization="fixed",
    )
    step = -.04j
    h = np.zeros((16, 16))
    for site in range(4):
        matrices = [X if i == site else I for i in range(4)]
        h += .2 * np.kron(np.kron(np.kron(*matrices[:2]), matrices[2]), matrices[3])
    for edge in EDGES:
        matrices = [Z if i in edge else I for i in range(4)]
        # Length-two periodic dimensions have two directed bonds per pair.
        h += .6 * np.kron(np.kron(np.kron(*matrices[:2]), matrices[2]), matrices[3])
    np.testing.assert_allclose(basis.compile_exp().partition_trace_exp(step),
                               np.trace(expm(step * h)), atol=2e-11)


def test_trace_budget_is_explicit_and_does_not_drop_collections():
    mpo, pepo = _square_products(3)
    for trace in (mpo.trace_exp, pepo.partition_trace_exp):
        with pytest.raises(ValueError, match="state_budget"):
            trace(.1, state_budget=1)
        np.testing.assert_allclose(trace(0), 16, atol=1e-12)



def test_trace_auto_mpo_aligns_host_identity_with_torch_parameter():
    torch = pytest.importorskip("torch")
    h = torch.tensor(.2, dtype=torch.float64, requires_grad=True)
    compiled = MPOClusterProductExpansion.from_local_terms(
        2, [MPOProductTerm((1,), (X,), MPOParameter("h"))],
        cluster_size=1,
    ).compile_exp()
    actual = compiled.trace_exp(.1, parameters={"h": h})
    reference = 2 * torch.trace(torch.matrix_exp(.1 * h * torch.tensor(X, dtype=h.dtype)))
    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(
        torch.autograd.grad(actual, h)[0],
        torch.autograd.grad(reference, h)[0],
    )


@pytest.mark.parametrize("representation", ["mpo", "pepo"])
def test_joint_two_site_trace_torch_gradients(representation):
    torch = pytest.importorskip("torch")
    h = torch.tensor(.23, dtype=torch.float64, requires_grad=True)
    step = torch.tensor(.11, dtype=torch.float64, requires_grad=True)
    if representation == "mpo":
        compiled = MPOClusterProductExpansion(
            2,
            (MPOClusterFactor([MPOProductTerm((i,), (X,), MPOParameter("h"))
                               for i in range(2)]),
             MPOClusterFactor([MPOProductTerm((0, 1), (Z, Z), .3)])),
            graph=ClusterLattice.from_edges(range(2), ((0, 1),)),
            cluster_size=2, factorization="fixed", cutoff=0.,
        ).compile_exp()
    else:
        compiled = PEPOClusterProductExpansion.from_bases((
            PauliPEPOBasis.compile(
                1, 2,
                [PauliPEPOTerm("onsite", "X", MPOParameter("h"), where=site)
                 for site in ((0, 0), (0, 1))],
                order=2, factorization="fixed",
            ),
            PauliPEPOBasis.compile(
                1, 2, [("edge", "ZZ", .3)], order=2, factorization="fixed",
            ),
        )).compile_exp()
    actual = compiled.trace_exp(step, parameters={"h": h})
    actual_grad = torch.autograd.grad(actual.real, (h, step))
    hx = torch.tensor(np.kron(X, I) + np.kron(I, X), dtype=h.dtype)
    hzz = torch.tensor(np.kron(Z, Z), dtype=h.dtype)
    expected = torch.trace(torch.matrix_exp(step * h * hx) @
                           torch.matrix_exp(step * .3 * hzz))
    expected_grad = torch.autograd.grad(expected.real, (h, step))
    torch.testing.assert_close(actual.real, expected.real)
    for got, want in zip(actual_grad, expected_grad):
        torch.testing.assert_close(got, want)


@pytest.mark.parametrize("representation", ["mpo", "pepo"])
def test_joint_two_site_trace_jax_jit_gradients(representation):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    if representation == "mpo":
        compiled = MPOClusterProductExpansion(
            2,
            (MPOClusterFactor([MPOProductTerm((i,), (X,), MPOParameter("h"))
                               for i in range(2)]),
             MPOClusterFactor([MPOProductTerm((0, 1), (Z, Z), .3)])),
            graph=ClusterLattice.from_edges(range(2), ((0, 1),)),
            cluster_size=2, factorization="fixed", cutoff=0.,
        ).compile_exp()
    else:
        compiled = PEPOClusterProductExpansion.from_bases((
            PauliPEPOBasis.compile(
                1, 2,
                [PauliPEPOTerm("onsite", "X", MPOParameter("h"), where=site)
                 for site in ((0, 0), (0, 1))],
                order=2, factorization="fixed",
            ),
            PauliPEPOBasis.compile(
                1, 2, [("edge", "ZZ", .3)], order=2, factorization="fixed",
            ),
        )).compile_exp()
    hx = jnp.asarray(np.kron(X, I) + np.kron(I, X))
    hzz = jnp.asarray(np.kron(Z, Z))
    traced = jax.jit(jax.value_and_grad(
        lambda h, step: jnp.real(compiled.trace_exp(
            step, parameters={"h": h})), argnums=(0, 1),
    ))
    reference = jax.value_and_grad(
        lambda h, step: jnp.real(jnp.trace(
            jax.scipy.linalg.expm(step * h * hx) @
            jax.scipy.linalg.expm(step * .3 * hzz)
        )), argnums=(0, 1),
    )
    for h, step in ((.0, .11), (.23, .11)):
        result, gradients = traced(jnp.asarray(h), jnp.asarray(step))
        expected, expected_gradients = reference(jnp.asarray(h), jnp.asarray(step))
        np.testing.assert_allclose(result, expected, rtol=2e-6)
        for got, want in zip(gradients, expected_gradients):
            np.testing.assert_allclose(got, want, rtol=2e-6, atol=2e-6)
