"""Compatibility and binding edge cases from the cluster API review."""

from dataclasses import fields, replace
import inspect

import numpy as np
import pytest
from scipy.linalg import expm

from pepsy.operators import (
    ClusterExpansionPlan, MPOBasis, MPOClusterProductExpansion, MPOParameter,
    PEPOClusterProductExpansion, PauliPEPOBasis,
)

pytestmark = [pytest.mark.integration, pytest.mark.operators]
X = np.array([[0., 1.], [1., 0.]])
Z = np.diag([1., -1.])


@pytest.mark.parametrize("cutoff", [{"order": 2}, {"cluster_size": 2}])
def test_dense_plan_dataclass_replace_preserves_legacy_order(cutoff):
    plan = ClusterExpansionPlan(1, 3, .3 * np.kron(Z, Z), .2 * X, **cutoff)
    refined = replace(plan, order=3)
    copied = replace(plan, lx=2)
    assert plan.order == plan.cluster_size == 2
    assert refined.order == refined.cluster_size == 3
    assert copied.order == copied.cluster_size == 2
    assert copied.lx == 2
    # Only the canonical field is stored, so replace cannot copy a stale alias.
    assert "cluster_size" not in {field.name for field in fields(plan)}
    signature = inspect.signature(ClusterExpansionPlan)
    assert signature.parameters["order"].default == 3
    assert signature.parameters["cluster_size"].kind is inspect.Parameter.KEYWORD_ONLY
    identity = np.eye(2)
    h = .2 * (np.kron(np.kron(X, identity), identity)
              + np.kron(np.kron(identity, X), identity)
              + np.kron(np.kron(identity, identity), X))
    h += .3 * (np.kron(np.kron(Z, Z), identity) + np.kron(np.kron(identity, Z), Z))
    actual = refined.exp(-.03j, materialize=True).to_dense()
    np.testing.assert_allclose(actual, expm(-.03j * h), atol=2e-12)


def _product(kind, coefficients=(.2, .3)):
    if kind == "mpo":
        bases = [MPOBasis.from_terms([((0,), (op,), coefficient)])
                 for op, coefficient in zip((X, Z), coefficients)]
        return MPOClusterProductExpansion.from_bases(
            bases, cluster_size=1, factorization="fixed", cutoff=0.,
        )
    bases = [PauliPEPOBasis(1, 1, [("onsite", word, coefficient)], cluster_size=1)
             for word, coefficient in zip("XZ", coefficients)]
    return PEPOClusterProductExpansion.from_bases(bases)


@pytest.mark.parametrize("kind", ["mpo", "pepo"])
@pytest.mark.parametrize("method", ["exp", "trace_exp"])
def test_none_factor_vector_keeps_default_terms(kind, method):
    product = _product(kind, (MPOParameter("h", default=.2), .3)).compile_exp()
    expected = expm(.1 * .2 * X) @ expm(.1 * .7 * Z)
    result = getattr(product, method)(.1, coefficients=[None, [.7]])
    if method == "exp":
        if hasattr(result, "to_mpo"):
            result = result.to_mpo()
        np.testing.assert_allclose(result.to_dense(), expected, atol=2e-12)
    else:
        np.testing.assert_allclose(result, np.trace(expected), atol=2e-12)


@pytest.mark.parametrize("kind", ["mpo", "pepo"])
@pytest.mark.parametrize("method", ["exp", "trace_exp"])
def test_binding_container_conflict_is_rejected_before_callbacks(kind, method):
    calls = []

    def coefficient(parameters):
        calls.append(parameters)
        return .2

    product = _product(kind, (coefficient, .3)).compile_exp()
    with pytest.raises(ValueError, match="(mutually exclusive|either parameters)"):
        getattr(product, method)(.1, {}, coefficients=[None, None])
    assert calls == []


def test_mpo_none_factor_vector_still_requires_missing_parameter():
    product = _product("mpo", (MPOParameter("h"), .3)).compile_exp()
    with pytest.raises(ValueError, match="parameters are required"):
        product.trace_exp(.1, coefficients=[None, [.7]])


def test_mpo_none_factor_vector_still_requires_callback_parameters():
    product = _product("mpo", (lambda params: params["h"], .3)).compile_exp()
    with pytest.raises(KeyError, match="require parameters"):
        product.trace_exp(.1, coefficients=[None, [.7]])


def test_legacy_graph_runtime_vectors_keep_bridged_term_supports():
    from pepsy.operators import ClusterLattice, MPOClusterFactor, MPOProductTerm

    product = MPOClusterProductExpansion(
        3,
        [MPOClusterFactor([MPOProductTerm((0, 2), (X, Z), .3)])],
        graph=ClusterLattice(3, [(0, 1), (1, 2)]),
        cluster_size=3,
        cutoff=0,
        graph_assembly="exact",
    )
    for coefficient in (.3, -.7):
        expected = expm(.1 * coefficient * np.kron(np.kron(X, np.eye(2)), Z))
        actual = product.exp(.1, coefficients=[coefficient], materialize=True).to_dense()
        np.testing.assert_allclose(actual, expected, atol=2e-12)
        np.testing.assert_allclose(
            product.trace_exp(.1, coefficients=[coefficient]), np.trace(expected), atol=2e-12
        )
    assert product.cache_info["coefficient_topology_compiled"]
