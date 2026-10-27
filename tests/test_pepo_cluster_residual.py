"""Ordered cluster PEPO residuals for trace-insertion measurements."""

import numpy as np
import pytest
from scipy.linalg import expm

from pepsy.operators import MPOParameter, PauliPEPOBasis, PEPOClusterProductExpansion


pytestmark = [pytest.mark.integration, pytest.mark.operators]


def _local_operator(operators):
    eye = np.eye(2)
    result = np.ones((1, 1))
    for site in range(4):
        result = np.kron(result, operators.get(site, eye))
    return result


@pytest.mark.parametrize("cyclic", [False, True])
def test_ordered_residual_matches_dense_open_and_periodic(cyclic):
    x = np.array([[0., 1.], [1., 0.]])
    z = np.diag([1., -1.])
    hx = .21 * sum(_local_operator({site: x}) for site in range(4))
    edges = ((0, 1), (2, 3), (0, 2), (1, 3))
    # A 2x2 periodic lattice has two physical bond occurrences per pair.
    hz = (2 if cyclic else 1) * .17 * sum(
        _local_operator({first: z, second: z}) for first, second in edges
    )
    first = PauliPEPOBasis.compile(
        2, 2, [("onsite", "X", .21)], order=4, cyclic=cyclic,
    )
    second = PauliPEPOBasis.compile(
        2, 2, [("edge", "ZZ", .17)], order=4, cyclic=cyclic,
    )
    compiled = PEPOClusterProductExpansion.from_bases((first, second)).compile_exp()
    step = -.08j
    actual = compiled.residual_exp(step)
    expected = expm(step * hx) @ expm(step * hz) - np.eye(16)
    np.testing.assert_allclose(actual.to_dense(), expected, atol=2e-13)
    np.testing.assert_allclose(
        actual.trace(actual.upper_inds, actual.lower_inds),
        np.trace(expected), atol=2e-13,
    )


@pytest.mark.parametrize("cyclic", [False, True])
def test_four_factor_defect_uses_one_joint_cluster_product(cyclic):
    x = np.array([[0., 1.], [1., 0.]])
    z = np.diag([1., -1.])
    hx = sum(_local_operator({site: x}) for site in range(4))
    edges = ((0, 1), (2, 3), (0, 2), (1, 3))
    hz = (2 if cyclic else 1) * sum(
        _local_operator({first: z, second: z}) for first, second in edges
    )
    coefficients = (.11, .07, -.09, -.05)
    terms = (("onsite", "X"), ("edge", "ZZ"),
             ("onsite", "Z"), ("onsite", "X"))
    bases = tuple(PauliPEPOBasis.compile(
        2, 2, [(support, word, coefficient)], order=4, cyclic=cyclic,
    ) for (support, word), coefficient in zip(terms, coefficients))
    step = .1j
    actual = PEPOClusterProductExpansion.from_bases(bases).residual_exp(step)
    hz_onsite = sum(_local_operator({site: z}) for site in range(4))
    expected = np.eye(16, dtype=complex)
    for generator in (.11 * hx, .07 * hz, -.09 * hz_onsite, -.05 * hx):
        expected = expected @ expm(step * generator)
    np.testing.assert_allclose(actual.to_dense(), expected - np.eye(16), atol=2e-13)


@pytest.mark.parametrize("cyclic", [False, True])
def test_residual_chi_caps_operator_after_identity_subtraction(cyclic):
    basis = PauliPEPOBasis.compile(
        2, 2, [("onsite", "X", .2), ("edge", "ZZ", .3)],
        order=2, cyclic=cyclic,
    )
    product = PEPOClusterProductExpansion.from_bases((basis, basis),
                                                      coefficients=(.7, -.2))
    uncapped = product.residual_exp(.1j)
    capped = product.residual_exp(.1j, chi=2)
    assert uncapped.max_bond() > 2
    assert capped.max_bond() <= 2
    assert np.isfinite(capped.to_dense()).all()
    with pytest.raises(ValueError, match="chi"):
        product.residual_exp(.1j, chi=0)


def test_residual_preserves_torch_parameter_gradient():
    torch = pytest.importorskip("torch")
    basis = PauliPEPOBasis.compile(
        1, 2, [("edge", "ZZ", MPOParameter("h"))], order=2,
    )
    compiled = PEPOClusterProductExpansion.from_bases((basis,)).compile_exp()
    strength = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
    residual = compiled.residual_exp(.11j, {"h": strength})
    value = residual.to_dense()[0, 0].real
    gradient, = torch.autograd.grad(value, strength)
    expected = (complex(.11j * np.exp(.11j * .3))).real
    np.testing.assert_allclose(gradient.item(), expected, atol=1e-12)
