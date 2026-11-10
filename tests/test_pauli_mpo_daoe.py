"""DAOE filtering against explicit coefficient-space reference calculations."""

import itertools
import math

import numpy as np
import pytest

from pepsy.operators import PauliMPO


def _coefficients(cores, einsum=np.einsum):
    value = cores[0][0].T
    for core in cores[1:]:
        value = einsum("...a,abp->...pb", value, core)
    return value[..., 0]


def _filter_weights(nsites, max_weight, gamma):
    weights = np.empty((4,) * nsites)
    for labels in itertools.product(range(4), repeat=nsites):
        weight = sum(label != 0 for label in labels)
        weights[labels] = math.exp(-gamma * max(weight - max_weight, 0))
    return weights


@pytest.mark.parametrize("nsites", [1, 2, 4])
@pytest.mark.parametrize("max_weight", [0, 1, 2, 4])
def test_daoe_sparse_and_native_match_every_pauli_coefficient(nsites, max_weight):
    rng = np.random.default_rng(91 + nsites)
    terms = [
        (rng.normal() + 1j * rng.normal(), "".join(labels))
        for labels in itertools.product("IXYZ", repeat=nsites)
    ]
    sparse = PauliMPO(nsites, terms, boundary="periodic")
    # Dense coefficient tensors, split exactly into cores, exercise arbitrary
    # bond superpositions rather than one disjoint path per Pauli word.
    native = sparse.compress_pauli(cutoff=0.0)
    before = _coefficients(native.to_pauli_cores())
    expected = before * _filter_weights(nsites, max_weight, 0.37)
    for operator in (sparse, native):
        result = operator.apply_daoe(max_weight=max_weight, gamma=0.37)
        np.testing.assert_allclose(_coefficients(result.to_pauli_cores()), expected, atol=2e-13)
        assert result.boundary == "periodic"
        assert result.trace(normalized=True) == pytest.approx(sparse.trace(normalized=True))
    np.testing.assert_array_equal(_coefficients(native.to_pauli_cores()), before)


def test_daoe_counts_bodyness_not_spatial_diameter():
    operator = PauliMPO(5, [(0.7, "XIIIZ"), (0.9, "XYZII")])
    result = dict((word, coefficient) for coefficient, word in operator.apply_daoe(
        max_weight=2, gamma=0.4,
    ).terms)
    assert result["XIIIZ"] == pytest.approx(0.7)
    assert result["XYZII"] == pytest.approx(0.9 * math.exp(-0.4))


def test_daoe_large_native_operator_never_materializes_words(monkeypatch):
    nsites = 64
    core = np.array([[[1.0, 1.0, 0.0, 0.0]]])
    operator = PauliMPO.from_pauli_cores([core] * nsites)

    def fail(*args, **kwargs):
        raise AssertionError("DAOE must not enumerate exponentially many words")

    monkeypatch.setattr(PauliMPO, "_materialize_core_terms", fail)
    result = operator.apply_daoe(max_weight=2, gamma=0.13)
    assert max(result.pauli_bond_dimensions) <= 3
    assert result.trace(normalized=True) == pytest.approx(1.0)
    query = PauliMPO(nsites, [(1.0, "X" * nsites)])
    assert query.inner(result, normalized=True) == pytest.approx(math.exp(-0.13 * 62))


def test_daoe_composition_noops_and_inplace_ownership():
    operator = PauliMPO(3, [(0.3, "III"), (0.7, "XYZ")]).compress_pauli(cutoff=0.0)
    original = _coefficients(operator.to_pauli_cores())
    twice = operator.apply_daoe(max_weight=1, gamma=0.2).apply_daoe(max_weight=1, gamma=0.3)
    once = operator.apply_daoe(max_weight=1, gamma=0.5)
    np.testing.assert_allclose(_coefficients(twice.to_pauli_cores()),
                               _coefficients(once.to_pauli_cores()), atol=1e-14)
    assert once.compression_report is None
    for max_weight, gamma in ((0, 0.0), (3, 0.5), (100, 0.5)):
        copy = operator.apply_daoe(max_weight=max_weight, gamma=gamma)
        assert copy is not operator
        assert not np.shares_memory(copy.to_pauli_cores(copy=False)[0],
                                    operator.to_pauli_cores(copy=False)[0])
        np.testing.assert_array_equal(_coefficients(copy.to_pauli_cores()), original)
    assert operator.apply_daoe(max_weight=1, gamma=0.5, inplace=True) is operator
    np.testing.assert_allclose(_coefficients(operator.to_pauli_cores()),
                               _coefficients(once.to_pauli_cores()), atol=1e-14)
    assert operator.compression_report is None


def test_daoe_preserves_integer_input_content_with_fractional_damping():
    operator = PauliMPO.from_pauli_cores([np.array([[[1, 2, 3, 4]]])])
    result = operator.apply_daoe(max_weight=0, gamma=0.7)
    assert result.to_pauli_cores()[0].dtype.kind == "f"
    np.testing.assert_allclose(result.to_pauli_cores()[0],
                               [[[1, 2 * np.exp(-0.7), 3 * np.exp(-0.7), 4 * np.exp(-0.7)]]])


@pytest.mark.parametrize("dtype_name", ["float32", "float64", "complex64", "complex128"])
def test_daoe_torch_gradients_match_explicit_coefficient_filter(dtype_name):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(31)
    cores = [
        torch.tensor(rng.normal(size=shape), dtype=getattr(torch, dtype_name), requires_grad=True)
        for shape in ((1, 2, 4), (2, 3, 4), (3, 1, 4))
    ]
    operator = PauliMPO.from_pauli_cores(cores)
    filtered = operator.apply_daoe(max_weight=1, gamma=0.6)
    actual = filtered.inner(filtered, normalized=True).real
    explicit = _coefficients(cores, torch.einsum)
    weights = torch.tensor(_filter_weights(3, 1, 0.6), dtype=cores[0].dtype)
    expected = (explicit * weights).abs().square().sum()
    tolerance = 2e-5 if dtype_name in {"float32", "complex64"} else 1e-11
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    actual_grads = torch.autograd.grad(actual, cores)
    expected_grads = torch.autograd.grad(expected, cores)
    for core, actual_grad, expected_grad in zip(cores, actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=tolerance, atol=tolerance)
        assert core.dtype == filtered.to_pauli_cores()[0].dtype
        assert core.device == filtered.to_pauli_cores()[0].device


def test_daoe_jax_jit_and_gradient_match_explicit_filter():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    rng = np.random.default_rng(32)
    cores = tuple(jnp.asarray(rng.normal(size=shape), dtype=jnp.float32)
                  for shape in ((1, 2, 4), (2, 3, 4), (3, 1, 4)))
    weights = jnp.asarray(_filter_weights(3, 1, 0.6), dtype=jnp.float32)

    def actual(arrays):
        operator = PauliMPO.from_pauli_cores(arrays)
        filtered = operator.apply_daoe(max_weight=1, gamma=0.6)
        return filtered.inner(filtered, normalized=True).real

    def expected(arrays):
        return jnp.sum(jnp.abs(_coefficients(arrays, jnp.einsum) * weights)**2)

    value, grads = jax.jit(jax.value_and_grad(actual))(cores)
    reference, reference_grads = jax.value_and_grad(expected)(cores)
    np.testing.assert_allclose(value, reference, rtol=2e-5)
    for grad, reference_grad in zip(grads, reference_grads):
        np.testing.assert_allclose(grad, reference_grad, rtol=2e-5, atol=2e-5)


def test_daoe_one_period_matches_dense_heisenberg_evolution_and_filter():
    scipy = pytest.importorskip("scipy.linalg")
    hamiltonian = PauliMPO(2, [(0.8, "ZZ"), (0.4, "XI"), (0.3, "IX")])
    initial = PauliMPO(2, [(1.0, "ZI")])
    unitary = scipy.expm(-0.17j * hamiltonian.to_dense())
    expected = PauliMPO.from_dense(unitary.conj().T @ initial.to_dense() @ unitary)
    expected_coefficients = _coefficients(expected.to_pauli_cores()) * _filter_weights(2, 1, 0.2)
    evolved = PauliMPO.from_pauli_cores(initial.to_pauli_cores()).apply_gate(
        unitary, (0, 1), mode="heisenberg",
    )
    filtered = evolved.apply_daoe(max_weight=1, gamma=0.2)
    compressed, report = filtered.compress_pauli(max_bond=4, cutoff=1e-14, return_report=True)
    np.testing.assert_allclose(_coefficients(compressed.to_pauli_cores()),
                               expected_coefficients, atol=1e-12)
    assert report.discarded_singular_weight < 1e-12


@pytest.mark.parametrize("max_weight", [-1, 1.5, True])
def test_daoe_rejects_invalid_weight(max_weight):
    with pytest.raises(ValueError, match="max_weight"):
        PauliMPO.identity(2).apply_daoe(max_weight=max_weight, gamma=0.1)


@pytest.mark.parametrize("gamma", [-0.1, np.nan, np.inf, -np.inf])
def test_daoe_rejects_invalid_damping(gamma):
    with pytest.raises(ValueError, match="gamma"):
        PauliMPO.identity(2).apply_daoe(max_weight=1, gamma=gamma)


@pytest.mark.parametrize("gamma", [True, 1j, "0.1", None])
def test_daoe_rejects_nonreal_parameters(gamma):
    with pytest.raises(TypeError, match="gamma"):
        PauliMPO.identity(2).apply_daoe(max_weight=1, gamma=gamma)
