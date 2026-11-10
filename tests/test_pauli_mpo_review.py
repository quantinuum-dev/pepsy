"""Independent algebra and backend regressions for native Pauli cores."""

import numpy as np
import pytest

from pepsy.operators import PauliMPO


def _coefficient_tensor(operator):
    cores = operator.to_pauli_cores(copy=False)
    value = cores[0][0].T
    for core in cores[1:]:
        value = np.einsum("...a,abp->...pb", value, core)
    return value[..., 0]


@pytest.mark.parametrize("nsites", [1, 2, 3, 4])
def test_native_subtraction_and_commutator_match_sparse_algebra(nsites):
    x = PauliMPO(nsites, [(1.0, "X" + "I" * (nsites - 1))])
    y = PauliMPO(nsites, [(0.7, "Y" + "I" * (nsites - 1))])
    native_x = PauliMPO.from_pauli_cores(x.to_pauli_cores())
    native_y = PauliMPO.from_pauli_cores(y.to_pauli_cores())
    np.testing.assert_allclose((native_x - native_y).to_dense(), (x - y).to_dense())
    np.testing.assert_allclose(
        native_x.commutator(native_y).to_dense(), x.commutator(y).to_dense(),
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_real_cores_convert_to_qubit_mpo_at_matching_precision(dtype):
    core = np.array([[[0.3, 0.4, 0.5, 0.6]]], dtype=dtype)
    operator = PauliMPO.from_pauli_cores([core])
    expected = np.array([[0.9, 0.4 - 0.5j], [0.4 + 0.5j, -0.3]])
    dense = operator.to_dense()
    assert dense.dtype == np.result_type(dtype, np.complex64)
    np.testing.assert_allclose(dense, expected, atol=1e-7)


@pytest.mark.parametrize("dtype_name", ["float32", "float64", "complex64", "complex128"])
def test_native_torch_readout_and_products_preserve_dtype_and_gradient(dtype_name):
    torch = pytest.importorskip("torch")
    dtype = getattr(torch, dtype_name)
    x = torch.tensor([[[0.0, 0.7, 0.0, 0.0]]], dtype=dtype, requires_grad=True)
    identity = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=dtype)
    operator = PauliMPO.from_pauli_cores([x, identity])
    assert operator.trace().dtype == dtype
    norm_squared = operator.inner(operator, normalized=True)
    assert norm_squared.dtype == dtype
    np.testing.assert_allclose(norm_squared.detach().cpu(), 0.49, atol=1e-7)
    (gradient,) = torch.autograd.grad(norm_squared.real, x)
    np.testing.assert_allclose(gradient.detach().cpu()[0, 0, 1], 1.4, atol=1e-7)
    product = operator @ operator
    expected_complex = (
        torch.complex128 if dtype_name in {"float64", "complex128"} else torch.complex64
    )
    assert product.to_pauli_cores()[0].dtype == expected_complex
    assert product.to_mpo().arrays[0].device == x.device
    np.testing.assert_allclose(
        product.to_dense().detach().cpu(), 0.49 * np.eye(4), atol=1e-7,
    )


@pytest.mark.parametrize("method", ["rsvd", "svds", "isvd"])
@pytest.mark.parametrize("form", ["right", "left", 1])
def test_partial_svd_reports_the_actual_discarded_operator_norm(method, form):
    rng = np.random.default_rng(21)
    operator = PauliMPO.from_pauli_cores([
        rng.normal(size=(1, 4, 4)).astype(complex),
        rng.normal(size=(4, 1, 4)).astype(complex),
    ])
    compressed, report = operator.compress_pauli(
        max_bond=1, method=method, oversample=0, n_iter=0,
        seed=7, cutoff=0.0, form=form, return_report=True,
    )
    error = 2 * np.linalg.norm(_coefficient_tensor(operator) - _coefficient_tensor(compressed))
    assert not report.exact
    assert report.discarded_singular_weight == pytest.approx(error)
    assert report.bond_reports[0].discarded_weight == pytest.approx(error)
    assert report.discarded_ranks == (3,)


def test_native_jax_dense_readout_preserves_jit_and_gradient():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    # Both Y coefficients and entangled virtual paths contribute to readout.
    rng = np.random.default_rng(92)
    cores = tuple(jnp.asarray(rng.normal(size=shape), dtype=jnp.float32)
                  for shape in ((1, 2, 4), (2, 1, 4)))
    paulis = np.array([np.eye(2), [[0, 1], [1, 0]],
                       [[0, -1j], [1j, 0]], np.diag([1, -1])])
    basis = jnp.asarray([[np.kron(p, q) for q in paulis] for p in paulis])

    def dense(arrays):
        return PauliMPO.from_pauli_cores(arrays).to_dense()

    def expected(arrays):
        coefficients = jnp.einsum("ap,aq->pq", arrays[0][0], arrays[1][:, 0])
        return jnp.einsum("pq,pqij->ij", coefficients, basis)

    np.testing.assert_allclose(jax.jit(dense)(cores), expected(cores), atol=1e-6)
    def norm(value):
        return jnp.sum(jnp.abs(value)**2)

    actual_grad = jax.jit(jax.grad(lambda arrays: norm(dense(arrays))))(cores)
    expected_grad = jax.grad(lambda arrays: norm(expected(arrays)))(cores)
    for actual, reference in zip(actual_grad, expected_grad):
        np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-5)


def test_real_native_gate_and_inplace_channel_stay_native(monkeypatch):
    from scipy.linalg import expm

    z = np.diag([1.0, -1.0])
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    gate = expm(-0.23j * x)
    operator = PauliMPO.from_pauli_cores([
        np.array([[[0.0, 0.0, 0.0, 1.0]]]),
        np.array([[[1.0, 0.0, 0.0, 0.0]]]),
    ]).compress_pauli(cutoff=0.0)
    expected = np.kron(gate.conj().T @ z @ gate, np.eye(2))

    def fail(*args, **kwargs):
        raise AssertionError("native gating/channel application must not expand words")

    monkeypatch.setattr(PauliMPO, "_materialize_core_terms", fail)
    result = operator.copy()
    assert result.apply_gate(gate, 0, mode="heisenberg", inplace=True) is result
    assert result.compression_report is None
    np.testing.assert_allclose(result.to_dense(), expected, atol=1e-14)
    assert operator.apply_channel([gate], 0, inplace=True) is operator
    assert operator.compression_report is None
    np.testing.assert_allclose(operator.to_dense(), expected, atol=1e-14)


@pytest.mark.parametrize("where", [True, 0.5, [0.5], [True], ["0"]])
def test_gate_support_rejects_noninteger_sites(where):
    operator = PauliMPO(2, [(1.0, "ZI")])
    with pytest.raises((TypeError, ValueError)):
        operator.apply_gate(np.eye(2), where)
