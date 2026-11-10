"""Explicit damping MPOs against independent coefficient-space references."""

from dataclasses import FrozenInstanceError
import itertools
import math

import numpy as np
import pytest

from pepsy.operators import PauliDampingMPO, PauliMPO


def test_damping_has_one_public_owner():
    import pepsy.operators as operators
    from pepsy.operators.pauli_damping import PauliDampingMPO as implementation

    assert operators.PauliDampingMPO is implementation
    assert "PauliDampingMPO" in operators.__all__
    assert operators.pauli_damping.PauliDampingMPO is implementation


def _coefficient_tensor(cores):
    value = cores[0][0].T
    for core in cores[1:]:
        value = np.einsum("...a,abp->...pb", value, core)
    return value[..., 0]


def _expected_diagonal(nsites, max_weight, gamma):
    return np.array([
        math.exp(-gamma * max(sum(p != 0 for p in labels) - max_weight, 0))
        for labels in itertools.product(range(4), repeat=nsites)
    ])


@pytest.mark.parametrize("nsites", [1, 2, 4])
@pytest.mark.parametrize("max_weight", [0, 1, 2, 4])
@pytest.mark.parametrize("gamma", [0.0, 0.37])
def test_explicit_superoperator_mpo_matches_every_string(nsites, max_weight, gamma):
    damping = PauliDampingMPO(nsites, max_weight=max_weight, gamma=gamma)
    mpo = damping.to_mpo()
    expected = _expected_diagonal(nsites, max_weight, gamma)
    np.testing.assert_allclose(mpo.to_dense(), np.diag(expected), atol=1e-15)
    np.testing.assert_allclose(
        _coefficient_tensor(damping.to_diagonal_cores()).ravel(), expected,
    )
    assert not mpo.cyclic
    assert all(mpo.phys_dim(i) == 4 for i in range(nsites))
    assert max(damping.bond_dimensions, default=1) <= max_weight + 1
    assert damping.is_identity == (gamma == 0 or max_weight >= nsites)
    for index, labels in enumerate(itertools.product("IXYZ", repeat=nsites)):
        assert damping.factor("".join(labels)) == pytest.approx(expected[index])


@pytest.mark.parametrize("nsites", [1, 2, 4])
def test_quimb_mpo_application_and_native_application_agree(nsites):
    import quimb.tensor as qtn

    rng = np.random.default_rng(231)
    bonds = [1] + [2] * (nsites - 1) + [1]
    cores = [rng.normal(size=(left, right, 4)) + 1j * rng.normal(size=(left, right, 4))
             for left, right in zip(bonds[:-1], bonds[1:])]
    operator = PauliMPO.from_pauli_cores(cores)
    arrays = []
    for site, core in enumerate(cores):
        array = core[0] if site == 0 else core
        if site == nsites - 1:
            array = array[0] if nsites == 1 else array[:, 0]
        arrays.append(array)
    vectorized = qtn.MatrixProductState(arrays, shape="lrp", site_ind_id="q{}")
    damping = PauliDampingMPO(nsites, max_weight=1, gamma=0.29)
    applied_mps = damping.to_mpo().apply(vectorized, compress=False)
    native = operator.apply_damping(damping)
    expected = _coefficient_tensor(cores).ravel() * _expected_diagonal(nsites, 1, 0.29)
    np.testing.assert_allclose(applied_mps.to_dense().ravel(), expected, atol=2e-14)
    np.testing.assert_allclose(_coefficient_tensor(native.to_pauli_cores()).ravel(),
                               expected, atol=2e-14)


def test_reusable_filter_owns_exported_arrays_and_preserves_inplace_contract():
    damping = PauliDampingMPO(3, max_weight=1, gamma=0.3)
    reference = damping.to_mpo().to_dense()
    damping.to_diagonal_cores()[0][...] = 0
    damping.to_diagonal_cores(like=np.zeros(1))[0][...] = 0
    damping.to_mpo().arrays[0][...] = 0
    np.testing.assert_array_equal(damping.to_mpo().to_dense(), reference)
    with pytest.raises(FrozenInstanceError):
        damping.gamma = 0.7

    initial = PauliMPO(3, [(0.4, "III"), (0.6, "XYZ")])
    for operator in (initial, initial.compress_pauli(cutoff=0.0)):
        expected = damping.apply(operator)
        actual = operator.copy()
        assert actual.apply_damping(damping, inplace=True) is actual
        assert actual.compression_report is None
        np.testing.assert_allclose(actual.to_dense(), expected.to_dense())
    np.testing.assert_array_equal(damping.to_mpo().to_dense(), reference)


def test_damping_then_compression_preserves_the_requested_target():
    rng = np.random.default_rng(133)
    operator = PauliMPO.from_pauli_cores([
        rng.normal(size=(1, 2, 4)), rng.normal(size=(2, 2, 4)),
        rng.normal(size=(2, 1, 4)),
    ])
    damping = PauliDampingMPO(3, max_weight=1, gamma=0.4)
    expected = _coefficient_tensor(operator.to_pauli_cores()).ravel()
    expected *= _expected_diagonal(3, 1, 0.4)
    filtered = damping.apply(operator)
    assert filtered.compression_report is None
    compressed, report = filtered.compress_pauli(
        max_bond=4, cutoff=0.0, renorm=False, return_report=True,
    )
    np.testing.assert_allclose(_coefficient_tensor(compressed.to_pauli_cores()).ravel(),
                               expected, atol=3e-14)
    assert report.discarded_singular_weight < 1e-12


def test_lossy_compression_error_is_measured_against_the_filtered_target():
    rng = np.random.default_rng(234)
    operator = PauliMPO.from_pauli_cores([
        rng.normal(size=(1, 3, 4)), rng.normal(size=(3, 3, 4)),
        rng.normal(size=(3, 3, 4)), rng.normal(size=(3, 1, 4)),
    ])
    target = _coefficient_tensor(operator.to_pauli_cores()).ravel()
    target *= _expected_diagonal(4, 1, 0.4)
    filtered = PauliDampingMPO(4, max_weight=1, gamma=0.4).apply(operator)
    compressed, report = filtered.compress_pauli(
        max_bond=1, cutoff=0.0, renorm=False, return_report=True,
    )
    error = 4 * np.linalg.norm(target - _coefficient_tensor(compressed.to_pauli_cores()).ravel())
    assert error > 1e-6
    assert not report.exact
    assert error <= report.discarded_singular_weight * (1 + 1e-12)


@pytest.mark.parametrize("gamma", [0.0, 0.12, 0.7])
def test_repeated_evolution_damping_and_compression_match_dense_reference(gamma):
    """Check a trajectory, conserved spin readout and amplitude loss together."""
    from scipy.linalg import expm

    paulis = np.array([np.eye(2), [[0, 1], [1, 0]],
                       [[0, -1j], [1j, 0]], np.diag([1, -1])])
    basis = np.array([np.kron(np.kron(a, b), c)
                      for a, b, c in itertools.product(paulis, repeat=3)])
    local_h = 0.5 * (np.kron(paulis[1], paulis[1]) + np.kron(paulis[2], paulis[2]))
    local_h += 0.2 * np.kron(paulis[3], paulis[3])
    gate = expm(-0.17j * local_h)
    full_gates = (np.kron(gate, np.eye(2)), np.kron(np.eye(2), gate))
    initial = np.kron(np.kron(np.eye(2), paulis[3]), np.eye(2))
    expected = initial.copy()
    operator = PauliMPO(3, [(1.0, "IZI")]).compress_pauli(cutoff=0.0)
    damping = PauliDampingMPO(3, max_weight=1, gamma=gamma)
    factors = _expected_diagonal(3, 1, gamma)
    queries = [PauliMPO(3, [(1.0, "I" * i + "Z" + "I" * (2 - i))])
               for i in range(3)]
    previous_norm = 1.0
    for _ in range(8):
        for site, full_gate in enumerate(full_gates):
            expected = full_gate.conj().T @ expected @ full_gate
            operator = operator.apply_gate(gate, (site, site + 1), mode="heisenberg")
            operator = operator.compress_pauli(max_bond=4, cutoff=0.0, renorm=False)
        coefficients = np.einsum("aij,ij->a", basis.conj(), expected) / 8
        expected = np.einsum("a,aij->ij", factors * coefficients, basis)
        operator = damping.apply(operator).compress_pauli(
            max_bond=4, cutoff=0.0, renorm=False,
        )
        np.testing.assert_allclose(operator.to_dense(), expected, atol=1e-13)
        correlations = np.array([query.inner(operator, normalized=True) for query in queries])
        expected_correlations = [np.trace(query.to_dense() @ expected) / 8 for query in queries]
        np.testing.assert_allclose(correlations, expected_correlations, atol=1e-13)
        assert correlations.sum() == pytest.approx(1.0, abs=1e-13)
        norm = operator.inner(operator, normalized=True).real
        assert norm <= previous_norm + 1e-13
        previous_norm = norm
    if gamma == 0.0:
        assert previous_norm == pytest.approx(1.0, abs=1e-13)
    else:
        assert previous_norm < 0.99


@pytest.mark.parametrize("complex_input", [False, True])
def test_reused_torch_filter_passes_finite_difference_gradcheck(complex_input):
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(912)
    dtype = torch.complex128 if complex_input else torch.float64
    cores = tuple(torch.randn(shape, generator=generator, dtype=dtype, requires_grad=True)
                  for shape in ((1, 2, 4), (2, 1, 4)))
    damping = PauliDampingMPO(2, max_weight=0, gamma=0.37)

    def objective(*arrays):
        filtered = damping.apply(PauliMPO.from_pauli_cores(arrays, copy=False))
        return filtered.inner(filtered, normalized=True).real

    assert torch.autograd.gradcheck(objective, cores, eps=1e-6, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_explicit_numpy_filter_matches_reference_precision(dtype):
    damping = PauliDampingMPO(2, max_weight=0, gamma=0.3)
    cores = damping.to_diagonal_cores(like=np.zeros(1, dtype=dtype))
    mpo = damping.to_mpo(like=np.zeros(1, dtype=dtype))
    assert all(core.dtype == dtype for core in cores)
    assert all(array.dtype == dtype for array in mpo.arrays)
    np.testing.assert_allclose(mpo.to_dense(),
                               np.diag(_expected_diagonal(2, 0, 0.3)), atol=1e-7)


@pytest.mark.parametrize("dtype_name", ["float32", "float64", "complex64", "complex128"])
def test_explicit_torch_filter_preserves_dtype_and_device(dtype_name):
    torch = pytest.importorskip("torch")
    reference = torch.zeros(1, dtype=getattr(torch, dtype_name))
    damping = PauliDampingMPO(2, max_weight=0, gamma=0.3)
    for array in (*damping.to_diagonal_cores(like=reference),
                  *damping.to_mpo(like=reference).arrays):
        assert array.dtype == reference.dtype
        assert array.device == reference.device


def test_explicit_filter_can_be_exported_inside_jax_jit():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    damping = PauliDampingMPO(2, max_weight=0, gamma=0.3)

    @jax.jit
    def evaluate(reference):
        cores = damping.to_diagonal_cores(like=reference)
        return jnp.einsum("arp,rbq->pq", cores[0], cores[1])

    diagonal = evaluate(jnp.zeros(1, dtype=jnp.float32))
    assert diagonal.dtype == jnp.float32
    np.testing.assert_allclose(diagonal.ravel(), _expected_diagonal(2, 0, 0.3), atol=1e-7)


def test_jax_filter_preserves_a_nondefault_device():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    devices = jax.devices("cpu")
    if len(devices) < 2:
        pytest.skip("requires two logical CPU devices; set XLA_FLAGS before Python startup")
    device = devices[1]
    core = jax.device_put(jnp.asarray([[[1.0, 2.0, 3.0, 4.0]]], dtype=jnp.float32), device)
    damping = PauliDampingMPO(2, max_weight=0, gamma=0.37)
    filtered = damping.apply(PauliMPO.from_pauli_cores([core, core]))
    for array in (*filtered.to_pauli_cores(), *damping.to_diagonal_cores(like=core),
                  *damping.to_mpo(like=core).arrays):
        assert array.device == device
        assert array.dtype == core.dtype
    expected = np.outer(np.asarray(core).ravel(), np.asarray(core).ravel()).ravel()
    expected *= _expected_diagonal(2, 0, 0.37)
    np.testing.assert_allclose(_coefficient_tensor(filtered.to_pauli_cores()).ravel(),
                               expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("nsites", [0, -1, True, 1.5])
def test_damping_rejects_invalid_chain_size(nsites):
    with pytest.raises(ValueError, match="nsites"):
        PauliDampingMPO(nsites, max_weight=1, gamma=0.1)


def test_damping_rejects_incompatible_application_and_readout():
    damping = PauliDampingMPO(2, max_weight=1, gamma=0.1)
    with pytest.raises(TypeError, match="operator"):
        damping.apply(np.eye(4))
    with pytest.raises(ValueError, match="nsites"):
        damping.apply(PauliMPO.identity(3))
    with pytest.raises(TypeError, match="inplace"):
        damping.apply(PauliMPO.identity(2), inplace=1)
    with pytest.raises(ValueError, match="word length"):
        damping.factor("X")
    with pytest.raises(ValueError, match="Pauli"):
        damping.factor("AA")
    with pytest.raises(TypeError, match="PauliDampingMPO"):
        PauliMPO.identity(2).apply_damping(np.eye(16))
