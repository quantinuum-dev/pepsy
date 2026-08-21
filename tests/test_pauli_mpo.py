"""Tests for the sparse Pauli-basis MPO front end."""

import numpy as np
import pytest

from pepsy.operators import (
    PauliBondCompressionReport,
    PauliCompressionReport,
    PauliMPO,
    decompose_pauli,
)


def _pauli_word_matrix(word):
    matrices = {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.diag([1.0, -1.0]).astype(complex),
    }
    result = matrices[word[0]]
    for label in word[1:]:
        result = np.kron(result, matrices[label])
    return result


def test_pauli_mpo_expands_local_strings_and_full_words():
    operator = PauliMPO.from_terms(
        3,
        [
            (0.35, "ZIZ"),
            (-1.0, "ZZ"),
            (1.5, "X"),
        ],
    )

    expected = {
        "ZIZ": 0.35,
        "ZZI": -1.0,
        "IZZ": -1.0,
        "XII": 1.5,
        "IXI": 1.5,
        "IIX": 1.5,
    }
    assert {word: coefficient for coefficient, word in operator.terms} == expected

    dense = sum(
        coefficient * _pauli_word_matrix(word)
        for coefficient, word in operator.terms
    )
    np.testing.assert_allclose(operator.to_dense(), dense)


def test_pauli_mpo_supports_periodic_translation_and_explicit_sites():
    periodic = PauliMPO.from_terms(3, [(1.0, "ZZ")], boundary="periodic")
    assert {
        word: coefficient for coefficient, word in periodic.terms
    } == {"ZZI": 1.0, "IZZ": 1.0, "ZIZ": 1.0}

    explicit = PauliMPO.from_terms(
        4,
        [(2.0, (0, 3), "YX")],
    )
    assert explicit.terms == ((2.0, "YIIX"),)


def test_pauli_mpo_algebra_uses_ixyz_phase_rules():
    x = PauliMPO.from_terms(1, [(1.0, "X")])
    y = PauliMPO.from_terms(1, [(1.0, "Y")])
    z = PauliMPO.from_terms(1, [(1.0, "Z")])

    assert (x @ y).terms == ((1j, "Z"),)
    assert (y @ x).terms == ((-1j, "Z"),)
    assert (x @ x).terms == ((1, "I"),)
    assert (x + x).terms == ((2.0, "X"),)
    assert (1.0j * y).dagger().terms == ((-1j, "Y"),)
    assert (x @ y - y @ x).terms == ((2j, "Z"),)
    assert z.transpose().terms == ((1.0, "Z"),)
    assert y.transpose().terms == ((-1.0, "Y"),)


def test_pauli_mpo_trace_inner_norm_and_partial_trace():
    operator = PauliMPO.from_terms(3, [(2.0, "III"), (1.0, "XII")])

    assert operator.trace() == pytest.approx(16.0)
    assert operator.trace(normalized=True) == pytest.approx(2.0)
    assert operator.inner(operator) == pytest.approx(40.0)
    assert operator.norm() == pytest.approx(np.sqrt(40.0))

    traced = operator.partial_trace((0,))
    assert traced.terms == ((4.0, "II"),)
    assert operator.partial_trace((1, 2), keep=True).terms == ((4.0, "II"),)
    assert operator.partial_trace((0, 1, 2)) == pytest.approx(16.0)


def test_pauli_mpo_identity_and_zero_compile_without_extra_identity():
    identity = PauliMPO.identity(2, coefficient=2.0)
    zero = PauliMPO.zero(2)
    np.testing.assert_allclose(identity.to_dense(), 2.0 * np.eye(4))
    np.testing.assert_allclose(zero.to_dense(), np.zeros((4, 4)))


def test_pauli_mpo_expectation_application_and_quimb_compression():
    qtn = pytest.importorskip("quimb.tensor")
    state = qtn.MPS_computational_state("000", dtype="complex128")
    operator = PauliMPO.from_terms(3, [(1.0, "ZII"), (2.0, "IZZ")])

    assert operator.expectation(state) == pytest.approx(3.0)
    applied = PauliMPO.from_terms(3, [(1.0, "XII")]).apply(state)
    expected = qtn.MPS_computational_state("100", dtype="complex128")
    np.testing.assert_allclose(applied.to_dense(), expected.to_dense())

    compressed, report = operator.compress(
        basis="mpo",
        max_bond=2,
        return_report=True,
    )
    assert isinstance(compressed, qtn.MatrixProductOperator)
    assert report.final_bond_dimensions == tuple(compressed.bond_sizes())
    assert hasattr(operator.to_mpo(), "pepsy_pauli_mpo")


def test_pauli_mpo_torch_scalar_preserves_norm_gradient():
    torch = pytest.importorskip("torch")
    coefficient = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    operator = PauliMPO.from_terms(2, [(coefficient, "XX")])

    norm = operator.norm()
    (gradient,) = torch.autograd.grad(norm, coefficient)
    assert torch.allclose(norm, torch.tensor(1.4, dtype=torch.float64))
    assert torch.allclose(gradient, torch.tensor(2.0, dtype=torch.float64))


def test_pauli_mpo_native_canonicalization_combines_and_prunes():
    operator = PauliMPO(
        2,
        [
            (1.0, "XX"),
            (2.0, "XX"),
            (1.0e-13, "ZZ"),
        ],
    )

    assert operator.terms == ((3.0, "XX"), (1.0e-13, "ZZ"))
    assert operator.canonicalize(atol=1.0e-12).terms == ((3.0, "XX"),)
    assert operator.simplify(rtol=1.0e-10).terms == ((3.0, "XX"),)
    assert operator.canonicalize().terms == operator.terms


def test_pauli_mpo_dense_decomposition_round_trips():
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    terms = decompose_pauli(h)
    assert tuple(word for _, word in terms) == ("X", "Z")
    np.testing.assert_allclose(
        [coefficient for coefficient, _ in terms],
        [1 / np.sqrt(2), 1 / np.sqrt(2)],
    )

    operator = PauliMPO.from_dense(h)
    np.testing.assert_allclose(operator.to_dense(), h)
    np.testing.assert_allclose(PauliMPO.from_matrix(h).to_dense(), h)


def test_pauli_mpo_apply_gate_matches_dense_left_right_and_conjugate():
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    operator = PauliMPO.from_terms(3, [(1.0, "IZI")])

    conjugated = operator.apply_gate(h, 1)
    expected_conjugated = np.kron(np.kron(_pauli_word_matrix("I"), h), np.eye(2))
    expected_conjugated = expected_conjugated @ _pauli_word_matrix("IZI") @ expected_conjugated.conj().T
    np.testing.assert_allclose(conjugated.to_dense(), expected_conjugated, atol=1e-12)

    left = operator.apply_gate(h, 1, mode="left")
    right = operator.apply_gate(h, 1, mode="right")
    gate_full = np.kron(np.kron(np.eye(2), h), np.eye(2))
    np.testing.assert_allclose(
        left.to_dense(), gate_full @ operator.to_dense(), atol=1e-12
    )
    np.testing.assert_allclose(
        right.to_dense(), operator.to_dense() @ gate_full, atol=1e-12
    )

    inplace = operator.copy()
    assert inplace.apply_gate(h, 1, inplace=True) is inplace
    np.testing.assert_allclose(inplace.to_dense(), conjugated.to_dense(), atol=1e-12)


def test_pauli_mpo_apply_gate_supports_noncontiguous_sites_and_channels():
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    operator = PauliMPO.from_terms(3, [(1.0, "XII")])
    hh = np.kron(h, h)
    transformed = operator.apply_gate(hh, (2, 0))
    gate_full = np.kron(np.kron(h, np.eye(2)), h)
    np.testing.assert_allclose(
        transformed.to_dense(),
        gate_full @ operator.to_dense() @ gate_full.conj().T,
        atol=1e-12,
    )

    gamma = 0.3
    kraus = (
        np.diag([1.0, np.sqrt(1.0 - gamma)]).astype(complex),
        np.array([[0.0, np.sqrt(gamma)], [0.0, 0.0]], dtype=complex),
    )
    identity = PauliMPO.identity(1)
    channel_identity = identity.apply_channel(kraus, 0)
    np.testing.assert_allclose(channel_identity.to_dense(), np.eye(2))


def test_pauli_mpo_native_cores_canonicalize_and_compress_without_leaving_basis():
    operator = PauliMPO.from_terms(
        4,
        [
            (0.5, "IIII"),
            (1.0, "XX"),
            (2.0, "ZZ"),
        ],
    )
    dense = operator.to_dense()

    canonical = operator.canonicalize_native()
    assert canonical.pauli_bond_dimensions == (4, 7, 7)
    for core in canonical.to_pauli_cores()[:-1]:
        matrix = core.transpose(0, 2, 1).reshape(-1, core.shape[1])
        np.testing.assert_allclose(matrix.conj().T @ matrix, np.eye(core.shape[1]))
    np.testing.assert_allclose(canonical.to_mpo().to_dense(), dense, atol=1e-12)

    compressed, report = operator.compress_pauli(max_bond=8, return_report=True)
    assert isinstance(report, PauliCompressionReport)
    assert compressed.compression_report is report
    assert report.exact
    assert max(compressed.pauli_bond_dimensions) <= 8
    np.testing.assert_allclose(compressed.to_mpo().to_dense(), dense, atol=1e-12)

    truncated, report = operator.compress(basis="native", max_bond=2, return_report=True)
    assert isinstance(truncated, PauliMPO)
    assert max(truncated.pauli_bond_dimensions) <= 2
    assert not report.exact
    assert all(core.shape[2] == 4 for core in truncated.to_pauli_cores())

    default_compressed = operator.compress(max_bond=8)
    assert isinstance(default_compressed, PauliMPO)
    assert max(default_compressed.pauli_bond_dimensions) <= 8


def test_pauli_mpo_native_product_grows_then_compresses_pauli_bonds():
    left = PauliMPO.from_terms(
        4,
        [(1.0, "XX"), (0.5, "Z"), (0.25j, "YXY")],
    ).compress_pauli(max_bond=3)
    right = PauliMPO.from_terms(
        4,
        [(2.0, "ZZ"), (-0.75, "X"), (0.5j, "YZY")],
    ).compress_pauli(max_bond=2)

    product = left @ right
    expected_bonds = tuple(
        left_bond * right_bond
        for left_bond, right_bond in zip(
            left.pauli_bond_dimensions,
            right.pauli_bond_dimensions,
        )
    )
    assert product.pauli_bond_dimensions == expected_bonds
    assert all(core.shape[2] == 4 for core in product.to_pauli_cores())
    np.testing.assert_allclose(
        product.to_dense(),
        left.to_dense() @ right.to_dense(),
        atol=1.0e-11,
    )

    compressed, report = product.compress(
        max_bond=3,
        cutoff=1.0e-12,
        return_report=True,
    )
    assert isinstance(compressed, PauliMPO)
    assert max(compressed.pauli_bond_dimensions) <= 3
    assert compressed.compression_report is report
    assert not report.exact


def test_pauli_mpo_native_core_constructor_round_trips_small_expansion():
    operator = PauliMPO.from_terms(3, [(1.25, "XII"), (-0.5j, "IZZ")])
    reconstructed = PauliMPO.from_pauli_cores(operator.to_pauli_cores())

    assert reconstructed.terms == operator.terms
    np.testing.assert_allclose(reconstructed.to_dense(), operator.to_dense())


def test_pauli_mpo_native_compression_matches_quimb_bond_policy():
    pytest.importorskip("quimb.tensor")
    operator = PauliMPO.from_terms(
        5,
        [(1.0, "XX"), (2.0, "ZZ"), (0.7, "X"), (0.2, "YXY")],
    )

    for form in ("right", "left"):
        quimb_mpo = operator.to_mpo()
        quimb_mpo.compress(form=form, max_bond=4, cutoff=1.0e-10)
        native = operator.compress_pauli(
            form=form,
            max_bond=4,
            cutoff=1.0e-10,
        )
        assert native.pauli_bond_dimensions == tuple(quimb_mpo.bond_sizes())

    for cutoff_mode in ("rel", "abs", "sum1", "sum2"):
        quimb_mpo = operator.to_mpo()
        quimb_mpo.compress(
            form="right",
            cutoff=1.0,
            cutoff_mode=cutoff_mode,
        )
        native = operator.compress_pauli(
            form="right",
            cutoff=1.0,
            cutoff_mode=cutoff_mode,
        )
        assert native.pauli_bond_dimensions == tuple(quimb_mpo.bond_sizes())

    inplace = operator.copy()
    returned = inplace.compress_pauli(max_bond=2, inplace=True)
    assert returned is inplace
    assert max(inplace.pauli_bond_dimensions) <= 2


def test_pauli_mpo_native_core_operations_do_not_need_word_expansion():
    operator = PauliMPO.from_terms(
        4,
        [(0.5, "IIII"), (1.0, "XX"), (2.0j, "YZZ")],
    ).compress_pauli(max_bond=4)
    dense = operator.to_dense()

    def fail_materialization(*args, **kwargs):
        raise AssertionError("core-native operation expanded Pauli words")

    operator._materialize_core_terms = fail_materialization
    np.testing.assert_allclose(operator.trace(), np.trace(dense))
    np.testing.assert_allclose(operator.dagger().to_dense(), dense.conj().T)
    np.testing.assert_allclose(operator.conjugate().to_dense(), dense.conj())
    np.testing.assert_allclose(operator.transpose().to_dense(), dense.T)
    np.testing.assert_allclose(operator.inner(operator), np.trace(dense.conj().T @ dense))
    np.testing.assert_allclose((operator + operator).to_dense(), 2.0 * dense)
    np.testing.assert_allclose((operator @ operator).to_dense(), dense @ dense, atol=1e-12)

    hadamard = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)
    transformed = operator.apply_gate(hadamard, 2)
    full_gate = np.kron(
        np.kron(np.kron(np.eye(2), np.eye(2)), hadamard),
        np.eye(2),
    )
    np.testing.assert_allclose(
        transformed.to_dense(),
        full_gate @ dense @ full_gate.conj().T,
        atol=1e-12,
    )

    reduced = operator.partial_trace((1, 3))
    dense_tensor = dense.reshape(2, 2, 2, 2, 2, 2, 2, 2)
    expected = np.trace(dense_tensor, axis1=1, axis2=5)
    expected = np.trace(expected, axis1=2, axis2=5).reshape(4, 4)
    np.testing.assert_allclose(reduced.to_dense(), expected, atol=1e-12)


def test_pauli_mpo_native_randomized_and_flat_reports_are_detailed():
    operator = PauliMPO.from_terms(
        6,
        [(1.0, "X"), (2.0, "ZZ"), (0.3, "YXY"), (0.1, "XYZXYZ")],
    )
    for method in ("rsvd", "svds", "isvd"):
        compressed, report = operator.compress_pauli(
            max_bond=3,
            method=method,
            seed=7,
            return_report=True,
        )
        assert max(compressed.pauli_bond_dimensions) <= 3
        assert isinstance(report.bond_reports[0], PauliBondCompressionReport)
        assert len(report.bond_reports) == operator.nsites - 1
        assert report.per_bond == report.bond_reports
        assert all(item.singular_values for item in report.bond_reports)

    flat, flat_report = operator.compress_pauli(
        max_bond=2,
        form="flat",
        return_report=True,
    )
    assert max(flat.pauli_bond_dimensions) <= 2
    assert flat_report.form == "flat"


def test_pauli_mpo_native_compression_preserves_torch_backend():
    torch = pytest.importorskip("torch")
    coefficient = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    operator = PauliMPO.from_terms(4, [(coefficient, "XX"), (1.0, "Z")])

    compressed = operator.compress_pauli(
        max_bond=2,
        method="rsvd",
        seed=3,
    )
    core = compressed.to_pauli_cores()[0]
    assert isinstance(core, torch.Tensor)
    assert core.device == coefficient.device
    assert core.requires_grad
    assert isinstance(compressed.trace(), torch.Tensor)
