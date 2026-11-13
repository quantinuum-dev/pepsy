"""Quimb cutoff semantics and protected DMT data in physical MPO units."""

import numpy as np
import pytest
from quimb.tensor.decomp import array_split
from scipy.linalg import expm

from pepsy.operators import PauliMPO
from pepsy.optimizers import PauliDMT
from pepsy.operators._pauli_dmt import _split
from test_pauli_dmt_large import product_mixture


@pytest.mark.parametrize('mode,cutoff', [
    ('rel', .25), ('abs', .25), ('sum1', .15),
    ('rsum1', .15), ('sum2', .05), ('rsum2', .05),
])
@pytest.mark.parametrize('scale', [1e-120, 1., 1e120])
def test_complement_cutoffs_match_quimb_and_preserve_physical_panels(mode, cutoff, scale):
    s = np.array([5., 4., 3., 2., 1., .8, .4, .2, .1, .05, .02, .01])
    matrix = scale * np.diag(s)
    protected = np.eye(12)[:, :4]
    power = 2 if mode == 'sum2' else 1
    physical_cutoff = cutoff * ((scale * 8.) ** power) if mode in {'abs', 'sum1', 'sum2'} else cutoff
    left, right, report = _split(
        matrix, protected, protected, max_bond=20, cutoff=physical_cutoff,
        cutoff_mode=mode, nsites=6, variant='operator', absorb='right',
        bond=0, diagnostics=True,
    )
    u, values, vh = array_split(np.diag(s[4:]), method='svd', absorb=None,
                                cutoff=cutoff, cutoff_mode=mode, max_bond=12,
                                renorm=False)
    expected = np.diag(s)
    expected[4:, 4:] = (u * values) @ vh
    np.testing.assert_allclose((left @ right) / scale, expected, atol=2e-13)
    np.testing.assert_allclose((protected.T @ left @ right) / scale,
                               protected.T @ np.diag(s), atol=2e-13)
    assert report.final_bond <= 20


@pytest.mark.parametrize('dtype,resolved', [('float32', 1e-6), ('float64', 1e-12),
                                             ('complex128', 1e-12)])
def test_auto_cutoff_uses_pepsy_dtype_policy_on_all_public_paths(dtype, resolved):
    rho = PauliMPO.identity(4, coefficient=1 / 16)
    rho = PauliMPO.from_pauli_cores([a.real.astype(dtype) for a in rho.to_pauli_cores()])
    _, report = rho.compress_dmt(max_bond=8, cutoff='auto', cutoff_mode='auto',
                                 return_report=True)
    assert report.cutoff == resolved and report.cutoff_mode == 'rel'
    _, report = rho.apply_gate_dmt(np.eye(4), (1, 2), max_bond=8, cutoff='auto',
                                   cutoff_mode='rsum2', return_report=True)
    assert report.cutoff == resolved and report.cutoff_mode == 'rsum2'
    run = PauliDMT(rho, max_bond=8, cutoff='auto', cutoff_mode='sum2')
    assert run.report.cutoff == resolved and run.report.cutoff_mode == 'sum2'
    run.apply_gate(np.eye(4), (1, 2))
    assert run.report.cutoff_mode == 'sum2'


@pytest.mark.parametrize('kwargs', [
    {'cutoff': 'unknown'}, {'cutoff': -1.}, {'cutoff': float('nan')},
    {'cutoff_mode': 'unknown'},
])
def test_invalid_cutoff_options_are_rejected(kwargs):
    with pytest.raises(ValueError, match='cutoff'):
        PauliMPO.identity(3).compress_dmt(max_bond=8, **kwargs)


def test_explicit_trace_guard_controls_public_gate_paths():
    traceless = PauliMPO.from_terms(3, [(1., 'XII')])
    assert not PauliDMT(traceless, max_bond=8, variant='operator').trace_guard_active
    assert PauliDMT(traceless, max_bond=8, variant='operator', trace_guard=True).trace_guard_active
    assert not PauliDMT(PauliMPO.identity(3), max_bond=8, trace_guard=False).trace_guard_active
    with pytest.raises(ValueError, match='trace_guard'):
        traceless.apply_gate_dmt(np.eye(4), (0, 1), max_bond=8,
                                 variant='operator', trace_guard='wrong')


def test_nonfinite_local_target_is_rejected_transactionally():
    rho = PauliMPO.identity(4, coefficient=1 / 16)
    before = rho.to_dense()
    gate = np.eye(4)
    gate[0, 0] = np.nan
    with pytest.raises(FloatingPointError):
        rho.apply_gate_dmt(gate, (0, 1), max_bond=8, inplace=True)
    np.testing.assert_array_equal(rho.to_dense(), before)


@pytest.mark.parametrize('backend', ['torch', 'jax'])
@pytest.mark.parametrize('mode,cutoff', [
    ('rel', .25), ('abs', .25), ('sum1', .15),
    ('rsum1', .15), ('sum2', .05), ('rsum2', .05),
])
def test_cutoff_modes_use_backend_native_svd(backend, mode, cutoff, request):
    values = np.array([5., 4., 3., 2., 1., .8, .4, .2, .1, .05, .02, .01])
    matrix, protected = np.diag(values), np.eye(12)[:, :4]
    if backend == 'torch':
        torch = pytest.importorskip('torch')
        convert = lambda x: torch.as_tensor(x, dtype=torch.float64)
        host = lambda x: x.detach().cpu().numpy()
    else:
        jax = pytest.importorskip('jax')
        previous = jax.config.x64_enabled
        request.addfinalizer(lambda: jax.config.update('jax_enable_x64', previous))
        jax.config.update('jax_enable_x64', True)
        import jax.numpy as jnp
        convert, host = jnp.asarray, np.asarray
    left, right, _ = _split(
        convert(matrix), convert(protected), convert(protected), max_bond=20,
        cutoff=cutoff, cutoff_mode=mode, variant='operator', absorb='right',
        bond=0, diagnostics=True,
    )
    u, s, vh = array_split(np.diag(values[4:]), method='svd', absorb=None,
                           cutoff=cutoff, cutoff_mode=mode, max_bond=12, renorm=False)
    expected = matrix.copy()
    expected[4:, 4:] = (u * s) @ vh
    np.testing.assert_allclose(host(left @ right), expected, atol=2e-12)


@pytest.mark.parametrize('mode', ['rel', 'abs', 'sum1', 'rsum1', 'sum2', 'rsum2'])
def test_cutoff_modes_preserve_local_data_after_compression_and_gate(mode):
    rho, _, _ = product_mixture(8, rank=16)
    compressed, report = rho.compress_dmt(max_bond=12, cutoff=.001,
                                          cutoff_mode=mode, return_report=True)
    assert any(r.input_rank > r.final_bond for r in report.bond_reports)
    h = np.kron([[0., 1.], [1., 0.]], np.diag([1., -1.]))
    gate = expm(-.12j * h)
    target = rho.apply_gate_dmt(gate, (3, 4), max_bond=64)
    result = rho.apply_gate_dmt(gate, (3, 4), max_bond=12,
                                cutoff=.001, cutoff_mode=mode)
    for j in range(6):
        window = range(j, j + 3)
        np.testing.assert_allclose(
            compressed.partial_trace(window, keep=True).to_dense(),
            rho.partial_trace(window, keep=True).to_dense(), atol=3e-12,
        )
        np.testing.assert_allclose(
            result.partial_trace(window, keep=True).to_dense(),
            target.partial_trace(window, keep=True).to_dense(), atol=3e-12,
        )


@pytest.mark.parametrize('variant', ['density', 'operator'])
@pytest.mark.parametrize('side', ['left', 'right'])
def test_full_protected_side_skips_unnecessary_positive_cutoff_truncation(variant, side):
    rng = np.random.default_rng(44)
    m = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    identity = np.eye(4, dtype=complex)
    restricted = identity[:, :2]
    lv, rv = (identity, restricted) if side == 'left' else (restricted, identity)
    left, right, report = _split(
        m, lv, rv, max_bond=12, cutoff=.9, cutoff_mode='rel',
        variant=variant, absorb='right', bond=0, diagnostics=True,
    )
    np.testing.assert_allclose(left @ right, m, atol=3e-14)
    assert report.discarded_weight == 0.
    assert report.residual_rank == 0
    assert max(report.protected_left_rank, report.protected_right_rank) == 4


@pytest.mark.parametrize('variant', ['density', 'operator'])
@pytest.mark.parametrize('where', [(0, 1), (2, 3)])
def test_boundary_gate_preserves_full_target_without_svd(monkeypatch, variant, where):
    import pepsy.operators._pauli_dmt as dmt
    rng = np.random.default_rng(43)
    x = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
    dense = x @ x.conj().T
    dense /= np.trace(dense)
    rho = PauliMPO.from_dense(dense, max_qubits=4)
    h = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    gate = expm(-.1j * (h + h.conj().T))
    full = np.kron(gate, np.eye(4)) if where[0] == 0 else np.kron(np.eye(4), gate)
    expected = full @ dense @ full.conj().T
    original, svds = dmt.array_split, []

    def observe(matrix, **kwargs):
        if kwargs.get('method') == 'svd':
            svds.append(matrix.shape)
        return original(matrix, **kwargs)

    monkeypatch.setattr(dmt, 'array_split', observe)
    result = rho.apply_gate_dmt(gate, where, max_bond=8, cutoff=.9, variant=variant)
    np.testing.assert_allclose(result.to_dense(), expected, atol=3e-13)
    assert not svds
