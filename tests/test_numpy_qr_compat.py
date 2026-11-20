"""Exact QR compatibility without calling the affected native GEQRF route."""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from pepsy.backends import _numpy_qr as compat
from pepsy.operators import _pauli_dmt as dmt


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('shape', [(544, 7), (7, 544), (544, 544)])
def test_pivoted_factors_restore_original_columns_and_preserve_input(monkeypatch, dtype, shape):
    monkeypatch.setattr(compat, '_affected_accelerate', lambda: True)
    monkeypatch.setattr(compat, '_warn_workaround', lambda: None)
    rng = np.random.default_rng(803)
    a = rng.normal(size=shape).astype(dtype)
    a[:, -1] = a[:, 0]  # rank deficiency must not require dropping a direction
    before = a.copy()
    q, r = compat._numpy_qr_workaround(a)
    tolerance = 2e-5 if dtype == np.float32 else 2e-12
    np.testing.assert_allclose(q @ r, a, atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(q.T @ q, np.eye(q.shape[1]), atol=tolerance, rtol=tolerance)
    np.testing.assert_array_equal(a, before)
    assert q.dtype == r.dtype == a.dtype


def test_scope_keeps_small_complex_and_other_backends_unchanged(monkeypatch):
    monkeypatch.setattr(compat, '_affected_accelerate', lambda: True)
    for a in (np.ones((8, 8)), np.ones((544, 2), dtype=complex),
              np.ones((544, 2), dtype=np.float16), [[1., 2.]]):
        assert compat._numpy_qr_workaround(a) is None
    monkeypatch.setattr(compat, '_affected_accelerate', lambda: False)
    assert compat._numpy_qr_workaround(np.ones((544, 2))) is None


@pytest.mark.parametrize('system,machine,version,blas,expected', [
    ('Darwin', 'arm64', '26.7.3', 'accelerate', True),
    ('Darwin', 'arm64', '26.7.3', 'openblas', False),
    ('Darwin', 'arm64', '27.0', 'accelerate', False),
    ('Darwin', 'x86_64', '26.7.3', 'accelerate', False),
    ('Linux', 'arm64', '26.7.3', 'accelerate', False),
])
def test_workaround_is_gated_by_platform_and_linked_blas(monkeypatch, system, machine, version, blas, expected):
    compat._affected_accelerate.cache_clear()
    monkeypatch.setattr(compat.platform, 'system', lambda: system)
    monkeypatch.setattr(compat.platform, 'machine', lambda: machine)
    monkeypatch.setattr(compat.platform, 'mac_ver', lambda: (version, (), ''))
    monkeypatch.setattr(np, 'show_config', lambda **kwargs: {'Build Dependencies': {'blas': {'name': blas}}})
    try:
        assert compat._affected_accelerate() is expected
    finally:
        # A mocked platform must not influence following native operations.
        compat._affected_accelerate.cache_clear()


@pytest.mark.parametrize('absorb', ['left', 'right'])
def test_dmt_qr_and_lq_avoid_unsafe_split_dispatch(monkeypatch, absorb):
    monkeypatch.setattr(compat, '_affected_accelerate', lambda: True)
    monkeypatch.setattr(compat, '_warn_workaround', lambda: None)

    def forbidden(*args, **kwargs):
        raise AssertionError('affected matrix reached unsafe split QR')

    monkeypatch.setattr(dmt, 'array_split', forbidden)
    a = np.random.default_rng(814).normal(size=(544, 9))
    left, right = dmt._qr(a) if absorb == 'right' else dmt._lq(a)
    np.testing.assert_allclose(left @ right, a, atol=2e-12)


def test_large_real_qr_has_clean_subprocess_exit_and_keeps_live_factors():
    # A Gram check in the corrupting process is insufficient: the old routine
    # can return valid factors and crash only when Python frees unrelated data.
    code = '''
import numpy as np
from pepsy.backends._numpy_qr import _qr_reduced
rng = np.random.default_rng(417)
saved = []
for _ in range(8):
    q, r = _qr_reduced(rng.normal(size=(768, 192)))
    saved.append((q, q.copy()))
    a = rng.normal(size=(768, 768))
    q, r = _qr_reduced(a)
    assert np.linalg.norm(q @ r - a) / np.linalg.norm(a) < 2e-12
    assert np.linalg.norm(q.T @ q - np.eye(768)) < 2e-10
    assert all(np.array_equal(x, y) for x, y in saved)
print('clean QR exit')
'''
    env = dict(os.environ, VECLIB_MAXIMUM_THREADS='1')
    result = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True,
                            text=True, timeout=45, cwd=Path(__file__).resolve().parents[1])
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'clean QR exit' in result.stdout
