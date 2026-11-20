"""Explicit CPU QR policy preserves Torch storage and restores registrations."""

import autoray as ar
import numpy as np
import pytest

from pepsy.backends import TorchLinalgConfig, get_torch_linalg_config


@pytest.mark.parametrize('dtype', ['float64', 'complex128'])
@pytest.mark.parametrize('absorb', ['left', 'right'])
def test_cpu_pivoted_policy_covers_autoray_and_quimb_without_native_qr(monkeypatch, dtype, absorb):
    torch = pytest.importorskip('torch')
    from quimb.tensor.decomp import array_split

    a = torch.randn((17, 11), dtype=getattr(torch, dtype))
    a[:, -1] = a[:, 0]
    before = a.clone()

    def forbidden(*args, **kwargs):
        raise AssertionError('explicit CPU policy called native Torch QR')

    with TorchLinalgConfig(cpu_qr='scipy_pivoted', quimb_split_drivers=True).activated():
        monkeypatch.setattr(torch.linalg, 'qr', forbidden)
        q, r = ar.do('linalg.qr', a, mode='reduced')
        torch.testing.assert_close(q @ r, a, atol=2e-13, rtol=2e-13)
        left, _, right = array_split(a, method='qr', absorb=absorb)
        torch.testing.assert_close(left @ right, a, atol=2e-13, rtol=2e-13)
        iso = left if absorb == 'right' else right.mH
        torch.testing.assert_close(iso.mH @ iso, torch.eye(iso.shape[1], dtype=a.dtype), atol=2e-13, rtol=2e-13)
        assert left.device == right.device == a.device and left.dtype == right.dtype == a.dtype
        torch.testing.assert_close(a, before, atol=0., rtol=0.)


def test_cpu_policy_is_forward_only_and_context_restores_previous_config():
    torch = pytest.importorskip('torch')
    previous = get_torch_linalg_config()
    policy = TorchLinalgConfig(cpu_qr='scipy_pivoted')
    assert policy.to_dict()['cpu_qr'] == 'scipy_pivoted'
    with policy.activated():
        with pytest.raises(RuntimeError, match='forward-only'):
            ar.do('linalg.qr', torch.eye(3, dtype=torch.float64, requires_grad=True))
    assert get_torch_linalg_config() == previous
    with pytest.raises(ValueError, match='cpu_qr'):
        TorchLinalgConfig(cpu_qr='unknown')


def test_complex_subnormal_qr_phase_is_finite_without_clipping_pivot():
    torch = pytest.importorskip('torch')
    from quimb.tensor.decomp import array_split

    tiny = np.nextafter(0., 1.) * 1024
    a = torch.diag(torch.tensor([1.+0j, complex(tiny)], dtype=torch.complex128))
    with TorchLinalgConfig(cpu_qr='scipy_pivoted', quimb_split_drivers=True).activated():
        left, _, right = array_split(a, method='qr', absorb='right')
    assert torch.isfinite(left).all() and torch.isfinite(right).all()
    torch.testing.assert_close(left @ right, a, atol=0., rtol=0.)
