"""Captured adaptive QR retains parameter-dependent singular-chart decisions."""

import numpy as np
import pytest


@pytest.mark.parametrize('shape', [(4, 3), (3, 4), (2, 4, 3)])
def test_captured_qr_vjp_reuses_graph_across_rank_changes(shape):
    torch = pytest.importorskip('torch')
    import autoray as ar
    from torch.fx.experimental.proxy_tensor import make_fx
    from pepsy.backends import TorchLinalgConfig

    generator = torch.Generator().manual_seed(71)
    a = torch.randn(shape, dtype=torch.complex128, generator=generator, requires_grad=True)
    weight = torch.randn(shape, dtype=torch.complex128, generator=generator)

    def joint(x):
        q, r = ar.do('linalg.qr', x)
        loss = ((q @ r)*weight.conj()).real.sum()
        return loss, torch.autograd.grad(loss, x)[0]

    with TorchLinalgConfig(mode='complex', stabilized=True, qr_rank_policy='adaptive').activated():
        graph = make_fx(joint)(a)
        compiled = torch.compile(graph, backend='eager', fullgraph=True)
        for x in (a, a*1.17, torch.zeros_like(a), a*torch.arange(shape[-1], dtype=torch.float64)):
            x = x.detach().requires_grad_()
            expected, gradient = joint(x)
            actual, captured = compiled(x.detach())
            np.testing.assert_allclose(actual.numpy(), expected.detach().numpy(), atol=1e-12)
            np.testing.assert_allclose(captured.numpy(), gradient.numpy(), atol=1e-8)
            assert torch.isfinite(captured).all()
