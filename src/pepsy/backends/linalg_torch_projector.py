"""First-order VJP for gauge-invariant losses of retained isometric factors."""

import torch
from torch.autograd.function import once_differentiable

from .projector_split import _retained_rank


class _ProjectorSplit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, cutoff, cutoff_mode, max_bond):
        from .config import TorchLinalgConfig, get_torch_linalg_config
        from .linalg_torch import _svd_forward

        config = get_torch_linalg_config() or TorchLinalgConfig()
        # Honor the active policy's forward driver, dtype and device. This
        # primitive supplies its own composed VJP, independently of the raw
        # singular-vector/QR backward registrations.
        u, s, vh = _svd_forward(
            a, driver=config.svd_driver, cpu_svd=config.cpu_svd,
            fallback=config.resolved_svd_fallback, stabilized=True,
        )
        r = _retained_rank(u, s, vh, cutoff=cutoff,
                           cutoff_mode=cutoff_mode, max_bond=max_bond)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(u, s, vh)
        ctx.rank = r
        if r == 0:
            # A zero matrix has no unique retained subspace. Its squared-norm
            # or zero-overlap loss still has a zero VJP; reject nonzero incoming
            # cotangents in backward rather than inventing a derivative.
            return torch.eye(a.shape[0], 1, dtype=a.dtype, device=a.device), a.new_zeros((1, a.shape[1]))
        return u[:, :r], s[:r, None] * vh[:r]

    @staticmethod
    @once_differentiable
    def backward(ctx, gq, gb):
        u, s, vh = ctx.saved_tensors
        r = ctx.rank
        if r == 0:
            if any(g is not None and bool((g != 0).any()) for g in (gq, gb)):
                raise RuntimeError("zero-matrix projector split has no differentiable subspace chart")
            return u.new_zeros((u.shape[0], vh.shape[1])), None, None, None

        q, sr, vr = u[:, :r], s[:r], vh[:r]
        gq = torch.zeros_like(q) if gq is None else gq
        gb = torch.zeros_like(vr) if gb is None else gb
        b = sr[:, None] * vr
        vertical = q.mH @ gq + b @ gb.mH
        # The composed rule cannot differentiate arbitrary singular-vector
        # observables. A vertical unitary rotation must have zero cotangent.
        # Use a matrix-product roundoff allowance, not a spectrum regularizer.
        scale = (q.abs().mT @ gq.abs() + b.abs() @ gb.abs().mT).amax()
        tolerance = 64 * torch.finfo(s.dtype).eps * (q.shape[0] + b.shape[1]) * scale
        if bool((vertical - vertical.mH).abs().amax() > tolerance):
            raise RuntimeError("projector split requires a gauge-invariant loss of its paired factors")
        result = q @ gb

        # Choose parallel transport within the retained subspace: Q^H dQ=0.
        # Internal rotations cancel in a gauge-invariant tensor-network loss,
        # including when retained singular values repeat. Only the gap between
        # retained and discarded subspaces enters the derivative.
        ud, sd, vd = u[:, r:], s[r:], vh[r:]
        if sd.numel():
            ratio = sd[:, None] / sr[None, :]
            gap = (1 - ratio) * (1 + ratio)
            if bool((gap <= max(u.shape[0], vh.shape[1]) * torch.finfo(s.dtype).eps).any()):
                raise RuntimeError("projector split has no resolved kept/discarded singular-value gap")
            # Evaluate U_d^H (gQ + A gB^H) spectrally. Forming A gB^H first
            # and subtracting/projecting large terms would amplify roundoff
            # when the retained spectrum includes small resolved values.
            h = ud.mH @ gq + sd[:, None] * (vd @ gb.mH)
            k = (h / sr[None, :]) / gap
            result = result + (ud @ k) @ vr + q @ (k * ratio).mH @ vd
        if u.shape[0] > u.shape[1]:
            # The remaining left-null space is outside the thin SVD. A's
            # contribution there vanishes algebraically, avoiding cancellation.
            h = gq - u @ (u.mH @ gq)
            result = result + (h / sr[None, :]) @ vr
        return result, None, None, None


def torch_projector_split(a, cutoff, cutoff_mode, max_bond):
    """Dense isometric factors with a composed, first-order subspace VJP."""
    return _ProjectorSplit.apply(a, cutoff, cutoff_mode, max_bond)
