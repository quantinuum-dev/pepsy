"""Array kernels for prepared dense simple update (one leading batch axis)."""

import autoray as ar


def _qr(x):
    q, r = ar.do('linalg.qr', x)
    diagonal = r.diagonal(dim1=-2, dim2=-1)
    zero = diagonal == 0
    phase = (diagonal + zero) / (diagonal.abs() + zero)
    return q * phase.conj().unsqueeze(-2), phase.unsqueeze(-1) * r


def _rank(s, cutoff, cutoff_mode, cap):
    import torch
    if cutoff <= 0:
        return s.new_full(s.shape[:-1], min(cap, s.shape[-1]), dtype=torch.int64)
    if cutoff_mode == 'abs':
        n = (s > cutoff).sum(-1)
    elif cutoff_mode == 'rel':
        n = (s > cutoff * s[..., :1]).sum(-1)
    else:
        power = 2 if cutoff_mode.endswith('2') else 1
        cumulative = (s**power).cumsum(-1)
        total = cumulative[..., -1:]
        threshold = total * (1-cutoff) if cutoff_mode.startswith('r') else total-cutoff
        n = (cumulative < threshold).sum(-1) + 1
    return n.clamp(min=1, max=cap)


def two_site(a, b, outer_a, outer_b, weight, gate, *, max_bond, cutoff, cutoff_mode,
             renorm_bonds, smudge):
    """Reduced SU for canonical (batch, exterior, bond, physical) tensors.

    Output keeps the input bond dimension only if every per-member cutoff
    decision agrees. The caller must inspect ``valid`` and replay adaptively
    when it does not. No rank/cutoff decision is silently specialized.
    """
    import torch
    from ._gauge_scale import dense_gauge_scale
    batch, na, bond, physical_a = a.shape
    nb, physical_b = b.shape[1], b.shape[-1]
    root = weight.sqrt()
    aw = a * outer_a[:, :, None, None] * root[:, None, :, None]
    bw = b * outer_b[:, :, None, None] * root[:, None, :, None]
    qa, ra = _qr(aw.reshape(batch, na, bond*physical_a))
    qb, rb = _qr(bw.reshape(batch, nb, bond*physical_b))
    ka, kb = ra.shape[1], rb.shape[1]
    core = torch.einsum('naip,nbiq,nuvpq->naubv',
        ra.reshape(batch, ka, bond, physical_a),
        rb.reshape(batch, kb, bond, physical_b),
        gate.reshape(batch, physical_a, physical_b, physical_a, physical_b))
    u, s, vh = ar.do('linalg.svd', core.reshape(batch, ka*physical_a, kb*physical_b))
    valid = _rank(s, cutoff, cutoff_mode, max_bond) == bond
    s = s[:, :bond]
    u = u[:, :, :bond].reshape(batch, ka, physical_a, bond)
    vh = vh[:, :bond, :].reshape(batch, bond, kb, physical_b)
    anew = torch.einsum('nxa,napi->nxip', qa, u)
    bnew = torch.einsum('nxb,nibq->nxiq', qb, vh)
    # Zero outer weights use the same support-projector inverse as gate_simple.
    ia = torch.where(outer_a != 0, outer_a, torch.ones_like(outer_a)).reciprocal()
    ib = torch.where(outer_b != 0, outer_b, torch.ones_like(outer_b)).reciprocal()
    anew = anew * (ia*(outer_a != 0))[:, :, None, None]
    bnew = bnew * (ib*(outer_b != 0))[:, :, None, None]
    scale = dense_gauge_scale(s, smudge) if renorm_bonds else s.new_ones((batch, 1))
    return anew, bnew, s/scale, scale.log10().squeeze(-1), valid
