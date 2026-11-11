"""Opt-in batches of independent, shape-compatible dense Torch ALS problems.

Each member retains its own conditioning decision, stopping and rank. Public
TorchLinalgConfig dispatch owns the batched QR/SVD forward and backward rules.
"""

import math


def _solve(gram, rhs, rcond):
    """Apply the sequential Cholesky certificate independently to each member."""
    import torch

    gram = (gram + gram.mH) * .5
    # Certification is a discrete decision. Keep failed Cholesky factors out
    # of autograd entirely: even a zero cotangent can yield NaN in their VJP.
    with torch.no_grad():
        factor, info = torch.linalg.cholesky_ex(gram, check_errors=False)
    identity = torch.eye(gram.shape[-1], dtype=gram.dtype, device=gram.device)
    # Failed Cholesky factors must never enter the triangular solve/backward.
    factor = torch.where((info == 0)[:, None, None], factor, identity)
    inverse = torch.linalg.solve_triangular(factor, identity.expand_as(gram), upper=False)
    trace = gram.diagonal(dim1=-2, dim2=-1).sum(-1).real
    floor = max(rcond, 1e-15)
    certificate = trace * inverse.abs().square().sum((-2, -1)) * floor
    probe = torch.cholesky_solve(rhs.detach(), factor)
    residual = torch.linalg.vector_norm(gram.detach() @ probe - rhs.detach(), dim=(-2, -1))
    denominator = (torch.linalg.vector_norm(gram.detach(), dim=(-2, -1))
                   * torch.linalg.vector_norm(probe, dim=(-2, -1))
                   + torch.linalg.vector_norm(rhs.detach(), dim=(-2, -1)))
    safe = ((info == 0) & torch.isfinite(certificate) & (trace > 0)
            & (certificate < .5) & torch.isfinite(probe).all((-2, -1))
            & (residual <= 128*torch.finfo(gram.real.dtype).eps*gram.shape[-1]*denominator))
    decisions = safe.detach().cpu().tolist()
    solution = torch.zeros_like(rhs)
    good = [i for i, valid in enumerate(decisions) if valid]
    if good:
        indices = torch.tensor(good, device=gram.device)
        factors = (torch.linalg.cholesky(gram[indices]) if gram.requires_grad
                   else factor[indices])
        solution = solution.index_copy(0, indices, torch.cholesky_solve(rhs[indices], factors))
    bad = [i for i, valid in enumerate(decisions) if not valid]
    if bad:
        indices = torch.tensor(bad, device=gram.device)
        values, vectors = torch.linalg.eigh(gram[indices])
        maximum = values.amax(-1)
        if not bool((torch.isfinite(values).all() & (maximum > 0).all()).detach()):
            raise ValueError('local ALS matrix has no finite positive support')
        values = values.clamp_min(maximum[:, None]*floor)
        fallback = vectors @ ((vectors.mH @ rhs[indices])/values[:, :, None])
        solution = solution.index_copy(0, indices, fallback)
    return solution, decisions


def _balance(left, right):
    import autoray as ar
    import torch

    count, a, p, rank = left.shape
    _, _, q, b = right.shape
    left_q, left_r = ar.do('linalg.qr', left.reshape(count, a*p, rank))
    right = torch.einsum('nst,ntqb->nsqb', left_r, right)
    right_q, right_r = ar.do('linalg.qr', right.reshape(count, rank, q*b).mH)
    left = left_q @ right_r.mH
    return left.reshape(count, a, p, rank), right_q.mH.reshape(count, rank, q, b)


def solve(problems, *, max_bond, max_iterations, tol, measure_fidelity, rcond=1e-12):
    """Fit compatible retained environments with independent complete-sweep stops."""
    import autoray as ar
    import torch
    from ._reduced_torch import _left_equations, _objective, _right_equations, _weighted_target
    from .reduced_update import ReducedALSSolution, _reduced_als_stop, _validated_local_fidelity

    target = torch.stack([p.target for p in problems])
    environment = torch.stack([p.environment.data for p in problems])
    count, a, p, q, b = target.shape
    u, singular, vh = ar.do('linalg.svd', target.reshape(count, a*p, q*b))
    rank = min(max_bond, singular.shape[-1])
    root = singular[:, :rank].sqrt()
    left = (u[:, :, :rank]*root[:, None, :]).reshape(count, a, p, rank)
    right = (root[:, :, None]*vh[:, :rank]).reshape(count, rank, q, b)
    weighted = _weighted_target(environment, target)
    norms = (target.conj()*weighted).sum((-4, -3, -2, -1)).real.detach().cpu().tolist()
    if any(not math.isfinite(x) or x <= 0 for x in norms):
        raise ValueError('Torch ALS requires a positive finite target norm')
    initial = _objective(environment, target, left, right)[0].detach().cpu().tolist()
    costs = [[value] for value in initial]
    counts = [dict(cholesky=0, eigh=0) for _ in problems]
    reasons = ['max_iterations']*count
    active = list(range(count))
    for _ in range(max_iterations):
        ids = torch.tensor(active, device=target.device)
        env, weight = environment[ids], weighted[ids]
        gram, rhs = _left_equations(env, weight, right[ids])
        solved, left_decisions = _solve(gram, rhs, rcond)
        lnew = solved.reshape(len(active), a, rank, p).transpose(-2, -1)
        gram, rhs = _right_equations(env, weight, lnew)
        solved, right_decisions = _solve(gram, rhs, rcond)
        rnew = solved.reshape(len(active), rank, b, q).transpose(-2, -1)
        lnew, rnew = _balance(lnew, rnew)
        left, right = left.index_copy(0, ids, lnew), right.index_copy(0, ids, rnew)
        values = _objective(env, target[ids], lnew, rnew)[0].detach().cpu().tolist()
        remaining = []
        for offset, i in enumerate(active):
            for decisions in (left_decisions, right_decisions):
                counts[i]['cholesky' if decisions[offset] else 'eigh'] += 1
            costs[i].append(values[offset])
            reason = _reduced_als_stop(costs[i][-2], costs[i][-1], norms[i], tol)
            if reason is None:
                remaining.append(i)
            else:
                reasons[i] = reason
        active = remaining
        if not active:
            break
    fidelities = [(None, 'not_requested')]*count
    if measure_fidelity:
        _, theta, delta = _objective(environment, target, left, right)
        out_norms = (theta.conj()*(weighted+delta)).sum((-4, -3, -2, -1)).real
        overlaps = (theta.conj()*weighted).sum((-4, -3, -2, -1)).abs()
        scalars = torch.stack((out_norms, overlaps), -1).detach().cpu().tolist()
        fidelities = [_validated_local_fidelity(n, norms[i], z, torch.finfo(target.real.dtype).eps)
                      for i, (n, z) in enumerate(scalars)]
    return tuple(ReducedALSSolution(
        left=left[i], right=right[i], costs=tuple(costs[i]), solver='torch',
        iterations=len(costs[i])-1,
        converged=reasons[i] not in {'max_iterations', 'nonfinite_cost'},
        termination_reason=reasons[i], local_fidelity=fidelities[i][0],
        local_infidelity=None if fidelities[i][0] is None else 1-fidelities[i][0],
        fidelity_status=fidelities[i][1], solver_report=dict(**counts[i], torch_backend=None,
            environment='open reduced tensor; independent cluster batch'),
        target_norm=norms[i]) for i in range(count))
