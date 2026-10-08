"""Reduced two-site full update (Lubasch et al., arXiv:1405.3259, III B).

Dense Torch implementation: QR/LQ reduction, positive norm environment,
environment gauges and alternating least squares. No dense full PEPS vector.
"""

import math

import quimb.tensor as qtn

from ...boundary.metrics import build_bra_ket, _retune_bdy_to_chi
from ...boundary._fit_policy import _FIT_QUIMB_MODES
from ...boundary.states import BdyMPS
from ...boundary.sweeps import CompBdy
from ...boundary._reuse import EnvironmentCache
from ...bp.reduced_update import (
    ReducedEnvironmentUpdateProblem, prepare_reduced_bond_pair, solve_reduced_als,
    _psd_project_metric, _svd_initial_factors,
)


def options(values=None):
    result = {'max_iterations': 50, 'rtol': 1e-9, 'rcond': None, 'gauge': True,
              'solver': 'auto'}
    values = dict(values or {})
    unknown = values.keys() - result.keys()
    if unknown:
        raise ValueError(f'Unknown full_update_kwargs: {sorted(unknown)}')
    result.update(values)
    if not isinstance(result['gauge'], bool):
        raise TypeError('full update gauge must be a bool')
    if result['solver'] not in ('auto', 'quimb', 'qr'):
        raise ValueError('full update solver must be auto, quimb or qr')
    if isinstance(result['max_iterations'], bool) or not isinstance(result['max_iterations'], int) or result['max_iterations'] < 1:
        raise ValueError('full update max_iterations must be a positive integer')
    for name in ('rtol', 'rcond'):
        value = result[name]
        if value is None and name == 'rcond':
            continue
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f'full update {name} must be positive and finite')
        result[name] = float(value)
    if result['rcond'] is not None and result['rcond'] >= 1:
        raise ValueError('full update rcond must be smaller than one')
    return result


class ReducedPair:
    """Keep all tensors outside the gated pair fixed, including QR/LQ factors."""

    def __init__(self, state, gate, where, chi):
        import torch

        self.state = state
        self.where = tuple(tuple(site) for site in where)
        if len(self.where) != 2 or any(len(site) != 2 for site in self.where):
            raise ValueError('full-update requires two coordinate sites')
        a, b = self.where
        if sum(abs(i - j) for i, j in zip(a, b)) != 1:
            raise ValueError('full-update requires a nearest-neighbor pair')
        self.tensors = state[a], state[b]
        if any(not isinstance(t.data, torch.Tensor) or t.data.requires_grad for t in state):
            raise TypeError('full-update currently supports non-differentiable dense Torch PEPS')
        if any(t.data.dtype not in (torch.complex64, torch.complex128) for t in state):
            raise TypeError('full-update requires Torch complex64 or complex128')
        self.phys = tuple(state.site_ind(*site) for site in self.where)
        bonds = tuple(self.tensors[0].bonds(self.tensors[1]))
        if len(bonds) != 1:
            raise ValueError('full-update requires exactly one bond between the pair')
        self.bond = bonds[0]
        self.external = tuple(tuple(i for i in t.inds if i not in (p, self.bond))
                              for t, p in zip(self.tensors, self.phys))
        self.shapes = tuple(tuple(t.ind_size(i) for i in ext)
                            for t, ext in zip(self.tensors, self.external))
        self.dims = tuple(t.ind_size(p) for t, p in zip(self.tensors, self.phys))
        # Reuse Pepsy's existing QR/LQ reduction without requesting BP messages.
        self.reduced = prepare_reduced_bond_pair(state, where=self.where,
                                                 input_mode='physical', run_bp=False)
        self.na, _, _, self.nb = self.reduced.theta_shape
        self.qa = self.reduced.q_left.transpose(*self.external[0], self.reduced.reduced_left_ind).data.reshape(-1, self.na)
        self.qb = self.reduced.q_right.transpose(self.reduced.reduced_right_ind, *self.external[1]).data.reshape(self.nb, -1)
        sample = self.tensors[0].data
        gate = torch.as_tensor(gate, dtype=sample.dtype, device=sample.device)
        if gate.requires_grad or gate.numel() != math.prod(self.dims) ** 2:
            raise ValueError('full-update requires a fixed two-site gate of matching size')
        if not bool(torch.isfinite(gate).all()):
            raise ValueError('full-update gate contains nonfinite values')
        gate = gate.reshape(*self.dims, *self.dims)
        self.theta = torch.einsum('PQpq,apqb->aPbQ', gate, self.reduced.theta_array())
        self.chi = int(chi)

    def split(self, theta, chi):
        left, right = _svd_initial_factors(theta.permute(0, 1, 3, 2), chi)
        return left, right.permute(0, 2, 1)

    def state_from(self, left, right):
        return self.reduced.reconstruct_tn(left, right.permute(0, 2, 1))

    def initial_states(self):
        guess = self.state_from(*self.split(self.theta, self.chi))
        target = self.state_from(*self.split(self.theta, min(self.na * self.dims[0], self.nb * self.dims[1])))
        return guess, target

    def environment(self, state, *, boundary=None, chi, contraction_opt, boundary_kwargs):
        """Contract the untouched strip using validated left/right boundary cuts."""
        self.contraction_opt = contraction_opt
        _, network = build_bra_ket(ket=state)
        kwargs = dict(boundary_kwargs)
        single_layer = kwargs.get('single_layer', False)
        def layout(tn):
            data = next(iter(tn)).data
            return tn.Lx, tn.Ly, str(data.dtype), str(data.device)
        if boundary is not None and (layout(boundary._tn_norm) != layout(network)
                                     or boundary._single_layer != single_layer):
            boundary = None
        if boundary is None:
            boundary = BdyMPS(tn_double=network, chi=chi, lazy=True, single_layer=single_layer)
            boundary.mps_b.environment_cache = EnvironmentCache()
        else:
            boundary._tn_norm = network.copy()
            block = kwargs.get('fit_mode') in {'dmrg2', 'two-site'} or kwargs.get('fit_block_size', 1) > 1
            _retune_bdy_to_chi(boundary, chi, 'full-update norm',
                              expand_growth=not (block or kwargs.get('fit_mode') in _FIT_QUIMB_MODES))
        axis = 'y' if self.where[0][1] == self.where[1][1] else 'x'
        coordinate = self.where[0][1 if axis == 'y' else 0]
        size = state.Ly if axis == 'y' else state.Lx
        n_iter = kwargs.pop('n_iter', 10)
        allowed = ('fit_mode', 'fit_layer_mode', 'fit_layer_order', 'layer_tags',
                   'fit_init_strategy', 'fit_init_seed', 'fit_block_size',
                   'fit_adaptive_sweeps', 'fit_sweep_sequence', 'fit_cutoff_mode',
                   'fit_compression_opts', 'fit_min_iter', 'fit_rtol', 'fit_patience')
        comp_opts = {key: value for key, value in kwargs.items() if key in allowed}
        comp_opts['fit_cutoff'] = kwargs.get('cutoff', 'auto')
        comp = CompBdy(network, boundary.mps_b, contraction_opt=contraction_opt,
                      fit_max_bond=kwargs.get('fit_max_bond') or chi, **comp_opts)
        strip = network.select(f'{axis.upper()}{coordinate}')
        for side, count in (('left', coordinate), ('right', size - 1 - coordinate)):
            for step in range(count):
                comp.move_step_bdy(pos=step, direction=f'{axis}_{side}', n_iter=n_iter)
            if count:
                strip = strip | boundary.mps_b[f'{axis.upper()}{count - 1}_{side[0]}']
        qa_ind, qb_ind = qtn.rand_uuid(), qtn.rand_uuid()
        for site, data, inds in (
            (self.where[0], self.qa.reshape(*self.shapes[0], self.na), (*self.external[0], qa_ind)),
            (self.where[1], self.qb.reshape(self.nb, *self.shapes[1]), (qb_ind, *self.external[1])),
        ):
            site_tag = state.site_tag(*site)
            strip.delete(site_tag)
            q = qtn.Tensor(data, inds=inds)
            bra = q.conj().reindex({ind: f'{ind}_*' for ind in inds})
            strip = strip | q | bra
        value, _ = strip.contract(all, output_inds=(f'{qa_ind}_*', f'{qb_ind}_*', qa_ind, qb_ind),
                                  optimize=contraction_opt, strip_exponent=True)
        return value.data.reshape(self.na * self.nb, self.na * self.nb), boundary

    def optimize(self, norm, controls):
        import torch

        controls = options(controls)
        real_dtype = norm.real.dtype
        rcond = controls['rcond'] or (1e-6 if real_dtype == torch.float32 else 1e-12)
        if not bool(torch.isfinite(norm).all()):
            raise ValueError('full-update norm environment is nonfinite')
        scale = torch.linalg.vector_norm(norm)
        if float(scale) == 0:
            raise ValueError('full-update norm environment is zero')
        norm = norm / scale
        _, _, clipped, values, vectors, positive = _psd_project_metric(norm, 0., return_spectrum=True)
        if float(positive.max()) <= 0:
            raise ValueError('full-update norm environment has no positive support')
        root = positive.sqrt()[:, None] * vectors.mH
        report = {
            'norm_antihermitian_relative': float(torch.linalg.vector_norm(norm - norm.mH)),
            'norm_min_eigenvalue': float(values.min()),
            'norm_negative_weight': float((-values.clamp_max(0)).sum() / positive.sum()),
            'rcond': rcond, 'environment_gauge_applied': False,
            'norm_clipped_eigenvalues': clipped,
        }
        theta = self.theta
        inv_a = torch.eye(self.na, dtype=norm.dtype, device=norm.device)
        inv_b = torch.eye(self.nb, dtype=norm.dtype, device=norm.device)
        ga, gb = inv_a, inv_b
        if controls['gauge']:
            x = root.reshape(-1, self.na, self.nb)
            _, ga = torch.linalg.qr(x.permute(0, 2, 1).reshape(-1, self.na), mode='reduced')
            _, gb = torch.linalg.qr(x.reshape(-1, self.nb), mode='reduced')
            sa, sb = torch.linalg.svdvals(ga), torch.linalg.svdvals(gb)
            if bool((sa[-1] > rcond * sa[0]) & (sb[-1] > rcond * sb[0])):
                inv_a, inv_b = torch.linalg.inv(ga), torch.linalg.inv(gb)
                theta = torch.einsum('ac,bd,cpdq->apbq', ga, gb, theta)
                root = torch.einsum('ecd,ca,db->eab', x, inv_a, inv_b).reshape(-1, self.na * self.nb)
                report['environment_gauge_applied'] = True
            else:
                ga, gb = inv_a, inv_b
        n = (root.mH @ root).reshape(self.na, self.nb, self.na, self.nb)
        n = n / torch.linalg.vector_norm(n)

        def inner(a, b):
            return torch.einsum('apbq,abcd,cpdq->', a.conj(), n, b)

        target_norm = inner(theta, theta).real
        if not bool(torch.isfinite(target_norm)) or float(target_norm) <= 0:
            raise ValueError('full-update target has no positive environment norm')

        def loss(left, right):
            delta = torch.einsum('apk,kbq->apbq', left, right) - theta
            value = float(inner(delta, delta).real / target_norm)
            return max(0., value) if math.isfinite(value) else math.inf

        left, right = self.split(theta, self.chi)
        # Gauge-SVD is usually a better initial guess, but retain the ordinary
        # rank-D split if it is better in this same positive environment.
        baseline_left, baseline_right = self.split(self.theta, self.chi)
        baseline_left = torch.einsum('ac,cpk->apk', ga, baseline_left)
        baseline_right = torch.einsum('bd,kdq->kbq', gb, baseline_right)
        baseline_loss = loss(baseline_left, baseline_right)
        if baseline_loss < loss(left, right):
            left, right = baseline_left, baseline_right
        history = [loss(left, right)]
        pair = self.reduced
        environment = qtn.Tensor(
            n.permute(2, 3, 0, 1),
            inds=(pair.reduced_left_ind, pair.reduced_right_ind,
                  pair.reduced_left_bra_ind, pair.reduced_right_bra_ind),
        )
        problem = ReducedEnvironmentUpdateProblem(
            pair=pair, environment=environment, target=theta.permute(0, 1, 3, 2),
        )
        from ...tensors.contractions import build_optimizer
        policy = getattr(self, 'contraction_opt', None)
        if policy is None:
            policy = build_optimizer(progbar=False)
        solution = solve_reduced_als(
            problem, max_bond=self.chi, max_iterations=controls['max_iterations'],
            tol=controls['rtol'], rcond=rcond, solver=controls['solver'],
            quimb_opts={'contract_optimize': policy, 'enforce_pos': True, 'progbar': False},
        )
        proposed_left, proposed_right = solution.left, solution.right.permute(0, 2, 1)
        current = loss(proposed_left, proposed_right)
        accepted = math.isfinite(current) and current <= history[-1]
        if accepted:
            left, right = proposed_left, proposed_right
            history.append(current)
        candidate = torch.einsum('apk,kbq->apbq', left, right)
        fidelity = abs(inner(candidate, theta)) ** 2 / (inner(candidate, candidate).real * target_norm)
        if not bool(torch.isfinite(fidelity)):
            raise ValueError('full-update candidate has an unusable positive-environment norm')
        # Undo the environment gauges, then balance the internal bond by SVD.
        left = torch.einsum('ac,cpk->apk', inv_a, left)
        right = torch.einsum('bd,kdq->kbq', inv_b, right)
        left, right = self.split(torch.einsum('apk,kbq->apbq', left, right), self.chi)
        report.update(backend='full-update', success=True, converged=None,
                      warmstart_loss=baseline_loss,
                      solver=solution.solver, solver_candidate_accepted=accepted,
                      max_iterations=controls['max_iterations'],
                      iterations=None if solution.solver == 'quimb' else (len(solution.costs) - 1) // 2,
                      loss_history=history,
                      solver_costs=[float(c) / float(target_norm) for c in solution.costs],
                      fidelity=float(fidelity.real), infidelity=max(0., min(1., 1. - float(fidelity.real))),
                      fidelity_convention='local positive-environment estimate; not global fidelity')
        return self.state_from(left, right), report
