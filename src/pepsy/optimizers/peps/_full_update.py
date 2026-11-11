"""Two-site full update (Lubasch et al., arXiv:1405.3259, III B).

Dense Torch/CuPy implementation: QR/LQ reduction, positive norm environment,
optional environment gauges and reduced ALS or pair L-BFGS. No dense full
PEPS vector. Full-tensor L-BFGS retains the environment as a tensor network.
"""

import math

import autoray as ar
import quimb.tensor as qtn

from ...backends import infer_backend_converter_from_sample, infer_backend_signature
from ...boundary.metrics import build_bra_ket, _retune_bdy_to_chi, _as_scaled_scalar
from ...boundary._fit_policy import _FIT_QUIMB_MODES
from ...boundary.states import BdyMPS
from ...boundary.sweeps import CompBdy
from ...boundary._reuse import EnvironmentCache
from ..._internal.cutoff import resolve_fit_rtol
from ...bp._backend import all_finite, dag, einsum, transpose
from ...bp.reduced_update import (
    ReducedEnvironmentUpdateProblem, prepare_reduced_bond_pair, solve_reduced_als,
    _psd_project_metric, _svd_initial_factors, _site_physical_inds, _reduced_gate_target,
)
from ...operators.gates import _is_operator_tensor_network, _normalize_gate_which, _pair_pepo_gate_entries


def options(values=None):
    result = {'max_iterations': 50, 'rtol': 'auto', 'rcond': None, 'gauge': False,
              'gauge_qr': 'qr',
              'tensor_mode': 'reduced', 'skip_exact': True,
              'solver': 'auto', 'refine_sweeps': 0, 'refine_rtol': 'auto',
              'refine_scope': 'strip', 'refine_solver': 'quimb',
              'refine_balance': False,
              'refine_max_matrix_size': 1024,
              'accumulate_local_infidelity': True}
    values = dict(values or {})
    unknown = values.keys() - result.keys()
    if unknown:
        raise ValueError(f'Unknown full_update_kwargs: {sorted(unknown)}')
    result.update(values)
    if result['refine_scope'] not in ('strip', 'layer'):
        raise ValueError('full update refine_scope must be strip or layer')
    if result['refine_solver'] not in ('quimb', 'pinv'):
        raise ValueError('full update refine_solver must be quimb or pinv')
    if not isinstance(result['refine_balance'], bool):
        raise TypeError('full update refine_balance must be a bool')
    limit = result['refine_max_matrix_size']
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError('refine_max_matrix_size must be a positive integer or None')
    if not isinstance(result['gauge'], bool):
        raise TypeError('full update gauge must be a bool')
    if result['gauge_qr'] not in ('qr', 'qr:auto'):
        raise ValueError("full update gauge_qr must be 'qr' or 'qr:auto'")
    if result['gauge_qr'] == 'qr:auto':
        if not result['gauge']:
            raise ValueError("gauge_qr='qr:auto' requires gauge=True")
    if not isinstance(result['skip_exact'], bool):
        raise TypeError('full update skip_exact must be a bool')
    if not isinstance(result['accumulate_local_infidelity'], bool):
        raise TypeError('full update accumulate_local_infidelity must be a bool')
    if (isinstance(result['refine_sweeps'], bool)
            or not isinstance(result['refine_sweeps'], int) or result['refine_sweeps'] < 0):
        raise ValueError('full update refine_sweeps must be a nonnegative integer')
    if result['refine_rtol'] != 'auto':
        result['refine_rtol'] = resolve_fit_rtol(result['refine_rtol'])
    if result['tensor_mode'] not in ('reduced', 'full'):
        raise ValueError('full update tensor_mode must be reduced or full')
    if result['solver'] not in ('auto', 'quimb', 'qr', 'lbfgs'):
        raise ValueError('full update solver must be auto, quimb, qr or lbfgs')
    if result['tensor_mode'] == 'full':
        if result['solver'] == 'auto':
            result['solver'] = 'lbfgs'
        if result['solver'] != 'lbfgs':
            raise ValueError('full tensor mode requires solver=lbfgs')
        if result['gauge']:
            raise ValueError('full tensor mode requires gauge=False')
    if isinstance(result['max_iterations'], bool) or not isinstance(result['max_iterations'], int) or result['max_iterations'] < 1:
        raise ValueError('full update max_iterations must be a positive integer')
    for name in ('rtol', 'rcond'):
        value = result[name]
        if name == 'rtol':
            if value != 'auto':
                result[name] = resolve_fit_rtol(value)
            continue
        if value is None and name == 'rcond':
            continue
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f'full update {name} must be positive and finite')
        result[name] = float(value)
    if result['rcond'] is not None and result['rcond'] >= 1:
        raise ValueError('full update rcond must be smaller than one')
    if result['gauge_qr'] == 'qr:auto' and not callable(getattr(qtn.decomp, 'qr_auto', None)):
        raise NotImplementedError("gauge_qr='qr:auto' requires Quimb's qr_auto")
    return result


def _environment_gauge_r(matrix, method):
    """Factor an environment unfolding without changing global QR policies."""
    if method == 'qr':
        return ar.do('linalg.qr', matrix)[1]
    # ReducedPair already rejects differentiable and native-symmetry inputs.
    # Only R is consumed; Quimb chooses CholeskyQR2 or its ordinary fallback
    # and preserves dtype/device without installing a new derivative rule.
    return qtn.decomp.qr_auto(matrix, absorb='rfactor')[2]


def boundary_strip(network, *, axis, coordinate, boundary, chi, contraction_opt, boundary_kwargs):
    """Retain one exact strip between validated compressed transverse cuts."""
    kwargs = dict(boundary_kwargs)
    single_layer = kwargs.get('single_layer', False)
    def layout(tn):
        return tn.Lx, tn.Ly, infer_backend_signature(next(iter(tn)).data)
    if boundary is not None and (layout(boundary._tn_norm) != layout(network)
                                 or boundary._single_layer != single_layer):
        boundary = None
    if boundary is None:
        boundary = BdyMPS(tn_double=network, chi=chi, lazy=True, single_layer=single_layer)
        boundary.mps_b.environment_cache = EnvironmentCache()
    else:
        boundary._tn_norm = network.copy()
        block = kwargs.get('fit_mode') in {'dmrg2', 'two-site'} or kwargs.get('fit_block_size', 1) > 1
        _retune_bdy_to_chi(boundary, chi, 'full-update strip',
                          expand_growth=not (block or kwargs.get('fit_mode') in _FIT_QUIMB_MODES))
    size = network.Ly if axis == 'y' else network.Lx
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
    traversal = []
    for side, count in (('left', coordinate), ('right', size - 1 - coordinate)):
        for step in range(count):
            cache = getattr(boundary.mps_b, 'environment_cache', None)
            hits = cache.hits if cache is not None else 0
            comp.move_step_bdy(pos=step, direction=f'{axis}_{side}', n_iter=n_iter)
            traversal.append(dict(direction=f'{axis}_{side}', step=step,
                                  coordinate=step if side == 'left' else size-1-step,
                                  cut=f'{axis.upper()}{step}_{side[0]}',
                                  action='reuse' if cache is not None and cache.hits > hits else 'build'))
        if count:
            strip = strip | boundary.mps_b[f'{axis.upper()}{count - 1}_{side[0]}']
    boundary._pepsy_full_update_traversal = traversal
    return strip, boundary


def full_update_entries(gates, state, which=None, *, fuse_pepo_legs=False):
    """Fuse adjacent opposite-leg actions on one bond; never move a gate."""
    yield from _pair_pepo_gate_entries(gates, state, which, fuse_pepo_legs=fuse_pepo_legs)


class ReducedPair:
    """Keep all tensors outside the gated pair fixed, including QR/LQ factors."""

    tensor_mode = 'reduced'

    def __init__(self, state, gate, where, chi, *, which=None):
        self.state = state
        self.where = tuple(tuple(site) for site in where)
        if len(self.where) != 2 or any(len(site) != 2 for site in self.where):
            raise ValueError('full-update requires two coordinate sites')
        a, b = self.where
        if sum(abs(i - j) for i, j in zip(a, b)) != 1:
            raise ValueError('full-update requires a nearest-neighbor pair')
        self.tensors = state[a], state[b]
        sample = self.tensors[0].data
        signature = infer_backend_signature(sample)
        if (signature[0] not in {'torch', 'cupy'}
                or any(infer_backend_signature(t.data) != signature
                       or getattr(t.data, 'requires_grad', False) for t in state)):
            raise TypeError('full-update requires non-differentiable dense Torch or CuPy PEPS/PEPO '
                            'with matching backend, dtype and device')
        if ar.get_dtype_name(sample) not in ('complex64', 'complex128'):
            raise TypeError('full-update requires complex64 or complex128')
        self.operator = _is_operator_tensor_network(state)
        both = which == 'both'
        which = 'both' if both else _normalize_gate_which(which)
        if which is not None and not self.operator:
            raise ValueError('full-update which selects upper/lower PEPO legs, not PEPS legs')
        self.which = (which or 'upper') if self.operator else None
        all_physical = tuple(_site_physical_inds(state, site) for site in self.where)
        physical = tuple(_site_physical_inds(state, site, self.which) for site in self.where)
        self.physical = physical
        self.phys = tuple(index for group in all_physical for index in group)
        self.gate_phys = (tuple(getattr(state, f'{"upper" if both else self.which}_ind')(*site) for site in self.where)
                          if self.operator else self.phys)
        bonds = tuple(self.tensors[0].bonds(self.tensors[1]))
        if len(bonds) != 1:
            raise ValueError('full-update requires exactly one bond between the pair')
        self.bond = bonds[0]
        self.external = tuple(tuple(i for i in t.inds if i not in (*group, self.bond))
                              for t, group in zip(self.tensors, physical))
        self.shapes = tuple(tuple(t.ind_size(i) for i in ext)
                            for t, ext in zip(self.tensors, self.external))
        physical_dims = tuple(tuple(t.ind_size(i) for i in group)
                              for t, group in zip(self.tensors, physical))
        self.dims = tuple(math.prod(group) for group in physical_dims)
        # Reuse Pepsy's existing QR/LQ reduction without requesting BP messages.
        self.reduced = prepare_reduced_bond_pair(state, where=self.where,
                                                 input_mode='physical', run_bp=False, which=self.which)
        self.na, _, _, self.nb = self.reduced.theta_shape
        self.qa = self.reduced.q_left.transpose(*self.external[0], self.reduced.reduced_left_ind).data.reshape(-1, self.na)
        self.qb = self.reduced.q_right.transpose(self.reduced.reduced_right_ind, *self.external[1]).data.reshape(self.nb, -1)
        if both and (not isinstance(gate, (tuple, list)) or len(gate) != 2):
            raise ValueError('paired PEPO update requires upper and lower gates')
        actions = list(zip(('upper', 'lower'), gate)) if both else [(self.which, gate)]
        validated = []
        for side, payload in actions:
            inds = (tuple(getattr(state, f'{side}_ind')(*site) for site in self.where)
                    if self.operator else self.phys)
            dims = tuple(t.ind_size(p) for t, p in zip(self.tensors, inds))
            payload = infer_backend_converter_from_sample(sample)(payload)
            if getattr(payload, 'requires_grad', False) or math.prod(payload.shape) != math.prod(dims)**2:
                raise ValueError('full-update requires a fixed two-site gate of matching size')
            if not all_finite(payload):
                raise ValueError('full-update gate contains nonfinite values')
            validated.append((side, payload.reshape(*dims, *dims)))
        self.gate = tuple(g for _, g in validated) if both else validated[0][1]
        theta = _reduced_gate_target(self.reduced, self.gate, self.which)
        self.theta = transpose(theta, (0, 1, 3, 2))
        self.chi = int(chi)
        self.initial_factors = None
        self._svd_guess_factors = None

    def split(self, theta, chi):
        left, right = _svd_initial_factors(transpose(theta, (0, 1, 3, 2)), chi)
        return left, transpose(right, (0, 2, 1))

    def state_from(self, left, right):
        return self.reduced.reconstruct_tn(left, transpose(right, (0, 2, 1)))

    def balance(self, left, right):
        """Balance a rank-chi result through QR/LQ and a chi-by-chi SVD.

        ALS already supplies low-rank factors. Splitting their reconstructed
        full pair repeats a much larger SVD without changing its rank cap.
        The small core retains the symmetric singular-value bond gauge.
        """
        rank = left.shape[-1]
        qa, ra = ar.do('linalg.qr', left.reshape(-1, rank))
        qb, rb = ar.do('linalg.qr', dag(transpose(right, (0, 2, 1)).reshape(rank, -1)))
        u, s, vh = ar.do('linalg.svd', ra @ dag(rb))
        roots = ar.do('sqrt', s)
        left = (qa @ (u[:, :len(s)] * roots)).reshape(self.na, self.dims[0], len(s))
        right = ((roots[:, None] * vh[:len(s)]) @ dag(qb)).reshape(len(s), self.dims[1], self.nb)
        return left, transpose(right, (0, 2, 1))

    def initial_states(self, guess=None, *, build_target=True):
        # One local SVD supplies the warm start and exact target. Determine
        # numerical support using machine precision, never the fit tolerance:
        # a small but genuinely discarded component must still trigger FU.
        matrix = transpose(self.theta, (0, 1, 3, 2)).reshape(
            self.na * self.dims[0], self.dims[1] * self.nb)
        u, s, vh = ar.do('linalg.svd', matrix)
        eps = 2.**-23 if ar.get_dtype_name(matrix) == 'complex64' else 2.**-52
        threshold = eps * max(matrix.shape) * float(s[0])
        self.target_rank = int(ar.do('sum', s > threshold))
        self.warmstart_exact = 0 < self.target_rank <= self.chi
        rank = max(1, min(self.chi, self.target_rank))
        total = float(ar.do('sum', s**2))
        self.discarded_weight = float(ar.do('sum', s[rank:]**2)) / total if total > 0 else 0.

        def factors(count):
            roots = ar.do('sqrt', s[:count])
            left = (u[:, :count] * roots).reshape(self.na, self.dims[0], count)
            right = (roots[:, None] * vh[:count]).reshape(count, self.dims[1], self.nb)
            return left, transpose(right, (0, 2, 1))

        self.initial_factors = None
        self._svd_guess_factors = None
        if guess is None or self.warmstart_exact:
            self._svd_guess_factors = factors(rank)
            guess = self.state_from(*self._svd_guess_factors)
        else:
            # SU uses a different internal-bond gauge but the same exterior
            # tensors. Project its active sites into the frozen QR/LQ bases.
            ta, tb = (guess[site] for site in self.where)
            size = ta.ind_size(self.bond)
            a = ta.transpose(*self.external[0], *self.physical[0], self.bond).data
            b = tb.transpose(self.bond, *self.physical[1], *self.external[1]).data
            left = einsum('ax,apk->xpk', self.qa.conj(), a.reshape(-1, self.dims[0], size))
            right = einsum('kpb,bx->kxp', b.reshape(size, self.dims[1], -1), self.qb.conj().T)
            self.initial_factors = left, right
            guess = self.state_from(left, right)
        self.target_bond_dimension = int(s.shape[0])
        target = self.state_from(*factors(len(s))) if build_target else None
        return guess, target

    def exact_report(self):
        """No exterior metric is needed when the local SVD loses no support."""
        return dict(backend='full-update', tensor_mode=self.tensor_mode, solver='svd',
                    reduced_physical_dimensions=self.dims,
                    success=True, converged=True, iterations=0, warmstart_loss=0.,
                    warmstart_fidelity=1., fidelity=1., infidelity=0., loss_history=[0.],
                    termination_reason='exact_svd', environment_gauge_applied=False,
                    norm_matrix_formed=False, target_rank=self.target_rank,
                    discarded_weight=self.discarded_weight,
                    fidelity_convention='untruncated local SVD; exact to floating-point precision')

    def normalize_target(self, norm):
        """Apply the target's global scale to every subsequent reconstruction.

        Magnitude stays in the network exponent, so the reduced least-squares
        objective only changes by a common factor and exterior tensors stay
        untouched. Keep the norm's phase on the active pair.
        """
        mantissa, exponent = _as_scaled_scalar(norm)
        magnitude = abs(complex(mantissa))
        if not math.isfinite(magnitude) or magnitude == 0 or not math.isfinite(exponent):
            raise ValueError('Cannot normalize a zero or nonfinite full-update target')
        self.reduced.tn.exponent -= .5 * (math.log10(magnitude) + exponent)
        self.theta = self.theta * (complex(mantissa) / magnitude)**-.5
        if self.initial_factors is not None:
            left, right = self.initial_factors
            self.initial_factors = left * (complex(mantissa) / magnitude)**-.5, right
        if self._svd_guess_factors is not None:
            left, right = self._svd_guess_factors
            self._svd_guess_factors = left * (complex(mantissa) / magnitude)**-.5, right

    def strip_environment(self, state, *, boundary=None, chi, contraction_opt, boundary_kwargs,
                          strip_cache=None):
        """Contract the untouched strip using validated left/right boundary cuts."""
        self.contraction_opt = contraction_opt
        _, network = build_bra_ket(ket=state)
        axis = 'y' if self.where[0][1] == self.where[1][1] else 'x'
        coordinate = self.where[0][1 if axis == 'y' else 0]
        strip, boundary = boundary_strip(network, axis=axis, coordinate=coordinate,
                                         boundary=boundary, chi=chi, contraction_opt=contraction_opt,
                                         boundary_kwargs=boundary_kwargs)
        sources = getattr(network, '_pepsy_boundary_sources', None)
        if strip_cache is not None and sources is not None:
            along = 0 if axis == 'y' else 1
            strip = strip_cache.reduce(
                strip, sources=sources, axis='X' if along == 0 else 'Y',
                coordinate=coordinate, length=state.Lx if along == 0 else state.Ly,
                active=tuple(site[along] for site in self.where), optimize=contraction_opt,
            )
        for site in self.where:
            strip.delete(state.site_tag(*site))
        # Every local scalar uses this same environment scale, which cancels
        # in the target-normalized objective and fidelity.
        strip.exponent = 0.
        return strip, boundary

    def environment(self, state, **kwargs):
        strip, boundary = self.strip_environment(state, **kwargs)
        contraction_opt = self.contraction_opt
        qa_ind, qb_ind = qtn.rand_uuid(), qtn.rand_uuid()
        for site, data, inds in (
            (self.where[0], self.qa.reshape(*self.shapes[0], self.na), (*self.external[0], qa_ind)),
            (self.where[1], self.qb.reshape(self.nb, *self.shapes[1]), (qb_ind, *self.external[1])),
        ):
            q = qtn.Tensor(data, inds=inds)
            # Untouched operator legs stay on Q and trace against the bra
            # locally. Only virtual/reduced indices need distinct bra labels.
            inner = set(state.inner_inds())
            bra = q.conj().reindex({ind: f'{ind}_*' for ind in inds
                                   if ind in inner or ind in {qa_ind, qb_ind}})
            strip = strip | q | bra
        value, _ = strip.contract(all, output_inds=(f'{qa_ind}_*', f'{qb_ind}_*', qa_ind, qb_ind),
                                  optimize=contraction_opt, strip_exponent=True)
        return value.data.reshape(self.na * self.nb, self.na * self.nb), boundary

    def optimize(self, norm, controls):
        controls = options(controls)
        tolerance = resolve_fit_rtol(controls['rtol'], dtype=norm.dtype)
        tolerance = 0. if tolerance is None else tolerance
        rcond = controls['rcond'] or (1e-6 if ar.get_dtype_name(norm) == 'complex64' else 1e-12)
        if not all_finite(norm):
            raise ValueError('full-update norm environment is nonfinite')
        scale = ar.do('linalg.norm', norm.reshape(-1))
        if float(scale) == 0:
            raise ValueError('full-update norm environment is zero')
        norm = norm / scale
        projected, _, clipped, values, vectors, positive = _psd_project_metric(norm, 0., return_spectrum=True)
        if float(positive.max()) <= 0:
            raise ValueError('full-update norm environment has no positive support')
        report = {
            'norm_matrix_dimension': self.na * self.nb,
            'reduced_physical_dimensions': self.dims,
            'norm_antihermitian_relative': float(ar.do('linalg.norm', norm - dag(norm))),
            'norm_min_eigenvalue': float(values.min()),
            'norm_negative_weight': float((-ar.do('clip', values, None, 0)).sum() / positive.sum()),
            'rcond': rcond, 'environment_gauge_applied': False,
            'gauge_qr': controls['gauge_qr'] if controls['gauge'] else None,
            'norm_clipped_eigenvalues': clipped,
        }
        theta = self.theta
        # Derive identities from existing arrays, preserving their device.
        inv_a = ar.do('diag', ar.do('ones_like', norm[:self.na, 0]))
        inv_b = ar.do('diag', ar.do('ones_like', norm[:self.nb, 0]))
        ga, gb = inv_a, inv_b
        if controls['gauge']:
            root = ar.do('sqrt', positive)[:, None] * dag(vectors)
            x = root.reshape(-1, self.na, self.nb)
            # Independent leg unfoldings of the same square root (Fig. 11).
            # We store N = root.H @ root; the paper uses X @ X.H, so its
            # QR/LQ pair appears as two QR factorizations in this convention.
            ga = _environment_gauge_r(
                transpose(x, (0, 2, 1)).reshape(-1, self.na), controls['gauge_qr'])
            gb = _environment_gauge_r(x.reshape(-1, self.nb), controls['gauge_qr'])
            # Quimb's values-only driver can select NumPy for Torch arrays.
            # Keep this small gauge-conditioning check on the input device.
            sa = ar.do('linalg.svd', ga)[1]
            sb = ar.do('linalg.svd', gb)[1]
            if bool((sa[-1] > rcond * sa[0]) & (sb[-1] > rcond * sb[0])):
                inv_a, inv_b = ar.do('linalg.inv', ga), ar.do('linalg.inv', gb)
                theta = einsum('ac,bd,cpdq->apbq', ga, gb, theta)
                root = einsum('ecd,ca,db->eab', x, inv_a, inv_b).reshape(-1, self.na * self.nb)
                report['environment_gauge_applied'] = True
            else:
                ga, gb = inv_a, inv_b
            n = dag(root) @ root
        else:
            # PSD projection already formed V diag(s+) V.H. Rebuilding the
            # same matrix through its square root adds a dense product per fit.
            n = projected
        n = n.reshape(self.na, self.nb, self.na, self.nb)
        n = n / ar.do('linalg.norm', n.reshape(-1))

        def inner(a, b):
            return einsum('apbq,abcd,cpdq->', a.conj(), n, b)

        target_norm = inner(theta, theta).real
        if not all_finite(target_norm) or float(target_norm) <= 0:
            raise ValueError('full-update target has no positive environment norm')

        def loss(left, right):
            delta = einsum('apk,kbq->apbq', left, right) - theta
            value = float(inner(delta, delta).real / target_norm)
            return max(0., value) if math.isfinite(value) else math.inf

        baseline = self.initial_factors
        if baseline is None:
            baseline = self._svd_guess_factors
        if baseline is None:
            baseline = self.split(self.theta, self.chi)
        baseline_left, baseline_right = baseline
        if self.initial_factors is not None:
            left, right = self.initial_factors
            left = einsum('ac,cpk->apk', ga, left)
            right = einsum('bd,kdq->kbq', gb, right)
        elif controls['gauge']:
            left, right = self.split(theta, self.chi)
        else:
            left, right = baseline_left, baseline_right
        # Compare gauge-SVD with the ordinary split in the same environment;
        # an explicitly supplied SU start is retained in either gauge.
        baseline_left = einsum('ac,cpk->apk', ga, baseline_left)
        baseline_right = einsum('bd,kdq->kbq', gb, baseline_right)
        baseline_loss = loss(baseline_left, baseline_right)
        baseline = einsum('apk,kbq->apbq', baseline_left, baseline_right)
        baseline_norm = float(inner(baseline, baseline).real)
        baseline_fidelity = (float(abs(inner(baseline, theta)) ** 2 / (baseline_norm * target_norm))
                             if baseline_norm > 0. else None)
        if baseline_loss < loss(left, right):
            left, right = baseline_left, baseline_right
        history = [loss(left, right)]
        solution = self.solve(n, theta, left, right, controls, tolerance, rcond)
        proposed_left, proposed_right = solution.left, transpose(solution.right, (0, 2, 1))
        current = loss(proposed_left, proposed_right)
        accepted = math.isfinite(current) and current <= history[-1]
        if accepted:
            left, right = proposed_left, proposed_right
            history.append(current)
        candidate = einsum('apk,kbq->apbq', left, right)
        fidelity = abs(inner(candidate, theta)) ** 2 / (inner(candidate, candidate).real * target_norm)
        if not all_finite(fidelity):
            raise ValueError('full-update candidate has an unusable positive-environment norm')
        # Undo the environment gauges, then balance the internal bond by SVD.
        left = einsum('ac,cpk->apk', inv_a, left)
        right = einsum('bd,kdq->kbq', inv_b, right)
        if controls['gauge']:
            left, right = self.split(einsum('apk,kbq->apbq', left, right), self.chi)
        else:
            left, right = self.balance(left, right)
        residual_converged = tolerance > 0 and history[-1] <= tolerance
        reason = ('residual_tolerance' if residual_converged else
                  solution.termination_reason if accepted else 'solver_candidate_rejected')
        report.update(backend='full-update', success=True, tensor_mode=self.tensor_mode,
                      converged=bool(residual_converged or (accepted and solution.converged)),
                      warmstart_loss=baseline_loss,
                      warmstart_fidelity=baseline_fidelity,
                      solver=solution.solver, solver_candidate_accepted=accepted,
                      max_iterations=controls['max_iterations'],
                      iterations=solution.iterations, rtol=tolerance,
                      termination_reason=reason,
                      solver_converged=solution.converged,
                      solver_termination_reason=solution.termination_reason,
                      stopping_rule=('normalized residual or L-BFGS stopping criteria'
                                     if solution.solver == 'lbfgs' else
                                     'normalized residual or complete-sweep cost change'),
                      loss_history=history,
                      solver_costs=[float(c) / float(target_norm) for c in solution.costs],
                      fidelity=float(fidelity.real), infidelity=max(0., min(1., 1. - float(fidelity.real))),
                      fidelity_convention='local positive-environment estimate; not global fidelity')
        return self.state_from(left, right), report

    def solve(self, n, theta, left, right, controls, tolerance, rcond):
        if controls['solver'] == 'lbfgs':
            from ._pair_lbfgs import solve_pair_lbfgs
            return solve_pair_lbfgs(n, theta, left, right,
                                    max_iterations=controls['max_iterations'], tol=tolerance)
        pair = self.reduced
        environment = qtn.Tensor(
            transpose(n, (2, 3, 0, 1)),
            inds=(pair.reduced_left_ind, pair.reduced_right_ind,
                  pair.reduced_left_bra_ind, pair.reduced_right_bra_ind),
        )
        problem = ReducedEnvironmentUpdateProblem(
            pair=pair, environment=environment, target=transpose(theta, (0, 1, 3, 2)),
        )
        from ...tensors.contractions import build_optimizer
        policy = getattr(self, 'contraction_opt', None)
        if policy is None:
            policy = build_optimizer(progbar=False)
        return solve_reduced_als(
            problem, max_bond=self.chi, max_iterations=controls['max_iterations'],
            tol=tolerance, rcond=rcond, solver=controls['solver'],
            monitor_convergence=True,
            initial_factors=(left, transpose(right, (0, 2, 1))),
            # Quimb's automatic large-tensor route uses a SciPy LinearOperator,
            # which cannot consume native Torch/CuPy dtypes. These reduced
            # environments are already dense; keep their solves native too.
            quimb_opts={'contract_optimize': policy, 'enforce_pos': True,
                        'dense_solve': True, 'progbar': False},
        )


class FullPair(ReducedPair):
    """Fit full sites through cached scalar TNs; never form a full norm matrix.

    Reduction is used only to initialize the guess and exact diagnostic target.
    The objective retains the original pair plus its explicit gate as a TN.
    """

    tensor_mode = 'full'

    def __init__(self, state, gate, where, chi, *, which=None):
        super().__init__(state, gate, where, chi, which=which)
        self.target_phase = 1.

    def normalize_target(self, norm):
        super().normalize_target(norm)
        mantissa, _ = _as_scaled_scalar(norm)
        self.target_phase *= (complex(mantissa) / abs(complex(mantissa)))**-.5

    def environment(self, state, **kwargs):
        from ._pair_objective import PairObjective
        env, boundary = self.strip_environment(state, **kwargs)
        tags = tuple(state.site_tag(*site) for site in self.where)
        # Keep the gate lazy, separate from the frozen pre-gate site tensors.
        target = self.state.select_any(tags)
        actions = list(zip(('upper', 'lower'), self.gate)) if self.which == 'both' else [(self.which, self.gate)]
        for i, (leg, payload) in enumerate(actions):
            inds = (tuple(getattr(self.state, f'{leg}_ind')(*site) for site in self.where)
                    if self.operator else self.phys)
            target = qtn.tensor_network_gate_inds(
                target, payload * (self.target_phase if i == 0 else 1.), inds, contract=False)
        objective = PairObjective(env, tuple(state[site] for site in self.where), target,
                                  physical_inds=self.phys, optimize=self.contraction_opt)
        return objective, boundary

    def optimize(self, objective, controls):
        controls = options(dict(controls, tensor_mode='full'))
        params, report = objective.optimize(controls)
        state = self.state.copy()
        state.exponent = self.reduced.tn.exponent
        for site, data in zip(self.where, params):
            state[site].modify(data=data)
        return state, report
