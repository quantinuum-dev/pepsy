"""Cached scalar pair objectives with fixed environments and explicit gates."""

import math

import autoray as ar
import cotengra as ctg
import quimb.tensor as qtn

from ..._internal.cutoff import resolve_fit_rtol
from ...backends import infer_backend_converter_from_sample
from ...solvers.gradient import GradientOptimizer


class PairObjective:
    """Own one gate's constants; reuse only the two variable site arrays.

    Cross-gate reuse belongs to the version-checked boundary/strip caches.
    Expressions and folded constants here live for one frozen local solve,
    so a different gate or environment cannot reuse stale numeric values.
    """

    def __init__(self, environment, tensors, target, *, physical_inds, optimize):
        import torch

        self.templates = tensors
        sample = tensors[0].data
        self.to_native = infer_backend_converter_from_sample(sample)
        if ar.infer_backend(sample) == 'cupy':
            sample = torch.empty((), dtype=getattr(torch, str(sample.dtype)),
                                 device=f'cuda:{sample.device.id}')
        convert = infer_backend_converter_from_sample(sample)
        # Snapshot constants and starting parameters, including conjugate views.
        def snapshot(x):
            return convert(x).detach().resolve_conj().clone()

        env = environment.copy()
        env.exponent = 0.
        env.apply_to_arrays(snapshot)
        target = target.copy()
        target.exponent = 0.
        target.apply_to_arrays(snapshot)
        self.params = {str(i): snapshot(t.data) for i, t in enumerate(tensors)}
        self.roundoff = 2e-5 if sample.dtype == torch.complex64 else 1e-10
        self.dtype = sample.dtype
        self.target_tensor_count = target.num_tensors
        self.evaluations = 0
        self.expression_builds = 0
        self.constant_counts = []
        self.largest_intermediates = []

        def bra(tn):
            return tn.conj().reindex({ix: f'{ix}_*' for ix in tn.ind_map
                                     if ix not in physical_inds})

        active = qtn.TensorNetwork([t.copy() for t in tensors])
        for t, array in zip(active, self.params.values()):
            t.modify(data=array)
        # Fixed target bra/ket bonds must remain disjoint from candidate bonds.
        target.reindex_({ix: qtn.rand_uuid() for ix in target.inner_inds()})
        target_bra = bra(target)
        self.target_norm = (target | env | target_bra).contract(all, optimize=optimize)
        self._check_norm(self.target_norm, 'target')
        self.target_norm = self.target_norm.real.detach()

        active_bra = bra(active)
        self.norm_expr = self._expression(
            tuple(active) + tuple(active_bra), tuple(env), optimize,
        )
        self.overlap_expr = self._expression(
            tuple(active_bra), tuple(env) + tuple(target), optimize,
        )

    def _expression(self, variables, constants, optimize):
        tensors = (*variables, *constants)
        inputs = tuple(t.inds for t in tensors)
        sizes = {ix: size for t in tensors for ix, size in zip(t.inds, t.shape)}
        tree = ctg.array_contract_tree(inputs, (), sizes, optimize=optimize)
        self.expression_builds += 1
        self.constant_counts.append(len(constants))
        self.largest_intermediates.append(tree.max_size())
        return ctg.array_contract_expression(
            inputs, (), sizes, optimize=tree,
            constants={i: t.data for i, t in enumerate(tensors) if i >= len(variables)},
            implementation='autoray', autojit=False,
        )

    def _check_norm(self, value, role):
        z = complex(value.detach())
        if (not math.isfinite(z.real) or not math.isfinite(z.imag) or z.real <= 0
                or abs(z.imag) > self.roundoff * z.real):
            raise FloatingPointError(f'full-tensor {role} norm is nonpositive, nonfinite or nonreal')

    def evaluate(self, params):
        a, b = params['0'], params['1']
        norm = self.norm_expr(a, b, a.conj(), b.conj())
        overlap = self.overlap_expr(a.conj(), b.conj())
        self.evaluations += 1
        self._check_norm(norm, 'candidate')
        loss = (norm.real - 2 * overlap.real + self.target_norm) / self.target_norm
        fidelity = abs(overlap)**2 / (norm.real * self.target_norm)
        loss_value, fid_value = float(loss.detach()), float(fidelity.detach())
        if (not math.isfinite(loss_value) or not math.isfinite(fid_value)
                or loss_value < -self.roundoff or fid_value > 1. + self.roundoff):
            raise FloatingPointError('full-tensor environment gives an invalid residual or fidelity')
        return loss, fidelity

    def loss(self, params):
        return self.evaluate(params)[0]

    def optimize(self, controls):
        tolerance = resolve_fit_rtol(controls['rtol'], dtype=self.dtype) or 0.
        initial, initial_fidelity = self.evaluate(self.params)
        initial, initial_fidelity = float(initial), float(initial_fidelity)
        params = self.params
        history = []
        reason = 'residual_tolerance'
        iterations = 0
        if not (tolerance > 0 and initial <= tolerance):
            # Same shared native-autodiff solver used by SweepOptimizer.
            result = GradientOptimizer(
                solver='lbfgs', n_steps=controls['max_iterations'],
                options={'ftol': tolerance, 'gtol': tolerance, 'restore_best': True,
                         'assume_nonnegative': True, 'best_neg_tol': self.roundoff},
            ).run(params_init=self.params, loss_fn=self.loss)
            params, history = result.params, result.history
            reason, iterations = result.convergence_reason, result.n_steps
        try:
            final, fidelity = self.evaluate(params)
            final, fidelity = float(final), float(fidelity)
            accepted = final <= initial
        except (FloatingPointError, RuntimeError):
            accepted = False
        if not accepted:
            params, final, fidelity = self.params, initial, initial_fidelity
            reason = 'solver_candidate_rejected'
        converged = ((tolerance > 0 and final <= tolerance)
                     or (accepted and reason.startswith('CONVERGENCE')))
        report = dict(
            backend='full-update', tensor_mode='full', solver='lbfgs',
            autodiff_backend='torch', success=True, converged=converged,
            warmstart_loss=max(0., initial), warmstart_fidelity=initial_fidelity,
            loss_history=[max(0., initial), max(0., final)], solver_costs=history,
            solver_candidate_accepted=accepted, solver_converged=converged,
            termination_reason=reason, solver_termination_reason=reason,
            iterations=iterations, max_iterations=controls['max_iterations'], rtol=tolerance,
            stopping_rule='normalized residual or L-BFGS stopping criteria',
            environment_gauge_applied=False, norm_matrix_formed=False,
            fidelity=fidelity, infidelity=max(0., min(1., 1.-fidelity)),
            fidelity_convention='local fixed-environment TN estimate; not global fidelity',
            contraction_cache=dict(expression_builds=self.expression_builds,
                                   constant_tensors=self.constant_counts,
                                   target_tensors=self.target_tensor_count,
                                   largest_intermediate_elements=self.largest_intermediates,
                                   target_norm_evaluations=1, evaluations=self.evaluations),
        )
        return tuple(self.to_native(params[str(i)].detach()) for i in range(2)), report
