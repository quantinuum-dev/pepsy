"""Fit one fixed, untruncated PEPO application through the existing solvers."""

from copy import deepcopy

from ...operators.gates import _resolve_gate_cutoff, _resolve_gate_cutoff_mode
from .._boundary_control import (
    BoundaryAccuracyError, BoundaryGrowth, boundary_growth_scope, chi_pair, merge_options,
)
from ..sweep.environments import uses_symmray_arrays
from ._timing import profile_run


class PepoFitMixin:
    """The gate-stream driver also owns one complete operator step per fit."""

    @profile_run
    @boundary_growth_scope
    def run_pepo(
        self, pepo, *, mode=None, unitary=False, normalize_chi=None, evaluation_chi=None,
        cutoff='auto', cutoff_mode='auto',
        normalize_initial=None, infidelity_tol='auto', progress=False,
        normalize_kwargs=None, infidelity_kwargs=None,
        sweep_kwargs=None, sweep_optimize_kwargs=None,
        global_kwargs=None, global_optimize_kwargs=None,
        accept_if_improved=None, improvement_tol=0., reset_traces=True,
        timing=False, timing_sync_device=False, step_callback=None,
    ):
        """Apply one full PEPO and fit its lazy exact target at PEPS cap ``chi``.

        The operator is never compressed here. Its site factors remain separate
        from the input ket; only a disposable warm start is contracted and
        compressed. Sweep mode uses ALS or column L-BFGS with DMRG/FIT boundaries.
        Global mode defaults to Pepsy's direct BRA--PEPO--KET boundaries.
        The queued gate stream is unused and preserved. Returned state becomes
        the input for the next call. Norm/overlap caps remain independent.

        ``unitary=True`` trusts the operator's unitarity and reuses the measured
        input PEPS norm instead of contracting the full target self-overlap.
        Supply ``infidelity_kwargs={'norm_target': 1.0}`` for an input known to
        have exact unit norm, avoiding that measurement as well.

        Sweep ALS uses the configured bounded failure-growth policy (automatic
        by default); global fitting retains fixed caps. Each attempt measures
        the target norm once and reuses it for fitting and acceptance. Dense
        arrays are supported; native symmetry needs separate validation.
        """
        del timing, timing_sync_device  # Consumed by the profiling decorator.
        run_mode = self.mode if mode is None else self._normalize_mode(mode)
        if run_mode not in {'sweep', 'global'}:
            raise ValueError('run_pepo requires sweep ALS or global mode')
        solver = self._apply_sweep_optimizer_options(merge_options(
            self.sweep_optimize_kwargs, sweep_optimize_kwargs))['optimizer']
        if run_mode == 'sweep' and str(solver).lower() not in {'als', 'scipy'}:
            raise ValueError('run_pepo sweep mode requires ALS or scipy column L-BFGS')
        if uses_symmray_arrays(self.state, pepo):
            raise NotImplementedError('run_pepo currently supports dense PEPS/PEPO arrays')
        if step_callback is not None and not callable(step_callback):
            raise TypeError('step_callback must be callable or None')
        if (getattr(pepo, 'Lx', None), getattr(pepo, 'Ly', None)) != (self.state.Lx, self.state.Ly):
            raise ValueError('PEPO and PEPS must have the same lattice shape')
        if not callable(getattr(pepo, 'apply', None)):
            raise TypeError('pepo must be a Quimb PEPO operator')
        for site in self.state.gen_site_coos():
            dim = self.state.ind_size(self.state.site_ind(*site))
            if (pepo.ind_size(pepo.upper_ind(*site)) != dim
                    or pepo.ind_size(pepo.lower_ind(*site)) != dim):
                raise ValueError('PEPO physical dimensions must match the PEPS')
        self._active_full_update = False
        cutoff = _resolve_gate_cutoff(self.state, cutoff)
        cutoff_mode = _resolve_gate_cutoff_mode(cutoff_mode)
        infidelity_tol = self._resolve_infidelity_tol(infidelity_tol)
        accept_if_improved = (self.accept_if_improved if accept_if_improved is None
                              else bool(accept_if_improved))
        norm_cap = self._boundary_chi_for_norm(normalize_chi, options=normalize_kwargs)
        metric_cap = self._boundary_chi_for_infidelity(evaluation_chi, options=infidelity_kwargs)
        uncapped_norm, uncapped_metric = norm_cap is None, metric_cap is None
        fit_cap = chi_pair(merge_options(self.sweep_kwargs, sweep_kwargs).get('chi', self.boundary_chi))
        policy = self._boundary_growth_options
        if self._boundary_convergence_auto and run_mode == 'sweep' and str(solver).lower() == 'als':
            policy = {'max_chi': 'auto'}
        if policy is not None:
            if run_mode != 'sweep' or str(solver).lower() != 'als':
                raise ValueError('on_failure boundary growth requires sweep ALS')
            self._validate_adaptive_fit_caps(sweep_kwargs)
            policies = (merge_options(self.normalize_kwargs, normalize_kwargs),
                        merge_options(self.infidelity_kwargs, infidelity_kwargs),
                        merge_options(self.sweep_kwargs, sweep_kwargs))
            if any(p.get('fit_max_bond') is not None for p in policies):
                raise ValueError('on_failure boundary growth requires fit_max_bond=None')
            if any(p.get(k) is not None for p in policies for k in ('bdy', 'bdy_target', 'bdy_overlap')):
                raise ValueError('on_failure boundary growth requires fresh boundary environments')
            metrics = fit_cap if uncapped_metric else chi_pair(metric_cap)
            norm = fit_cap[0] if uncapped_norm else chi_pair(norm_cap)[0]
            caps = (max(fit_cap[0], metrics[0], norm), max(fit_cap[1], metrics[1]))
            if self._boundary_growth is None:
                self._boundary_growth = BoundaryGrowth(caps, **policy)
            else:
                self._boundary_growth.retain(caps)
            self._active_boundary_growth = self._boundary_growth
            self._boundary_growth.run_count += 1
        elif (not self._boundary_convergence_auto
              and self.boundary_convergence is not None and run_mode == 'sweep'):
            raise ValueError('run_pepo supports fixed caps or on_failure growth; '
                             'set boundary_convergence=False or "on_failure"')

        def options():
            caps = fit_cap if self._active_boundary_growth is None else self._active_boundary_growth.chi
            nc = norm_cap if self._active_boundary_growth is None else (None if uncapped_norm else caps[0])
            mc = metric_cap if self._active_boundary_growth is None else (None if uncapped_metric else caps)
            norm_policy = self._metric_boundary_options(self.normalize_kwargs, normalize_kwargs)
            return nc, mc, merge_options(normalize_kwargs,
                {'chi': nc, 'cutoff': norm_policy.get('cutoff', 0.)}), caps

        if reset_traces:
            self._reset_traces()
        nc, mc, norm_options, caps = options()
        self._ensure_initial_normalized(normalize_initial=normalize_initial,
                                        normalize_kwargs=norm_options, normalize_chi=nc)
        # Reserve PEPO for operator factors. A compressed previous output can
        # inherit it from its warm start, but is now a single ket layer.
        source = self.state.copy()
        for tag in ('BRA', 'KET', 'PEPO'):
            if tag in source.tags:
                source.drop_tags(tag)
        operator = pepo.copy()
        operator.add_tag('PEPO')
        target = operator.apply(source, contract=False)
        target.mangle_inner_('__pepsy_pepo_target')
        snapshot = self.state.copy()
        growth = self._active_boundary_growth
        event_start = 0 if growth is None else len(growth.events)
        metric_options = merge_options(infidelity_kwargs)
        metric_options.pop('evaluation_max_retries', None)
        metric_options.setdefault('cutoff', self._metric_boundary_options(
            self.infidelity_kwargs).get('cutoff', 0.))
        metric_options.setdefault('norm_target', self.infidelity_kwargs.get('norm_target'))
        while True:
            nc, mc, norm_options, caps = options()
            if growth is not None:
                self.boundary_chi = caps
            try:
                # Applying/contracting this copy constructs a guess, never the
                # target. This also gives ALS nonzero support on new bonds.
                warm = operator.apply(source, contract=True, compress=True,
                                      max_bond=self.chi, cutoff=cutoff, cutoff_mode=cutoff_mode)
                for tag in ('BRA', 'KET', 'PEPO'):
                    if tag in warm.tags:
                        warm.drop_tags(tag)
                self._normalize_state(warm, normalize_chi=nc, normalize_kwargs=norm_options)
                attempt_metric_options = dict(metric_options)
                if unitary and attempt_metric_options.get('norm_target') is None:
                    # Unitarity preserves the input norm, including finite-chi
                    # normalization drift. Measure the two-layer input rather
                    # than the four-layer (U psi).H (U psi), once at these caps.
                    attempt_metric_options['norm_target'] = self._measure_target_norm(
                        source, evaluation_chi=mc, metric_kwargs=attempt_metric_options)
                pre = self.estimate_infidelity(warm, target, evaluation_chi=mc,
                                               evaluation_max_retries=0, **attempt_metric_options)
                target_norm = self._last_target_norm
                reliable = not self.evaluation_records[-1].get('clipped_infidelity', False)
                summary = None
                final, reason, post = warm, 'below_tol', pre
                if not reliable or pre > infidelity_tol:
                    # Small singular channels of the full operator target can
                    # be essential near a product state. Dropping them while
                    # seeding one-site FIT causes errors that larger chi cannot
                    # repair; default to rank-only boundary truncation here.
                    sweep_options = merge_options(
                        {'cutoff': self.boundary_kwargs.get('cutoff', 0.)},
                        self.sweep_kwargs, sweep_kwargs, {'chi': caps})
                    loss_options = merge_options({'mode': 'direct',
                        'layer_tags': ('BRA', 'KET'), 'overlap_layer_tags': ('BRA', 'PEPO', 'KET'),
                        'canonize': False, 'cutoff': 0.},
                        self.global_kwargs.get('loss_kwargs'),
                        (global_kwargs or {}).get('loss_kwargs'))
                    effective_global = merge_options(global_kwargs, {'loss_kwargs': loss_options})
                    axes = (('x',) if self.update_style == 'row' else ('y',)
                            if self.update_style == 'column' or str(solver).lower() == 'scipy'
                            else ('y', 'x'))
                    sweep_run_options = merge_options({'axes': axes},
                        self.sweep_optimize_kwargs, sweep_optimize_kwargs)
                    candidate, _, summary = self._optimize_state(
                        warm.copy(), target, mode=run_mode, target_norm=target_norm,
                        progress=progress, sweep_progress=self.sweep_progress,
                        normalize_chi=nc, cutoff=cutoff,
                        sweep_kwargs=sweep_options, sweep_optimize_kwargs=sweep_run_options,
                        global_kwargs=effective_global, global_optimize_kwargs=global_optimize_kwargs,
                    )
                    if summary is not None and summary.get('success') is False:
                        reason = 'optimizer_failed'
                    else:
                        self._normalize_state(candidate, normalize_chi=nc, normalize_kwargs=norm_options)
                        post = self.estimate_infidelity(candidate, target, evaluation_chi=mc,
                            evaluation_max_retries=0,
                            **merge_options(metric_options, {'norm_target': target_norm}))
                        if accept_if_improved and reliable and post >= pre - improvement_tol:
                            reason = 'optimizer_rejected'
                            post = pre
                        else:
                            final, reason = candidate, 'optimized'
                self.state = final
                fidelity, geometric = self._record_fidelity_progress(post)
                record = dict(step=1, mode=run_mode, optimizer=solver if run_mode == 'sweep' else 'global',
                    target_representation='pepo', pepo_max_bond=pepo.max_bond(), unitary=bool(unitary),
                    normalize_chi=nc, evaluation_chi=mc, effective_evaluation_chi=self._last_evaluation_chi,
                    pre_infidelity=pre, final_infidelity=post, reason=reason, fidelity=fidelity,
                    geometric_fidelity=geometric, optimizer_result=summary,
                    boundary_growth=None if growth is None else {
                        'chi': growth.chi, 'max_chi': growth.max_chi,
                        'events': deepcopy(growth.events[event_start:])})
                self.step_records.append(record)
                self.last_result = {**record, 'state': final}
                if step_callback is not None:
                    step_callback(deepcopy(record))
                return self.state
            except BoundaryAccuracyError as failure:
                if growth is None:
                    raise
                self._increase_boundary_chi(failure, snapshot, step=1)
