"""PEPS sweep construction and dispatch, separate from gate-stream replay."""

from collections.abc import Mapping

from ..sweep.environments import normalize_boundary_engine, uses_symmray_arrays
from .._boundary_control import merge_options as _merge_opts
from .._boundary_control import metric_failure
from .._boundary_control import UNSET_METRIC_CHI as _UNSET_METRIC_CHI
from ._strip_batch import reuse_boundary_guesses
from ._timing import timed_phase

_DEFAULT_SWEEP_OPTIMIZE_KWARGS = {"n_round_trips": 2, "renormalize": False}


class SweepFitMixin:
    """Construct one fixed-cap fit; replay owns failure-triggered retries."""

    @timed_phase("sweep")
    def _optimize_with_sweep(
        self,
        state,
        target,
        *,
        progress,
        target_norm=1.0,
        sweep_progress=None,
        normalize_chi=_UNSET_METRIC_CHI,
        sweep_kwargs=None,
        sweep_optimize_kwargs=None,
    ):
        sweep_kwargs = dict(sweep_kwargs or {})
        checked_boundaries = sweep_kwargs.pop('_checked_boundaries', None)
        checked_target = sweep_kwargs.pop('_checked_target', None)
        is_strip = '_active_strip' in (sweep_optimize_kwargs or {})
        # build_bra_ket already separates colliding ket/bra bonds. Keep the
        # calibrated labels when transferring environments into this fitter.
        target_opt = (checked_target if checked_boundaries is not None
                      else self._copy_and_mangle_target(target))
        if is_strip:
            # Stable disjoint labels permit compatible overlap guesses to
            # survive subsequent strip batches. No numerical cuts are reused.
            target_opt = target.copy()
            target_opt.mangle_inner_('__pepsy_strip_target')
        boundary_init_kwargs, boundary_opt_kwargs, strip_exponent = self._sweep_boundary_kwargs(
            progress=progress,
            sweep_progress=sweep_progress,
        )
        init_kwargs = {
            "chi": self.boundary_chi,
            "target_norm": target_norm,
            "contraction_opt": self.contraction_opt,
            # run() already normalized the compressed warm start.
            "renormalize_state": False,
            "boundary_engine": self.boundary_engine,
            **boundary_init_kwargs,
        }
        if self.boundary_options:
            init_kwargs["boundary_options"] = dict(self.boundary_options)
        init_kwargs.update(self.sweep_kwargs)
        init_kwargs.update(dict(sweep_kwargs or {}))
        if checked_boundaries is not None:
            init_kwargs.update(checked_boundaries)
        if init_kwargs.get("local_contraction_opt") is None:
            # One reusable search cache across gate batches and run() calls.
            # This plans local objectives only, never caches contraction values.
            init_kwargs["local_contraction_opt"] = self._get_sweep_contraction_optimizer()
        init_kwargs.setdefault("evaluation_negative_tol", self.evaluation_negative_tol)
        # ``full_simplify`` is not backend-safe for Symmray block trees. Make
        # the supported default explicit here so SweepOptimizer does not need
        # to correct it (and warn) during construction. An explicit
        # ``simplify=True`` in sweep_kwargs remains visible and is corrected by
        # SweepOptimizer with its actionable compatibility warning.
        if uses_symmray_arrays(state, target_opt) and "simplify" not in init_kwargs:
            init_kwargs["simplify"] = False
        boundary_engine = normalize_boundary_engine(
            init_kwargs.get("boundary_engine", "auto"),
            state,
            target_opt,
        )
        init_kwargs["boundary_engine"] = boundary_engine
        normalize_payload = _merge_opts(
            {"chi": self._boundary_chi_for_norm()
             if normalize_chi is _UNSET_METRIC_CHI else normalize_chi},
            init_kwargs.get("normalize_kwargs"),
        )
        if boundary_engine == "quimb-mps":
            normalize_payload = _merge_opts(
                {"method": "mps", "mode_": "mps", "balance_bonds": False},
                normalize_payload,
            )
        if strip_exponent is not None:
            normalize_payload = _merge_opts(
                normalize_payload,
                {"strip_exponent": bool(strip_exponent)},
            )
        if normalize_payload is not None:
            init_kwargs["normalize_kwargs"] = normalize_payload

        sweeper = self._make_sweep_optimizer(
            state=state,
            state_target=target_opt,
            **init_kwargs,
        )
        sweeper._boundary_growth = self._active_boundary_growth
        guess_reuse = (reuse_boundary_guesses(sweeper, self._strip_guess_boundaries)
                       if is_strip and sweeper.fit_init_strategy.startswith('reuse-') else None)

        opt_kwargs = _merge_opts(
            _DEFAULT_SWEEP_OPTIMIZE_KWARGS,
            boundary_opt_kwargs,
            self.sweep_optimize_kwargs,
            sweep_optimize_kwargs,
        )
        opt_kwargs = self._apply_sweep_optimizer_options(opt_kwargs)
        if self._active_boundary_growth is not None:
            # The controller already selected the constructor pair. A legacy
            # optimize-time cap must not silently reset a retained component.
            opt_kwargs['chi'] = None
        # Internal start/end diagnostics must use the same target norm as
        # the local objective. Sweep's standalone diagnostic default measures
        # it anew; its existing diagnostic options let this driver reuse it.
        opt_kwargs["debug_loss_kwargs"] = _merge_opts(
            {"norm_target": init_kwargs["target_norm"]},
            opt_kwargs.get("debug_loss_kwargs"),
        )
        if sweep_progress is not None:
            # A top-level explicit setting wins over legacy nested
            # ``sweep_optimize_kwargs={"progress": ...}`` values.
            opt_kwargs["progress"] = bool(sweep_progress)
            opt_kwargs["progress_position"] = 1 if progress and sweep_progress else 0
            opt_kwargs["progress_leave"] = False
        opt_kwargs.setdefault("progress", bool(progress))
        active_strip = opt_kwargs.pop('_active_strip', None)
        if opt_kwargs:
            sweeper.set_optimize_kwargs(**opt_kwargs)

        if active_strip is None:
            result = sweeper.run()
        else:
            axis, index = active_strip
            strip_options = dict(sweeper.optimize_kwargs)
            axes = tuple(strip_options.pop('axes', (axis,)))
            if axes != (axis,):
                raise ValueError('strip sweep axes must match the gated strip axis')
            strip_options['solver'] = strip_options.pop('optimizer', 'als')
            strip_options['solver_options'] = strip_options.pop('optimizer_options', None)
            # The driver owns default pre/post metrics. Explicit diagnostic
            # requests still run through the common sweep lifecycle.
            strip_options.setdefault('compute_initial_loss', False)
            strip_options.setdefault('compute_final_loss', False)
            result = sweeper._run_global_sweeps(
                axes=(axis,), start_index=index, retain_boundary_guesses=True,
                **strip_options,
            )
            if sweeper.boundary_engine == 'dmrg':
                self._strip_guess_boundaries = result.pop('boundary_guesses', None)
        if self._active_boundary_growth is not None and isinstance(result, Mapping):
            exact_diagnostic = (opt_kwargs.get('debug', False)
                                and opt_kwargs.get('debug_loss_mode', 'exact') == 'exact')
            if not exact_diagnostic:
                # Requested diagnostics have already been contracted. Inspect
                # the raw final value even if a finite best iterate exists.
                for key in ('loss_before', 'raw_loss_after'):
                    failure = metric_failure({'infidelity': result.get(key)},
                                             tolerance=max(1e-10, self.evaluation_negative_tol),
                                             roundoff=1e-10)
                    if failure is not None:
                        failure.context.update(stage='sweep_diagnostic', diagnostic=key)
                        raise failure
        self._append_fit_diagnostics(getattr(sweeper, "fit_diagnostics", None))
        best_state = result.get("best_state") if isinstance(result, Mapping) else None
        state_out = best_state if best_state is not None else sweeper.state
        final_infidelity = None
        if isinstance(result, Mapping) and result.get('success') is not False:
            final_infidelity = result.get("best_loss")
            if final_infidelity is None:
                final_infidelity = result.get("loss_after")
        summary = self._summarize_sweep_result(result)
        if guess_reuse is not None:
            summary['boundary_guesses_reused'] = guess_reuse
        if checked_boundaries is not None:
            summary['environment_reuse'] = {
                name: boundary.mps_b.environment_cache.report()
                for name, boundary in checked_boundaries.items()
            }
        return state_out, self._clean_optimizer_infidelity(final_infidelity, summary), summary
