"""Mode dispatch and control-delimited MPS replay orchestration.

These operations receive the live optimizer explicitly and call its hooks so
state ownership, timing, and subclass overrides remain on ``MpsOptimizer``.
"""

from .._stream_events import _CONTROL_EVENT_NAMES


DEFAULT_FIT_INIT_STRATEGY = "guess_src"


def execute_mode(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    self,
    G_seq,
    where_seq,
    event_seq,
    *,
    logical_where_seq=None,
    n_iter,
    progbar,
    cutoff,
    cutoff_mode,
    mpo_cutoff_mode=None,
    k_2q_batch,
    normalize_every,
    normalize_final,
    normalize_eps,
    non_unitary,
    submpo_method,
    compression_seed=None,
    compression_opts=None,
    mix_strict=False,
    fit_min_iter=2,
    fit_rtol=None,
    fit_patience=2,
    mix_sticky_nonfinite=False,
    fit_block_size=2,
    fit_adaptive_sweeps=2,
    fit_sweep_sequence="RL",
    fit_max_span="auto",
    target_cutoff=0.0,
    fit_target_strategy="auto",
    fit_mpo_guess=True,
    fit_init_strategy=DEFAULT_FIT_INIT_STRATEGY,
    fit_init_rand_strength=0.0,
    fit_init_seed=0,
    fit_single_pair_fast_path=False,
    fit_single_pair_n_iter=None,
    finite_check=False,
    fit_overlap_diagnostics=False,
    stabilize_unitary=False,
    quality_check_every=None,
    quality_check_repair=True,
):
    """Dispatch a gate/subMPO segment to the active mode backend.

    This is the mode-specific core of :meth:`run`; ``G_seq``/``where_seq``/
    ``event_seq`` must contain only ``"gate"``/``"submpo"`` events. Control
    events (measure/cap/reset) are handled by :meth:`_run_segmented`.
    """
    # The stream-install boundary already normalized and validated these
    # payloads. Do not repeat that work for each control-delimited segment.
    # Dispatch directly to the mode implementations, which share the same
    # validated stream but have different state contracts: DMRG owns local
    # variational targets, MPO owns Quimb compression, swap/perm own endpoint
    # movement, and exact deliberately bypasses canonical-center bookkeeping.
    if self.mode == "dmrg":
        self._timed_call("dmrg.prepare", self._prepare_dmrg_state)
        self._timed_call(
            "dmrg.replay",
            self._run_dmrg,
            G_seq,
            where_seq,
            event_seq=event_seq,
            n_iter=n_iter,
            progbar=progbar,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            k_2q_batch=k_2q_batch,
            normalize_every=normalize_every,
            normalize_final=normalize_final,
            normalize_eps=normalize_eps,
            non_unitary=non_unitary,
            fit_min_iter=fit_min_iter,
            fit_rtol=fit_rtol,
            fit_patience=fit_patience,
            fit_finite_check=finite_check,
            fit_block_size=fit_block_size,
            fit_adaptive_sweeps=fit_adaptive_sweeps,
            fit_sweep_sequence=fit_sweep_sequence,
            fit_max_span=fit_max_span,
            target_cutoff=target_cutoff,
            fit_target_strategy=fit_target_strategy,
            fit_mpo_guess=fit_mpo_guess,
            fit_init_strategy=fit_init_strategy,
            fit_init_rand_strength=fit_init_rand_strength,
            fit_init_seed=fit_init_seed,
            fit_single_pair_fast_path=fit_single_pair_fast_path,
            fit_single_pair_n_iter=fit_single_pair_n_iter,
            finite_check=finite_check,
            fit_overlap_diagnostics=fit_overlap_diagnostics,
            stabilize_unitary=stabilize_unitary,
            quality_check_every=quality_check_every,
            quality_check_repair=quality_check_repair,
        )
        return self.p

    if self.mode == "mix":
        self._timed_call(
            "mix.replay",
            self._run_mix,
            G_seq,
            where_seq,
            event_seq,
            logical_where_seq=logical_where_seq,
            n_iter=n_iter,
            k_2q_batch=k_2q_batch,
            progbar=progbar,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            submpo_method=submpo_method,
            compression_seed=compression_seed,
            compression_opts=compression_opts,
            mix_strict=mix_strict,
            fit_min_iter=fit_min_iter,
            fit_rtol=fit_rtol,
            fit_patience=fit_patience,
            sticky_nonfinite=mix_sticky_nonfinite,
            fit_block_size=fit_block_size,
            fit_adaptive_sweeps=fit_adaptive_sweeps,
            fit_sweep_sequence=fit_sweep_sequence,
            fit_max_span=fit_max_span,
            target_cutoff=target_cutoff,
            fit_target_strategy=fit_target_strategy,
            fit_init_strategy=fit_init_strategy,
            fit_init_rand_strength=fit_init_rand_strength,
            fit_init_seed=fit_init_seed,
            fit_single_pair_fast_path=fit_single_pair_fast_path,
            finite_check=finite_check,
            fit_overlap_diagnostics=fit_overlap_diagnostics,
            stabilize_unitary=stabilize_unitary,
            non_unitary=non_unitary,
            quality_check_every=quality_check_every,
            quality_check_repair=quality_check_repair,
        )
        return self.p

    if self._is_mpo_mode(self.mode):
        # Report the selected compression method rather than the internal
        # implementation family, including selection through a legacy alias.
        replay_name = self._mode_mpo_method(self.mode)
        self._timed_call(
            f"{replay_name}.replay",
            self._run_mpo,
            G_seq,
            where_seq,
            event_seq,
            progbar=progbar,
            cutoff=cutoff,
            cutoff_mode=mpo_cutoff_mode,
            normalize_every=normalize_every,
            normalize_final=normalize_final,
            normalize_eps=normalize_eps,
            non_unitary=non_unitary,
            submpo_method=submpo_method,
            compression_seed=compression_seed,
            compression_opts=compression_opts,
            stabilize_unitary=stabilize_unitary,
        )
        return self.p

    if self.mode == "swap":
        self._timed_call(
            "swap.replay",
            self._run_swap,
            G_seq,
            where_seq,
            progbar=progbar,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            normalize_every=normalize_every,
            normalize_final=normalize_final,
            normalize_eps=normalize_eps,
            non_unitary=non_unitary,
            stabilize_unitary=stabilize_unitary,
        )
        return self.p

    if self.mode == "perm":
        self._timed_call(
            "perm.replay",
            self._run_perm,
            G_seq,
            where_seq,
            progbar=progbar,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            normalize_every=normalize_every,
            normalize_final=normalize_final,
            normalize_eps=normalize_eps,
            non_unitary=non_unitary,
            stabilize_unitary=stabilize_unitary,
        )
        return self.p

    if self.mode == "svd":
        self._timed_call(
            "svd.replay",
            self._run_svd,
            G_seq,
            where_seq,
            progbar=progbar,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            normalize_every=normalize_every,
            normalize_final=normalize_final,
            normalize_eps=normalize_eps,
            non_unitary=non_unitary,
            stabilize_unitary=stabilize_unitary,
        )
        return self.p

    if self.mode == "exact-batch":
        self._timed_call(
            "exact-batch.replay",
            self._run_exact_batch,
            G_seq,
            where_seq,
            progbar=progbar,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
        )
        return self.p

    if self.mode == "exact":
        self._timed_call(
            "exact.replay",
            self._run_exact,
            G_seq,
            where_seq,
            progbar=progbar,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
        )
        return self.p

    raise ValueError(f"Unknown mode: {self.mode}")


def run_segmented(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    self,
    G_seq,
    where_seq,
    event_seq,
    *,
    logical_where_seq=None,
    progbar,
    cutoff,
    cutoff_mode,
    measure_renormalize,
    where_is_physical=False,
    mode_kwargs,
):
    """Replay a stream containing measure/cap/reset control events.

    Consecutive ``"gate"``/``"submpo"`` events are grouped into segments run
    through :meth:`_execute_mode` (using the active mode), while control
    events are applied directly to ``self.p`` between segments so the same
    stream works in every mode. ``cap`` events change the MPS length, so
    later event site labels refer to the shortened chain.

    ``where_seq`` holds the execution locations (already mapped into the
    active layout order when a layout is used); ``logical_where_seq`` holds
    the matching user-facing locations for bookkeeping such as recorded
    measurement sites. When no layout is active the two are identical.
    ``where_is_physical`` prevents persistent-layout locations from being
    mapped a second time by the control-event dispatcher.
    """
    if logical_where_seq is None:
        logical_where_seq = where_seq
    seg_G = []
    seg_where = []
    seg_logical_where = []
    seg_event = []

    def flush():
        # Control events are state boundaries: prior gates must be committed
        # before their probability is evaluated, and later gates must see the
        # updated state. Never batch a segment across a control event.
        if seg_G:
            self._execute_mode(
                list(seg_G),
                list(seg_where),
                list(seg_event),
                logical_where_seq=list(seg_logical_where),
                progbar=progbar,
                **mode_kwargs,
            )
            seg_G.clear()
            seg_where.clear()
            seg_logical_where.clear()
            seg_event.clear()

    for payload, where, logical_where, event_type in zip(
        G_seq, where_seq, logical_where_seq, event_seq
    ):
        if event_type in _CONTROL_EVENT_NAMES:
            flush()
            self._apply_control_event(
                event_type,
                payload,
                where,
                record_where=logical_where,
                where_is_physical=where_is_physical,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                measure_renormalize=measure_renormalize,
                mode_kwargs=mode_kwargs,
            )
        else:
            seg_G.append(payload)
            seg_where.append(where)
            seg_logical_where.append(logical_where)
            seg_event.append(event_type)

    flush()
    return self.p


def run_prepared(
    self,
    G_seq,
    where_seq,
    event_seq,
    *,
    logical_where_seq,
    mode_kwargs,
    has_control,
    has_cap,
    layout_plan,
    layout_order_tuple,
    persistent_layout_active,
    measure_renormalize,
    progbar,
    timing,
    timing_sync_device,
):
    """Execute validated replay and restore a temporary layout on every exit."""
    layout_current_order = None
    # Option validation has completed before the live state is reordered.
    if layout_plan is not None and not persistent_layout_active:
        layout_current_order = self._reorder_mps_to_logical_order(layout_order_tuple)
        if has_cap:
            # Caps compact the register using the logical label at each live
            # position, even when their execution locations are physical.
            self._set_site_order(layout_current_order)

    def replay():
        if has_control:
            return self._run_segmented(
                G_seq,
                where_seq,
                event_seq,
                logical_where_seq=logical_where_seq,
                progbar=progbar,
                cutoff=mode_kwargs["cutoff"],
                cutoff_mode=mode_kwargs["cutoff_mode"],
                measure_renormalize=measure_renormalize,
                where_is_physical=persistent_layout_active,
                mode_kwargs=mode_kwargs,
            )
        return self._execute_mode(
            G_seq,
            where_seq,
            event_seq,
            logical_where_seq=logical_where_seq,
            progbar=progbar,
            **mode_kwargs,
        )

    try:
        return self._run_with_fit_copy_policy(
            replay,
            enabled=timing,
            finite_check=mode_kwargs["finite_check"],
            event_count=len(G_seq),
            sync_device=timing_sync_device,
        )
    finally:
        if layout_current_order is not None:
            current_order = (
                tuple(self.logical_order)
                if has_cap
                else tuple(layout_current_order)
            )
            self._reorder_mps_to_logical_order(
                tuple(range(int(getattr(self.p, "L", 0)))),
                current_order=current_order,
            )
            self._set_site_order(
                tuple(range(int(getattr(self.p, "L", 0))))
            )
            self._normalize_visible_mps_order()
