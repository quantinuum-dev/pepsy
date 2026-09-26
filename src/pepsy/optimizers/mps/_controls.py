"""Measurement, reset, cap, and conditional control execution for MPS replay.

Operations receive the live owner explicitly and retain dispatch through its hooks.
"""

from __future__ import annotations

import math
import autoray as ar
import numpy as np
import quimb.tensor as qtn
from .._stream_events import (
    _CONTROL_EVENT_NAMES,
    _RESET_FLIP_AXES,
    _control_event_contains_cap,
    _resolve_conditional,
)
from ..._internal.quimb import run_seeded_quimb as _run_seeded_quimb
from ._streams import _PAULI_1Q, _normalize_gate_queue
from .compression import (
    _MPO_METHODS_NEED_INTERIOR_WORKAROUND,
    _apply_submpo_with_interior_workaround,
    _is_interior_submpo_span,
)
from ...backends import infer_backend_signature as _array_backend_signature

_CONTROL_CLIFFORDS = {
    "H": np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
    "HY": np.array([[1, -1j], [1, 1j]], dtype=complex) / np.sqrt(2),
    "CX": np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    ),
}


def _apply_control_event(self, *args, **kwargs):
    """Apply one control event, optionally recording its stage time."""
    if self._timing_state is None:
        return self._apply_control_event_impl(*args, **kwargs)
    name = args[0] if args else kwargs.get("name", "unknown")
    return self._timed_call(
        f"control.{name}",
        self._apply_control_event_impl,
        *args,
        **kwargs,
    )


def _apply_control_event_impl(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    self,
    name,
    payload,
    where,
    *,
    record_where=None,
    where_is_physical=False,
    cutoff,
    cutoff_mode,
    measure_renormalize,
    mode_kwargs=None,
):
    """Apply one measure/cap/reset control event to ``self.p``."""
    if record_where is None:
        record_where = where
    self._ensure_mps_state()
    self._ensure_tracked_center()
    if name == "conditional":
        record_index, expected = _resolve_conditional(
            payload, len(self.measurements)
        )
        record = self.measurements[record_index]
        outcome = int(getattr(record, "outcome", record[2]))
        if int(outcome < 0) != expected:
            return
        action_payloads, action_wheres, action_types = _normalize_gate_queue(
            (payload["action"],)
        )
        if len(action_payloads) != 1:
            raise ValueError(
                "conditional action must normalize to exactly one stream entry."
            )
        action_where = action_wheres[0]
        action_type = action_types[0]
        # The parent conditional's ``where`` is the same logical support
        # as ``action_where``, but has already passed through any transient
        # or persistent layout mapping. Resolve it only after a true
        # predicate so removed sites on false branches remain harmless.
        action_execution_where = (
            tuple(int(site) for site in where)
            if where_is_physical
            else self._logical_to_physical_where(where)
        )
        if action_type in _CONTROL_EVENT_NAMES:
            self._apply_control_event(
                action_type,
                action_payloads[0],
                action_execution_where,
                record_where=action_where,
                where_is_physical=True,
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                measure_renormalize=measure_renormalize,
                mode_kwargs=mode_kwargs,
            )
        else:
            action_payload = action_payloads[0]
            if action_type == "submpo" and tuple(
                map(int, action_where)
            ) != action_execution_where:
                action_payload = self._copy_submpo_for_layout(
                    action_payload,
                    dict(zip(action_where, action_execution_where)),
                    action_where,
                )
            # Ordinary mode backends receive physical execution
            # locations, but perm replay owns the logical-to-physical
            # translation inside its gate kernel. Passing the already
            # mapped location there would translate a conditional action
            # a second time after an earlier lazy swap.
            mode_where = (
                action_where
                if self.mode == "perm" and not where_is_physical
                else action_execution_where
            )
            self._execute_mode(
                [action_payload],
                [mode_where],
                [action_type],
                logical_where_seq=[action_where],
                progbar=False,
                # The predicate selects a gate, not a new solver policy.
                # Reuse the same validated settings as ordinary segments,
                # including named DMRG schedules and explicit FIT guesses.
                **mode_kwargs,
            )
        return
    # Resolve physical sites only for an executed control. A false
    # conditional may mention a site removed by a preceding cap.
    execution_where = (
        tuple(int(site) for site in where)
        if where_is_physical
        else self._logical_to_physical_where(where)
    )
    if name == "measure":
        self._apply_measure_event(
            payload["pauli"],
            execution_where,
            payload.get("outcome"),
            record_where=record_where,
            renormalize=measure_renormalize,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            mode_kwargs=mode_kwargs,
        )
    elif name == "cap":
        logical_site = int(record_where[0])
        physical_site = int(execution_where[0])
        self._apply_cap_event(
            execution_where,
            payload["vec"],
            payload.get("absorb", "left"),
        )
        self._apply_effective_cap(physical_site)
        self._update_permutation_after_cap(logical_site, physical_site)
    elif name == "reset":
        self._apply_reset_event(
            execution_where,
            payload.get("axes"),
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
        )
    elif name == "measure_reset":
        self._apply_measure_reset_event(
            payload["axes"],
            execution_where,
            payload["outcomes"],
            record_where=record_where,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            mode_kwargs=mode_kwargs,
        )
    else:  # pragma: no cover - guarded by parsing
        raise ValueError(f"Unknown control event {name!r}.")

    if name != "cap":
        self._record_effective_event(execution_where, event_type=name)


def _control_operator(self, name):
    """Return an owned small control matrix on the current state backend.

    Only fixed Pauli/Clifford constants are cached, for one backend,
    device and dtype at a time. Copies keep Quimb and trajectory branches
    from mutating a cached operand. Native arrays still use the existing
    metadata-aware conversion boundary, which rejects dense promotion.
    """
    signature = _array_backend_signature(self._state_backend_like())
    cache = self._control_operator_cache
    if cache is None or cache[0] != signature:
        cache = self._control_operator_cache = (signature, {})
    operators = cache[1]
    if name not in operators:
        source = _PAULI_1Q[name] if name in _PAULI_1Q else _CONTROL_CLIFFORDS[name]
        operators[name] = self._to_state_backend(source)
    return ar.do("copy", operators[name])


def _one_site_projector(self, axis, outcome):
    """Assemble a Pauli projector without transferring a new host array."""
    return 0.5 * (
        self._control_operator("I") + outcome * self._control_operator(axis)
    )


def _pauli_operator(self, pauli, where):
    """Return the dense Pauli operator (numpy) for ``pauli`` on ``where``."""
    chars = [c for c in str(pauli).upper() if not c.isspace()]
    if len(chars) != len(where):
        raise ValueError(
            f"pauli string {pauli!r} has {len(chars)} axes but where {where!r} "
            f"has {len(where)} site(s)."
        )
    try:
        op = _PAULI_1Q[chars[0]]
        for axis in chars[1:]:
            op = np.kron(op, _PAULI_1Q[axis])
    except KeyError as exc:  # pragma: no cover - guarded by dict lookup
        raise ValueError(f"unknown Pauli axis in {pauli!r}.") from exc
    return op


def _build_pauli_projector_submpo(self, pauli, where, outcome):
    """Build ``(I + outcome * P) / 2`` as a bond-two windowed sub-MPO.

    The dense projector is retained for native Symmray/fermionic states,
    where a dense MPO cannot carry the target charge and dummy-mode
    metadata. Dense MPS states use the two product branches directly:
    ``0.5 * I`` and ``0.5 * outcome * P``.
    """
    if len(where) < 2:
        return None
    if self._replay_has_symmray_data(self.p) or self.p.isfermionic():
        return None

    chars = [c for c in str(pauli).upper() if not c.isspace()]
    sites = tuple(int(site) for site in where)
    if len(chars) != len(sites):
        raise ValueError(
            f"pauli string {pauli!r} has {len(chars)} axes but where "
            f"{where!r} has {len(sites)} site(s)."
        )
    if len(set(sites)) != len(sites):
        raise ValueError("measurement sites must be unique.")
    if any(axis not in _PAULI_1Q for axis in chars):
        raise ValueError(f"unknown Pauli axis in {pauli!r}.")

    axes_by_site = dict(zip(sites, chars))
    span = tuple(range(min(sites), max(sites) + 1))
    identity = self._control_operator("I")
    zero = ar.do("zeros_like", identity)
    local_operators = {
        axis: self._control_operator(axis) for axis in set(chars) | {"I"}
    }
    arrays = []

    for position, site in enumerate(span):
        local = local_operators[axes_by_site.get(site, "I")]
        if position == 0:
            tensor = ar.do("stack", (identity, local))
        elif position == len(span) - 1:
            tensor = ar.do("stack", (0.5 * identity, 0.5 * int(outcome) * local))
        else:
            tensor = ar.do("stack", (
                ar.do("stack", (identity, zero)),
                ar.do("stack", (zero, local)),
            ))
        arrays.append(tensor)

    submpo = qtn.MatrixProductOperator(
        arrays,
        sites=span,
        L=int(self.p.L),
        shape="lrud",
        upper_ind_id=self.ind_id,
        lower_ind_id="b{}",
        site_tag_id="I{}",
    )
    return submpo, span


def _apply_submpo_with_method(
    self,
    p,
    submpo,
    where,
    *,
    method,
    cutoff,
    cutoff_mode,
    info=None,
    seed=None,
):
    """Apply and compress a sub-MPO using the selected Quimb method."""
    if info is None:
        info = self._info_for_state(p)
    method = self._normalize_submpo_method(method)
    compress_opts = self._submpo_compress_opts(
        method,
        cutoff=cutoff,
        cutoff_mode=cutoff_mode,
    )
    # The direct API is preferred for a full-chain or ordinary local
    # payload. A partitioned interior payload needs the local workaround
    # only for wrappers whose nested call assumes every chain site has a
    # matching tag; keeping the partition local avoids both tag failures
    # and unnecessary full-chain contraction work.
    if (
        method in _MPO_METHODS_NEED_INTERIOR_WORKAROUND
        and _is_interior_submpo_span(p, where)
    ):
        _apply_submpo_with_interior_workaround(
            p,
            submpo,
            where,
            chi=self.chi,
            method=method,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            info=info,
            inplace_mpo=False,
            seed=seed,
            **{
                key: value
                for key, value in compress_opts.items()
                if key == "optimize"
            },
        )
    else:
        _run_seeded_quimb(
            seed,
            p.gate_with_submpo_,
            submpo,
            where=where,
            method=method,
            max_bond=self.chi,
            info=info,
            inplace_mpo=False,
            **compress_opts,
        )
    return p


def _apply_dense_operator(self, p, op, where, *, max_bond, cutoff, cutoff_mode, info=None):
    """Apply a dense operator ``op`` on ``where`` sites of MPS ``p`` in place.

    ``info`` is the canonicalization tracking dict; it defaults to
    ``self.info_c`` for operations on ``self.p`` and should be an isolated
    dict when acting on a throwaway copy so the tracked centre is preserved.
    """
    if info is None:
        info = self._info_for_state(p)
    where = tuple(int(site) for site in where)
    op_b = self._to_state_backend(op)
    if len(where) == 1:
        self._apply_gate(
            p,
            op_b,
            where,
            contract=True,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            inplace=True,
        )
    else:
        p.gate_nonlocal_(
            op_b,
            where,
            dims=None,
            max_bond=max_bond,
            info=info,
            method="direct",
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
        )
    return p


def _state_expectation(self, pauli, where):
    """Return the normalized expectation ``<P> = Re <psi|P|psi> / <psi|psi>``.

    For small supports exposing ``local_expectation_canonical``, move
    the tracked orthogonality centre around the support and contract only
    the local reduced density matrix. Larger dense supports use a parity
    circuit to avoid an exponential operator. The fallback preserves
    compatibility with older Quimb versions without that method.
    """
    if len(where) > 2 and not self._replay_has_symmray_data(self.p):
        positive, negative = self._pauli_amplitude_probabilities(pauli, where)
        return positive - negative
    return self._state_operator_expectation(self._pauli_operator(pauli, where), where)


def _state_operator_expectation(self, op, where):
    """Contract a normalized local operator using the live center metadata."""
    p = self.p
    op = self._to_state_backend(op)
    local_expectation = getattr(p, "local_expectation_canonical", None)
    if callable(local_expectation):
        return self._real_float(
            local_expectation(
                op,
                tuple(int(site) for site in where),
                normalized=True,
                info=self.info_c,
                optimize=self.contraction_opt,
            )
        )

    # Compatibility path for older Quimb releases without local MPS
    # expectation support.
    p_op = p.copy()
    self._apply_dense_operator(
        p_op, op, where, max_bond=None, cutoff=0.0, cutoff_mode="abs", info={}
    )
    overlap = (p.H & p_op).contract(
        all, output_inds=(), optimize=self.contraction_opt
    )
    norm_sq = (p.H & p).contract(
        all, output_inds=(), optimize=self.contraction_opt
    )
    norm_val = self._real_float(norm_sq)
    if norm_val == 0.0:
        return 0.0
    return self._real_float(overlap) / norm_val


def _measurement_probabilities(self, pauli, where):
    """Return both Born weights, retaining small positive branches."""
    if not self._replay_has_symmray_data(self.p):
        return self._pauli_amplitude_probabilities(pauli, where)
    expectation = self._state_expectation(pauli, where)
    p_plus = min(max(0.5 * (1.0 + expectation), 0.0), 1.0)
    p_minus = 1.0 - p_plus
    if min(p_plus, p_minus) < 1e-8:
        # Subtracting an expectation near +/-1 loses relative precision
        # in rare branches. Contract that branch's projector directly;
        # ordinary measurements keep their single expectation contraction.
        outcome = 1 if p_plus < p_minus else -1
        op = self._pauli_operator(pauli, where)
        projector = 0.5 * (np.eye(op.shape[0], dtype=complex) + outcome * op)
        probability = min(max(self._state_operator_expectation(projector, where), 0.0), 1.0)
        return (probability, 1.0 - probability) if outcome > 0 else (1.0 - probability, probability)
    return p_plus, p_minus


def _pauli_amplitude_probabilities(self, pauli, where):
    """Measure local projected amplitudes, without a dense reduced state.

    A disposable Clifford circuit collects a Pauli product's parity on
    one qubit. Only one- and two-qubit gates are formed, with no truncation.
    Squaring projected amplitudes avoids cancellation in Tr(rho P).
    """
    chars = [axis for axis in str(pauli).upper() if not axis.isspace()]
    if len(chars) != len(where) or len(set(where)) != len(where):
        raise ValueError("Pauli axes must match a support of unique sites.")
    if any(axis not in _PAULI_1Q for axis in chars):
        raise ValueError(f"unknown Pauli axis in {pauli!r}.")
    axes = dict(zip(where, chars))
    sites = sorted(site for site, axis in axes.items() if axis != "I")
    if not sites:
        return 1.0, 0.0
    anchor = sites[0]
    self.canonize_mps(self.p, anchor)
    if len(sites) == 1:
        tensor = self.p[anchor]
        measured_axis = axes[anchor]
    else:
        state = self.p.copy()
        state.exponent = 0.0
        info = dict(self.info_c)
        for site in sites:
            axis = axes[site]
            if axis == "Z":
                continue
            rotation = self._control_operator("H" if axis == "X" else "HY")
            self._apply_dense_operator(state, rotation, (site,), max_bond=None,
                                       cutoff=0., cutoff_mode="abs", info=info)
        # Reduce along the ordered support rather than repeatedly crossing
        # the full span. The last parity lives at the leftmost site.
        for control, target in zip(reversed(sites[1:]), reversed(sites[:-1])):
            self._apply_dense_operator(state, self._control_operator("CX"), (control, target),
                                       max_bond=None, cutoff=0., cutoff_mode="abs", info=info)
        self.canonize_mps(state, anchor, info=info)
        tensor = state[anchor]
        measured_axis = "Z"
    scale = tensor.norm()
    normalized = tensor / scale
    weights = []
    for sign in (1, -1):
        projector = self._one_site_projector(measured_axis, sign)
        projected = normalized.gate(projector, self.p.site_ind(anchor))
        amplitude = self._real_float(ar.do("abs", projected.norm()))
        weights.append(amplitude * amplitude)
    total = sum(weights)
    if total <= 0.0 or not math.isfinite(total):
        raise FloatingPointError("Measurement requires a finite nonzero state norm.")
    return tuple(weight / total for weight in weights)


def _scaled_norm_value(norm, exponent=0.0):
    """Reconstruct a display norm, saturating beyond float range."""
    norm = float(abs(norm))
    if norm == 0.0 or not math.isfinite(norm) or exponent == 0.0:
        return norm
    logarithm = math.log(norm) + float(exponent) * math.log(10.)
    return math.inf if logarithm > math.log(np.finfo(float).max) else math.exp(logarithm)


def _control_state_norm(self, *, include_exponent=True):
    """Read the represented control-state norm from its tracked center."""
    if self.mode in {"exact", "exact-batch"}:
        raw_state = self.p.copy()
        raw_state.exponent = 0.0
        norm = self._real_float(ar.do("abs", raw_state.norm()))
        return self._scaled_norm_value(norm, self.p.exponent) if include_exponent else norm
    current = self._current_orthog(self.p)
    raw_norm, _center = self._retained_center_norm(self.p, current)
    norm = self._real_float(ar.do("abs", raw_norm))
    return self._scaled_norm_value(norm, self.p.exponent) if include_exponent else norm


def _recanonize_center(self, site, *, renormalize):
    """Move the orthogonality centre to ``site`` and track it exactly.

    Canonicalizes from the currently tracked centre (never a blind scan) so
    ``site`` becomes a single-site orthogonality centre, records it in
    ``info_c``, and, when ``renormalize`` is set, rescales that centre tensor
    to unit norm (its Frobenius norm equals the represented state norm).
    """
    site = int(site)
    self.p.canonize(
        [site],
        cur_orthog=self._current_orthog(self.p),
        info=self.info_c,
    )
    self.info_c["cur_orthog"] = (site, site)
    if not renormalize:
        return
    center = self.p[self.p.site_tag(site)]
    norm = self._real_float(center.norm())
    if norm > 0.0:
        center.modify(data=center.data / norm)
    if hasattr(self.p, "exponent"):
        self.p.exponent = 0.0


def _finish_measurement_center(self, site, *, renormalize):
    """Track and optionally normalize a post-measurement center."""
    site = int(site)
    current = self._current_orthog(self.p)
    if current != (site, site):
        self.p.canonize(
            [site],
            cur_orthog=current,
            info=self.info_c,
        )
    self.info_c["cur_orthog"] = (site, site)
    if not renormalize:
        return
    center = self.p[self.p.site_tag(site)]
    norm = self._real_float(center.norm())
    if norm > 0.0:
        center.modify(data=center.data / norm)
    if hasattr(self.p, "exponent"):
        self.p.exponent = 0.0


def _apply_measure_event(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    self,
    pauli,
    where,
    outcome,
    *,
    record_where=None,
    renormalize,
    cutoff,
    cutoff_mode,
    norm_kind="measure",
    mode_kwargs=None,
):
    """Measure Pauli ``pauli`` on ``where``, collapse, and record the result.

    ``where`` is the execution location; ``record_where`` (defaulting to
    ``where``) is the user-facing location stored in :attr:`measurements`.
    """
    if record_where is None:
        record_where = where
    # Compute the physical Born probability before constructing or applying
    # any localizer. The localizer is a Clifford change of Pauli frame; it
    # can make the same observable look simpler, but its post-frame
    # expectation is not the probability of the original branch.
    p_plus, p_minus = self._measurement_probabilities(pauli, where)
    if outcome is None:
        m = 1 if self._rng.random() < p_plus else -1
    else:
        m = 1 if int(outcome) >= 0 else -1
    prob = p_plus if m > 0 else p_minus
    if outcome is not None and prob <= 0.0:
        raise ValueError(
            f"forced measure outcome {outcome} has zero probability ({prob:.2e})."
        )
    # Move the orthogonality centre to the (anchor) collapse site so the
    # projector acts at the centre and truncation/renormalization stay
    # local and exactly tracked.
    anchor = min(int(site) for site in where)
    self.canonize_mps(self.p, anchor)
    input_norm = self._control_state_norm(include_exponent=False)
    input_exponent = self._real_float(self.p.exponent)

    collapse_center = None
    projector_submpo = self._build_pauli_projector_submpo(
        pauli,
        where,
        m,
    )
    if projector_submpo is not None:
        # Dense multi-site projectors stay as a bond-two MPO. DMRG receives
        # it as a lazy exact target; other MPS modes use their selected
        # Quimb compression method directly. This keeps target formation
        # separate from output compression and avoids a dense 2**k matrix.
        submpo, span = projector_submpo
        if self.mode == "dmrg" and mode_kwargs is not None:
            projected_norm, collapse_center = self._run_dmrg_measurement(
                submpo,
                span,
                n_iter=mode_kwargs["n_iter"],
                cutoff=cutoff,
                cutoff_mode=cutoff_mode,
                fit_min_iter=mode_kwargs["fit_min_iter"],
                fit_rtol=mode_kwargs["fit_rtol"],
                fit_patience=mode_kwargs["fit_patience"],
                fit_block_size=mode_kwargs["fit_block_size"],
                fit_adaptive_sweeps=mode_kwargs["fit_adaptive_sweeps"],
                fit_sweep_sequence=mode_kwargs["fit_sweep_sequence"],
                target_cutoff=mode_kwargs["target_cutoff"],
                fit_target_strategy=mode_kwargs["fit_target_strategy"],
                fit_mpo_guess=mode_kwargs["fit_mpo_guess"],
                fit_init_strategy=mode_kwargs["fit_init_strategy"],
                fit_init_rand_strength=mode_kwargs["fit_init_rand_strength"],
                fit_init_seed=mode_kwargs["fit_init_seed"],
                fit_single_pair_fast_path=mode_kwargs[
                    "fit_single_pair_fast_path"
                ],
                finite_check=mode_kwargs.get("finite_check", False),
                fit_overlap_diagnostics=mode_kwargs[
                    "fit_overlap_diagnostics"
                ],
                measurement_index=len(self.measurements),
            )
            # FIT returns a raw center norm. Keep its exponent separate
            # until the event ratio cancels the represented input scale.
        else:
            method = (
                self._mode_mpo_method(self.mode)
                if self._is_mpo_mode(self.mode)
                else "direct"
            )
            method_cutoff_mode = cutoff_mode
            if mode_kwargs is not None and self._is_mpo_mode(self.mode):
                method_cutoff_mode = mode_kwargs.get(
                    "mpo_cutoff_mode",
                    cutoff_mode,
                )
            self._apply_submpo_with_method(
                self.p,
                submpo,
                span,
                method=method,
                cutoff=cutoff,
                cutoff_mode=method_cutoff_mode,
                info=self.info_c,
            )
            projected_norm = self._control_state_norm(include_exponent=False)
    else:
        if len(where) == 1 and not self._replay_has_symmray_data(self.p):
            axis = str(pauli).strip().upper()
            projector = self._one_site_projector(axis, m)
        else:
            op = self._pauli_operator(pauli, where)
            dim = op.shape[0]
            projector = 0.5 * (np.eye(dim, dtype=complex) + m * op)
        self._apply_dense_operator(
            self.p,
            projector,
            where,
            max_bond=self.chi,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
        )
        projected_norm = self._control_state_norm(include_exponent=False)
    self._record_norm_event(
        norm_kind,
        # ``prob`` is the physical branch factor, while ``projected_norm``
        # is the norm after the selected approximate compression route.
        # The norm event records both effects without counting the branch
        # probability as compression infidelity.
        expected_norm=input_norm * math.sqrt(float(prob)),
        observed_norm=projected_norm,
        expected_exponent=input_exponent,
        observed_exponent=self._real_float(self.p.exponent),
        where=where,
        branch_probability=prob,
        physical_boundary=True,
        renormalized=renormalize,
    )
    self._finish_measurement_center(
        anchor if collapse_center is None else collapse_center,
        renormalize=renormalize,
    )
    self.measurements.append(
        (str(pauli), tuple(int(site) for site in record_where), int(m), float(prob))
    )
    return m


def _apply_basis_flip(self, q, axis, *, cutoff, cutoff_mode):
    """Flip the ``-axis`` eigenstate at site ``q`` to the ``+axis`` eigenstate."""
    flip_axis = _RESET_FLIP_AXES[axis]
    self._apply_dense_operator(
        self.p,
        self._control_operator(flip_axis),
        (q,),
        max_bond=self.chi,
        cutoff=cutoff,
        cutoff_mode=cutoff_mode,
    )
    # A single-site gate at the centre keeps the centre at q.
    self.info_c["cur_orthog"] = (q, q)


def _apply_reset_event(self, where, axes=None, *, cutoff, cutoff_mode):
    """Reset each qubit in ``where`` to the requested + Pauli eigenstate."""
    if axes is None:
        axes = ("Z",) * len(where)
    for site, axis in zip(where, axes):
        q = int(site)
        p_plus, p_minus = self._measurement_probabilities(axis, (q,))
        m = 1 if self._rng.random() < p_plus else -1
        if self._replay_has_symmray_data(self.p):
            projector = 0.5 * (
                np.eye(2, dtype=complex) + m * _PAULI_1Q[axis]
            )
        else:
            projector = self._one_site_projector(axis, m)
        # Centre at q, collapse, renormalize, and (if needed) flip |1> -> |0>,
        # keeping the tracked centre at q throughout.
        self.canonize_mps(self.p, q)
        input_norm = self._control_state_norm(include_exponent=False)
        input_exponent = self._real_float(self.p.exponent)
        self._apply_dense_operator(
            self.p,
            projector,
            (q,),
            max_bond=self.chi,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
        )
        projected_norm = self._control_state_norm(include_exponent=False)
        branch_probability = p_plus if m > 0 else p_minus
        self._record_norm_event(
            "reset",
            expected_norm=input_norm * math.sqrt(float(branch_probability)),
            observed_norm=projected_norm,
            expected_exponent=input_exponent,
            observed_exponent=self._real_float(self.p.exponent),
            where=(q,),
            branch_probability=branch_probability,
            physical_boundary=True,
            renormalized=True,
        )
        self._recanonize_center(q, renormalize=True)
        if m < 0:
            self._apply_basis_flip(
                q, axis, cutoff=cutoff, cutoff_mode=cutoff_mode
            )
    return self.p


def _apply_measure_reset_event(  # pylint: disable=too-many-arguments
    self,
    axes,
    where,
    outcomes,
    *,
    record_where,
    cutoff,
    cutoff_mode,
    mode_kwargs=None,
):
    """Measure each target, record it, then reset it to the + Pauli eigenstate."""
    record_sites = tuple(int(site) for site in record_where)
    for axis, site, record_site, outcome in zip(
        axes, where, record_sites, outcomes
    ):
        q = int(site)
        m = self._apply_measure_event(
            axis,
            (q,),
            outcome,
            record_where=(record_site,),
            renormalize=True,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            norm_kind="measure_reset",
            mode_kwargs=mode_kwargs,
        )
        if m < 0:
            self._apply_basis_flip(
                q, axis, cutoff=cutoff, cutoff_mode=cutoff_mode
            )
    return self.p


def _apply_cap_event(self, where, vec, absorb):
    """Contract site ``where``'s physical index with ``vec`` and shorten the MPS."""
    (q,) = (int(site) for site in where)
    p = self.p
    L = int(p.L)
    if not 0 <= q < L:
        raise ValueError(
            f"cap site {q} is outside the MPS range [0, {L})."
        )
    if L <= 1:
        raise ValueError("cannot cap the only site of a length-1 MPS.")

    self._invalidate_replay_metadata()
    # Cap vectors are state contractions rather than operator payloads.
    # The stream parser historically normalized them to complex dtype,
    # which made a real Torch MPS fail at the contraction boundary even
    # for the ordinary real vectors used to sum or project a binary leg.
    # Preserve genuinely complex caps, but discard an exactly-zero
    # imaginary part so the vector follows the live MPS backend/dtype.
    vec_arr = np.asarray(vec).ravel()
    if np.iscomplexobj(vec_arr) and np.all(np.imag(vec_arr) == 0):
        vec_arr = np.asarray(np.real(vec_arr))
    phys_ind = p.site_ind(q)
    phys_dim = p.ind_size(phys_ind)
    if vec_arr.shape[0] != phys_dim:
        raise ValueError(
            f"cap vector length {vec_arr.shape[0]} does not match the "
            f"physical dimension {phys_dim} of site {q}."
        )

    site_ind_id = p.site_ind_id
    site_tag_id = p.site_tag_id
    if absorb == "left":
        neighbour = q - 1 if q > 0 else q + 1
    else:
        neighbour = q + 1 if q < L - 1 else q - 1

    # Move the orthogonality centre onto the absorbing neighbour first: the
    # capped site is then an isometry adjacent to the centre, so merging it
    # in leaves the centre exactly on the (renumbered) neighbour. This keeps
    # the tracked centre exact without any rescan.
    self.canonize_mps(p, neighbour)
    new_center = neighbour if neighbour < q else neighbour - 1

    cap_tensor = qtn.Tensor(self._to_state_backend(vec_arr), inds=(phys_ind,))
    site_tensor = p[p.site_tag(q)]
    neighbour_tensor = p[p.site_tag(neighbour)]
    merged = qtn.tensor_contract(site_tensor, cap_tensor, neighbour_tensor)

    p.delete(p.site_tag(q))
    p.delete(p.site_tag(neighbour))
    merged.modify(tags=(p.site_tag(neighbour),))
    p |= merged

    # Renumber every site above the removed one down by one position.
    temp_reindex = {}
    temp_retag = {}
    for old in range(q + 1, L):
        temp_reindex[site_ind_id.format(old)] = f"__pepsy_cap_k{old - 1}"
        temp_retag[site_tag_id.format(old)] = f"__pepsy_cap_I{old - 1}"
    if temp_reindex:
        p.reindex_(temp_reindex)
    if temp_retag:
        p.retag_(temp_retag)
    final_reindex = {
        f"__pepsy_cap_k{i}": site_ind_id.format(i) for i in range(q, L - 1)
    }
    final_retag = {
        f"__pepsy_cap_I{i}": site_tag_id.format(i) for i in range(q, L - 1)
    }
    if final_reindex:
        p.reindex_(final_reindex)
    if final_retag:
        p.retag_(final_retag)

    capped = p.view_as_(
        qtn.MatrixProductState,
        L=L - 1,
        cyclic=False,
        site_ind_id=site_ind_id,
        site_tag_id=site_tag_id,
    )
    self.p = self._install_represented_norm(capped)
    self.info_c["cur_orthog"] = (new_center, new_center)
    # A raw cap can change the physical norm without any truncation.
    # Preserve accumulated compression loss, but let the next unitary
    # segment establish its baseline from this shorter state. No extra
    # norm contraction or diagnostic scan is needed at the cap boundary.
    self._invalidate_unitary_norm_baseline()
    self._mps_length_history.append(int(self.p.L))
    self.cap_history.append(
        {
            "physical_site": int(q),
            "old_length": int(L),
            "new_length": int(self.p.L),
            "absorb": str(absorb),
        }
    )
    return self.p


def _validate_event_stream_for_run(self, G_seq, where_seq, event_seq):
    """Validate queued event metadata before replay."""
    if not (len(G_seq) == len(where_seq) == len(event_seq)):
        raise ValueError(
            "MpsOptimizer event stream metadata is inconsistent: "
            "payloads, wheres, and event types must have the same length."
        )

    unknown = sorted(set(event_seq) - {"gate", "submpo"} - _CONTROL_EVENT_NAMES)
    if unknown:
        raise ValueError(f"Unknown MPS stream event type(s): {unknown!r}.")

    has_submpo = any(event_type == "submpo" for event_type in event_seq)
    if has_submpo and not (
        self._is_mpo_mode(self.mode) or self.mode == "dmrg"
    ):
        raise ValueError(
            "subMPO stream events require an MPO or DMRG mode."
        )

    has_cap = any(
        _control_event_contains_cap(event_type, payload)
        for payload, event_type in zip(G_seq, event_seq)
    )
    if not has_submpo:
        return

    # ``cap`` events shorten the MPS mid-stream, so a static site-range
    # check against the initial length is unreliable; those events are
    # validated dynamically as they are applied.
    L = int(getattr(self.p, "L", 0))
    for step, (where, event_type) in enumerate(
        zip(where_seq, event_seq),
        start=1,
    ):
        if event_type != "submpo":
            continue
        if len(set(where)) != len(where):
            raise ValueError(
                f"subMPO event at step {step} has repeated site(s): {where!r}."
            )
        if has_cap:
            continue
        out_of_range = [site for site in where if site < 0 or site >= L]
        if out_of_range:
            raise ValueError(
                f"subMPO event at step {step} references site(s) outside "
                f"the MPS range [0, {L}): {out_of_range!r}."
            )
