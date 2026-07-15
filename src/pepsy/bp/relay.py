"""Relay-BP: disordered-memory 1-norm belief propagation on a tensor network.

This wraps quimb's **1-norm** belief propagation (``L1BP`` / ``HV1BP`` /
``D1BP``) for nonnegative / partition-function-style contractions. Plain
:func:`one_norm_bp` supports all three; the per-node disordered-memory
:func:`relay_bp` extension supports the directed ``L1BP`` and ``D1BP`` message
layouts only:

* **Disordered memory.**  Each tensor node ``i`` is given a random *memory
  strength* ``gamma_i`` (drawn i.i.d. per node, possibly **negative**).  After
  each message-passing sweep the outgoing messages of node ``i`` are mixed with
  their previous value, ``gamma_i * old + (1 - gamma_i) * new``.  The
  heterogeneity (disorder) breaks the symmetric fixed points on which plain BP
  oscillates.  quimb's own ``damping`` hook is a *uniform* ``(old, new)``
  callable, so the per-node strength is applied here around quimb's
  :meth:`iterate` on the public ``messages`` dict.
* **Relay.**  Several BP *legs* are run; each leg warm-starts from the previous
  leg's messages but re-draws the ``gamma`` disorder, and the best-converged
  fixed point over all legs is returned.

pepsy PLAN.md section 1 ("Convergence robustness -- disordered-memory /
relay-BP").  This is the tensor-network generalisation of tensy's standalone
Tanner-graph ``tensy.decoders.RelayBpDecoder``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import autoray as ar
import numpy as np

__all__ = ["RelayBPResult", "one_norm_bp", "relay_bp"]

_ONE_NORM_CLASSES = {"l1bp": "L1BP", "hv1bp": "HV1BP", "d1bp": "D1BP"}
_RELAY_METHODS = {"l1bp", "d1bp"}


def _method_key(method: str) -> str:
    """Validate and normalize a public 1-norm BP method name."""
    key = str(method).lower()
    if key not in _ONE_NORM_CLASSES:
        raise ValueError(
            f"method must be one of {sorted(_ONE_NORM_CLASSES)}; got {method!r}"
        )
    return key


def _bp_class(method: str):
    """Return the quimb 1-norm BP class for ``method``."""
    from quimb.tensor import belief_propagation as _bp

    key = _method_key(method)
    return getattr(_bp, _ONE_NORM_CLASSES[key])


def _message_data(message):
    """Return the array carried by either quimb message representation.

    ``L1BP`` stores individual ``Tensor`` messages, while ``D1BP`` and
    ``HV1BP`` expose backend arrays (the latter in batched containers).
    Checking for ``modify`` deliberately avoids ``ndarray.data``, which is a
    memory-view rather than the message array.
    """
    return message.data if hasattr(message, "modify") else message


def _set_message(bp, key, message, data) -> None:
    """Replace one message, supporting Tensor and bare-array BPs."""
    if hasattr(message, "modify"):
        message.modify(data=data)
    else:
        bp.messages[key] = data


def _snapshot(messages):
    """Detach any quimb BP message representation.

    ``L1BP`` and ``D1BP`` expose a dictionary of messages, while ``HV1BP``
    exposes a pair of dictionaries containing batched arrays.  Preserve that
    public representation so a snapshot can be passed straight back as
    ``init_messages`` to :func:`one_norm_bp`.
    """
    if isinstance(messages, Mapping):
        return {key: _snapshot(value) for key, value in messages.items()}
    if isinstance(messages, tuple):
        return tuple(_snapshot(value) for value in messages)
    return ar.do("copy", _message_data(messages))


def _message_shape(message) -> tuple[int, ...]:
    """Return a backend-independent message shape for warm-start checks."""
    return tuple(ar.do("shape", _message_data(message)))


def _validate_message_tree(template, snapshot, path="messages") -> None:
    """Check that a snapshot exactly matches a BP message layout."""
    if isinstance(template, Mapping):
        if not isinstance(snapshot, Mapping):
            raise ValueError(f"{path} has an incompatible container type")
        if set(template) != set(snapshot):
            raise ValueError(
                f"{path} does not match this tensor-network topology; "
                "warm starts require exactly the same message keys"
            )
        for key, value in template.items():
            _validate_message_tree(value, snapshot[key], f"{path}[{key!r}]")
        return

    if isinstance(template, tuple):
        if not isinstance(snapshot, tuple) or len(template) != len(snapshot):
            raise ValueError(f"{path} has an incompatible container layout")
        for i, (value, saved) in enumerate(zip(template, snapshot)):
            _validate_message_tree(value, saved, f"{path}[{i}]")
        return

    if _message_shape(template) != _message_shape(snapshot):
        raise ValueError(
            f"{path} has shape {_message_shape(snapshot)}, expected "
            f"{_message_shape(template)} for this tensor-network topology"
        )


def _set_messages(bp, messages) -> None:
    """Warm-start a BP object from a snapshot of message arrays.

    The snapshot must have exactly the same keys, nesting, and shapes as the
    destination BP object, i.e. the tensor-network *topology* is unchanged and
    only tensor values differ (as when reusing a fixed point across successive
    rounds / shots).  Rejecting partial snapshots avoids silently mixing stale
    and freshly initialized messages.
    """
    _validate_message_tree(bp.messages, messages)
    if not isinstance(bp.messages, Mapping):
        # HV1BP owns batched arrays rather than individual mutable message
        # objects, so replacing the public state is the supported restore path.
        bp.messages = _snapshot(messages)
        return

    for key, message in bp.messages.items():
        _set_message(bp, key, message, ar.do("copy", messages[key]))


def _bp_constructor_kwargs(method_key: str, damping, update, bp_opts):
    """Adapt Pepsy's common API to the differing public quimb constructors."""
    if method_key == "hv1bp":
        if update not in (None, "parallel"):
            raise ValueError("HV1BP only supports update='parallel'")
        return {"damping": damping, **bp_opts}

    if update is None:
        update = "sequential"
    return {"damping": damping, "update": update, **bp_opts}


def _run_options(tol_abs, tol_rolling_diff) -> dict[str, Any]:
    """Forward strict convergence controls to quimb's public ``run`` API."""
    return {
        "tol_abs": tol_abs,
        "tol_rolling_diff": tol_rolling_diff,
    }


def _strict_converged(max_mdiff: float, tol: float, tol_abs: float | None) -> bool:
    """Whether the final residual, rather than rolling convergence, is small."""
    threshold = tol if tol_abs is None else tol_abs
    return bool(np.isfinite(max_mdiff) and max_mdiff < threshold)


def _relay_message_sources(bp, method_key: str) -> dict:
    """Map each relay message key to the tensor/site that sends it.

    Quimb's lazy ``L1BP`` messages are keyed by ``(source, target)``.  Dense
    ``D1BP`` messages instead use ``(bond_index, destination_tensor)``; the
    source is the other endpoint of that bond.  Relay-BP disorder is per
    *source node*, not per bond or destination.
    """
    keys = tuple(bp.messages)
    if any((not isinstance(key, tuple)) or len(key) != 2 for key in keys):
        raise ValueError(
            "relay_bp requires directed dictionary messages; "
            f"method={method_key!r} uses a different message layout"
        )

    if method_key == "l1bp":
        return {key: key[0] for key in keys}

    if method_key != "d1bp":  # guarded at the public entry point
        raise ValueError(f"relay_bp does not support method={method_key!r}")

    sources = {}
    for index, destination in keys:
        tids = tuple(bp.tn.ind_map.get(index, ()))
        if len(tids) != 2 or destination not in tids:
            raise ValueError(
                "D1BP relay requires a closed pairwise tensor graph; "
                f"invalid message key {(index, destination)!r}"
            )
        sources[index, destination] = tids[1] if tids[0] == destination else tids[0]
    return sources


@dataclass
class RelayBPResult:
    """Result of a (relay-)BP run.

    Attributes
    ----------
    bp :
        The quimb BP object restored to the best-scoring fixed point; use
        ``bp.messages`` as a gauge / to feed loop corrections.
    converged :
        Whether the best leg reached the *absolute* message tolerance. This is
        stricter than quimb's optional rolling-difference stopping criterion.
    quimb_converged :
        Whether quimb declared the best plain leg converged. This can be true
        from its rolling-difference criterion even when ``converged`` is false.
    iterations :
        Iterations taken by the best leg.
    max_mdiff :
        Final maximum message update distance of the best leg.
    num_legs_run :
        Number of relay legs executed.
    """

    bp: Any
    converged: bool
    iterations: int
    max_mdiff: float
    num_legs_run: int
    quimb_converged: bool | None = None

    def contract(self, **kwargs):
        """BP estimate of the contracted tensor-network scalar."""
        return self.bp.contract(**kwargs)

    @property
    def messages(self):
        """The fixed-point messages (BP gauge), live on the BP object."""
        return self.bp.messages

    def snapshot(self):
        """Detached copy of the fixed-point messages.

        Pass this as ``init_messages`` to :func:`one_norm_bp` / :func:`relay_bp`
        on the next round / shot (same tensor-network topology) to warm-start
        BP from the previous fixed point.
        """
        return _snapshot(self.bp.messages)


def one_norm_bp(
    tn,
    *,
    method: str = "l1bp",
    max_iterations: int = 1000,
    tol: float = 5e-6,
    tol_abs: float | None = None,
    tol_rolling_diff: float | None = 0.0,
    damping: float = 0.0,
    update: str | None = None,
    diis: bool | dict[str, Any] = True,
    progbar: bool = False,
    init_messages: Any | None = None,
    **bp_opts,
) -> RelayBPResult:
    """Run plain quimb 1-norm belief propagation to a fixed point.

    Uses quimb's own :meth:`run` loop and returns a :class:`RelayBPResult`.
    Absolute residual convergence is the default: ``tol_rolling_diff=0.0``
    disables quimb's plateau/oscillation early-stop heuristic. Pass a positive
    value explicitly if that heuristic is desired. ``HV1BP`` is automatically
    run in its required parallel mode; ``L1BP`` and ``D1BP`` default to
    sequential updates. Pass ``init_messages`` (a previous
    :meth:`RelayBPResult.snapshot`) to warm-start an identical topology.
    """
    method_key = _method_key(method)
    if not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if method_key == "d1bp":
        from .gauges import _validate_d1_graph

        _validate_d1_graph(tn)
    bp_class = _bp_class(method_key)
    bp = bp_class(
        tn,
        **_bp_constructor_kwargs(method_key, damping, update, bp_opts),
    )
    if init_messages is not None:
        _set_messages(bp, init_messages)
    info: dict = {}
    bp.run(
        max_iterations=max_iterations,
        tol=tol,
        diis=diis,
        progbar=progbar,
        info=info,
        **_run_options(tol_abs, tol_rolling_diff),
    )
    max_mdiff = float(info.get("max_mdiff", float("nan")))
    return RelayBPResult(
        bp=bp,
        converged=_strict_converged(max_mdiff, tol, tol_abs),
        iterations=int(info.get("iterations", 0)),
        max_mdiff=max_mdiff,
        num_legs_run=1,
        quimb_converged=bool(info.get("converged", False)),
    )


def relay_bp(
    tn,
    *,
    method: str = "l1bp",
    num_relays: int = 6,
    max_iterations: int = 200,
    gamma_range: tuple[float, float] = (-0.3, 0.9),
    tol: float = 5e-6,
    tol_abs: float | None = None,
    tol_rolling_diff: float | None = 0.0,
    damping: float = 0.0,
    update: str | None = None,
    memory_first_leg: bool = False,
    diis: bool | dict[str, Any] = True,
    init_messages: Any | None = None,
    seed: int | None = None,
    **bp_opts,
) -> RelayBPResult:
    """Relay-BP with per-node disordered memory (arXiv:2506.01779).

    Parameters
    ----------
    tn :
        A quimb ``TensorNetwork`` (nonnegative / factor network).
    method :
        Which BP representation to use (``"l1bp"`` or ``"d1bp"``).
        ``HV1BP`` is intentionally unsupported: its batched arrays do not
        expose a per-source message update for disordered memory.
    num_relays :
        Number of relay legs.  Each leg warm-starts from the previous leg's
        messages and re-draws the ``gamma`` disorder; the best-converged fixed
        point over all legs is returned.
    max_iterations :
        Message-passing sweeps per leg.
    gamma_range :
        Range from which per-node memory strengths are drawn uniformly and
        i.i.d. for each memory leg. It must satisfy ``min <= max < 1``;
        negative values give symmetry-breaking anti-memory.
    memory_first_leg :
        If ``False`` (default) the first leg is plain BP (no memory), so an
        easy / well-behaved network still returns the exact plain-BP fixed
        point; memory legs then try to rescue harder cases.
    init_messages :
        Optional previous :meth:`RelayBPResult.snapshot` to warm-start BP from
        an earlier round / shot with the same tensor-network topology.
    seed :
        Seed for the memory-disorder RNG.

    Returns
    -------
    RelayBPResult
    """
    method_key = _method_key(method)
    if method_key not in _RELAY_METHODS:
        raise ValueError(
            "relay_bp supports only 'l1bp' and 'd1bp'; HV1BP's batched "
            "message representation cannot apply per-node memory safely"
        )
    if not isinstance(num_relays, (int, np.integer)) or num_relays < 1:
        raise ValueError("num_relays must be a positive integer")
    if not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    try:
        gamma_min, gamma_max = map(float, gamma_range)
    except (TypeError, ValueError) as exc:
        raise ValueError("gamma_range must contain two finite floats") from exc
    if not (
        np.isfinite(gamma_min)
        and np.isfinite(gamma_max)
        and gamma_min <= gamma_max < 1.0
    ):
        raise ValueError(
            "gamma_range must satisfy finite min <= max < 1.0; gamma=1 "
            "would freeze a message and fake convergence"
        )
    if method_key == "d1bp":
        from .gauges import _validate_d1_graph

        _validate_d1_graph(tn)

    bp_class = _bp_class(method_key)
    rng = np.random.default_rng(seed)
    bp = bp_class(
        tn,
        **_bp_constructor_kwargs(method_key, damping, update, bp_opts),
    )

    if init_messages is not None:
        _set_messages(bp, init_messages)

    sources = _relay_message_sources(bp, method_key)
    nodes = tuple(dict.fromkeys(sources.values()))

    best: tuple | None = None
    for leg in range(num_relays):
        use_memory = memory_first_leg or leg > 0
        if not use_memory:
            # Plain leg: use quimb's DIIS-accelerated run.
            info: dict = {}
            bp.run(
                max_iterations=max_iterations,
                tol=tol,
                diis=diis,
                info=info,
                **_run_options(tol_abs, tol_rolling_diff),
            )
            iteration = int(info.get("iterations", 0))
            max_mdiff = float(info.get("max_mdiff", float("nan")))
            converged = _strict_converged(max_mdiff, tol, tol_abs)
            quimb_converged = bool(info.get("converged", False))
        else:
            # Out-of-band per-node memory injection invalidates D1BP's local
            # convergence bookkeeping, so a memory leg must recompute every
            # message. Keep quimb's default local convergence for an initial
            # plain leg: on some sequential D1BP graphs, forcing full sweeps
            # changes the update dynamics into a two-cycle even from a good
            # SU warm start.
            if hasattr(bp, "local_convergence"):
                bp.local_convergence = False
            gamma = {
                node: float(rng.uniform(gamma_min, gamma_max)) for node in nodes
            }
            converged = False
            quimb_converged = None
            iteration = 0
            max_mdiff = float("inf")
            for iteration in range(1, max_iterations + 1):
                prev = _snapshot(bp.messages)
                bp.iterate(tol=tol)
                sweep_mdiff = 0.0
                for key, message in bp.messages.items():
                    g = gamma[sources[key]]
                    data = _message_data(message)
                    if g != 0.0:
                        data = g * prev[key] + (1.0 - g) * data
                        _set_message(bp, key, message, data)
                    # Use the BP's selected distance metric on the post-memory
                    # messages, rather than silently switching to an L-infinity
                    # residual in relay legs.
                    delta = float(bp._distance_fn(prev[key], data))
                    sweep_mdiff = max(sweep_mdiff, delta)
                max_mdiff = sweep_mdiff
                if sweep_mdiff < tol:
                    converged = True
                    break

        score = (0 if converged else 1, max_mdiff)
        if best is None or score < best[0]:
            best = (
                score,
                converged,
                iteration,
                max_mdiff,
                quimb_converged,
                _snapshot(bp.messages),
            )
        # relay: keep bp.messages as the warm start for the next leg.

    _, converged, iteration, max_mdiff, quimb_converged, snapshot = best
    _set_messages(bp, snapshot)
    return RelayBPResult(
        bp=bp,
        converged=converged,
        iterations=iteration,
        max_mdiff=max_mdiff,
        num_legs_run=num_relays,
        quimb_converged=quimb_converged,
    )
