"""Relay-BP: disordered-memory belief propagation on a tensor network.

This wraps quimb's **1-norm** belief propagation (``L1BP`` / ``HV1BP`` /
``D1BP``) for nonnegative / partition-function-style contractions, plus its
dense **2-norm** ``D2BP`` for wavefunction / PEPS-like networks. Plain
:func:`one_norm_bp` supports the three 1-norm implementations; the per-node
disordered-memory :func:`relay_bp` extension supports the directed ``L1BP``,
``D1BP``, and ``D2BP`` message layouts:

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

For D2BP, relay memory is restricted to a convex mixture (``0 <= gamma < 1``)
so that positive-semidefinite density-matrix messages stay positive
semidefinite.  The anti-memory values allowed for 1-norm relay BP are therefore
not available in the 2-norm path.

Pepsy project plan section 1 ("Convergence robustness -- disordered-memory /
relay-BP").  This is the tensor-network generalisation of tensy's standalone
Tanner-graph ``tensy.decoders.RelayBpDecoder``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import autoray as ar
import numpy as np

__all__ = [
    "BPState",
    "BPUpdateResult",
    "RelayBPResult",
    "one_norm_bp",
    "relay_bp",
    "two_norm_bp",
]

_ONE_NORM_CLASSES = {"l1bp": "L1BP", "hv1bp": "HV1BP", "d1bp": "D1BP"}
_RELAY_CLASSES = {**_ONE_NORM_CLASSES, "d2bp": "D2BP"}
_RELAY_METHODS = {"l1bp", "d1bp", "d2bp"}


def _method_key(method: str, classes: Mapping[str, str]) -> str:
    """Validate and normalize a public BP method name against ``classes``."""
    key = str(method).lower()
    if key not in classes:
        raise ValueError(
            f"method must be one of {sorted(classes)}; got {method!r}"
        )
    return key


def _bp_class(method_key: str):
    """Return the quimb BP class for a validated method key."""
    from quimb.tensor import belief_propagation as _bp

    return getattr(_bp, _RELAY_CLASSES[method_key])


def _message_data(message):
    """Return the array carried by either quimb message representation.

    ``L1BP`` stores individual ``Tensor`` messages, while ``D1BP``, ``D2BP``,
    and ``HV1BP`` expose backend arrays (the latter in batched containers).
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

    ``L1BP``, ``D1BP``, and ``D2BP`` expose a dictionary of messages, while
    ``HV1BP`` exposes a pair of dictionaries containing batched arrays.
    Preserve that public representation so a snapshot can be passed straight
    back as ``init_messages`` to :func:`one_norm_bp` or :func:`relay_bp`.
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

    Quimb's lazy ``L1BP`` messages are keyed by ``(source, target)``. Dense
    ``D1BP`` and ``D2BP`` messages instead use
    ``(bond_index, destination_tensor)``; the source is the other endpoint of
    that bond. Relay-BP disorder is per *source node*, not per bond or
    destination.
    """
    keys = tuple(bp.messages)
    if any((not isinstance(key, tuple)) or len(key) != 2 for key in keys):
        raise ValueError(
            "relay_bp requires directed dictionary messages; "
            f"method={method_key!r} uses a different message layout"
        )

    if method_key == "l1bp":
        return {key: key[0] for key in keys}

    if method_key not in {"d1bp", "d2bp"}:  # guarded at the public entry point
        raise ValueError(f"relay_bp does not support method={method_key!r}")

    sources = {}
    for index, destination in keys:
        tids = tuple(bp.tn.ind_map.get(index, ()))
        if len(tids) != 2 or destination not in tids:
            raise ValueError(
                "dense relay BP requires a closed pairwise tensor graph; "
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


def _d1_topology_signature(tn):
    """Return the stable pairwise topology identity used by D1BP warm starts."""
    return (
        frozenset(tn.tensor_map),
        frozenset(
            (index, frozenset(tids)) for index, tids in tn.ind_map.items()
        ),
    )


@dataclass
class BPUpdateResult:
    """Outcome of :meth:`BPState.update_local`.

    ``result`` is the updated D1BP state. ``updated_tids`` records the tensor
    sources whose outgoing messages were recomputed. A finite ``radius`` is a
    deliberately truncated light-cone update, so it is *not* a global BP
    fixed-point certificate; ``boundary_tids`` are the next sources that would
    need an update to extend that light cone.
    """

    result: RelayBPResult
    updated_tids: tuple[Any, ...]
    boundary_tids: tuple[Any, ...]
    radius: int | None
    fully_converged: bool
    used_local_scheduler: bool


@dataclass
class BPState:
    """Reusable D1BP fixed point for value-only changes on one TN topology.

    Construct this from a converged :class:`RelayBPResult` and call
    :meth:`update_local` after changing a known set of tensor values. It
    restores the old messages and seeds Quimb D1BP's incremental scheduler at
    exactly those tensors. With ``radius=None`` (the default), propagation
    continues until it reaches the new fixed point. With a finite radius it
    performs a bounded light-cone update, useful when only local observables
    are subsequently queried.

    The caller must list every tensor whose data changed, and the TN topology
    (tensor ids, index names, and bonds) must remain unchanged. This state is
    intentionally D1BP-only: its one-message-per-directed-bond layout makes
    the affected region explicit and is the layout shared with SU gauges.
    """

    result: RelayBPResult
    damping: float | None = None
    update: str | None = None
    bp_options: dict[str, Any] = field(default_factory=dict)
    _topology_signature: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.result.bp.__class__.__name__ != "D1BP":
            raise ValueError("BPState currently requires a D1BP result")

        bp = self.result.bp
        self._topology_signature = _d1_topology_signature(bp.tn)
        if self.damping is None:
            self.damping = float(getattr(bp, "_damping", 0.0))
        if self.update is None:
            self.update = getattr(bp, "update", "sequential")

        # Preserve the numerical convention of the supplied fixed point by
        # default; callers can deliberately override these at construction.
        defaults = {
            "normalize": getattr(bp, "_normalize", None),
            "distance": getattr(bp, "_distance", None),
            "local_convergence": getattr(bp, "local_convergence", True),
            "contract_every": getattr(bp, "contract_every", None),
        }
        defaults.update(self.bp_options)
        self.bp_options = defaults

    @classmethod
    def from_result(cls, result: RelayBPResult, **kwargs) -> "BPState":
        """Create a reusable state from a converged D1BP result.

        The result need not be rejected solely because a caller chose to keep
        an exploratory BP approximation, but a local update inherits that
        approximation. For fixed-point guarantees, construct the state from a
        result with ``result.converged`` true.
        """
        return cls(result=result, **kwargs)

    def update_local(
        self,
        tn,
        changed_tids,
        *,
        radius: int | None = None,
        max_iterations: int = 1000,
        tol: float = 5e-6,
        tol_abs: float | None = None,
        damping: float | None = None,
        update: str | None = None,
        **bp_overrides,
    ) -> BPUpdateResult:
        """Warm-update D1BP after changing tensor values in ``changed_tids``.

        ``radius`` counts propagation hops beyond the changed tensors. A value
        of zero recomputes only their outgoing messages; ``None`` continues
        until D1BP's local scheduler is empty and returns a fixed-point
        certificate. Unlike a full :func:`one_norm_bp` call, this path does
        not use DIIS because its extrapolated history is nonlocal.
        """
        if not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        if radius is not None and (
            not isinstance(radius, (int, np.integer)) or radius < 0
        ):
            raise ValueError("radius must be a nonnegative integer or None")
        if _d1_topology_signature(tn) != self._topology_signature:
            raise ValueError(
                "BPState belongs to a different tensor-network topology; "
                "local warm updates require the same tensor ids and bonds"
            )

        changed_tids = tuple(dict.fromkeys(changed_tids))
        if not changed_tids:
            raise ValueError("changed_tids must contain at least one tensor id")
        missing = set(changed_tids).difference(tn.tensor_map)
        if missing:
            raise ValueError(f"changed_tids are not present in the TN: {missing}")

        from .gauges import _validate_d1_graph

        _validate_d1_graph(tn)
        bp_cls = _bp_class("d1bp")
        opts = dict(self.bp_options)
        opts.update(bp_overrides)
        if not opts.get("local_convergence", True):
            raise ValueError(
                "BPState.update_local requires local_convergence=True; "
                "otherwise Quimb schedules a full BP sweep"
            )
        bp = bp_cls(
            tn,
            **_bp_constructor_kwargs(
                "d1bp",
                self.damping if damping is None else damping,
                self.update if update is None else update,
                opts,
            ),
        )
        _set_messages(bp, self.result.snapshot())

        # Current Quimb D1BP exposes ``touched`` as its concrete scheduler state:
        # it computes sources in this set, then enqueues only destinations
        # whose messages changed above tolerance. Feature-detect it so a
        # future upstream change fails safely rather than silently pretending
        # to have performed a local update.
        try:
            bp.touched = type(bp.touched)(changed_tids)
        except (AttributeError, TypeError):
            full = one_norm_bp(
                tn,
                method="d1bp",
                max_iterations=max_iterations,
                tol=tol,
                tol_abs=tol_abs,
                damping=self.damping if damping is None else damping,
                update=self.update if update is None else update,
                init_messages=self.result.snapshot(),
                **opts,
            )
            self.result = full
            return BPUpdateResult(
                result=full,
                updated_tids=tuple(tn.tensor_map),
                boundary_tids=(),
                radius=radius,
                fully_converged=full.converged,
                used_local_scheduler=False,
            )

        threshold = tol if tol_abs is None else tol_abs
        updated_tids: list[Any] = []
        max_mdiff = float("inf")
        boundary_tids: tuple[Any, ...] = ()
        fully_converged = False
        iterations = 0
        max_hops = None if radius is None else radius + 1

        for iterations in range(1, max_iterations + 1):
            active = tuple(bp.touched)
            updated_tids.extend(active)
            info = bp.iterate(tol=threshold)
            max_mdiff = float(info.get("max_mdiff", float("nan")))
            boundary_tids = tuple(bp.touched)

            if not boundary_tids:
                fully_converged = bool(
                    np.isfinite(max_mdiff) and max_mdiff < threshold
                )
                break
            if max_hops is not None and iterations >= max_hops:
                break

        result = RelayBPResult(
            bp=bp,
            converged=fully_converged,
            iterations=iterations,
            max_mdiff=max_mdiff,
            num_legs_run=1,
            quimb_converged=None,
        )
        self.result = result
        return BPUpdateResult(
            result=result,
            updated_tids=tuple(dict.fromkeys(updated_tids)),
            boundary_tids=boundary_tids,
            radius=radius,
            fully_converged=fully_converged,
            used_local_scheduler=True,
        )


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
    method_key = _method_key(method, _ONE_NORM_CLASSES)
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


def two_norm_bp(
    tn,
    *,
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
    """Run plain quimb D2BP to a fixed point on a wavefunction-like TN.

    D2BP contracts the 2-norm internally, so ``tn`` must be the physical
    state/PEPS-like tensor network, not its explicitly doubled norm network.
    The result uses the same wrapper as :func:`one_norm_bp` and
    :func:`relay_bp`, including detached message snapshots for warm starts.
    """
    if not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    bp = _bp_class("d2bp")(
        tn,
        **_bp_constructor_kwargs("d2bp", damping, update, bp_opts),
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
    gamma_range: tuple[float, float] | None = None,
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
        A quimb ``TensorNetwork``. ``L1BP`` / ``D1BP`` expect a nonnegative
        factor network; ``D2BP`` expects a wavefunction / PEPS-like network.
    method :
        Which BP representation to use (``"l1bp"``, ``"d1bp"``, or
        ``"d2bp"``). ``HV1BP`` is intentionally unsupported: its batched
        arrays do not expose a per-source message update for disordered
        memory.
    num_relays :
        Number of relay legs.  Each leg warm-starts from the previous leg's
        messages and re-draws the ``gamma`` disorder; the best-converged fixed
        point over all legs is returned.
    max_iterations :
        Message-passing sweeps per leg.
    gamma_range :
        Range from which per-node memory strengths are drawn uniformly and
        i.i.d. for each memory leg. For 1-norm BP it must satisfy
        ``min <= max < 1``; negative values give symmetry-breaking
        anti-memory. For D2BP it must satisfy ``0 <= min <= max < 1`` to
        preserve the positive-semidefinite message cone. Defaults to
        ``(-0.3, 0.9)`` for 1-norm BP and ``(0.0, 0.9)`` for D2BP.
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
    method_key = _method_key(method, _RELAY_CLASSES)
    if method_key not in _RELAY_METHODS:
        raise ValueError(
            "relay_bp supports only 'l1bp', 'd1bp', and 'd2bp'; HV1BP's batched "
            "message representation cannot apply per-node memory safely"
        )
    if not isinstance(num_relays, (int, np.integer)) or num_relays < 1:
        raise ValueError("num_relays must be a positive integer")
    if not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if gamma_range is None:
        gamma_range = (0.0, 0.9) if method_key == "d2bp" else (-0.3, 0.9)
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
    if method_key == "d2bp" and gamma_min < 0.0:
        raise ValueError(
            "D2BP relay requires gamma_range with 0 <= min <= max < 1.0 "
            "to preserve positive-semidefinite messages"
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
            # Out-of-band per-node memory injection invalidates quimb's local
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
