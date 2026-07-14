"""Relay-BP: disordered-memory 1-norm belief propagation on a tensor network.

This wraps quimb's **1-norm** belief propagation (``L1BP`` / ``HV1BP`` /
``D1BP``) -- the right family for the nonnegative / partition-function-style
contractions decoding needs -- and adds the **relay-BP** convergence-robustness
extension of Müller et al. (*Improved belief propagation is sufficient for
real-time decoding of quantum memory*, arXiv:2506.01779):

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

from dataclasses import dataclass
from typing import Any

import autoray as ar
import numpy as np

__all__ = ["RelayBPResult", "one_norm_bp", "relay_bp"]

_ONE_NORM_CLASSES = {"l1bp": "L1BP", "hv1bp": "HV1BP", "d1bp": "D1BP"}


def _bp_class(method: str):
    """Return the quimb 1-norm BP class for ``method``."""
    from quimb.tensor import belief_propagation as _bp

    key = str(method).lower()
    if key not in _ONE_NORM_CLASSES:
        raise ValueError(
            f"method must be one of {sorted(_ONE_NORM_CLASSES)}; got {method!r}"
        )
    return getattr(_bp, _ONE_NORM_CLASSES[key])


def _message_data(message):
    """Return the array carried by either quimb message representation.

    ``L1BP`` and ``HV1BP`` store messages as ``Tensor`` objects, whereas
    ``D1BP`` stores the bare backend arrays directly in ``messages``.
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


def _snapshot(messages) -> dict:
    """Backend-agnostic copy of the current message arrays keyed by bond."""
    return {
        key: ar.do("copy", _message_data(message))
        for key, message in messages.items()
    }


def _set_messages(bp, messages) -> None:
    """Warm-start a BP object from a snapshot of message arrays.

    Only keys present in both are set; their shapes must match, i.e. the
    tensor-network *topology* is unchanged and only the tensor values differ
    (as when reusing a fixed point across successive rounds / shots).
    """
    for key, message in bp.messages.items():
        data = messages.get(key)
        if data is not None:
            _set_message(bp, key, message, ar.do("copy", data))


@dataclass
class RelayBPResult:
    """Result of a (relay-)BP run.

    Attributes
    ----------
    bp :
        The quimb BP object restored to the best-scoring fixed point; use
        ``bp.messages`` as a gauge / to feed loop corrections.
    converged :
        Whether the best leg reached the message tolerance.
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

    def contract(self, **kwargs):
        """BP estimate of the contracted tensor-network scalar."""
        return self.bp.contract(**kwargs)

    @property
    def messages(self):
        """The fixed-point messages (BP gauge), live on the BP object."""
        return self.bp.messages

    def snapshot(self) -> dict:
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
    damping: float = 0.0,
    update: str = "sequential",
    diis: bool = True,
    progbar: bool = False,
    init_messages: dict | None = None,
    **bp_opts,
) -> RelayBPResult:
    """Run plain quimb 1-norm belief propagation to a fixed point.

    Uses quimb's own :meth:`run` loop, so ``diis`` DIIS acceleration and the
    rolling-difference convergence checks are available; returns a
    :class:`RelayBPResult`.  Pass ``init_messages`` (a previous
    :meth:`RelayBPResult.snapshot`) to warm-start from an earlier round.
    Use ``update="parallel"`` for Jacobi (parallelisable) sweeps.
    """
    bp_class = _bp_class(method)
    bp = bp_class(tn, damping=damping, update=update, **bp_opts)
    if init_messages:
        _set_messages(bp, init_messages)
    info: dict = {}
    bp.run(
        max_iterations=max_iterations, tol=tol, diis=diis, progbar=progbar, info=info
    )
    return RelayBPResult(
        bp,
        bool(info.get("converged", False)),
        int(info.get("iterations", 0)),
        float(info.get("max_mdiff", float("nan"))),
        1,
    )


def relay_bp(
    tn,
    *,
    method: str = "l1bp",
    num_relays: int = 6,
    max_iterations: int = 200,
    gamma_range: tuple[float, float] = (-0.3, 0.9),
    tol: float = 5e-6,
    damping: float = 0.0,
    update: str = "sequential",
    memory_first_leg: bool = False,
    diis: bool = True,
    init_messages: dict | None = None,
    seed: int | None = None,
    **bp_opts,
) -> RelayBPResult:
    """Relay-BP with per-node disordered memory (arXiv:2506.01779).

    Parameters
    ----------
    tn :
        A quimb ``TensorNetwork`` (nonnegative / factor network).
    method :
        Which 1-norm quimb BP to use (``"l1bp"``, ``"hv1bp"``, ``"d1bp"``).
    num_relays :
        Number of relay legs.  Each leg warm-starts from the previous leg's
        messages and re-draws the ``gamma`` disorder; the best-converged fixed
        point over all legs is returned.
    max_iterations :
        Message-passing sweeps per leg.
    gamma_range :
        Inclusive range the per-node memory strengths are drawn from (uniform,
        i.i.d. per node, re-drawn each memory leg).  Negative values give
        symmetry-breaking anti-memory.
    memory_first_leg :
        If ``False`` (default) the first leg is plain BP (no memory), so an
        easy / well-behaved network still returns the exact plain-BP fixed
        point; memory legs then try to rescue harder cases.
    init_messages :
        Optional previous :meth:`RelayBPResult.snapshot` to warm-start BP from
        an earlier round / shot (same tensor-network topology).  Combined with
        ``update="parallel"`` (Jacobi sweeps) and, for ``hv1bp``, a
        ``thread_pool=`` passed via ``**bp_opts``, this is how successive
        rounds reuse messages and parallelise.
    seed :
        Seed for the memory-disorder RNG.

    Returns
    -------
    RelayBPResult
    """
    bp_class = _bp_class(method)
    rng = np.random.default_rng(seed)
    # Out-of-band per-node memory injection invalidates quimb's
    # local-convergence bookkeeping, so recompute every message each sweep.
    bp_opts.setdefault("local_convergence", False)
    bp = bp_class(tn, damping=damping, update=update, **bp_opts)

    if init_messages:
        _set_messages(bp, init_messages)

    keys = list(bp.messages)
    if any((not isinstance(k, tuple)) or len(k) != 2 for k in keys):
        raise ValueError(
            "relay_bp disordered memory targets the lazy 1-norm BPs "
            "(l1bp / d1bp) whose messages are keyed by (source, target); "
            f"method={method!r} uses a different message layout."
        )
    nodes = sorted({k[0] for k in keys})

    best: tuple | None = None
    for leg in range(num_relays):
        use_memory = memory_first_leg or leg > 0
        if not use_memory:
            # Plain leg: use quimb's DIIS-accelerated run.
            info: dict = {}
            bp.run(max_iterations=max_iterations, tol=tol, diis=diis, info=info)
            converged = bool(info.get("converged", False))
            iteration = int(info.get("iterations", 0))
            max_mdiff = float(info.get("max_mdiff", float("nan")))
        else:
            gamma = {node: float(rng.uniform(*gamma_range)) for node in nodes}
            converged = False
            iteration = 0
            max_mdiff = float("inf")
            for iteration in range(1, max_iterations + 1):
                prev = _snapshot(bp.messages)
                bp.iterate(tol=tol)
                sweep_mdiff = 0.0
                for key, message in bp.messages.items():
                    g = gamma[key[0]]
                    data = _message_data(message)
                    if g != 0.0:
                        data = g * prev[key] + (1.0 - g) * data
                        _set_message(bp, key, message, data)
                    # post-mix change of this message (proper convergence metric)
                    delta = float(ar.do("max", ar.do("abs", data - prev[key])))
                    sweep_mdiff = max(sweep_mdiff, delta)
                max_mdiff = sweep_mdiff
                if sweep_mdiff < tol:
                    converged = True
                    break

        score = (0 if converged else 1, max_mdiff)
        if best is None or score < best[0]:
            best = (score, converged, iteration, max_mdiff, _snapshot(bp.messages))
        # relay: keep bp.messages as the warm start for the next leg.

    _, converged, iteration, max_mdiff, snapshot = best
    _set_messages(bp, snapshot)
    return RelayBPResult(bp, converged, iteration, max_mdiff, num_relays)
