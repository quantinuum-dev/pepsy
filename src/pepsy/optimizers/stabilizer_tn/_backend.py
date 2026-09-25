"""Backend scalar bookkeeping and small classical decisions for STN replay."""

import autoray as ar
import numpy as np

from ...backends import to_float


def array_namespace(like):
    """Use cached early dispatch when available, otherwise Autoray dispatch."""
    factory = getattr(ar, "get_namespace", None)
    return factory(like=like) if callable(factory) else ar.numpy


def scalar_to_host(value):
    """Materialize diagnostic scalars only at a caller-requested boundary."""
    if isinstance(value, dict):
        return {key: scalar_to_host(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(scalar_to_host(item) for item in value)
    if getattr(value, "shape", None) == ():
        return to_float(value, real=True)
    return value


def diagnostic_scalar(value):
    """Detach bookkeeping while keeping the available real precision."""
    value = ar.do("stop_gradient", ar.do("real", value))
    if ar.infer_backend(value) in {"torch", "cupy"}:
        device = getattr(getattr(value, "device", None), "type", None)
        value = ar.astype(value, "float32" if device == "mps" else "float64")
    return value


def compression_event(before, after, segment_log, completed_log, previous_loss, *, step, kind):
    """Compute the segmented MPS retained-norm ledger with no host decisions.

    Norms are base-ten logarithms. Invalid update records are masked on the
    device and discarded by the public getter, preserving the host policy.
    """
    xp = array_namespace(after)
    raw = 2. * (after - before) * np.log(10.)
    valid = xp.logical_and(before != -np.inf,
                           xp.logical_not(xp.logical_or(xp.isnan(before), xp.isnan(raw))))
    local_log = xp.clip(raw, None, 0.)
    local = xp.exp(local_log)
    # Preserve the host double's complete-loss threshold, rather than
    # prematurely losing the ledger when a float32 display value underflows.
    zero_log = np.log(np.nextafter(0., 1.)) - np.log(2.)
    segment = xp.where(local_log <= zero_log, -np.inf, segment_log + local_log)
    segment = xp.where(valid, segment, segment_log)
    cumulative = completed_log + segment
    current = xp.where(valid, -xp.expm1(segment), previous_loss)
    return segment, current, {
        "step": step, "kind": kind, "valid": valid,
        "expected_norm": xp.exp(before * np.log(10.)),
        "observed_norm": xp.exp(after * np.log(10.)),
        "fidelity_raw": xp.exp(raw),
        "local_fidelity": local, "local_infidelity": -xp.expm1(local_log),
        "segment_fidelity": xp.exp(segment), "segment_infidelity": -xp.expm1(segment),
        "cumulative_fidelity": xp.exp(cumulative),
        "cumulative_infidelity": -xp.expm1(cumulative),
        "cumulative_compression_fidelity": xp.exp(cumulative),
        "cumulative_compression_infidelity": -xp.expm1(cumulative),
        "stabilized": False,
    }


def stabilizer_product_eigenstate(vector, *, tol=1e-10):
    """Classify a qubit using backend reductions and three host Bloch values."""
    vector = ar.do("reshape", vector, (-1,))
    if ar.shape(vector) != (2,):
        return None
    xp = array_namespace(vector)
    weights = xp.real(xp.conj(vector) * vector)
    norm_sq = xp.sum(weights)
    cross = xp.conj(vector[0]) * vector[1]
    bloch = xp.stack((2. * xp.real(cross), 2. * xp.imag(cross), weights[0] - weights[1]))
    bloch = bloch / xp.where(norm_sq > tol * tol, norm_sq, 1.)
    # A classical Stim tableau update needs this small branch decision. The
    # coefficient tensor and its normalization never leave the device.
    values = np.asarray(ar.to_numpy(xp.stop_gradient(bloch)))
    index = int(np.argmax(np.abs(values)))
    if abs(abs(values[index]) - 1.) > tol or not np.all(np.isfinite(values)):
        return None
    if any(abs(value) > tol for j, value in enumerate(values) if j != index):
        return None
    return "XYZ"[index], (1 if values[index] >= 0. else -1)
