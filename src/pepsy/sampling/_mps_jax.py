"""Dense JAX MPS conditionals; imported only for JAX-backed source tensors.

Compiled functions receive numerical arrays as runtime arguments, so code is
reused across sampler instances/refreshes without retaining stale tensors.
"""

from functools import partial
from numbers import Integral
import secrets

import jax
import jax.numpy as jnp
import numpy as np


def require(valid, message):
    """Validate once at the API boundary, or only on failure while tracing."""
    if isinstance(valid, jax.core.Tracer):
        def fail(_):
            def raise_invalid(flag):
                # vmap may execute both cond branches: inspect the actual
                # predicate rather than raising for a valid batched member.
                if not bool(flag):
                    raise ValueError(message)
            jax.debug.callback(raise_invalid, valid)
        jax.lax.cond(valid, lambda _: None, fail, operand=None)
    elif not bool(valid):
        raise ValueError(message)


def _safe(value):
    return jnp.where(value > 0, value, jnp.ones_like(value))


def _rescale(vector):
    scale = jnp.max(jnp.abs(vector), axis=1)
    return vector / _safe(scale)[:, None], jnp.log(_safe(scale))


@jax.jit
def _prepare(arrays):
    # Positive scalar gauges cancel from normalized probabilities/amplitudes.
    # Rescale both leaves and suffixes before any long products can overflow.
    scales = tuple(jnp.max(jnp.abs(array)) for array in arrays)
    valid = jnp.all(jnp.stack([jnp.all(jnp.isfinite(array)) & (scale > 0)
                              for array, scale in zip(arrays, scales)]))
    arrays = tuple(array / _safe(scale) for array, scale in zip(arrays, scales))
    right = [None] * (len(arrays) + 1)
    right[-1] = jnp.ones((1, 1), dtype=arrays[0].dtype)
    exponent = jnp.zeros((), dtype=arrays[0].real.dtype)
    for i in range(len(arrays) - 1, -1, -1):
        array = arrays[i]
        temporary = jnp.einsum('bc,dsc->dsb', right[i + 1], array)
        environment = jnp.einsum('asb,dsb->ad', array.conj(), temporary)
        scale = jnp.max(jnp.abs(environment))
        valid = valid & jnp.isfinite(scale) & (scale > 0)
        right[i] = environment / _safe(scale)
        exponent = exponent + jnp.log(_safe(scale))
    norm = right[0][0, 0].real
    valid = valid & jnp.isfinite(norm) & (norm > 0)
    return (arrays, tuple(right), exponent + jnp.log(_safe(norm))), valid


def prepare(arrays):
    plan, valid = _prepare(arrays)
    require(valid, 'MPS must have a finite non-zero norm.')
    return plan


def key(seed, reference):
    if seed is None:
        seed = secrets.randbits(32)
    if isinstance(seed, Integral):
        result = jax.random.key(int(seed))
    else:
        # Accept typed keys and legacy uint32[2] keys, including traced keys.
        jax.random.key_data(seed)
        result = seed
    device = getattr(reference, 'device', None)
    return jax.device_put(result, device) if device is not None else result


def _distribution(vector, array, right):
    amps = jnp.einsum('na,asb->nsb', vector, array)
    weights = jnp.sum((amps.conj() @ right) * amps, axis=2).real
    valid = jnp.all(jnp.isfinite(weights), axis=1)
    weights = jnp.maximum(weights, 0)
    totals = jnp.sum(weights, axis=1)
    valid = valid & (totals > 0) & jnp.isfinite(totals)
    probs = weights / _safe(totals)[:, None]
    # An invalid row cannot be sampled, but keep its categorical logits defined
    # until the single returned validity flag raises at the public boundary.
    fallback = jax.nn.one_hot(jnp.zeros(probs.shape[0], dtype=jnp.int32),
                              probs.shape[1], dtype=probs.dtype)
    return amps, jnp.where(valid[:, None], probs, fallback), valid


@partial(jax.jit, static_argnames=('n_samples',))
def draw(plan, rng, n_samples):
    arrays, right, _log_norm = plan
    vector = jnp.ones((n_samples, 1), dtype=arrays[0].dtype)
    log_probability = jnp.zeros(n_samples, dtype=arrays[0].real.dtype)
    valid = jnp.array(True)
    rows = jnp.arange(n_samples)
    configs = []
    for site, array in enumerate(arrays):
        amps, probs, good = _distribution(vector, array, right[site + 1])
        # High mode avoids the float32 low-Gumbel loss of rare categories.
        choices = jax.random.categorical(jax.random.fold_in(rng, site),
            jnp.log(probs), axis=-1, mode='high').astype(jnp.int32)
        chosen = probs[rows, choices]
        log_probability = log_probability + jnp.log(_safe(chosen))
        vector, _shift = _rescale(amps[rows, choices, :])
        valid = valid & jnp.all(good) & jnp.all(chosen > 0)
        configs.append(choices)
    return jnp.stack(configs, axis=1), jnp.exp(log_probability), valid


def configs(value, arrays):
    value = jnp.asarray(value, dtype=jnp.int32)
    device = getattr(arrays[0], 'device', None)
    if device is not None:
        value = jax.device_put(value, device)
    if value.ndim != 2 or value.shape[1] != len(arrays):
        raise ValueError(f'configs must have shape (batch, L={len(arrays)})')
    valid = jnp.array(True)
    for i, array in enumerate(arrays):
        valid = valid & jnp.all((value[:, i] >= 0) & (value[:, i] < array.shape[1]))
    require(valid, 'configs contain invalid physical indices.')
    return value


@jax.jit
def probabilities(plan, value):
    arrays, right, _log_norm = plan
    vector = jnp.ones((value.shape[0], 1), dtype=arrays[0].dtype)
    result = jnp.zeros(value.shape[0], dtype=arrays[0].real.dtype)
    alive = jnp.ones(value.shape[0], dtype=bool)
    valid = jnp.array(True)
    rows = jnp.arange(value.shape[0])
    for site, array in enumerate(arrays):
        amps, probs, good = _distribution(vector, array, right[site + 1])
        choices = value[:, site]
        chosen = probs[rows, choices]
        valid = valid & jnp.all(good | ~alive)
        alive = alive & good & (chosen > 0)
        result = result + jnp.log(_safe(chosen))
        vector, _shift = _rescale(amps[rows, choices, :])
    return jnp.where(alive, jnp.exp(result), 0), valid


@jax.jit
def amplitudes(plan, value):
    arrays, _right, log_norm = plan
    vector = jnp.ones((value.shape[0], 1), dtype=arrays[0].dtype)
    exponent = jnp.zeros(value.shape[0], dtype=arrays[0].real.dtype)
    for site, array in enumerate(arrays):
        selected = jnp.transpose(array, (1, 0, 2))[value[:, site]]
        vector = jnp.einsum('na,nab->nb', vector, selected)
        vector, shift = _rescale(vector)
        exponent = exponent + shift
    final = vector[:, 0]
    magnitude = jnp.abs(final)
    phase = final / _safe(magnitude)
    return phase * jnp.exp(exponent + jnp.log(_safe(magnitude)) - log_norm / 2)


def validate_arrays(arrays):
    """Metadata-only validation; never stage tensors through NumPy."""
    dtype = np.dtype(arrays[0].dtype)
    if dtype not in (np.dtype('float32'), np.dtype('float64'),
                     np.dtype('complex64'), np.dtype('complex128')):
        raise ValueError('native JAX MPS sampling requires float32/64 or complex64/128')
    device = None
    for array in arrays:
        if array.dtype != dtype:
            raise ValueError('JAX MPS tensors must share one dtype')
        current = getattr(array, 'device', None)
        if current is not None:
            if device is not None and device != current:
                raise ValueError('JAX MPS tensors must share one device')
            device = current
