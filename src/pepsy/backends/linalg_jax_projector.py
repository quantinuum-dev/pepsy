"""Fixed-rank JAX subspace factors with a composed differential.

The horizontal factor differential represents gauge-invariant functions of
the pair, including reconstruction. It does not differentiate a chosen
singular-vector gauge. Only the kept/discarded gap must be resolved.
"""

from functools import partial

import jax
import jax.numpy as jnp


def _adjoint(x):
    return jnp.conj(x.T)


def _polar_decomposition(a):
    """Scale-safe square polar factors, including a rank-chart check."""
    scale = jnp.max(jnp.abs(a))
    safe_scale = jnp.where(scale > 0, scale, jnp.ones_like(scale))
    u, s, vh = jnp.linalg.svd(a / safe_scale, full_matrices=False)
    resolved = (scale > 0) & jnp.all(s > a.shape[0]*jnp.finfo(s.dtype).eps*s[0])
    unitary = (u @ vh)*jnp.where(resolved, 1., jnp.nan)
    return unitary, s, vh, safe_scale


@jax.custom_jvp
def polar_unitary_jax(a):
    """Unitary polar factor with a smooth first-order, repeated-spectrum JVP.

    This private square-matrix primitive is used for FIT gauge alignment.
    Singular overlaps return nonfinite factors so the outer solve rejects
    the invalid chart. A polar factor depends on singular-value sums, never
    the internal differences that make raw singular-vector AD undefined.
    """
    return _polar_decomposition(a)[0]


@polar_unitary_jax.defjvp
def _polar_jvp(primals, tangents):
    (a,), (da,) = primals, tangents
    q, s, vh, scale = _polar_decomposition(a)
    scaled_tangent = da / scale
    skew = _adjoint(q) @ scaled_tangent - _adjoint(scaled_tangent) @ q
    core = vh @ skew @ _adjoint(vh)
    denominator = s[:, None] + s[None, :]
    omega = _adjoint(vh) @ (core / jnp.where(denominator > 0, denominator, 1.)) @ vh
    return q, q @ omega


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def jax_projector_split(a, rank):
    """Return isometric Q and B at an explicit, static retained rank."""
    u, s, vh = jnp.linalg.svd(a, full_matrices=False)
    return u[:, :rank], s[:rank, None] * vh[:rank]


@jax_projector_split.defjvp
def _projector_jvp(rank, primals, tangents):
    (a,), (da,) = primals, tangents
    u, s, vh = jnp.linalg.svd(a, full_matrices=False)
    q, sr, vr = u[:, :rank], s[:rank], vh[:rank]
    b = sr[:, None] * vr
    scale = jnp.maximum(s[0], jnp.finfo(s.dtype).tiny)
    resolved = jnp.all(sr / scale > max(a.shape) * jnp.finfo(s.dtype).eps)
    safe_sr = jnp.where(sr > 0, sr, jnp.ones_like(sr))
    # Parallel transport: Q^H dQ=0. Rotations inside a repeated retained
    # multiplet cancel between Q and B and never enter this differential.
    ud, sd, vd = u[:, rank:], s[rank:], vh[rank:]
    if rank < s.size:
        ratio = sd[:, None] / safe_sr[None, :]
        gap = (1 - ratio) * (1 + ratio)
        resolved = resolved & jnp.all(gap > max(a.shape) * jnp.finfo(s.dtype).eps)
        safe_gap = jnp.where(gap > 0, gap, jnp.ones_like(gap))
        core = _adjoint(ud) @ da @ _adjoint(vr)
        cross = _adjoint(_adjoint(q) @ da @ _adjoint(vd))
        mixing = ((core + ratio * cross) / safe_sr[None, :]) / safe_gap
        dq = ud @ mixing
    else:
        dq = jnp.zeros_like(q)
    if a.shape[0] > s.size:
        dav = da @ _adjoint(vr)
        dq = dq + (dav - u @ (_adjoint(u) @ dav)) / safe_sr[None, :]
    db = _adjoint(dq) @ a + _adjoint(q) @ da
    # A closed rank cut or an inactive retained dimension is not a smooth
    # chart. Propagate an explicit invalid gradient, never a finite surrogate.
    dq = dq * jnp.where(resolved, 1., jnp.nan)
    db = db * jnp.where(resolved, 1., jnp.nan)
    return (q, b), (dq, db)


def svd_reconstruct_jax(a, *, max_bond):
    """Reconstruct a fixed-rank matrix with a repeated-subspace-safe gradient.

    Both JVPs and VJPs are available. Retained repeated singular values are
    supported when the retained spectrum is positive and its cut is gapped.
    Closed gaps produce nonfinite derivatives. The forward is native truncated
    SVD. ``max_bond`` is a positive static integer; leading batch axes are allowed.
    """
    from numbers import Integral

    if isinstance(max_bond, bool) or not isinstance(max_bond, Integral) or max_bond < 1:
        raise ValueError("max_bond must be a positive static integer")
    if a.ndim < 2 or min(a.shape[-2:]) < 1:
        raise ValueError("a must contain nonempty matrices")
    rank = min(int(max_bond), *a.shape[-2:])
    if rank == min(a.shape[-2:]):
        # Without truncation the physical map is the identity, including at
        # zero/rank-deficient inputs where separate factors have no chart.
        return a
    if a.ndim > 2:
        shape = a.shape
        values = jax.vmap(lambda x: svd_reconstruct_jax(x, max_bond=rank))(
            a.reshape((-1, *shape[-2:]))
        )
        return values.reshape(shape)
    q, b = jax_projector_split(a, rank)
    return q @ b
