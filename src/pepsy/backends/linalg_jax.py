"""JAX-side linalg registrations with truncation-safe VJP rules."""

import autoray as ar
import jax
import jax.numpy as jnp
from jax import custom_vjp


_SVD_REGISTERED = False
_SVD_REGISTERED_FUNCTION = None
_NATIVE_QR = ar.get_lib_fn("jax", "linalg.qr")


@custom_vjp
def _svd_jax(A):
    """Thin JAX SVD with a Quimb-truncation-safe backward rule.

    Quimb's ``svd_truncated`` can pass cotangents only for the singular-vector
    columns it retained. The custom VJP restores those leading columns to the
    full thin-SVD output shape before applying a relative-regularized
    pullback. This is important for approximate tensor
    contractions, where a fixed ``max_bond`` is the normal JIT-compatible
    path.
    """
    return jnp.linalg.svd(A, full_matrices=False)

def h(x):
    """Return the conjugate transpose of ``x`` (Hermitian transpose)."""
    return jnp.conj(jnp.swapaxes(x, -1, -2))


@custom_vjp
def _qr_jax(a):
    return tuple(_NATIVE_QR(a))


def _qr_fwd(a):
    q, r = _NATIVE_QR(a)
    return (q, r), (a, q, r)


def _qr_bwd(residual, tangents):
    """Reduced QR VJP, with a finite extension at singular QR charts.

    Match Pepsy's Torch policy: preserve the exact QR forward and native
    derivative at nonzero pivots; only singular blocks use a relative
    Tikhonov right inverse. JAX complex cotangents use the conjugate of the
    convention used in the corresponding Torch rule.
    """
    a, q, r = residual
    dq, dr = (jnp.conj(t) for t in tangents)
    scale = jnp.max(jnp.abs(a))
    eps = max(1e-6, jnp.finfo(a.real.dtype).eps**.5)
    # A small nonzero pivot still has a defined QR derivative. Regularizing
    # it by relative magnitude destroys cancellations in composed MPS costs.
    # Use primal pivots (not cotangents) so the pullback remains linear.
    singular = jnp.any(jnp.diag(r) == 0)

    def native(_):
        # vmap lowers cond into masked branches with stop_gradient on the
        # operands. That prevents transposing the pullback in GMRES. Use a
        # primal-selected, nonsingular dummy chart for inactive native blocks
        # and select between the two linear pullbacks below instead.
        safe_a = jnp.where(singular, jnp.eye(*a.shape, dtype=a.dtype), a)
        _, pullback = jax.vjp(lambda x: tuple(_NATIVE_QR(x)), safe_a)
        return pullback(tangents)[0]

    def regularized(_):
        gradient = dr @ h(r) - h(q) @ dq
        shift = eps*jnp.where(scale == 0, 1., scale)

        def solve(rhs, tri):
            gram = h(tri) @ tri + shift**2*jnp.eye(tri.shape[-1], dtype=a.dtype)
            return h(jnp.linalg.solve(gram, h(rhs @ tri)))

        m, n = a.shape
        if m >= n:
            upper = jnp.triu(gradient)
            projected = upper + h(upper)
            projected = projected - jnp.diag(jnp.real(jnp.diag(upper)))
            # Keep the directly known R cotangent outside the regularized
            # solve. With an exact inverse this is the same QR formula, but
            # regularizing it too erases composed derivatives at zero pivots
            # (even a linear loss of q @ r when q spans the full space).
            base = q @ dr
            gradient = base + solve(q @ projected + dq - base @ h(r), r)
        else:
            skew = jnp.tril(h(gradient) - gradient)
            if jnp.iscomplexobj(a):
                skew = skew - .5j*jnp.diag(jnp.imag(jnp.diag(skew)))
            leading = solve(q @ skew, r[:, :m])
            gradient = jnp.pad(leading, ((0, 0), (0, n-m))) + q @ dr
        return jnp.conj(jnp.where(scale == 0, jnp.zeros_like(a), gradient))

    # A pullback must be linear in its incoming cotangents. Selecting a
    # branch by whether those cotangents or their resulting gradient are
    # finite breaks transposition in JAX's implicit linear solvers. Select
    # the chart exclusively from primal QR pivots instead.
    return (jnp.where(singular, regularized(None), native(None)),)


_qr_jax.defvjp(_qr_fwd, _qr_bwd)


def qr_jax(a, mode="reduced"):
    """Rank-aware reduced QR; other modes retain native JAX behavior."""
    if mode != "reduced":
        return jnp.linalg.qr(a, mode=mode)
    if a.ndim > 2:
        shape = a.shape
        q, r = jax.vmap(_qr_jax)(a.reshape((-1, *shape[-2:])))
        return (q.reshape((*shape[:-2], *q.shape[-2:])),
                r.reshape((*shape[:-2], *r.shape[-2:])))
    return _qr_jax(a)


def _restore_truncated_tangent(tangent, full, *, axis):
    """Pad a leading-rank Quimb cotangent to a thin-SVD output shape."""
    axis %= full.ndim
    if tangent is None:
        return jnp.zeros_like(full)
    if tangent.shape == full.shape:
        return tangent

    if tangent.ndim != full.ndim:
        raise TypeError(
            "SVD cotangent rank does not match the corresponding thin-SVD "
            f"output: got {tangent.shape!r}, expected {full.shape!r}."
        )
    for dim, (actual, expected) in enumerate(zip(tangent.shape, full.shape)):
        if dim != axis and actual != expected:
            raise TypeError(
                "SVD cotangent shape is incompatible with the corresponding "
                f"thin-SVD output: got {tangent.shape!r}, expected "
                f"{full.shape!r}."
            )
    if tangent.shape[axis] > full.shape[axis]:
        raise TypeError(
            "SVD cotangent has more singular-vector components than the "
            f"thin-SVD output: got {tangent.shape!r}, expected "
            f"{full.shape!r}."
        )

    slices = [slice(None)] * full.ndim
    slices[axis] = slice(0, tangent.shape[axis])
    return jnp.zeros_like(full).at[tuple(slices)].set(tangent)


def jaxsvd_fwd(A):
    """Forward rule for :func:`svd_jax`, retaining full thin-SVD shapes."""
    outputs = jnp.linalg.svd(A, full_matrices=False)
    return outputs, (A, outputs)


def jaxsvd_bwd(residual, tangents):
    """Full-spectrum pullback, exact outside unresolved spectral gaps.

    Retaining discarded vectors avoids the zero-discarded-spectrum
    approximation discussed in Francuz et al., arXiv:2311.11894. At an
    unresolved gap this is a bounded surrogate, not an exact derivative of
    a rank selection that cuts a degenerate multiplet.
    """
    A, outputs = residual
    U, S, Vh = outputs
    dU, dS, dVh = tangents
    cotangents = (
        _restore_truncated_tangent(dU, U, axis=-1),
        _restore_truncated_tangent(dS, S, axis=-1),
        _restore_truncated_tangent(dVh, Vh, axis=-2),
    )
    if S.shape[-1] == 0:
        return (jnp.zeros_like(A),)
    scale = jnp.max(S, axis=-1, keepdims=True)
    # Use the same compact stabilizer as Torch, with no squared spectrum.
    def reciprocal(x, spectral_scale, rtol=1e-6):
        threshold = jnp.maximum(spectral_scale * rtol, jnp.finfo(S.dtype).tiny)
        resolved = jnp.abs(x) >= threshold
        safe = jnp.where(resolved, x, jnp.ones_like(x))
        y = jnp.clip(x, -threshold, threshold) / threshold
        return jnp.where(resolved, 1 / safe, (2 * y - y**3) / threshold)

    sj, si = S[..., None, :], S[..., :, None]
    off_diagonal = 1 - jnp.eye(S.shape[-1], dtype=S.dtype)
    gap = reciprocal(sj - si, scale[..., None]) * off_diagonal
    total = reciprocal(sj + si, scale[..., None]) * off_diagonal
    inverse = reciprocal(S, scale, max(A.shape[-2:]) * jnp.finfo(S.dtype).eps)
    V = h(Vh)

    def differential(dA):
        # JAX fixes the diagonal phase in U. Transpose this real-linear map
        # with JAX itself so complex cotangent conventions stay consistent.
        core = h(U) @ dA @ V
        ds = jnp.diagonal(core.real, axis1=-2, axis2=-1)
        phase = .5 * (core - h(core)) * jnp.eye(S.shape[-1], dtype=S.dtype) * inverse[..., None, :]
        du = U @ (.5 * ((gap + total) * core + (gap - total) * h(core)) + phase)
        dv = V @ (.5 * ((gap - total) * core + (gap + total) * h(core)))
        if A.shape[-2] > S.shape[-1]:
            dav = dA @ V
            du = du + (dav - U @ (h(U) @ dav)) * inverse[..., None, :]
        if A.shape[-1] > S.shape[-1]:
            dahu = h(dA) @ U
            dv = dv + (dahu - V @ (Vh @ dahu)) * inverse[..., None, :]
        return du, ds, h(dv)

    return jax.linear_transpose(differential, A)(cotangents)


_svd_jax.defvjp(jaxsvd_fwd, jaxsvd_bwd)


def svd_jax(A, full_matrices=False, compute_uv=True, hermitian=False):
    """Truncation-safe, relative-regularized thin SVD.

    Options outside the thin, general-matrix rule use JAX's native SVD.
    Explicit ``full_matrices=False`` follows the same custom VJP as omission.
    The forward decomposition is unchanged. Reverse derivatives are exact at
    resolved spectra; unresolved gaps use a bounded surrogate.
    A degeneracy across the retained/discarded boundary remains nonsmooth.
    """
    if full_matrices or not compute_uv or hermitian:
        return jnp.linalg.svd(
            A, full_matrices=full_matrices, compute_uv=compute_uv,
            hermitian=hermitian,
        )
    return _svd_jax(A)


def _native_svd_jax(A, *args, **kwargs):
    """Use native JAX SVD with Pepsy's thin-factor default."""
    kwargs.setdefault("full_matrices", False)
    return jnp.linalg.svd(A, *args, **kwargs)


def _register_svd_jax(function):
    """Register one JAX SVD implementation and remember the active rule."""
    global _SVD_REGISTERED  # pylint: disable=global-statement
    global _SVD_REGISTERED_FUNCTION  # pylint: disable=global-statement
    if _SVD_REGISTERED and _SVD_REGISTERED_FUNCTION is function:
        return
    ar.register_function("jax", "linalg.svd", function)
    _SVD_REGISTERED = True
    _SVD_REGISTERED_FUNCTION = function


def reg_native_svd_jax():
    """Register native JAX thin SVD in autoray."""
    _register_svd_jax(_native_svd_jax)


def reset_jax_linalg_registrations():
    """Restore native JAX SVD and clear Pepsy's registration cache."""
    global _SVD_REGISTERED  # pylint: disable=global-statement
    global _SVD_REGISTERED_FUNCTION  # pylint: disable=global-statement
    _SVD_REGISTERED = False
    _SVD_REGISTERED_FUNCTION = None
    reg_native_svd_jax()
    if ar.get_lib_fn("jax", "linalg.qr") is not _NATIVE_QR:
        ar.register_function("jax", "linalg.qr", _NATIVE_QR)


def reg_complex_svd_jax():
    """Register the truncation-safe JAX thin-SVD implementation in autoray."""
    _register_svd_jax(svd_jax)


def reg_rel_svd_jax():
    """Register the truncation-safe JAX thin-SVD implementation in autoray."""
    reg_complex_svd_jax()


def reg_real_svd_jax():
    """Register the truncation-safe JAX thin-SVD implementation in autoray."""
    reg_complex_svd_jax()
