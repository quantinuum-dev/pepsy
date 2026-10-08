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
    full thin-SVD output shape before delegating the actual derivative to
    JAX's maintained SVD pullback. This is important for approximate tensor
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
    derivative at resolved pivots; only singular blocks use a relative
    Tikhonov right inverse. JAX complex cotangents use the conjugate of the
    convention used in the corresponding Torch rule.
    """
    a, q, r = residual
    dq, dr = (jnp.conj(t) for t in tangents)
    scale = jnp.max(jnp.abs(a))
    eps = max(1e-6, jnp.finfo(a.real.dtype).eps**.5)
    singular = jnp.any(jnp.diag(r) == 0)

    def native(_):
        _, pullback = jax.vjp(lambda x: tuple(_NATIVE_QR(x)), a)
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
            gradient = solve(q @ projected + dq, r)
        else:
            skew = jnp.tril(h(gradient) - gradient)
            if jnp.iscomplexobj(a):
                skew = skew - .5j*jnp.diag(jnp.imag(jnp.diag(skew)))
            leading = solve(q @ skew, r[:, :m])
            gradient = jnp.pad(leading, ((0, 0), (0, n-m))) + q @ dr
        return jnp.conj(jnp.where(scale == 0, jnp.zeros_like(a), gradient))

    def adaptive(_):
        candidate = native(None)
        return jax.lax.cond(jnp.all(jnp.isfinite(candidate)),
                            lambda _: candidate, regularized, operand=None)

    active = jnp.any(dq != 0) | jnp.any(dr != 0)
    return (jax.lax.cond(active,
        lambda _: jax.lax.cond(singular, regularized, adaptive, operand=None),
        lambda _: jnp.zeros_like(a), operand=None),)


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
    """Differentiate a thin SVD after restoring Quimb's truncated tangents."""
    A, outputs = residual
    U, S, Vh = outputs
    dU, dS, dVh = tangents
    cotangents = (
        _restore_truncated_tangent(dU, U, axis=-1),
        _restore_truncated_tangent(dS, S, axis=-1),
        _restore_truncated_tangent(dVh, Vh, axis=-2),
    )
    _, pullback = jax.vjp(
        lambda matrix: jnp.linalg.svd(matrix, full_matrices=False),
        A,
    )
    cotangent_tree = jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(outputs),
        cotangents,
    )
    return pullback(cotangent_tree)


_svd_jax.defvjp(jaxsvd_fwd, jaxsvd_bwd)


def svd_jax(A, full_matrices=False, compute_uv=True, hermitian=False):
    """Truncation-safe thin SVD with standard JAX decomposition options.

    Options outside the thin, general-matrix rule use JAX's native SVD.
    Explicit ``full_matrices=False`` follows the same custom VJP as omission.
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
