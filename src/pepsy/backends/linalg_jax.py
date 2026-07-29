"""JAX-side linalg registrations with truncation-safe VJP rules."""

import autoray as ar
import jax
import jax.numpy as jnp
from jax import custom_vjp


@custom_vjp
def svd_jax(A):
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
    return jnp.conj(jnp.transpose(x))


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


svd_jax.defvjp(jaxsvd_fwd, jaxsvd_bwd)


def reg_complex_svd_jax():
    """Register the truncation-safe JAX thin-SVD implementation in autoray."""
    ar.register_function("jax", "linalg.svd", svd_jax)


def reg_rel_svd_jax():
    """Register the truncation-safe JAX thin-SVD implementation in autoray."""
    reg_complex_svd_jax()


def reg_real_svd_jax():
    """Register the truncation-safe JAX thin-SVD implementation in autoray."""
    reg_complex_svd_jax()
