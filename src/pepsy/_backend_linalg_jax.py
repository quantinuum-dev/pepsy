"""JAX-side linalg registrations with custom VJP rules."""

import autoray as ar
import jax.numpy as jnp
from jax import custom_vjp


@custom_vjp
def svd_jax(A):
    """JAX SVD primitive wrapped with a custom VJP definition."""
    return jnp.linalg.svd(A, full_matrices=False)


def _safe_reciprocal(x, epsilon=1.0e-12):
    """Regularized reciprocal used by the JAX SVD backward expressions."""
    return x / (x * x + epsilon)


def h(x):
    """Return the conjugate transpose of ``x`` (Hermitian transpose)."""
    return jnp.conj(jnp.transpose(x))


def jaxsvd_fwd(A):
    """Forward rule for :func:`svd_jax` custom VJP."""
    u, s, v = svd_jax(A)
    return (u, s, v), (u, s, v)


def jaxsvd_bwd(residual, tangents):
    """Backward rule for :func:`svd_jax` custom VJP."""
    U, S, V = residual
    du, ds, dv = tangents

    dU = jnp.conj(du)
    dS = jnp.conj(ds)
    dV = jnp.transpose(dv)

    ms = jnp.diag(S)
    ms1 = jnp.diag(_safe_reciprocal(S))
    dAs = U @ jnp.diag(dS) @ V

    F = S * S - (S * S)[:, None]
    F = _safe_reciprocal(F) - jnp.diag(jnp.diag(_safe_reciprocal(F)))

    J = F * (h(U) @ dU)
    dAu = U @ (J + h(J)) @ ms @ V

    K = F * (V @ dV)
    dAv = U @ ms @ (K + h(K)) @ V

    O = h(dU) @ U @ ms1
    dAc = -1 / 2.0 * U @ (jnp.diag(jnp.diag(O - jnp.conj(O)))) @ V

    dAv = dAv + U @ ms1 @ h(dV) @ (jnp.eye(jnp.size(V[1, :])) - h(V) @ V)
    dAu = dAu + (jnp.eye(jnp.size(U[:, 1])) - U @ h(U)) @ dU @ ms1 @ V
    grad_a = jnp.conj(dAv + dAu + dAs + dAc)
    return (grad_a,)


svd_jax.defvjp(jaxsvd_fwd, jaxsvd_bwd)


def reg_complex_svd_jax():
    """Register the custom JAX SVD implementation in autoray."""
    ar.register_function("jax", "linalg.svd", svd_jax)
