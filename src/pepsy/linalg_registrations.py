"""Autodiff-safe linear algebra registrations for torch and jax backends.

This module provides custom SVD/QR gradients and then registers them with
``autoray`` so higher-level tensor code can use the stabilized routines.
"""

import autoray as ar
import jax.numpy as jnp
import scipy.linalg
import torch
from jax import custom_vjp

# pylint: disable=abstract-method,arguments-differ,bad-staticmethod-argument,bare-except,line-too-long,multiple-statements,not-callable,superfluous-parens,too-many-branches,too-many-locals,too-many-statements,unnecessary-semicolon,unused-variable,using-constant-test
# jax.config.update("jax_enable_x64", True)



def safe_inverse(x, eps_abs=1.0e-12):
    """Return a smooth reciprocal-like map ``x / (x**2 + eps_abs)``.

    This is used as a regularized inverse in singular-value expressions where
    exact reciprocal factors may become numerically unstable.
    """
    eps_abs=1.0e-12
    return x / (x ** 2 + eps_abs)


def safe_inverse_2(x, eps):
    """Return a clamped reciprocal used for real nonnegative values.

    Values below ``eps`` are clipped before inversion.
    """
    return x.clamp_min(eps).reciprocal()


class SVD(torch.autograd.Function):
    """Custom torch SVD with stabilized backward for near-degenerate spectra.

    The forward computes ``torch.linalg.svd`` and the backward adds standard
    regularization terms used in tensor-network optimization workflows.
    """

    @staticmethod
    def forward(ctx, A):
        if A.is_cuda:
            U, S, Vh = torch.linalg.svd(A, full_matrices=False, driver='gesvd')
        else:
            U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        # A = U @ diag(S) @ Vh
        ctx.save_for_backward(U, S, Vh)

        return U, S, Vh


    @staticmethod
    def backward(ctx, gu, gsigma, gvh):
        r"""
        param gu: gradient on U
        param gsigma: gradient on S
        type gsigma: torch.Tensor
        param gv: gradient on V

        Computes backward gradient for SVD, adopted from
        https://github.com/pytorch/pytorch/blob/v1.10.2/torch/csrc/autograd/FunctionsManual.cpp

        For complex-valued input there is an additional term, see

            * https://giggleliu.github.io/2019/04/02/einsumbp.html
            * https://arxiv.org/abs/1909.02659

        The backward is regularized following

            * https://github.com/wangleiphy/tensorgrad/blob/master/tensornets/adlib/svd.py
            * https://arxiv.org/abs/1903.09650

        using

        .. math::
            S_i/(S^2_i-S^2_j) = (F_{ij}+G_{ij})/2\ \ \textrm{and}\ \ S_j/(S^2_i-S^2_j) = (F_{ij}-G_{ij})/2

        where

        .. math::
            F_{ij}=1/(S_i-S_j),\ G_{ij}=1/(S_i+S_j)
        """

        # TORCH_CHECK(compute_uv,
        #    "svd_backward: Setting compute_uv to false in torch.svd doesn't compute singular matrices, ",
        #    "and hence we cannot compute backward. Please use torch.svd(compute_uv=True)");

        diagnostics = None

        u, sigma, vh= ctx.saved_tensors
        m= u.size(0) # first dim of original tensor A = u sigma v^\dag
        n= vh.size(1) # second dim of A
        k= sigma.size(0)
        scaled_eps= 1.e-12

        #
        if (u.size(-2)!=u.size(-1)) or (vh.size(-2)!=vh.size(-1)):
            # We ignore the free subspace here because possible base vectors cancel
            # each other, e.g., both -v and +v are valid base for a dimension.
            # Don't assume behavior of any particular implementation of svd.
            u = u.narrow(-1, 0, k)
            vh = vh.narrow(-2, 0, k)
            if not (gu is None): gu = gu.narrow(-1, 0, k)
            if not (gvh is None): gvh = gvh.narrow(-2, 0, k)


        if not (gsigma is None):
            # computes u @ diag(gsigma) @ vh
            sigma_term = u * gsigma.unsqueeze(-2) @ vh
        else:
            sigma_term = torch.zeros(m,n,dtype=u.dtype,device=u.device)
        # in case that there are no gu and gvh, we can avoid the series of kernel
        # calls below
        if (gu is None) and (gvh is None):
            if not (diagnostics is None):
                print(f"{diagnostics} {sigma_term.abs().max()} {sigma.max()}")
            return sigma_term, None, None, None


        # sigma_inv= safe_inverse_2(sigma.clone(), sigma_scale*eps)
        # sigma_inv= safe_inverse(sigma.clone(), eps_abs=sigma_scale*eps)
        sigma_inv= safe_inverse(sigma.clone(), eps_abs= scaled_eps)

        F = sigma.unsqueeze(-2) - sigma.unsqueeze(-1)
        F = safe_inverse(F, eps_abs= scaled_eps)
        F.diagonal(0,-2,-1).fill_(0)

        G = sigma.unsqueeze(-2) + sigma.unsqueeze(-1)
        G = safe_inverse(G, eps_abs= scaled_eps)
        G.diagonal(0,-2,-1).fill_(0)

        uh= u.conj().transpose(-2,-1)
        if not (gu is None):
            guh = gu.conj().transpose(-2, -1);
            u_term = u @ ( (F+G).mul( uh @ gu - guh @ u) ) * 0.5
            if m > k:
                # projection operator onto subspace orthogonal to span(U) defined as I - UU^H
                proj_on_ortho_u = -u @ uh
                proj_on_ortho_u.diagonal(0, -2, -1).add_(1);
                u_term = u_term + proj_on_ortho_u @ (gu * sigma_inv.unsqueeze(-2))
            u_term = u_term @ vh
        else:
            u_term = torch.zeros(m,n,dtype=u.dtype,device=u.device)

        v= vh.conj().transpose(-2,-1)
        if not (gvh is None):
            gv = gvh.conj().transpose(-2, -1);
            v_term = ( (F-G).mul(vh @ gv - gvh @ v) ) @ vh * 0.5
            if n > k:
                # projection operator onto subspace orthogonal to span(V) defined as I - VV^H
                proj_on_v_ortho =  -v @ vh
                proj_on_v_ortho.diagonal(0, -2, -1).add_(1);
                v_term = v_term + sigma_inv.unsqueeze(-1) * (gvh @ proj_on_v_ortho)
            v_term = u @ v_term
        else:
            v_term = torch.zeros(m,n,dtype=u.dtype,device=u.device)


        # // for complex-valued input there is an additional term
        # // https://giggleliu.github.io/2019/04/02/einsumbp.html
        # // https://arxiv.org/abs/1909.02659
        dA= u_term + sigma_term + v_term
        if u.is_complex() or v.is_complex():
            L= (uh @ gu).diagonal(0,-2,-1)
            L.real.zero_()
            L.imag.mul_(sigma_inv)
            imag_term= (u * L.unsqueeze(-2)) @ vh
            dA= dA + imag_term

        if diagnostics is not None:
            print(f"{diagnostics} {dA.abs().max()} {sigma.max()}")

        return dA, None, None, None





# def safe_inverse(x, eps_abs: float = 1.0e-12):
#     """
#     Smooth, sign/phase-preserving 'inverse-like' regularizer.

#     For real x:   x / (x^2 + eps)
#     For complex x: x / (|x|^2 + eps)  where |x|^2 = x * conj(x)

#     This matches your intent for F and G (built from real singular values),
#     and is also well-defined if ever called on complex tensors.
#     """
#     if torch.is_complex(x):
#         denom = x * x.conj() + eps_abs
#     else:
#         denom = x * x + eps_abs
#     return x / denom


# def safe_inverse_2(x, eps: float):
#     """
#     Hard reciprocal with clamp. Only appropriate for nonnegative real tensors.
#     (Not appropriate for F = 1/(s_i - s_j) which can be negative.)
#     """
#     return x.clamp_min(eps).reciprocal()


# class SVD(torch.autograd.Function):
#     @staticmethod
#     def forward(A):
#         # Do NOT force driver='gesvd' on CUDA: it can be much slower.
#         # Let PyTorch choose the best driver.
#         U, S, Vh = torch.linalg.svd(A, full_matrices=False)
#         return U, S, Vh

#     @staticmethod
#     def setup_context(ctx, inputs, output):
#         # Required by functorch transforms (vmap, grad, jvp, jacrev, ...)
#         (A,) = inputs
#         U, S, Vh = output
#         ctx.save_for_backward(U, S, Vh)
#         ctx.set_materialize_grads(False)

#     @staticmethod
#     def backward(ctx, gu, gsigma, gvh):
#         """
#         Backward for A -> (U, S, Vh).

#         Shapes (batched-safe):
#           A:      (..., m, n)
#           U:      (..., m, k)
#           S:      (..., k)
#           Vh:     (..., k, n)
#           k = min(m, n)

#         Returns:
#           dA:     (..., m, n)
#         """
#         u, sigma, vh = ctx.saved_tensors
#         eps = 1.0e-12

#         # Batch-safe sizes
#         m = u.size(-2)
#         k = u.size(-1)
#         n = vh.size(-1)
#         batch_shape = u.shape[:-2]

#         # sigma_term = U * diag(gsigma) @ Vh
#         if gsigma is not None:
#             sigma_term = (u * gsigma.unsqueeze(-2)) @ vh
#         else:
#             sigma_term = torch.zeros((*batch_shape, m, n), dtype=u.dtype, device=u.device)

#         # If only singular values have grad, stop early
#         if gu is None and gvh is None:
#             return (sigma_term,)

#         # Your chosen regularizer for sigma^{-1}
#         sigma_inv = safe_inverse(sigma.clone(), eps_abs=eps)  # (..., k)

#         # Build F and G from singular values (real), and apply safe_inverse to both
#         F = sigma.unsqueeze(-2) - sigma.unsqueeze(-1)         # (..., k, k)
#         F = safe_inverse(F, eps_abs=eps)

#         G = sigma.unsqueeze(-2) + sigma.unsqueeze(-1)         # (..., k, k)
#         G = safe_inverse(G, eps_abs=eps)

#         # Zero diagonals WITHOUT in-place diagonal writes (transform-friendly)
#         eye = torch.eye(k, dtype=F.dtype, device=F.device).view(*(1,) * len(batch_shape), k, k)
#         F = F * (1.0 - eye)
#         G = G * (1.0 - eye)

#         uh = u.conj().transpose(-2, -1)                       # (..., k, m)

#         # ---- U term
#         if gu is not None:
#             guh = gu.conj().transpose(-2, -1)                 # (..., k, m)

#             # (uh @ gu - guh @ u): (..., k, k)
#             skew_u = (uh @ gu) - (guh @ u)
#             u_term = u @ ((F + G) * skew_u) * 0.5             # (..., m, k)

#             if m > k:
#                 # (I - U U^H) @ (gu * sigma_inv)
#                 X = gu * sigma_inv.unsqueeze(-2)              # (..., m, k)
#                 X = X - u @ (uh @ X)                          # (..., m, k)
#                 u_term = u_term + X

#             u_term = u_term @ vh                              # (..., m, n)
#         else:
#             u_term = torch.zeros((*batch_shape, m, n), dtype=u.dtype, device=u.device)

#         # ---- V term
#         v = vh.conj().transpose(-2, -1)                       # (..., n, k)

#         if gvh is not None:
#             gv = gvh.conj().transpose(-2, -1)                 # (..., n, k)

#             # (vh @ gv - gvh @ v): (..., k, k)
#             skew_v = (vh @ gv) - (gvh @ v)
#             v_term = (((F - G) * skew_v) @ vh) * 0.5          # (..., k, n)

#             if n > k:
#                 # gvh @ (I - V V^H)  (right-projection) without forming I:
#                 # Z = gvh - (gvh @ V) @ Vh
#                 Z = gvh - (gvh @ v) @ vh                      # (..., k, n)
#                 v_term = v_term + sigma_inv.unsqueeze(-1) * Z

#             v_term = u @ v_term                               # (..., m, n)
#         else:
#             v_term = torch.zeros((*batch_shape, m, n), dtype=u.dtype, device=u.device)

#         dA = u_term + sigma_term + v_term

#         # Complex correction term (only meaningful if input is complex and gu is present)
#         if (u.is_complex() or vh.is_complex()) and (gu is not None):
#             L = (uh @ gu).diagonal(0, -2, -1)                 # (..., k)
#             # Keep imaginary part only and scale by sigma_inv
#             L = torch.complex(torch.zeros_like(L.real), L.imag * sigma_inv)
#             dA = dA + (u * L.unsqueeze(-2)) @ vh

#         # Only one tensor input (A), so only one gradient returned
#         return (dA,)

#     @staticmethod
#     def vmap(info, in_dims, A):
#         """
#         vmap rule. torch.linalg.svd is already batched, so we just apply
#         the same Function to the batched tensor.
#         """
#         (A_bdim,) = in_dims

#         if A_bdim is None:
#             U, S, Vh = SVD.apply(A)
#             return (U, S, Vh), (None, None, None)

#         if A_bdim != 0:
#             A = A.movedim(A_bdim, 0)

#         U, S, Vh = SVD.apply(A)
#         return (U, S, Vh), (0, 0, 0)


# def svd_custom(A):
#     return SVD.apply(A)






class SVD_real(torch.autograd.Function):
    """Real-valued SVD autograd function with scipy fallback on failure.

    This variant is intended for real tensors and keeps signs consistent across
    repeated decompositions to reduce optimization noise.
    """

    @staticmethod
    def forward(self, A):
        try:
            U, S, V = torch.svd(A)
        except:
            if True:
                print('trouble in torch gesdd routine, falling back to gesvd')
            U, S, V = scipy.linalg.svd(A.detach().numpy(), full_matrices=False, lapack_driver='gesvd')
            U = torch.from_numpy(U)
            S = torch.from_numpy(S)
            V = torch.from_numpy(V.T)

        # make SVD result sign-consistent across multiple runs
        for idx in range(U.size()[1]):
            if max(torch.max(U[:,idx]), torch.min(U[:,idx]), key=abs) < 0.0:
                U[:,idx] *= -1.0
                V[:,idx] *= -1.0

        self.save_for_backward(U, S, V)
        return U, S, V.t()

    @staticmethod
    def backward(self, dU, dS, dV):
        U, S, V = self.saved_tensors
        dV = dV.t()
        Vt = V.t()
        Ut = U.t()
        M = U.size(0)
        N = V.size(0)
        NS = len(S)

        F = (S - S[:, None])
        F = safe_inverse(F)
        F.diagonal().fill_(0)

        G = (S + S[:, None])
        #G.diagonal().fill_(np.inf)
        #G = 1/G
        G = safe_inverse(G)
        G.diagonal().fill_(0)

        UdU = Ut @ dU
        VdV = Vt @ dV

        Su = (F+G)*(UdU-UdU.t())/2
        Sv = (F-G)*(VdV-VdV.t())/2

        dA = U @ (Su + Sv + torch.diag(dS)) @ Vt
        if (M>NS):
            #dA = dA + (torch.eye(M, dtype=dU.dtype, device=dU.device) - U@Ut) @ (dU/S) @ Vt
            dA = dA + (torch.eye(M, dtype=dU.dtype, device=dU.device) - U@Ut) @ (dU*safe_inverse(S)) @ Vt
        if (N>NS):
            #dA = dA + (U/S) @ dV.t() @ (torch.eye(N, dtype=dU.dtype, device=dU.device) - V@Vt)
            dA = dA + (U*safe_inverse(S)) @ dV.t() @ (torch.eye(N, dtype=dU.dtype, device=dU.device) - V@Vt)

        return dA



class QR_real(torch.autograd.Function):
    """Real-valued QR autograd function using a custom backward pass.

    Handles both square and rectangular ``R`` branches used by this project.
    """

    @staticmethod
    def forward(self, A):
        Q, R = torch.linalg.qr(A, )
        self.save_for_backward(A, Q, R)
        return Q, R

    @staticmethod
    def backward(self, dq, dr):
        A, q, r = self.saved_tensors
        if r.shape[0] == r.shape[1]:
            return _simple_qr_backward(q, r, dq ,dr)
        M, N = r.shape
        B = A[:,M:]
        dU = dr[:,:M]
        dD = dr[:,M:]
        U = r[:,:M]
        da = _simple_qr_backward(q, U, dq+B@dD.t(), dU)
        db = q@dD
        return torch.cat([da, db], 1)

def _simple_qr_backward(q, r, dq, dr):
    """Compute the QR gradient for the square-``R`` case.

    Parameters are ``Q, R`` and their upstream gradients ``dQ, dR``.
    """
    if r.shape[-2] != r.shape[-1]:
        raise NotImplementedError("QrGrad not implemented when ncols > nrows "
                          "or full_matrices is true and ncols != nrows.")

    qdq = q.t() @ dq
    qdq_ = qdq - qdq.t()
    rdr = r @ dr.t()
    rdr_ = rdr - rdr.t()
    tril = torch.tril(qdq_ + rdr_)

    def _TriangularSolve(x, r):
        """Equiv to x @ torch.inverse(r).t() if r is upper-tri."""
        res = torch.linalg.solve_triangular(r.T, x.T, upper=True).T
        return res

    grad_a = q @ (dr + _TriangularSolve(tril, r))
    grad_b = _TriangularSolve(dq - q @ qdq, r)
    return grad_a + grad_b





class QR_complex(torch.autograd.Function):
    """Complex-valued QR autograd function using Hermitian symmetrization.

    The backward enforces the appropriate Hermitian structure of the complex
    QR differential.
    """

    @staticmethod
    def forward(ctx, A):
        Q, R = torch.linalg.qr(A)
        ctx.save_for_backward(A, Q, R)
        return Q, R

    @staticmethod
    def backward(ctx, dQ, dR):
        A, Q, R = ctx.saved_tensors

        # Compute Hermitian conjugates
        Qh = Q.conj().transpose(-2, -1)
        Rh = R.conj().transpose(-2, -1)

        # M = R @ dR^† - Q^† @ dQ
        M = R @ dR.conj().transpose(-2, -1) - Qh @ dQ

        # sym_h(M) = 0.5 * (M + M^†)
        sym_h_M = 0.5 * (M + M.conj().transpose(-2, -1))

        # R^{-†} = (R^H)^{-1}
        R_inv_h = torch.linalg.solve(Rh, torch.eye(Rh.size(-1), dtype=R.dtype, device=R.device))

        # Final gradient: dA = (dQ + Q @ sym_h_M) @ R^{-†}
        dA = (dQ + Q @ sym_h_M) @ R_inv_h

        return dA



@custom_vjp
def svd_jax(A):
    """JAX SVD primitive wrapped with a custom VJP definition.

    Returns ``(U, S, V)`` with ``full_matrices=False``.
    """
    return jnp.linalg.svd(A, full_matrices=False)


def _safe_reciprocal(x, epsilon=1e-12):
    """Regularized reciprocal used by the JAX SVD backward expressions.

    Uses ``x / (x*x + epsilon)`` to avoid hard singularities.
    """
    return x / (x * x + epsilon)


def h(x):
    """Return the conjugate transpose of ``x`` (Hermitian transpose)."""
    return jnp.conj(jnp.transpose(x))


def jaxsvd_fwd(A):
    """Forward rule for :func:`svd_jax` custom VJP.

    Stores ``(U, S, V)`` as residuals for the backward rule.
    """
    u, s, v = svd_jax(A)
    return (u, s, v), (u, s, v)


def jaxsvd_bwd(r, tangents):
    """Backward rule for :func:`svd_jax` custom VJP.

    Combines singular-value, singular-vector, and gauge-fixing terms to produce
    a stable gradient with respect to the input matrix.
    """
    U, S, V = r
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
    """Register the custom JAX SVD implementation in autoray.

    After registration, ``autoray`` calls this VJP-aware primitive for JAX SVD.
    """
    ar.register_function('jax', 'linalg.svd', svd_jax)




def reg_complex_svd_torch():
    """Register the complex torch SVD autograd implementation in autoray."""
    ar.register_function('torch', 'linalg.svd', SVD.apply)

def reg_real_svd_torch():
    """Register the real torch SVD autograd implementation in autoray."""
    ar.register_function('torch', 'linalg.svd', SVD_real.apply)

def reg_real_qr_torch():
    """Register the real torch QR autograd implementation in autoray."""
    ar.register_function('torch', 'linalg.qr', QR_real.apply)

def reg_complex_qr_torch():
    """Register the complex torch QR autograd implementation in autoray."""
    ar.register_function('torch', 'linalg.qr', QR_complex.apply)
