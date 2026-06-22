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

_SVD_EPS_REL = 1.0e-6


def safe_inverse(x, eps_abs=1.0e-12, *, eps_rel=0.0, eps_scale=None):
    """Regularized reciprocal: ``x / (x**2 + eps_abs)``.

    Avoids division-by-zero singularities in SVD backward expressions
    where exact reciprocals of singular-value differences can diverge.

    Parameters
    ----------
    x : torch.Tensor
        Input values (typically singular-value differences or sums).
    eps_abs : float, default=1e-12
        Additive regularization constant.

    Returns
    -------
    torch.Tensor
        Smooth approximation to ``1/x``.
    """
    eps = x.new_tensor(eps_abs)
    if eps_scale is not None and eps_rel:
        eps_scale = torch.as_tensor(eps_scale, dtype=x.dtype, device=x.device)
        eps = torch.maximum(eps, (eps_rel * eps_scale) ** 2)
    return x / (x ** 2 + eps)


def safe_inverse_2(x, eps):
    """Clamped reciprocal for real nonnegative values.

    Clips values below ``eps`` before inverting. Only appropriate for
    nonnegative inputs (not for ``F = 1/(s_i - s_j)`` which can be negative).

    Parameters
    ----------
    x : torch.Tensor
        Nonnegative input values.
    eps : float
        Minimum clamp threshold.

    Returns
    -------
    torch.Tensor
        Clamped reciprocal.
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
        m= u.size(-2) # row dimension of original tensor A = u sigma v^\dag
        n= vh.size(-1) # column dimension of A
        k= sigma.size(-1)
        eps_abs= torch.finfo(sigma.dtype).tiny
        sigma_scale= sigma.detach().amax(dim=-1, keepdim=True)
        pair_scale= sigma_scale.unsqueeze(-1)

        #
        if (u.size(-1)!=k) or (vh.size(-2)!=k):
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
            sigma_term = torch.zeros((*sigma.shape[:-1],m,n),dtype=u.dtype,device=u.device)
        # in case that there are no gu and gvh, we can avoid the series of kernel
        # calls below
        if (gu is None) and (gvh is None):
            if not (diagnostics is None):
                print(f"{diagnostics} {sigma_term.abs().max()} {sigma.max()}")
            return sigma_term


        # sigma_inv= safe_inverse_2(sigma.clone(), sigma_scale*eps)
        # sigma_inv= safe_inverse(sigma.clone(), eps_abs=sigma_scale*eps)
        sigma_inv= safe_inverse(
            sigma.clone(),
            eps_abs=eps_abs,
            eps_rel=_SVD_EPS_REL,
            eps_scale=sigma_scale,
        )

        F = sigma.unsqueeze(-2) - sigma.unsqueeze(-1)
        F = safe_inverse(
            F,
            eps_abs=eps_abs,
            eps_rel=_SVD_EPS_REL,
            eps_scale=pair_scale,
        )
        F.diagonal(0,-2,-1).fill_(0)

        G = sigma.unsqueeze(-2) + sigma.unsqueeze(-1)
        G = safe_inverse(
            G,
            eps_abs=eps_abs,
            eps_rel=_SVD_EPS_REL,
            eps_scale=pair_scale,
        )
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
            u_term = torch.zeros((*sigma.shape[:-1],m,n),dtype=u.dtype,device=u.device)

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
            v_term = torch.zeros((*sigma.shape[:-1],m,n),dtype=u.dtype,device=u.device)


        # // for complex-valued input there is an additional term
        # // https://giggleliu.github.io/2019/04/02/einsumbp.html
        # // https://arxiv.org/abs/1909.02659
        dA= u_term + sigma_term + v_term
        if (u.is_complex() or v.is_complex()) and gu is not None:
            phase_diag= (uh @ gu).diagonal(0,-2,-1)
            L= 1j * phase_diag.imag * sigma_inv
            imag_term= (u * L.unsqueeze(-2)) @ vh
            dA= dA + imag_term

        if diagnostics is not None:
            print(f"{diagnostics} {dA.abs().max()} {sigma.max()}")

        return dA


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


def reg_stop_gradient_torch():
    """Register torch stop-gradient helper in autoray.

    Registers ``stop_gradient`` as ``x.detach().clone()`` so backend-agnostic
    code can call ``ar.do("stop_gradient", x)`` on torch tensors.
    """
    ar.register_function('torch', 'stop_gradient', lambda x: x.detach().clone())


def stop_grad(x):
    """Backend-agnostic stop-gradient wrapper.

    This uses the registered ``autoray`` backend op when available. If a
    backend has no registered implementation, the input is returned unchanged.
    """
    try:
        return ar.do("stop_gradient", x)
    except Exception:
        return x


# Register torch stop-gradient by default so callers don't need per-script boilerplate.
reg_stop_gradient_torch()
