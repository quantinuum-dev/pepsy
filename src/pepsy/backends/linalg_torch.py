"""Torch-side linalg registrations with stabilized autodiff rules."""

import autoray as ar
import numpy as np
import torch

try:
    import scipy.linalg as scipy_linalg  # pylint: disable=import-outside-toplevel
except ImportError:  # pragma: no cover - optional dependency
    scipy_linalg = None

# pylint: disable=abstract-method,arguments-differ,bad-staticmethod-argument,bare-except,line-too-long,multiple-statements,not-callable,superfluous-parens,too-many-branches,too-many-locals,too-many-statements,unnecessary-semicolon,unused-variable,using-constant-test

_SVD_EPS_REL = 1.0e-6


def safe_inverse(x, eps_abs=1.0e-12, *, eps_rel=0.0, eps_scale=None):
    """Regularized reciprocal: ``x / (x**2 + eps)``.

    When ``eps_scale`` is supplied, ``eps`` is at least
    ``(eps_rel * eps_scale)**2``. This lets spectral backpropagation use a
    scale-aware regularization while preserving the original absolute-only API.
    """
    eps = x.new_tensor(eps_abs)
    if eps_scale is not None and eps_rel:
        eps_scale = torch.as_tensor(eps_scale, dtype=x.dtype, device=x.device)
        eps = torch.maximum(eps, (eps_rel * eps_scale) ** 2)
    return x / (x ** 2 + eps)


def safe_inverse_2(x, eps):
    """Clamped reciprocal for real nonnegative values."""
    return x.clamp_min(eps).reciprocal()


def _scipy_gesvd(A, exc):
    """Compute a thin SVD through SciPy's more robust ``gesvd`` driver."""
    if scipy_linalg is None:
        raise RuntimeError(
            "torch.linalg.svd failed and scipy is unavailable for gesvd fallback."
        ) from exc

    A_np = A.detach().cpu().numpy()
    batch_shape = A_np.shape[:-2]
    m, n = A_np.shape[-2:]
    k = min(m, n)

    if batch_shape:
        flat = A_np.reshape((-1,) + A_np.shape[-2:])
        if flat.shape[0] == 0:
            U_np = np.empty(batch_shape + (m, k), dtype=A_np.dtype)
            S_np = np.empty(batch_shape + (k,), dtype=A_np.real.dtype)
            Vh_np = np.empty(batch_shape + (k, n), dtype=A_np.dtype)
        else:
            parts = [
                scipy_linalg.svd(
                    mat,
                    full_matrices=False,
                    lapack_driver="gesvd",
                )
                for mat in flat
            ]
            U_np = np.stack([part[0] for part in parts]).reshape(
                batch_shape + parts[0][0].shape
            )
            S_np = np.stack([part[1] for part in parts]).reshape(
                batch_shape + parts[0][1].shape
            )
            Vh_np = np.stack([part[2] for part in parts]).reshape(
                batch_shape + parts[0][2].shape
            )
    else:
        U_np, S_np, Vh_np = scipy_linalg.svd(
            A_np,
            full_matrices=False,
            lapack_driver="gesvd",
        )

    U = torch.from_numpy(np.asarray(U_np)).to(device=A.device, dtype=A.dtype)
    S = torch.from_numpy(np.asarray(S_np)).to(device=A.device, dtype=A.real.dtype)
    Vh = torch.from_numpy(np.asarray(Vh_np)).to(device=A.device, dtype=A.dtype)
    return U, S, Vh


class SVD(torch.autograd.Function):
    """Torch SVD with a relative-regularized reverse-mode rule.

    The rectangular real-SVD terms follow Townsend's reverse update, with the
    singular-gap and inverse-singular-value reciprocals regularized as
    ``x / (x**2 + eps)``. The scale-aware ``eps`` keeps the stabilizer relative
    to the current singular spectrum, which is the Lorentzian broadening used in
    differentiable tensor-network SVDs. Complex inputs additionally include the
    phase/gauge term from the complex-valued SVD backward formula.
    """

    @staticmethod
    def forward(ctx, A):
        try:
            if A.is_cuda:
                U, S, Vh = torch.linalg.svd(A, full_matrices=False, driver="gesvd")
            else:
                U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        except Exception as exc:  # pragma: no cover - backend failure dependent
            U, S, Vh = _scipy_gesvd(A, exc)
        ctx.save_for_backward(U, S, Vh)
        return U, S, Vh

    @staticmethod
    def backward(ctx, gu, gsigma, gvh):
        diagnostics = None

        u, sigma, vh = ctx.saved_tensors
        m = u.size(-2)
        n = vh.size(-1)
        k = sigma.size(-1)
        eps_abs = torch.finfo(sigma.dtype).tiny
        sigma_scale = sigma.detach().amax(dim=-1, keepdim=True)
        pair_scale = sigma_scale.unsqueeze(-1)

        if (u.size(-1) != k) or (vh.size(-2) != k):
            u = u.narrow(-1, 0, k)
            vh = vh.narrow(-2, 0, k)
            if not (gu is None):
                gu = gu.narrow(-1, 0, k)
            if not (gvh is None):
                gvh = gvh.narrow(-2, 0, k)

        if not (gsigma is None):
            sigma_term = u * gsigma.unsqueeze(-2) @ vh
        else:
            sigma_term = torch.zeros(
                (*sigma.shape[:-1], m, n),
                dtype=u.dtype,
                device=u.device,
            )

        if (gu is None) and (gvh is None):
            if not (diagnostics is None):
                print(f"{diagnostics} {sigma_term.abs().max()} {sigma.max()}")
            return sigma_term

        sigma_inv = safe_inverse(
            sigma.clone(),
            eps_abs=eps_abs,
            eps_rel=_SVD_EPS_REL,
            eps_scale=sigma_scale,
        )

        # Townsend's F+/F- terms, written as 1/(s_j - s_i) and
        # 1/(s_i + s_j), with relative Lorentzian broadening.
        F = sigma.unsqueeze(-2) - sigma.unsqueeze(-1)
        F = safe_inverse(
            F,
            eps_abs=eps_abs,
            eps_rel=_SVD_EPS_REL,
            eps_scale=pair_scale,
        )
        F.diagonal(0, -2, -1).fill_(0)

        G = sigma.unsqueeze(-2) + sigma.unsqueeze(-1)
        G = safe_inverse(
            G,
            eps_abs=eps_abs,
            eps_rel=_SVD_EPS_REL,
            eps_scale=pair_scale,
        )
        G.diagonal(0, -2, -1).fill_(0)

        uh = u.conj().transpose(-2, -1)
        if not (gu is None):
            guh = gu.conj().transpose(-2, -1)
            u_term = u @ ((F + G).mul(uh @ gu - guh @ u)) * 0.5
            if m > k:
                proj_on_ortho_u = -u @ uh
                proj_on_ortho_u.diagonal(0, -2, -1).add_(1)
                u_term = u_term + proj_on_ortho_u @ (gu * sigma_inv.unsqueeze(-2))
            u_term = u_term @ vh
        else:
            u_term = torch.zeros(
                (*sigma.shape[:-1], m, n),
                dtype=u.dtype,
                device=u.device,
            )

        v = vh.conj().transpose(-2, -1)
        if not (gvh is None):
            gv = gvh.conj().transpose(-2, -1)
            v_term = ((F - G).mul(vh @ gv - gvh @ v)) @ vh * 0.5
            if n > k:
                proj_on_v_ortho = -v @ vh
                proj_on_v_ortho.diagonal(0, -2, -1).add_(1)
                v_term = v_term + sigma_inv.unsqueeze(-1) * (gvh @ proj_on_v_ortho)
            v_term = u @ v_term
        else:
            v_term = torch.zeros(
                (*sigma.shape[:-1], m, n),
                dtype=u.dtype,
                device=u.device,
            )

        dA = u_term + sigma_term + v_term
        if (u.is_complex() or v.is_complex()) and gu is not None:
            # Complex-SVD gauge correction from arXiv:1909.02659.
            phase_diag = (uh @ gu).diagonal(0, -2, -1)
            L = 1j * phase_diag.imag * sigma_inv
            imag_term = (u * L.unsqueeze(-2)) @ vh
            dA = dA + imag_term

        if diagnostics is not None:
            print(f"{diagnostics} {dA.abs().max()} {sigma.max()}")

        return dA


class SVD_real(torch.autograd.Function):
    """Real-valued SVD with the relative-regularized reverse-mode rule.

    This is the real-only counterpart of :class:`SVD`. It shares the robust
    forward path (``gesvd`` driver on CUDA plus a batched SciPy ``gesvd``
    fallback) and the scale-aware Lorentzian broadening of the singular-gap and
    inverse-singular-value reciprocals, and it supports rectangular and batched
    inputs. Only the complex phase/gauge term of :class:`SVD` is dropped, since
    real orthogonal factors carry no gauge freedom.
    """

    @staticmethod
    def forward(ctx, A):
        try:
            if A.is_cuda:
                U, S, Vh = torch.linalg.svd(A, full_matrices=False, driver="gesvd")
            else:
                U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        except Exception as exc:  # pragma: no cover - backend failure dependent
            U, S, Vh = _scipy_gesvd(A, exc)
        ctx.save_for_backward(U, S, Vh)
        return U, S, Vh

    @staticmethod
    def backward(ctx, gu, gsigma, gvh):
        u, sigma, vh = ctx.saved_tensors
        m = u.size(-2)
        n = vh.size(-1)
        k = sigma.size(-1)
        eps_abs = torch.finfo(sigma.dtype).tiny
        sigma_scale = sigma.detach().amax(dim=-1, keepdim=True)
        pair_scale = sigma_scale.unsqueeze(-1)

        if (u.size(-1) != k) or (vh.size(-2) != k):
            u = u.narrow(-1, 0, k)
            vh = vh.narrow(-2, 0, k)
            if not (gu is None):
                gu = gu.narrow(-1, 0, k)
            if not (gvh is None):
                gvh = gvh.narrow(-2, 0, k)

        if not (gsigma is None):
            sigma_term = u * gsigma.unsqueeze(-2) @ vh
        else:
            sigma_term = torch.zeros(
                (*sigma.shape[:-1], m, n),
                dtype=u.dtype,
                device=u.device,
            )

        if (gu is None) and (gvh is None):
            return sigma_term

        sigma_inv = safe_inverse(
            sigma.clone(),
            eps_abs=eps_abs,
            eps_rel=_SVD_EPS_REL,
            eps_scale=sigma_scale,
        )

        # Townsend's F+/F- terms, written as 1/(s_j - s_i) and
        # 1/(s_i + s_j), with relative Lorentzian broadening.
        F = sigma.unsqueeze(-2) - sigma.unsqueeze(-1)
        F = safe_inverse(
            F,
            eps_abs=eps_abs,
            eps_rel=_SVD_EPS_REL,
            eps_scale=pair_scale,
        )
        F.diagonal(0, -2, -1).fill_(0)

        G = sigma.unsqueeze(-2) + sigma.unsqueeze(-1)
        G = safe_inverse(
            G,
            eps_abs=eps_abs,
            eps_rel=_SVD_EPS_REL,
            eps_scale=pair_scale,
        )
        G.diagonal(0, -2, -1).fill_(0)

        ut = u.transpose(-2, -1)
        if not (gu is None):
            gut = gu.transpose(-2, -1)
            u_term = u @ ((F + G).mul(ut @ gu - gut @ u)) * 0.5
            if m > k:
                proj_on_ortho_u = -u @ ut
                proj_on_ortho_u.diagonal(0, -2, -1).add_(1)
                u_term = u_term + proj_on_ortho_u @ (gu * sigma_inv.unsqueeze(-2))
            u_term = u_term @ vh
        else:
            u_term = torch.zeros(
                (*sigma.shape[:-1], m, n),
                dtype=u.dtype,
                device=u.device,
            )

        v = vh.transpose(-2, -1)
        if not (gvh is None):
            gv = gvh.transpose(-2, -1)
            v_term = ((F - G).mul(vh @ gv - gvh @ v)) @ vh * 0.5
            if n > k:
                proj_on_v_ortho = -v @ vh
                proj_on_v_ortho.diagonal(0, -2, -1).add_(1)
                v_term = v_term + sigma_inv.unsqueeze(-1) * (gvh @ proj_on_v_ortho)
            v_term = u @ v_term
        else:
            v_term = torch.zeros(
                (*sigma.shape[:-1], m, n),
                dtype=u.dtype,
                device=u.device,
            )

        return u_term + sigma_term + v_term


class QR_real(torch.autograd.Function):
    """Real-valued QR autograd function using a custom backward pass."""

    @staticmethod
    def forward(self, A):
        Q, R = torch.linalg.qr(A)
        self.save_for_backward(A, Q, R)
        return Q, R

    @staticmethod
    def backward(self, dq, dr):
        A, q, r = self.saved_tensors
        if r.shape[0] == r.shape[1]:
            return _simple_qr_backward(q, r, dq, dr)
        M, _N = r.shape
        B = A[:, M:]
        dU = dr[:, :M]
        dD = dr[:, M:]
        U = r[:, :M]
        da = _simple_qr_backward(q, U, dq + B @ dD.t(), dU)
        db = q @ dD
        return torch.cat([da, db], 1)


def _simple_qr_backward(q, r, dq, dr):
    """Compute the QR gradient for the square-``R`` case."""
    if r.shape[-2] != r.shape[-1]:
        raise NotImplementedError(
            "QrGrad not implemented when ncols > nrows "
            "or full_matrices is true and ncols != nrows."
        )

    qdq = q.t() @ dq
    qdq_ = qdq - qdq.t()
    rdr = r @ dr.t()
    rdr_ = rdr - rdr.t()
    tril = torch.tril(qdq_ + rdr_)

    def _triangular_solve(x, tri):
        return torch.linalg.solve_triangular(tri.T, x.T, upper=True).T

    grad_a = q @ (dr + _triangular_solve(tril, r))
    grad_b = _triangular_solve(dq - q @ qdq, r)
    return grad_a + grad_b


class QR_complex(torch.autograd.Function):
    """Complex-valued QR autograd function using Hermitian symmetrization."""

    @staticmethod
    def forward(ctx, A):
        Q, R = torch.linalg.qr(A)
        ctx.save_for_backward(A, Q, R)
        return Q, R

    @staticmethod
    def backward(ctx, dQ, dR):
        _A, Q, R = ctx.saved_tensors

        Qh = Q.conj().transpose(-2, -1)
        Rh = R.conj().transpose(-2, -1)

        M = R @ dR.conj().transpose(-2, -1) - Qh @ dQ
        sym_h_M = 0.5 * (M + M.conj().transpose(-2, -1))
        R_inv_h = torch.linalg.solve(
            Rh,
            torch.eye(Rh.size(-1), dtype=R.dtype, device=R.device),
        )
        dA = (dQ + Q @ sym_h_M) @ R_inv_h
        return dA


def reg_rel_svd_torch():
    """Register the relative-regularized torch SVD rule in autoray."""
    ar.register_function("torch", "linalg.svd", SVD.apply)


def reg_complex_svd_torch():
    """Register the complex torch SVD autograd implementation in autoray.

    This compatibility name installs the same relative-regularized SVD rule as
    :func:`reg_rel_svd_torch`.
    """
    reg_rel_svd_torch()


def reg_real_svd_torch():
    """Register the real torch SVD autograd implementation in autoray."""
    ar.register_function("torch", "linalg.svd", SVD_real.apply)


def reg_real_qr_torch():
    """Register the real torch QR autograd implementation in autoray."""
    ar.register_function("torch", "linalg.qr", QR_real.apply)


def reg_complex_qr_torch():
    """Register the complex torch QR autograd implementation in autoray."""
    ar.register_function("torch", "linalg.qr", QR_complex.apply)


def reg_stop_gradient_torch():
    """Register torch stop-gradient helper in autoray."""
    ar.register_function("torch", "stop_gradient", lambda x: x.detach().clone())


def stop_grad(x):
    """Backend-agnostic stop-gradient wrapper."""
    try:
        return ar.do("stop_gradient", x)
    except Exception:
        return x


reg_stop_gradient_torch()
