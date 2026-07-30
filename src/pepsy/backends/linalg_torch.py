"""Torch-side linalg registrations with stabilized autodiff rules."""

import warnings

import autoray as ar
import numpy as np
import torch

try:
    import scipy.linalg as scipy_linalg  # pylint: disable=import-outside-toplevel
except ImportError:  # pragma: no cover - optional dependency
    scipy_linalg = None

# pylint: disable=abstract-method,arguments-differ,bad-staticmethod-argument,bare-except,line-too-long,multiple-statements,not-callable,superfluous-parens,too-many-branches,too-many-locals,too-many-statements,unnecessary-semicolon,unused-variable,using-constant-test

_SVD_EPS_REL = 1.0e-6
_REGISTERED_FUNCTIONS = {}
_QR_RANK_POLICIES = {"warn", "native", "error"}
_QR_RANK_POLICY = "warn"
_QR_RANK_TOL_FACTOR = 1.0


def _same_callable(left, right):
    """Compare plain functions and class-bound autograd methods robustly."""
    if left is right:
        return True
    left_func = getattr(left, "__func__", None)
    right_func = getattr(right, "__func__", None)
    return (
        left_func is not None
        and right_func is not None
        and left_func is right_func
        and getattr(left, "__self__", None)
        is getattr(right, "__self__", None)
    )


def _register_once(name, function):
    """Register one Torch autoray function once per active implementation."""
    if _same_callable(_REGISTERED_FUNCTIONS.get(name), function):
        return
    ar.register_function("torch", name, function)
    _REGISTERED_FUNCTIONS[name] = function


def _configure_qr_rank_policy(policy="warn", rank_tol_factor=1.0):
    """Configure the real-QR response to detected rank deficiency."""
    if policy not in _QR_RANK_POLICIES:
        choices = ", ".join(sorted(_QR_RANK_POLICIES))
        raise ValueError(f"rank_policy must be one of: {choices}")
    try:
        rank_tol_factor = float(rank_tol_factor)
    except (TypeError, ValueError) as exc:
        raise TypeError("rank_tol_factor must be a positive finite number") from exc
    if not np.isfinite(rank_tol_factor) or rank_tol_factor <= 0.0:
        raise ValueError("rank_tol_factor must be a positive finite number")

    global _QR_RANK_POLICY  # pylint: disable=global-statement
    global _QR_RANK_TOL_FACTOR  # pylint: disable=global-statement
    _QR_RANK_POLICY = policy
    _QR_RANK_TOL_FACTOR = rank_tol_factor


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
    """Real QR with a custom full-rank backward and native rank fallback."""

    @staticmethod
    def forward(self, A):
        Q, R = torch.linalg.qr(A)
        diagonal = torch.diagonal(R, dim1=-2, dim2=-1).abs()
        scale = R.abs().amax(dim=(-2, -1))
        tolerance = (
            _QR_RANK_TOL_FACTOR
            * torch.finfo(A.dtype).eps
            * max(A.shape[-2:])
            * scale
        )
        rank_deficient = (diagonal <= tolerance.unsqueeze(-1)).any(dim=-1)
        if bool(rank_deficient.any().item()):
            message = (
                "QR_real detected a rank-deficient input; native Torch QR "
                "backward may be ill-conditioned."
            )
            if _QR_RANK_POLICY == "error":
                raise RuntimeError(message)
            if _QR_RANK_POLICY == "warn":
                warnings.warn(message, RuntimeWarning, stacklevel=2)
        self.save_for_backward(A, Q, R, rank_deficient)
        return Q, R

    @staticmethod
    def backward(self, dq, dr):
        A, q, r, rank_deficient = self.saved_tensors
        if bool(rank_deficient.any().item()):
            return _native_qr_backward(A, dq, dr)
        m, _n = r.shape[-2:]
        if m == _n:
            return _simple_qr_backward(q, r, dq, dr)
        B = A[..., :, m:]
        dU = dr[..., :, :m]
        dD = dr[..., :, m:]
        U = r[..., :, :m]
        da = _simple_qr_backward(
            q,
            U,
            dq + B @ dD.transpose(-2, -1),
            dU,
        )
        db = q @ dD
        return torch.cat([da, db], dim=-1)


def _simple_qr_backward(q, r, dq, dr):
    """Compute the QR gradient for the square-``R`` case."""
    if r.shape[-2] != r.shape[-1]:
        raise NotImplementedError(
            "QrGrad not implemented when ncols > nrows "
            "or full_matrices is true and ncols != nrows."
        )

    qdq = q.transpose(-2, -1) @ dq
    qdq_ = qdq - qdq.transpose(-2, -1)
    rdr = r @ dr.transpose(-2, -1)
    rdr_ = rdr - rdr.transpose(-2, -1)
    tril = torch.tril(qdq_ + rdr_)

    def _triangular_solve(x, tri):
        # Solve with the upper-triangular R factor. Using R.T here gives the
        # wrong reverse-mode rule and does not generalize to batched inputs.
        return torch.linalg.solve_triangular(
            tri,
            x.transpose(-2, -1),
            upper=True,
        ).transpose(-2, -1)

    grad_a = q @ (dr + _triangular_solve(tril, r))
    grad_b = _triangular_solve(dq - q @ qdq, r)
    return grad_a + grad_b


def _native_qr_backward(A, dq, dr):
    """Recompute native Torch QR to obtain a reliable VJP fallback."""
    with torch.enable_grad():
        replay = A.detach().requires_grad_(True)
        q, r = torch.linalg.qr(replay)
        dq = torch.zeros_like(q) if dq is None else dq
        dr = torch.zeros_like(r) if dr is None else dr
        return torch.autograd.grad(
            (q, r),
            replay,
            (dq, dr),
        )[0]


class QR_complex(torch.autograd.Function):
    """Complex-valued QR wrapper using native Torch's conjugate-aware VJP."""

    @staticmethod
    def forward(ctx, A):
        Q, R = torch.linalg.qr(A)
        ctx.save_for_backward(A)
        return Q, R

    @staticmethod
    def backward(ctx, dQ, dR):
        (A,) = ctx.saved_tensors
        return _native_qr_backward(A, dQ, dR)


def _native_svd(A, *args, **kwargs):
    """Use native Torch SVD with the thin-factor default expected by Pepsy."""
    kwargs.setdefault("full_matrices", False)
    return torch.linalg.svd(A, *args, **kwargs)


def reg_native_svd_torch():
    """Register native Torch SVD with Pepsy's thin-factor default."""
    _register_once("linalg.svd", _native_svd)


def reset_torch_linalg_registrations():
    """Restore native Torch SVD/QR mappings and clear Pepsy's cache."""
    _REGISTERED_FUNCTIONS.clear()
    _configure_qr_rank_policy()
    reg_native_svd_torch()
    reg_complex_qr_torch()


def reg_rel_svd_torch():
    """Register the relative-regularized torch SVD rule in autoray."""
    _register_once("linalg.svd", SVD.apply)


def reg_complex_svd_torch():
    """Register the complex torch SVD autograd implementation in autoray.

    This compatibility name installs the same relative-regularized SVD rule as
    :func:`reg_rel_svd_torch`.
    """
    reg_rel_svd_torch()


def reg_real_svd_torch():
    """Register the real torch SVD autograd implementation in autoray."""
    _register_once("linalg.svd", SVD_real.apply)


def reg_real_qr_torch(*, rank_policy="warn", rank_tol_factor=1.0):
    """Register real QR with a configurable rank-deficiency policy."""
    _configure_qr_rank_policy(rank_policy, rank_tol_factor)
    _register_once("linalg.qr", QR_real.apply)


def reg_complex_qr_torch():
    """Register native Torch QR for complex inputs.

    The explicit :class:`QR_complex` compatibility wrapper is safe but
    recomputes native QR during its backward pass, so Autoray uses native QR
    directly for the faster default path.
    """
    _register_once("linalg.qr", torch.linalg.qr)


def _stop_gradient_torch(x):
    """Return a detached, independent Torch tensor."""
    return x.detach().clone()


def reg_stop_gradient_torch():
    """Register torch stop-gradient helper in autoray."""
    _register_once(
        "stop_gradient",
        _stop_gradient_torch,
    )


def stop_grad(x):
    """Backend-agnostic stop-gradient wrapper."""
    try:
        return ar.do("stop_gradient", x)
    except Exception:
        return x


reg_stop_gradient_torch()
