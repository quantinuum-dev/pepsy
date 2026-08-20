"""Torch-side linalg registrations with configurable SVD policies."""

import contextvars
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
_QR_EPS_REL = 1.0e-6
_REGISTERED_FUNCTIONS = {}
_SVD_WRAPPERS = {}
_QR_WRAPPERS = {}
_SVD_FORWARD_OPTIONS = contextvars.ContextVar(
    "pepsy_torch_svd_forward_options",
    default=None,
)
_SVD_DRIVERS = {"auto", "gesvdj", "gesvda", "gesvd"}
_CPU_SVD_BACKENDS = {"torch", "scipy_gesdd", "scipy_gesvd"}
_SVD_FALLBACKS = {"auto", "none", "scipy_gesdd", "scipy_gesvd"}
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


def _handle_qr_rank_policy(rank_deficient):
    """Apply the configured response to a detected QR rank deficiency."""
    if not bool(rank_deficient.any().item()):
        return
    message = (
        "Torch QR detected a rank-deficient input; the ordinary backward "
        "derivative is not well-conditioned."
    )
    if _QR_RANK_POLICY == "error":
        raise RuntimeError(message)
    if _QR_RANK_POLICY == "warn":
        warnings.warn(message, RuntimeWarning, stacklevel=2)


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


def _scipy_svd(A, lapack_driver="gesvd", exc=None):
    """Compute a thin CPU SVD through an explicit SciPy LAPACK driver."""
    if scipy_linalg is None:
        message = (
            "SciPy is required for the requested CPU SVD backend or fallback."
        )
        if exc is not None:
            raise RuntimeError(message) from exc
        raise ImportError(message)

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
                    lapack_driver=lapack_driver,
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
            lapack_driver=lapack_driver,
        )

    U = torch.from_numpy(np.asarray(U_np)).to(device=A.device, dtype=A.dtype)
    S = torch.from_numpy(np.asarray(S_np)).to(device=A.device, dtype=A.real.dtype)
    Vh = torch.from_numpy(np.asarray(Vh_np)).to(device=A.device, dtype=A.dtype)
    return U, S, Vh


def _scipy_gesvd(A, exc):
    """Compute a thin SVD through SciPy's robust ``gesvd`` fallback."""
    return _scipy_svd(A, lapack_driver="gesvd", exc=exc)


def _resolve_svd_options(*, driver="auto", cpu_svd="torch", fallback="auto"):
    """Validate and normalize the low-level Torch SVD policy."""
    if driver not in _SVD_DRIVERS:
        choices = ", ".join(sorted(_SVD_DRIVERS))
        raise ValueError(f"svd_driver must be one of: {choices}")
    if cpu_svd not in _CPU_SVD_BACKENDS:
        choices = ", ".join(sorted(_CPU_SVD_BACKENDS))
        raise ValueError(f"cpu_svd must be one of: {choices}")
    if fallback not in _SVD_FALLBACKS:
        choices = ", ".join(sorted(_SVD_FALLBACKS))
        raise ValueError(f"svd_fallback must be one of: {choices}")
    return driver, cpu_svd, fallback


def _svd_forward(
    A,
    *,
    driver="auto",
    cpu_svd="torch",
    fallback="auto",
    stabilized=False,
):
    """Execute one configured thin SVD, including explicit CPU fallbacks."""
    driver, cpu_svd, fallback = _resolve_svd_options(
        driver=driver,
        cpu_svd=cpu_svd,
        fallback=fallback,
    )
    if fallback == "auto":
        fallback = "scipy_gesvd" if stabilized else "none"

    if A.device.type == "cpu" and cpu_svd != "torch":
        if A.requires_grad and not stabilized:
            raise RuntimeError(
                "cpu_svd={!r} is forward-only for stabilized=False. Use "
                "cpu_svd='torch' or stabilized=True for autodiff.".format(cpu_svd)
            )
        return _scipy_svd(A, lapack_driver=cpu_svd.removeprefix("scipy_"))

    kwargs = {"full_matrices": False}
    if A.is_cuda:
        if driver == "auto":
            # Preserve the historical stabilized path's robust CUDA choice,
            # while native Torch keeps its own default (currently gesvdj with
            # a gesvd fallback in supported CUDA builds).
            if stabilized:
                kwargs["driver"] = "gesvd"
        else:
            kwargs["driver"] = driver

    try:
        return torch.linalg.svd(A, **kwargs)
    except Exception as exc:  # pragma: no cover - backend failure dependent
        if fallback == "none":
            raise
        return _scipy_svd(A, lapack_driver=fallback.removeprefix("scipy_"), exc=exc)


def _configured_svd_call(function, A, *, driver, cpu_svd, fallback):
    """Call a regularized SVD with options scoped to this synchronous forward."""
    token = _SVD_FORWARD_OPTIONS.set((driver, cpu_svd, fallback))
    try:
        return function.apply(A)
    finally:
        _SVD_FORWARD_OPTIONS.reset(token)


def _validate_stabilized_svd_call(args, kwargs):
    """Validate the thin-SVD arguments accepted by the custom Torch VJP."""
    kwargs = dict(kwargs)
    if len(args) > 1:
        raise TypeError(
            "stabilized Torch SVD accepts at most one positional option: "
            "full_matrices"
        )
    if args and "full_matrices" in kwargs:
        raise TypeError("full_matrices was passed both positionally and by keyword")
    full_matrices = args[0] if args else kwargs.pop("full_matrices", False)
    if full_matrices:
        raise NotImplementedError(
            "Pepsy's stabilized Torch SVD only supports thin SVD "
            "(full_matrices=False)."
        )
    if kwargs:
        unexpected = next(iter(kwargs))
        raise TypeError(
            f"stabilized Torch SVD got an unexpected keyword argument {unexpected!r}"
        )


def _validate_reduced_qr_call(args, kwargs):
    """Validate the reduced-QR arguments accepted by the custom Torch VJP."""
    kwargs = dict(kwargs)
    if len(args) > 1:
        raise TypeError(
            "stabilized Torch QR accepts at most one positional option: mode"
        )
    if args and "mode" in kwargs:
        raise TypeError("mode was passed both positionally and by keyword")
    mode = args[0] if args else kwargs.pop("mode", "reduced")
    if mode != "reduced":
        raise NotImplementedError(
            "Pepsy's stabilized Torch QR only supports mode='reduced'."
        )
    if kwargs:
        unexpected = next(iter(kwargs))
        raise TypeError(
            f"stabilized Torch QR got an unexpected keyword argument {unexpected!r}"
        )


def _get_stabilized_svd_call(function_class, *, driver, cpu_svd, fallback):
    """Build a keyword-compatible thin-SVD dispatcher for Autoray."""
    def function(A, *args, **kwargs):
        _validate_stabilized_svd_call(args, kwargs)
        return _configured_svd_call(
            function_class,
            A,
            driver=driver,
            cpu_svd=cpu_svd,
            fallback=fallback,
        )

    function.__name__ = (
        f"{function_class.__name__}_stabilized_svd_"
        f"{driver}_{cpu_svd}_{fallback}"
    )
    return function


def _get_stabilized_qr_call(function_class):
    """Build a keyword-compatible reduced-QR dispatcher for Autoray."""
    cached = _QR_WRAPPERS.get(function_class)
    if cached is not None:
        return cached

    def function(A, *args, **kwargs):
        _validate_reduced_qr_call(args, kwargs)
        return function_class.apply(A)

    function.__name__ = f"{function_class.__name__}_stabilized_qr"
    _QR_WRAPPERS[function_class] = function
    return function


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
        options = _SVD_FORWARD_OPTIONS.get()
        if options is None:
            options = ("auto", "torch", "scipy_gesvd")
        U, S, Vh = _svd_forward(
            A,
            driver=options[0],
            cpu_svd=options[1],
            fallback=options[2],
            stabilized=True,
        )
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
        if A.is_complex():
            raise TypeError("SVD_real requires a real Torch tensor.")
        options = _SVD_FORWARD_OPTIONS.get()
        if options is None:
            options = ("auto", "torch", "scipy_gesvd")
        U, S, Vh = _svd_forward(
            A,
            driver=options[0],
            cpu_svd=options[1],
            fallback=options[2],
            stabilized=True,
        )
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
    """Real QR with a custom full-rank and rank-aware backward."""

    @staticmethod
    def forward(self, A):
        if A.is_complex():
            raise TypeError("QR_real requires a real Torch tensor.")
        Q, R = torch.linalg.qr(A)
        diagonal = torch.diagonal(R, dim1=-2, dim2=-1).abs()
        scale = R.abs().amax(dim=(-2, -1))
        tolerance = (
            _QR_RANK_TOL_FACTOR
            * _qr_regularization_relative_eps(A.dtype)
            * scale
        )
        rank_deficient = (diagonal <= tolerance.unsqueeze(-1)).any(dim=-1)
        if bool(rank_deficient.any().item()):
            message = (
                "QR_real detected a rank-deficient input; using the "
                "regularized QR backward rule."
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
            if _QR_RANK_POLICY == "native":
                return _native_qr_backward(A, dq, dr)
            return _regularized_qr_backward(A, q, r, dq, dr, rank_deficient)
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


def _qr_regularization_relative_eps(real_dtype):
    """Return the shared relative threshold and VJP regularization scale."""
    # The normal-equation right inverse squares the condition number. The
    # square-root machine-epsilon floor keeps it resolvable for float32 while
    # retaining the requested 1e-6 relative stabilization for float64/complex128.
    return max(_QR_EPS_REL, torch.finfo(real_dtype).eps ** 0.5)


class QR_real_safe(torch.autograd.Function):
    """Real QR with a finite VJP at a singular unpivoted QR chart.

    A zero diagonal entry of unpivoted ``R`` makes Torch's QR derivative
    undefined. This can occur in a symmetry-resolved PEPS block even when the
    complete wide matrix has useful physical rank. The forward QR remains
    exact; only its VJP replaces ``R^{-H}`` by a scale-relative regularized
    right inverse. Exactly zero blocks keep the explicit zero-VJP convention.
    """

    @staticmethod
    def forward(ctx, A):
        if A.is_complex():
            raise TypeError("QR_real_safe requires a real Torch tensor.")
        Q, R = torch.linalg.qr(A)
        diagonal = torch.diagonal(R, dim1=-2, dim2=-1).abs()
        scale = R.abs().amax(dim=(-2, -1))
        tolerance = _QR_RANK_TOL_FACTOR * _qr_regularization_relative_eps(A.dtype) * scale
        rank_deficient = (diagonal <= tolerance.unsqueeze(-1)).any(dim=-1)
        _handle_qr_rank_policy(rank_deficient)
        ctx.save_for_backward(A, Q, R, rank_deficient)
        return Q, R

    @staticmethod
    def backward(ctx, dQ, dR):
        A, Q, R, rank_deficient = ctx.saved_tensors
        if bool(rank_deficient.any().item()):
            if _QR_RANK_POLICY == "native":
                return _native_qr_backward(A, dQ, dR)
            return _regularized_qr_backward(A, Q, R, dQ, dR, rank_deficient)
        if R.shape[-1] > R.shape[-2]:
            return _native_qr_backward(A, dQ, dR)
        return _simple_qr_backward(Q, R, dQ, dR)


class QR_complex_safe(torch.autograd.Function):
    """Complex QR with a finite VJP at a singular unpivoted QR chart."""

    @staticmethod
    def forward(ctx, A):
        Q, R = torch.linalg.qr(A)
        diagonal = torch.diagonal(R, dim1=-2, dim2=-1).abs()
        scale = R.abs().amax(dim=(-2, -1))
        real_dtype = A.real.dtype if A.is_complex() else A.dtype
        tolerance = _QR_RANK_TOL_FACTOR * _qr_regularization_relative_eps(real_dtype) * scale
        rank_deficient = (diagonal <= tolerance.unsqueeze(-1)).any(dim=-1)
        _handle_qr_rank_policy(rank_deficient)
        ctx.save_for_backward(A, Q, R, rank_deficient)
        return Q, R

    @staticmethod
    def backward(ctx, dQ, dR):
        A, Q, R, rank_deficient = ctx.saved_tensors
        if bool(rank_deficient.any().item()):
            if _QR_RANK_POLICY == "native":
                return _native_qr_backward(A, dQ, dR)
            return _regularized_qr_backward(A, Q, R, dQ, dR, rank_deficient)
        return _native_qr_backward(A, dQ, dR)


def _simple_qr_backward(q, r, dq, dr):
    """Compute the QR gradient for the square-``R`` case."""
    if r.shape[-2] != r.shape[-1]:
        raise NotImplementedError(
            "QrGrad not implemented when ncols > nrows "
            "or full_matrices is true and ncols != nrows."
        )

    dq = torch.zeros_like(q) if dq is None else dq
    dr = torch.zeros_like(r) if dr is None else dr

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


def _regularized_qr_right_solve(rhs, r, shift):
    """Compute ``rhs @ R^{-H}`` with a scale-relative Tikhonov inverse."""
    gram = r.mH @ r
    size = r.shape[-1]
    identity = torch.eye(size, dtype=r.dtype, device=r.device)
    gram = gram + shift.square()[..., None, None] * identity
    # For ``shift == 0`` this is algebraically the ordinary right triangular
    # solve. For a singular pivot it is
    # ``rhs @ R @ (R^H R + shift**2 I)^{-1}``.
    return torch.linalg.solve(gram, (rhs @ r).mH).mH


def _qr_syminv_adjoint(x):
    """Adjoint of the QR upper-triangular Hermitian projection."""
    output = x + x.mH
    diagonal = output.diagonal(0, -2, -1)
    if output.is_complex():
        diagonal.real.mul_(0.5)
    else:
        diagonal.mul_(0.5)
    return output


def _qr_trilim_inv_adjoint_skew(x):
    """Adjoint QR skew projection for a wide reduced QR factorization."""
    output = (x - x.mH).tril()
    if output.is_complex():
        output.diagonal(0, -2, -1).imag.mul_(0.5)
    return output


def _regularized_qr_backward(a, q, r, dq, dr, singular_pivot):
    """Finite extension of Torch's reduced QR VJP at a singular pivot.

    The full-rank part exactly follows PyTorch's QR backward formula. For each
    singular unpivoted block, its final ``R^{-H}`` solve is replaced by the
    Tikhonov right inverse with a scale-relative shift. This preserves the
    exact forward QR and only chooses a bounded derivative where the
    mathematical QR chart has no unique derivative. An exactly zero input block
    retains a zero VJP, since it has neither a preferred QR gauge nor an
    intrinsic scale for the regularizer.
    """
    if (dq is None) and (dr is None):
        return torch.zeros_like(a)

    if dq is None:
        gradient = dr @ r.mH
    elif dr is None:
        gradient = -q.mH @ dq
    else:
        gradient = dr @ r.mH - q.mH @ dq

    block_scale = a.detach().abs().amax(dim=(-2, -1))
    zero_block = block_scale == 0
    scale_for_shift = torch.where(
        zero_block,
        torch.ones_like(block_scale),
        block_scale,
    )
    real_dtype = a.real.dtype if a.is_complex() else a.dtype
    relative_shift = _qr_regularization_relative_eps(real_dtype)
    shift = torch.where(
        singular_pivot,
        relative_shift * scale_for_shift,
        torch.zeros_like(scale_for_shift),
    )
    m = q.shape[-2]
    n = r.shape[-1]
    if m >= n:
        gradient = q @ _qr_syminv_adjoint(gradient.triu())
        if dq is not None:
            gradient = gradient + dq
        gradient = _regularized_qr_right_solve(gradient, r, shift)
    else:
        gradient = q @ _qr_trilim_inv_adjoint_skew(-gradient)
        r_leading = r.narrow(-1, 0, m)
        gradient = _regularized_qr_right_solve(gradient, r_leading, shift)
        gradient = torch.cat(
            (
                gradient,
                torch.zeros(
                    *gradient.shape[:-1],
                    n - m,
                    dtype=gradient.dtype,
                    device=gradient.device,
                ),
            ),
            dim=-1,
        )
        if dr is not None:
            gradient = gradient + q @ dr

    return torch.where(zero_block[..., None, None], torch.zeros_like(gradient), gradient)


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


def _native_svd_configured(A, *args, driver="auto", cpu_svd="torch", **kwargs):
    """Run native Torch SVD with an explicit backend policy."""
    kwargs.setdefault("full_matrices", False)
    if A.device.type == "cpu" and cpu_svd != "torch":
        if A.requires_grad:
            raise RuntimeError(
                "cpu_svd={!r} is forward-only for stabilized=False. Use "
                "cpu_svd='torch' or stabilized=True for autodiff.".format(cpu_svd)
            )
        return _scipy_svd(
            A,
            lapack_driver=cpu_svd.removeprefix("scipy_"),
        )
    if A.is_cuda and driver != "auto":
        kwargs.setdefault("driver", driver)
    return torch.linalg.svd(A, *args, **kwargs)


def _get_native_svd_wrapper(*, driver="auto", cpu_svd="torch"):
    """Return a cached Autoray wrapper for one native SVD policy."""
    _resolve_svd_options(driver=driver, cpu_svd=cpu_svd, fallback="none")
    if driver == "auto" and cpu_svd == "torch":
        return _native_svd
    key = ("native", driver, cpu_svd)
    function = _SVD_WRAPPERS.get(key)
    if function is None:
        def function(A, *args, **kwargs):
            return _native_svd_configured(
                A,
                *args,
                driver=driver,
                cpu_svd=cpu_svd,
                **kwargs,
            )

        function.__name__ = f"native_svd_{driver}_{cpu_svd}"
        _SVD_WRAPPERS[key] = function
    return _SVD_WRAPPERS[key]


def _get_stabilized_svd_wrapper(
    *,
    mode="complex",
    driver="auto",
    cpu_svd="torch",
    fallback="scipy_gesvd",
):
    """Return a cached regularized SVD wrapper for one explicit policy."""
    _resolve_svd_options(
        driver=driver,
        cpu_svd=cpu_svd,
        fallback=fallback,
    )
    function_class = SVD_real if mode == "real" else SVD
    key = ("stabilized", mode, driver, cpu_svd, fallback)
    function = _SVD_WRAPPERS.get(key)
    if function is None:
        function = _get_stabilized_svd_call(
            function_class,
            driver=driver,
            cpu_svd=cpu_svd,
            fallback=fallback,
        )
        _SVD_WRAPPERS[key] = function
    return _SVD_WRAPPERS[key]


def reg_native_svd_torch(*, svd_driver="auto", cpu_svd="torch"):
    """Register native Torch SVD with an optional backend policy.

    ``svd_driver`` applies only to CUDA tensors. ``cpu_svd`` can explicitly
    select Torch's native CPU LAPACK path or a SciPy LAPACK driver. SciPy
    backends are forward-only when the input requires gradients.
    """
    _register_once(
        "linalg.svd",
        _get_native_svd_wrapper(driver=svd_driver, cpu_svd=cpu_svd),
    )


def reset_torch_linalg_registrations():
    """Restore native Torch SVD/QR mappings and clear Pepsy's cache."""
    _REGISTERED_FUNCTIONS.clear()
    _configure_qr_rank_policy()
    reg_native_svd_torch()
    reg_complex_qr_torch()


def reg_rel_svd_torch(
    *,
    svd_driver="auto",
    cpu_svd="torch",
    svd_fallback="scipy_gesvd",
):
    """Register the relative-regularized Torch SVD rule in Autoray."""
    _register_once(
        "linalg.svd",
        _get_stabilized_svd_wrapper(
            mode="complex",
            driver=svd_driver,
            cpu_svd=cpu_svd,
            fallback=svd_fallback,
        ),
    )


def reg_complex_svd_torch(**kwargs):
    """Register the complex torch SVD autograd implementation in autoray.

    This compatibility name installs the same relative-regularized SVD rule as
    :func:`reg_rel_svd_torch`.
    """
    reg_rel_svd_torch(**kwargs)


def reg_real_svd_torch(
    *,
    svd_driver="auto",
    cpu_svd="torch",
    svd_fallback="scipy_gesvd",
):
    """Register the real Torch SVD autograd implementation in Autoray."""
    _register_once(
        "linalg.svd",
        _get_stabilized_svd_wrapper(
            mode="real",
            driver=svd_driver,
            cpu_svd=cpu_svd,
            fallback=svd_fallback,
        ),
    )


def reg_real_qr_torch(*, rank_policy="warn", rank_tol_factor=1.0):
    """Register real QR with a configurable rank-deficiency policy."""
    _configure_qr_rank_policy(rank_policy, rank_tol_factor)
    _register_once("linalg.qr", _get_stabilized_qr_call(QR_real))


def reg_complex_qr_torch(
    *,
    stabilized=False,
    rank_policy="warn",
    rank_tol_factor=1.0,
):
    """Register native or rank-safe Torch QR.

    The native path is the fast default. The stabilized path uses the finite
    rank-deficient VJP and accepts the same reduced-QR keyword arguments as
    the native Torch operation.
    """
    if not stabilized:
        _register_once("linalg.qr", torch.linalg.qr)
        return
    _configure_qr_rank_policy(rank_policy, rank_tol_factor)
    _register_once("linalg.qr", _get_stabilized_qr_call(QR_complex_safe))


def reg_quimb_torch_split_drivers(
    mode="real",
    *,
    svd_driver="auto",
    cpu_svd="torch",
    svd_fallback="scipy_gesvd",
    qr_rank_policy=None,
    qr_rank_tol_factor=None,
):
    """Advanced helper for Quimb raw-block split drivers.

    Quimb's composed ``qr_stabilized`` and ``svd_truncated`` drivers receive
    raw Torch blocks from Symmray and call the Torch namespace directly. That
    path bypasses Autoray's ordinary ``linalg.qr`` and ``linalg.svd``
    registrations, so PEPS autodiff needs backend-specific Quimb drivers as
    well. Use :func:`pepsy.register_torch_linalg` with
    ``quimb_split_drivers=True`` in application code; this helper remains for
    backend implementation and compatibility. Missing Quimb is allowed because
    it is optional outside tensor-network optimization.
    """
    if mode not in {"real", "complex"}:
        raise ValueError("mode must be 'real' or 'complex'")
    if qr_rank_policy is not None:
        _configure_qr_rank_policy(
            qr_rank_policy,
            1.0 if qr_rank_tol_factor is None else qr_rank_tol_factor,
        )
    try:
        import quimb.tensor.decomp as qd  # pylint: disable=import-outside-toplevel
    except ImportError:  # pragma: no cover - quimb is an optional integration
        return False

    def qr_torch(x, absorb=qd.get_U_sVH, stabilized=True, **kwargs):
        del kwargs
        absorb_i = qd._ABSORB_MAP[absorb]
        transposed = absorb_i in (qd.get_Us_VH, qd.get_Us, qd.get_VH)
        if transposed:
            x = x.swapaxes(-2, -1)
            absorb_i = qd._ABSORB_TRANSPOSE_MAP[absorb_i]

        if mode == "real" and not x.is_complex():
            Q, R = QR_real_safe.apply(x)
        elif mode == "complex" and x.is_complex():
            Q, R = QR_complex_safe.apply(x)
        else:
            Q, R = torch.linalg.qr(x)

        if stabilized:
            # The sign/phase is a discrete gauge choice. It should not enter
            # the VJP. ``phase(0)`` must be one: setting it to zero for a
            # rank-deficient block erases the associated Q column and R row,
            # so the nominally lossless QR split no longer reconstructs its
            # input. This matches Quimb's backend-generic ``sgn`` rule.
            diagonal = torch.diagonal(R, dim1=-2, dim2=-1)
            phase = torch.where(
                diagonal == 0,
                torch.ones_like(diagonal),
                torch.sgn(diagonal),
            ).detach()
            Q = Q * torch.conj(phase)[..., None, :]
            R = phase[..., :, None] * R

        left = None if absorb_i == qd.get_sVH else Q
        right = None if absorb_i == qd.get_U else R
        if transposed:
            left, right = right, left
            if left is not None:
                left = left.swapaxes(-2, -1)
            if right is not None:
                right = right.swapaxes(-2, -1)
        return left, None, right

    def svd_torch(
        x,
        cutoff=-1.0,
        cutoff_mode=qd.cutoff_mode_rsum2,
        max_bond=-1,
        absorb=qd.get_Usq_sqVH,
        renorm=0,
        info=None,
        **kwargs,
    ):
        del kwargs
        absorb_i = qd._ABSORB_MAP[absorb]
        cutoff_mode_i = qd._CUTOFF_MODE_MAP[cutoff_mode]
        if mode == "real" and not x.is_complex():
            svd_function = _get_stabilized_svd_wrapper(
                mode="real",
                driver=svd_driver,
                cpu_svd=cpu_svd,
                fallback=svd_fallback,
            )
        else:
            svd_function = _get_stabilized_svd_wrapper(
                mode="complex",
                driver=svd_driver,
                cpu_svd=cpu_svd,
                fallback=svd_fallback,
            )
        U, s, VH = svd_function(x)
        return qd._trim_and_renorm_svd_result(
            U,
            s,
            VH,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode_i,
            max_bond=max_bond,
            absorb=absorb_i,
            renorm=renorm,
            info=info,
            xp=qd.get_namespace(x),
        )

    try:
        qd.qr_stabilized.register("torch", qr_torch)
        qd.svd_truncated.register("torch", svd_torch)
    except AttributeError:  # pragma: no cover - older Quimb integration
        return False
    return True


def reset_quimb_torch_split_drivers():
    """Restore Quimb's default Torch split drivers if Quimb is installed."""
    try:
        import quimb.tensor.decomp as qd  # pylint: disable=import-outside-toplevel
    except ImportError:  # pragma: no cover - quimb is an optional integration
        return False
    try:
        qd.qr_stabilized.register("torch", qd.qr_stabilized._default_fn)
        qd.svd_truncated.register("torch", qd.svd_truncated._default_fn)
    except AttributeError:  # pragma: no cover - older Quimb integration
        return False
    return True


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
