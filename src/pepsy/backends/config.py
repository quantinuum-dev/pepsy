"""Backend configuration and optional autodiff registrations."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import autoray as ar
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

__all__ = [
    "build_backend", "backend_torch", "backend_numpy", "backend_cupy", "backend_jax",
    "TorchLinalgConfig", "get_torch_linalg_config",
    "register_torch_linalg", "register_jax_linalg", "reg_native_svd_torch",
    "reg_native_svd_jax", "reg_rel_svd_torch", "reg_real_svd_torch",
    "reg_complex_svd_torch", "reg_real_qr_torch", "reg_complex_qr_torch",
    "reg_rel_svd_jax", "reg_real_svd_jax", "reg_complex_svd_jax",
    "reset_linalg_registrations",
    "reg_stop_gradient_torch", "stop_grad", "set_default_array_backend",
    "get_default_array_backend", "set_default_grad_backend",
    "get_default_grad_backend", "reset_default_backends",
]

_DEFAULT_ARRAY_BACKEND = None
_DEFAULT_GRAD_BACKEND = None
_ACTIVE_TORCH_LINALG_CONFIG = None


@dataclass(frozen=True)
class TorchLinalgConfig:
    """Explicit policy for Pepsy's process-global Torch linalg dispatch.

    ``register()`` installs the complete policy: SVD forward selection,
    stabilized SVD reverse-mode differentiation when requested, QR
    differentiation, and optional Quimb raw-block split drivers. Keeping these
    decisions in one immutable object is important because Autoray and Quimb
    registrations are process-global.

    Parameters
    ----------
    mode : {"complex", "real"}, default="complex"
        Select the stabilized SVD/QR rule when ``stabilized=True``.
    stabilized : bool, default=False
        Use Pepsy's relative-regularized SVD VJP and the configured QR policy.
        Set this to ``True`` for Torch autodiff through difficult or
        rank-deficient tensor-network splits. The default keeps native Torch
        forward and backward behavior and is the recommended setting for
        ordinary non-differentiable simulation.
    svd_driver : {"auto", "gesvdj", "gesvda", "gesvd"}, default="auto"
        CUDA cuSOLVER driver. ``"auto"`` leaves the choice to Torch. The
        approximate ``"gesvda"`` driver requires ``allow_approximate=True``.
        This option has no effect on CPU tensors.
    cpu_svd : {"torch", "scipy_gesdd", "scipy_gesvd"}, default="torch"
        CPU forward SVD implementation. SciPy choices are useful for explicit
        LAPACK experimentation and forward-only MPS runs. With
        ``stabilized=True``, the custom backward remains available.
    svd_fallback : {"auto", "none", "scipy_gesdd", "scipy_gesvd"}, default="auto"
        Backend used after a Torch SVD failure. ``"auto"`` means no fallback
        for native SVD and ``"scipy_gesvd"`` for stabilized SVD.
    allow_approximate : bool, default=False
        Safety acknowledgement required for CUDA's approximate ``gesvda``.
    qr_rank_policy : {"warn", "native", "error"}, default="warn"
        Rank-deficiency response for the stabilized real QR VJP. ``"warn"``
        reports a potentially ill-conditioned native fallback, ``"native"``
        accepts it silently, and ``"error"`` stops the optimization.
    qr_rank_tol_factor : float, default=1.0
        Scale-relative multiplier used by the stabilized real QR rule.
    quimb_split_drivers : bool, default=False
        Also install the configured safe SVD/QR split drivers into Quimb's
        raw-block registry. Raw Symmray blocks bypass Autoray, so this must be
        ``True`` for Torch-autodiff PEPS/Symmray workflows. Dense workflows
        normally leave it disabled.

    Notes
    -----
    Autoray and Quimb registrations are process-global. Prefer
    ``config.register()`` at application startup, or ``with
    config.activated():`` for a scoped experiment.
    """

    mode: str = "complex"
    stabilized: bool = False
    svd_driver: str = "auto"
    cpu_svd: str = "torch"
    svd_fallback: str = "auto"
    allow_approximate: bool = False
    qr_rank_policy: str = "warn"
    qr_rank_tol_factor: float = 1.0
    quimb_split_drivers: bool = False

    def __post_init__(self):
        mode = str(self.mode).strip().lower()
        svd_driver = "auto" if self.svd_driver is None else str(self.svd_driver)
        cpu_svd = "torch" if self.cpu_svd is None else str(self.cpu_svd)
        svd_fallback = "auto" if self.svd_fallback is None else str(self.svd_fallback)
        qr_rank_policy = str(self.qr_rank_policy).strip().lower()
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "svd_driver", svd_driver)
        object.__setattr__(self, "cpu_svd", cpu_svd)
        object.__setattr__(self, "svd_fallback", svd_fallback)
        object.__setattr__(self, "qr_rank_policy", qr_rank_policy)
        self.validate()

    def validate(self):
        """Validate this policy and return ``self`` for fluent setup."""
        if self.mode not in {"complex", "real"}:
            raise ValueError("mode must be 'complex' or 'real'")
        if self.svd_driver not in {"auto", "gesvdj", "gesvda", "gesvd"}:
            raise ValueError(
                "svd_driver must be one of: auto, gesvdj, gesvda, gesvd"
            )
        if self.cpu_svd not in {"torch", "scipy_gesdd", "scipy_gesvd"}:
            raise ValueError(
                "cpu_svd must be one of: torch, scipy_gesdd, scipy_gesvd"
            )
        if self.svd_fallback not in {
            "auto", "none", "scipy_gesdd", "scipy_gesvd"
        }:
            raise ValueError(
                "svd_fallback must be one of: auto, none, scipy_gesdd, scipy_gesvd"
            )
        if self.svd_driver == "gesvda" and not self.allow_approximate:
            raise ValueError(
                "svd_driver='gesvda' is approximate; pass "
                "allow_approximate=True to enable it explicitly."
            )
        if self.qr_rank_policy not in {"warn", "native", "error"}:
            raise ValueError("qr_rank_policy must be one of: warn, native, error")
        try:
            qr_factor = float(self.qr_rank_tol_factor)
        except (TypeError, ValueError) as exc:
            raise TypeError("qr_rank_tol_factor must be a positive finite number") from exc
        if not np.isfinite(qr_factor) or qr_factor <= 0.0:
            raise ValueError("qr_rank_tol_factor must be a positive finite number")
        object.__setattr__(self, "qr_rank_tol_factor", qr_factor)
        return self

    @property
    def resolved_svd_fallback(self):
        """Return the concrete fallback selected by ``svd_fallback``."""
        if self.svd_fallback != "auto":
            return self.svd_fallback
        return "scipy_gesvd" if self.stabilized else "none"

    @property
    def approximate(self):
        """Whether this policy permits an approximate CUDA SVD driver."""
        return self.svd_driver == "gesvda"

    @property
    def exact(self):
        """Whether the selected SVD driver is a non-approximate algorithm."""
        return not self.approximate

    def to_dict(self):
        """Return JSON-friendly policy fields and resolved decisions."""
        return {
            "mode": self.mode,
            "stabilized": self.stabilized,
            "svd_driver": self.svd_driver,
            "cpu_svd": self.cpu_svd,
            "svd_fallback": self.svd_fallback,
            "resolved_svd_fallback": self.resolved_svd_fallback,
            "allow_approximate": self.allow_approximate,
            "approximate": self.approximate,
            "exact": self.exact,
            "qr_rank_policy": self.qr_rank_policy,
            "qr_rank_tol_factor": self.qr_rank_tol_factor,
            "quimb_split_drivers": self.quimb_split_drivers,
        }

    def describe(self):
        """Return policy fields together with available runtime backends."""
        info = self.to_dict()
        info["torch_version"] = None if torch is None else torch.__version__
        info["cuda_available"] = bool(torch is not None and torch.cuda.is_available())
        try:
            from . import linalg_torch as lr  # pylint: disable=import-outside-toplevel

            info["scipy_available"] = lr.scipy_linalg is not None
        except ImportError:  # pragma: no cover - optional dependency
            info["scipy_available"] = False
        return info

    def register(self):
        """Install the SVD, QR, and optional Quimb split policy.

        This is intentionally the one high-level registration operation. It
        keeps the forward SVD choice, the autodiff SVD rule, and the QR rule
        on the same policy instead of allowing separate registrations to
        silently disagree.
        """
        _install_torch_linalg_config(self)
        return self

    @contextmanager
    def activated(self):
        """Temporarily install this policy and restore the previous policy."""
        previous = get_torch_linalg_config()
        if not self.quimb_split_drivers:
            from . import linalg_torch as lr  # pylint: disable=import-outside-toplevel

            lr.reset_quimb_torch_split_drivers()
        self.register()
        try:
            yield self
        finally:
            if previous is None:
                reset_linalg_registrations(backend="torch")
            else:
                if not previous.quimb_split_drivers:
                    from . import linalg_torch as lr  # pylint: disable=import-outside-toplevel

                    lr.reset_quimb_torch_split_drivers()
                previous.register()


def get_torch_linalg_config():
    """Return the last Pepsy Torch linalg policy, or ``None`` if unknown."""
    return _ACTIVE_TORCH_LINALG_CONFIG

def _patch_unhashable_device_namespace_key():
    """Patch autoray namespace cache keys for unhashable backend device objects."""
    try:
        import autoray  # pylint: disable=import-outside-toplevel
        import autoray.autoray as ar_core  # pylint: disable=import-outside-toplevel
    except ImportError:  # pragma: no cover
        return

    if getattr(ar_core, "_pepsy_unhashable_device_patch", False):
        return

    original_get_namespace = ar_core.get_namespace

    def _safe_get_namespace(like=None, device=None, dtype=None, submodule=None):
        if (device is None) and (like is not None) and (not isinstance(like, str)):
            try:
                device = like.device
            except AttributeError:
                device = None

        if device is not None:
            try:
                hash(device)
            except TypeError:
                dev_id = getattr(device, "id", None)
                device = f"device:{dev_id}" if dev_id is not None else str(device)

        return original_get_namespace(
            like=like,
            device=device,
            dtype=dtype,
            submodule=submodule,
        )

    ar_core.get_namespace = _safe_get_namespace
    autoray.get_namespace = _safe_get_namespace

    try:
        import quimb.tensor.decomp as qtn_decomp  # pylint: disable=import-outside-toplevel

        qtn_decomp.get_namespace = _safe_get_namespace
    except Exception:  # pragma: no cover
        pass

    ar_core._pepsy_unhashable_device_patch = True


def _validate_backend_callable(name, fn):
    if fn is not None and not callable(fn):
        raise TypeError(f"{name} must be callable or None")


def set_default_array_backend(array_backend):
    """Set package-wide default array backend caster.

    Parameters
    ----------
    array_backend : callable | None
        Function mapping arrays to a target backend. ``None`` clears default.
    """
    _validate_backend_callable("array_backend", array_backend)
    global _DEFAULT_ARRAY_BACKEND  # pylint: disable=global-statement
    _DEFAULT_ARRAY_BACKEND = array_backend


def get_default_array_backend():
    """Return package-wide default array backend caster, or ``None``."""
    return _DEFAULT_ARRAY_BACKEND


def set_default_grad_backend(grad_backend):
    """Set package-wide default gradient backend caster.

    Parameters
    ----------
    grad_backend : callable | None
        Function mapping arrays to trainable backend tensors.
    """
    _validate_backend_callable("grad_backend", grad_backend)
    global _DEFAULT_GRAD_BACKEND  # pylint: disable=global-statement
    _DEFAULT_GRAD_BACKEND = grad_backend


def get_default_grad_backend():
    """Return package-wide default gradient backend caster, or ``None``."""
    return _DEFAULT_GRAD_BACKEND


def reset_default_backends():
    """Clear package-wide backend defaults."""
    global _DEFAULT_ARRAY_BACKEND  # pylint: disable=global-statement
    global _DEFAULT_GRAD_BACKEND  # pylint: disable=global-statement
    _DEFAULT_ARRAY_BACKEND = None
    _DEFAULT_GRAD_BACKEND = None


def backend_torch(device="cpu", dtype=None, requires_grad=False):
    """Return a converter that materializes arrays as torch tensors."""
    if torch is None:  # pragma: no cover - exercised in no-torch CI
        raise ImportError(
            "backend_torch requires optional dependency 'torch'. "
            "Install it with: pip install pepsy[torch] (or pip install torch)."
        )

    def cast_array(x, device=device, dtype=dtype, requires_grad=requires_grad):

        # Symmray can use complex NumPy containers for real-valued blocks. If
        # their imaginary part is identically zero, remove it before asking
        # Torch for a real tensor. This preserves the requested float64 path
        # without emitting Torch's misleading "discards the imaginary part"
        # warning. Non-zero imaginary parts retain the historical conversion
        # behavior below.
        target_is_real = (
            dtype is not None
            and not torch.empty((), dtype=dtype).is_complex()
        )
        if target_is_real:
            if isinstance(x, torch.Tensor):
                if x.is_complex() and not bool(torch.any(x.imag != 0)):
                    x = x.real
            elif np.iscomplexobj(x):
                x_np = np.asarray(x)
                if not np.any(x_np.imag != 0):
                    x = x_np.real

        if isinstance(x, torch.Tensor):
            out = x.detach() if requires_grad else x

            if dtype is None:
                out = out.to(device=device)
            else:
                out = out.to(device=device, dtype=dtype)

        else:
            if dtype is None:
                out = torch.as_tensor(x, device=device)
            else:
                out = torch.as_tensor(x, dtype=dtype, device=device)

        # Trainable tensors must be floating or complex
        if requires_grad and not (out.is_floating_point() or out.is_complex()):
            out = out.to(dtype=torch.float64)

        if requires_grad:
            out.requires_grad_(True)
        else:
            out.requires_grad_(False)

        return out

    return cast_array


def build_backend(device="cpu", dtype=None, requires_grad=False, *, set_default=True):
    """Build the standard Torch array backend, defaulting to CPU.

    The existing :func:`backend_torch` name remains unchanged.  This helper is
    the concise public entry point for workflows that want one backend
    converter and one package-wide default::

        import pepsy as py
        to_backend = py.build_backend()  # Torch CPU

    Parameters are forwarded to :func:`backend_torch`.  By default the
    resulting converter is also installed as Pepsy's default array backend;
    pass ``set_default=False`` when only the returned converter should be
    used.  Explicit ``to_backend=`` / ``array_backend=`` arguments continue to
    take precedence in individual APIs.
    """

    converter = backend_torch(
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )
    if set_default:
        set_default_array_backend(converter)
    return converter


def backend_numpy(dtype=np.float64):
    """Return a converter that materializes arrays as NumPy arrays."""

    def cast_array(x, dtype=dtype):
        return np.array(x, dtype=dtype)

    return cast_array


def backend_cupy(device=None, dtype=None):
    """Return a converter that materializes arrays as CuPy arrays.

    Parameters
    ----------
    device : int | cupy.cuda.Device | None, optional
        Target CUDA device. If ``None``, use CuPy's current device.
    dtype : dtype-like | torch.dtype | None, optional
        Target CuPy dtype. If ``None``, infer from input. Torch dtypes are
        accepted and internally mapped to CuPy-compatible dtypes.
    """
    try:
        import cupy as cp  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "backend_cupy requires optional dependency 'cupy'. "
            "Install it with: pip install cupy-cuda12x (or your CUDA variant)."
        ) from exc

    _patch_unhashable_device_namespace_key()

    target_device = device
    if isinstance(target_device, int):
        target_device = cp.cuda.Device(target_device)

    if torch is not None and isinstance(dtype, torch.dtype):
        torch_to_cupy = {
            torch.complex128: cp.complex128,
            torch.complex64: cp.complex64,
            torch.float64: cp.float64,
            torch.float32: cp.float32,
            torch.float16: cp.float16,
            torch.int64: cp.int64,
            torch.int32: cp.int32,
            torch.int16: cp.int16,
            torch.int8: cp.int8,
            torch.uint8: cp.uint8,
            torch.bool: cp.bool_,
        }
        if dtype not in torch_to_cupy:
            raise ValueError(
                f"backend_cupy does not support torch dtype {dtype!r}."
            )
        dtype = torch_to_cupy[dtype]

    def cast_array(x, device=target_device, dtype=dtype):
        if device is None:
            return cp.asarray(x, dtype=dtype)
        with device:
            return cp.asarray(x, dtype=dtype)

    return cast_array


def backend_jax(device="cpu", dtype=None):
    """Return a converter that materializes arrays as JAX arrays.

    Parameters
    ----------
    device : str | jax.Device | None, optional
        Target device. Strings ``"cpu"``, ``"cuda"``/``"gpu"`` (optionally with
        an index, e.g. ``"cuda:1"``) are resolved against ``jax.devices``. A
        ``jax.Device`` instance is used as-is. ``None`` leaves placement to
        JAX's default.
    dtype : str | jax.numpy.dtype | None, optional
        Target dtype, e.g. ``"float64"`` or ``jnp.complex128``. ``None`` infers
        from the input.

    Notes
    -----
    JAX arrays are immutable and have no ``requires_grad`` flag; gradients in
    JAX flow via tracing (``jax.grad`` / ``jax.value_and_grad``). This
    converter therefore does not expose a ``requires_grad`` argument.
    """
    try:
        import jax  # pylint: disable=import-outside-toplevel
        import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "backend_jax requires optional dependency 'jax'. "
            "Install it with: pip install jax (or jax[cuda12])."
        ) from exc

    def _resolve_device(dev):
        if dev is None or not isinstance(dev, str):
            return dev
        s = dev.lower()
        if ":" in s:
            kind, idx_s = s.split(":", 1)
            idx = int(idx_s)
        else:
            kind, idx = s, 0
        if kind == "cuda":
            kind = "gpu"
        try:
            return jax.devices(kind)[idx]
        except (RuntimeError, IndexError) as err:
            raise ValueError(
                f"backend_jax: device {dev!r} not available; "
                f"jax.devices() = {jax.devices()}"
            ) from err

    target_device = _resolve_device(device)
    if dtype is None:
        target_dtype = None
    else:
        # Canonicalize dtypes under current JAX x64 policy so requests like
        # float64/complex128 do not emit truncation warnings when x64 is off.
        target_dtype = jax.dtypes.canonicalize_dtype(jnp.dtype(dtype))

    def cast_array(x, device=target_device, dtype=target_dtype):
        # Coerce non-JAX inputs to a NumPy-compatible form through Autoray so
        # Torch, CuPy, and other registered backends share one host boundary.
        try:
            if ar.infer_backend(x) != "jax":
                x = ar.to_numpy(x)
        except Exception:
            # Let JAX handle custom array-likes that Autoray cannot infer.
            pass
        arr = jnp.asarray(x, dtype=dtype)
        if device is not None:
            arr = jax.device_put(arr, device)
        return arr

    return cast_array


def _install_torch_linalg_config(config):
    """Install one validated Torch linalg policy."""
    if torch is None:  # pragma: no cover - exercised in no-torch CI
        raise ImportError(
            "Torch linalg configuration requires optional dependency 'torch'. "
            "Install it with: pip install pepsy[torch] (or pip install torch)."
        )
    from ..backends import linalg_torch as lr  # pylint: disable=import-outside-toplevel

    # SVD registration: native Torch is the fast/default path, while the
    # stabilized classes provide finite reverse-mode derivatives at repeated
    # or tiny singular values. ``svd_driver`` and ``cpu_svd`` only select the
    # forward decomposition; they do not silently change the tensor dtype.
    fallback = config.resolved_svd_fallback
    if config.stabilized:
        if config.mode == "complex":
            lr.reg_rel_svd_torch(
                svd_driver=config.svd_driver,
                cpu_svd=config.cpu_svd,
                svd_fallback=fallback,
            )
        else:
            lr.reg_real_svd_torch(
                svd_driver=config.svd_driver,
                cpu_svd=config.cpu_svd,
                svd_fallback=fallback,
            )
        # QR registration: the real rule adds rank diagnostics and a finite
        # VJP for singular pivots. Complex ordinary Autoray QR stays native
        # for speed; its safe complex rule is used by the Quimb raw-block
        # split path below when that path is enabled.
        if config.mode == "real":
            lr.reg_real_qr_torch(
                rank_policy=config.qr_rank_policy,
                rank_tol_factor=config.qr_rank_tol_factor,
            )
        else:
            lr.reg_complex_qr_torch()
    else:
        lr.reg_native_svd_torch(
            svd_driver=config.svd_driver,
            cpu_svd=config.cpu_svd,
        )
        lr.reg_complex_qr_torch()

    # Quimb registration: raw Symmray blocks bypass Autoray, so PEPS
    # autodiff needs the same stabilized SVD/QR choices installed in Quimb's
    # ``svd_truncated`` and ``qr_stabilized`` registries as well.
    if config.quimb_split_drivers:
        lr.reg_quimb_torch_split_drivers(
            mode=config.mode,
            svd_driver=config.svd_driver,
            cpu_svd=config.cpu_svd,
            # Quimb's optional raw-block driver is the stabilized custom
            # split path even when the ordinary Autoray path is native.
            svd_fallback=fallback if config.stabilized else "scipy_gesvd",
        )

    global _ACTIVE_TORCH_LINALG_CONFIG  # pylint: disable=global-statement
    _ACTIVE_TORCH_LINALG_CONFIG = config


def register_torch_linalg(
    mode="complex",
    *,
    stabilized=False,
    svd_driver="auto",
    cpu_svd="torch",
    svd_fallback="auto",
    allow_approximate=False,
    qr_rank_policy="warn",
    qr_rank_tol_factor=1.0,
    quimb_split_drivers=False,
):
    """Configure the canonical Torch linalg rules used by Pepsy.

    This is the public configuration entry point. The lower-level
    ``reg_*_torch`` helpers are retained for advanced and compatibility uses;
    do not combine them in ordinary PEPS workflows.

    Parameters
    ----------
    mode : {"complex", "real"}, default="complex"
        Select the real or complex stabilized rule when ``stabilized=True``.
    stabilized : bool, default=False
        Keep native Torch SVD/QR by default. Set this to ``True`` to install
        Pepsy's relative-regularized SVD and validated real-QR rules.
    svd_driver : {"auto", "gesvdj", "gesvda", "gesvd"}, default="auto"
        CUDA cuSOLVER driver. ``"gesvda"`` is approximate and requires
        ``allow_approximate=True``. CPU tensors ignore this option.
    cpu_svd : {"torch", "scipy_gesdd", "scipy_gesvd"}, default="torch"
        CPU SVD implementation. SciPy choices are forward-only for native
        ``stabilized=False`` registration, and require SciPy at runtime.
    svd_fallback : {"auto", "none", "scipy_gesdd", "scipy_gesvd"}, default="auto"
        SVD failure fallback. ``"auto"`` means ``"none"`` for native SVD and
        ``"scipy_gesvd"`` for stabilized SVD.
    allow_approximate : bool, default=False
        Explicitly acknowledge the accuracy tradeoff of ``svd_driver="gesvda"``.
    qr_rank_policy : {"warn", "native", "error"}, default="warn"
        Response to rank-deficient inputs when stabilized real QR is active.
    qr_rank_tol_factor : float, default=1.0
        Multiplier for the scale-aware real-QR rank threshold.
    quimb_split_drivers : bool, default=False
        Also register Pepsy's stable Torch drivers for Quimb's
        ``qr_stabilized`` and ``svd_truncated`` split paths. Enable this for
        Torch-autodiff PEPS workflows with native Symmray blocks. The drivers
        use the same ``mode`` and a zero-safe QR phase convention
        ``phase(0)=1``. It is opt-in because this is a process-global Quimb
        registration. Passing ``False`` leaves any previously installed Quimb
        drivers unchanged; use :func:`reset_linalg_registrations` to restore
        Quimb's defaults explicitly.
    """
    config = TorchLinalgConfig(
        mode=mode,
        stabilized=stabilized,
        svd_driver=svd_driver,
        cpu_svd=cpu_svd,
        svd_fallback=svd_fallback,
        allow_approximate=allow_approximate,
        qr_rank_policy=qr_rank_policy,
        qr_rank_tol_factor=qr_rank_tol_factor,
        quimb_split_drivers=quimb_split_drivers,
    )
    return config.register()


def register_jax_linalg(*, stabilized=False):
    """Register native or truncation-safe JAX SVD in Autoray.

    Parameters
    ----------
    stabilized : bool, default=False
        Keep native thin SVD by default. Set this to ``True`` to install the
        custom VJP that restores cotangents from Quimb fixed-rank truncation.
    """
    try:
        __import__("jax")
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "register_jax_linalg requires optional dependency 'jax'. "
            "Install it with: pip install jax jaxlib."
        ) from exc
    from ..backends import linalg_jax as lr  # pylint: disable=import-outside-toplevel

    if stabilized:
        lr.reg_complex_svd_jax()
    else:
        lr.reg_native_svd_jax()


def reset_linalg_registrations(backend="all"):
    """Restore native linalg mappings and clear Pepsy registration caches.

    Parameters
    ----------
    backend : {"torch", "jax", "all"}, default="all"
        Which optional backend registration cache to reset. ``"all"`` skips
        optional backends that are not installed.
    """
    if backend not in {"torch", "jax", "all"}:
        raise ValueError("backend must be one of: all, jax, torch")

    if backend in {"torch", "all"}:
        if torch is None:
            if backend == "torch":
                raise ImportError(
                    "reset_linalg_registrations(backend='torch') requires "
                    "optional dependency 'torch'."
                )
        else:
            from ..backends import linalg_torch as lr  # pylint: disable=import-outside-toplevel

            lr.reset_torch_linalg_registrations()
            lr.reset_quimb_torch_split_drivers()
            global _ACTIVE_TORCH_LINALG_CONFIG  # pylint: disable=global-statement
            _ACTIVE_TORCH_LINALG_CONFIG = TorchLinalgConfig()

    if backend in {"jax", "all"}:
        try:
            __import__("jax")
        except ImportError:
            if backend == "jax":
                raise ImportError(
                    "reset_linalg_registrations(backend='jax') requires "
                    "optional dependency 'jax'."
                )
        else:
            from ..backends import linalg_jax as lr  # pylint: disable=import-outside-toplevel

            lr.reset_jax_linalg_registrations()


def reg_native_svd_torch():
    """Advanced compatibility helper; prefer ``register_torch_linalg``."""
    if torch is None:  # pragma: no cover - exercised in no-torch CI
        raise ImportError(
            "reg_native_svd_torch requires optional dependency 'torch'. "
            "Install it with: pip install pepsy[torch] (or pip install torch)."
        )
    from ..backends import linalg_torch as lr  # pylint: disable=import-outside-toplevel

    lr.reg_native_svd_torch()
    global _ACTIVE_TORCH_LINALG_CONFIG  # pylint: disable=global-statement
    _ACTIVE_TORCH_LINALG_CONFIG = TorchLinalgConfig()


def reg_native_svd_jax():
    """Register native JAX thin SVD in autoray."""
    try:
        __import__("jax")
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "reg_native_svd_jax requires optional dependency 'jax'. "
            "Install it with: pip install jax jaxlib."
        ) from exc
    from ..backends import linalg_jax as lr  # pylint: disable=import-outside-toplevel

    lr.reg_native_svd_jax()


def reg_rel_svd_torch():
    """Advanced compatibility helper for stabilized Torch SVD.

    The registered autoray ``torch`` SVD uses Townsend's rectangular SVD
    reverse-mode update, Lorentzian broadening of singular-value denominators
    from differentiable tensor-network practice, and the complex phase/gauge
    correction for complex-valued SVDs.
    """
    if torch is None:  # pragma: no cover - exercised in no-torch CI
        raise ImportError(
            "reg_rel_svd_torch requires optional dependency 'torch'. "
            "Install it with: pip install pepsy[torch] (or pip install torch)."
        )
    from ..backends import linalg_torch as lr  # pylint: disable=import-outside-toplevel

    lr.reg_rel_svd_torch()
    global _ACTIVE_TORCH_LINALG_CONFIG  # pylint: disable=global-statement
    _ACTIVE_TORCH_LINALG_CONFIG = TorchLinalgConfig(
        mode="complex",
        stabilized=True,
    )


def reg_complex_svd_torch():
    """Advanced compatibility helper for the stabilized complex Torch SVD.

    Compatibility wrapper for :func:`reg_rel_svd_torch`.
    """
    if torch is None:  # pragma: no cover - exercised in no-torch CI
        raise ImportError(
            "reg_complex_svd_torch requires optional dependency 'torch'. "
            "Install it with: pip install pepsy[torch] (or pip install torch)."
        )
    from ..backends import linalg_torch as lr  # pylint: disable=import-outside-toplevel

    lr.reg_rel_svd_torch()
    global _ACTIVE_TORCH_LINALG_CONFIG  # pylint: disable=global-statement
    _ACTIVE_TORCH_LINALG_CONFIG = TorchLinalgConfig(
        mode="complex",
        stabilized=True,
    )


def reg_real_svd_torch():
    """Advanced compatibility helper for the stabilized real Torch SVD.

    This is the real counterpart of :func:`reg_rel_svd_torch`. It shares the
    robust forward path (``gesvd`` driver on CUDA plus a batched SciPy ``gesvd``
    fallback), the same Townsend rectangular reverse-mode update, and the
    scale-aware Lorentzian broadening of the singular-value denominators, while
    dropping the complex phase/gauge correction. It supports rectangular and
    batched real inputs.
    """
    if torch is None:  # pragma: no cover - exercised in no-torch CI
        raise ImportError(
            "reg_real_svd_torch requires optional dependency 'torch'. "
            "Install it with: pip install pepsy[torch] (or pip install torch)."
        )
    from ..backends import linalg_torch as lr  # pylint: disable=import-outside-toplevel

    lr.reg_real_svd_torch()
    global _ACTIVE_TORCH_LINALG_CONFIG  # pylint: disable=global-statement
    _ACTIVE_TORCH_LINALG_CONFIG = TorchLinalgConfig(
        mode="real",
        stabilized=True,
    )


def reg_complex_qr_torch():
    """Advanced compatibility helper; prefer ``register_torch_linalg``."""
    if torch is None:  # pragma: no cover - exercised in no-torch CI
        raise ImportError(
            "reg_complex_qr_torch requires optional dependency 'torch'. "
            "Install it with: pip install pepsy[torch] (or pip install torch)."
        )
    from ..backends import linalg_torch as lr  # pylint: disable=import-outside-toplevel

    lr.reg_complex_qr_torch()


def reg_real_qr_torch(*, rank_policy="warn", rank_tol_factor=1.0):
    """Advanced compatibility helper; prefer ``register_torch_linalg``."""
    if torch is None:  # pragma: no cover - exercised in no-torch CI
        raise ImportError(
            "reg_real_qr_torch requires optional dependency 'torch'. "
            "Install it with: pip install pepsy[torch] (or pip install torch)."
        )
    from ..backends import linalg_torch as lr  # pylint: disable=import-outside-toplevel

    lr.reg_real_qr_torch(
        rank_policy=rank_policy,
        rank_tol_factor=rank_tol_factor,
    )


def reg_complex_svd_jax():
    """Register complex JAX SVD custom-VJP rule in autoray."""
    try:
        __import__("jax")
    except ImportError as exc:  # pragma: no cover - exercised in no-jax CI
        raise ImportError(
            "reg_complex_svd_jax requires optional dependency 'jax'. "
            "Install it with: pip install jax jaxlib."
        ) from exc

    from ..backends import linalg_jax as lr  # pylint: disable=import-outside-toplevel

    lr.reg_complex_svd_jax()


def reg_rel_svd_jax():
    """Register JAX SVD with Pepsy's truncation-safe VJP rule in autoray."""
    try:
        __import__("jax")
    except ImportError as exc:  # pragma: no cover - exercised in no-jax CI
        raise ImportError(
            "reg_rel_svd_jax requires optional dependency 'jax'. "
            "Install it with: pip install jax jaxlib."
        ) from exc

    from ..backends import linalg_jax as lr  # pylint: disable=import-outside-toplevel

    lr.reg_rel_svd_jax()


def reg_real_svd_jax():
    """Register JAX SVD's truncation-safe VJP rule for real workloads."""
    try:
        __import__("jax")
    except ImportError as exc:  # pragma: no cover - exercised in no-jax CI
        raise ImportError(
            "reg_real_svd_jax requires optional dependency 'jax'. "
            "Install it with: pip install jax jaxlib."
        ) from exc

    from ..backends import linalg_jax as lr  # pylint: disable=import-outside-toplevel

    lr.reg_real_svd_jax()


def reg_stop_gradient_torch():
    """Register torch stop-gradient helper in autoray."""
    if torch is None:  # pragma: no cover - exercised in no-torch CI
        raise ImportError(
            "reg_stop_gradient_torch requires optional dependency 'torch'. "
            "Install it with: pip install pepsy[torch] (or pip install torch)."
        )
    from ..backends import linalg_torch as lr  # pylint: disable=import-outside-toplevel

    lr.reg_stop_gradient_torch()


def stop_grad(x):
    """Return ``x`` detached from autograd when the backend supports it.

    This is the public convenience wrapper for backend-agnostic code that
    otherwise would need to repeat ``ar.do("stop_gradient", x)`` boilerplate.
    """
    try:
        from ..backends import linalg_torch as lr  # pylint: disable=import-outside-toplevel

        return lr.stop_grad(x)
    except Exception:
        try:
            return ar.do("stop_gradient", x)
        except Exception:
            return x
