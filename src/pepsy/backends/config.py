"""Backend configuration and optional autodiff registrations."""

from __future__ import annotations

import autoray as ar
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

__all__ = [
    "build_backend", "backend_torch", "backend_numpy", "backend_cupy", "backend_jax",
    "register_torch_linalg", "reg_rel_svd_torch", "reg_real_svd_torch",
    "reg_complex_svd_torch", "reg_real_qr_torch", "reg_complex_qr_torch",
    "reg_rel_svd_jax", "reg_real_svd_jax", "reg_complex_svd_jax",
    "reg_stop_gradient_torch", "stop_grad", "set_default_array_backend",
    "get_default_array_backend", "set_default_grad_backend",
    "get_default_grad_backend", "reset_default_backends",
]

_DEFAULT_ARRAY_BACKEND = None
_DEFAULT_GRAD_BACKEND = None

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
        # Coerce non-JAX inputs (incl. torch tensors) to a numpy-compatible
        # form first so jnp.asarray accepts them on any backend.
        if torch is not None and isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        arr = jnp.asarray(x, dtype=dtype)
        if device is not None:
            arr = jax.device_put(arr, device)
        return arr

    return cast_array


def register_torch_linalg(mode="complex"):
    """Register custom torch linalg gradients in autoray.

    Parameters
    ----------
    mode : {"complex", "real"}, default="complex"
        Which SVD/QR registrations to install.
    """
    if torch is None:  # pragma: no cover - exercised in no-torch CI
        raise ImportError(
            "register_torch_linalg requires optional dependency 'torch'. "
            "Install it with: pip install pepsy[torch] (or pip install torch)."
        )
    from ..backends import linalg_torch as lr  # pylint: disable=import-outside-toplevel

    if mode == "complex":
        lr.reg_rel_svd_torch()
        lr.reg_complex_qr_torch()
        return
    if mode == "real":
        lr.reg_real_svd_torch()
        lr.reg_real_qr_torch()
        return
    raise ValueError("mode must be 'complex' or 'real'")


def reg_rel_svd_torch():
    """Register torch SVD with a stable relative-regularized backward rule.

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


def reg_complex_svd_torch():
    """Register complex torch SVD autograd rule in autoray.

    Compatibility wrapper for :func:`reg_rel_svd_torch`.
    """
    if torch is None:  # pragma: no cover - exercised in no-torch CI
        raise ImportError(
            "reg_complex_svd_torch requires optional dependency 'torch'. "
            "Install it with: pip install pepsy[torch] (or pip install torch)."
        )
    from ..backends import linalg_torch as lr  # pylint: disable=import-outside-toplevel

    lr.reg_rel_svd_torch()


def reg_real_svd_torch():
    """Register the real-only torch SVD autograd rule in autoray.

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


def reg_complex_qr_torch():
    """Register complex torch QR autograd rule in autoray."""
    if torch is None:  # pragma: no cover - exercised in no-torch CI
        raise ImportError(
            "reg_complex_qr_torch requires optional dependency 'torch'. "
            "Install it with: pip install pepsy[torch] (or pip install torch)."
        )
    from ..backends import linalg_torch as lr  # pylint: disable=import-outside-toplevel

    lr.reg_complex_qr_torch()


def reg_real_qr_torch():
    """Register real torch QR autograd rule in autoray."""
    if torch is None:  # pragma: no cover - exercised in no-torch CI
        raise ImportError(
            "reg_real_qr_torch requires optional dependency 'torch'. "
            "Install it with: pip install pepsy[torch] (or pip install torch)."
        )
    from ..backends import linalg_torch as lr  # pylint: disable=import-outside-toplevel

    lr.reg_real_qr_torch()


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
