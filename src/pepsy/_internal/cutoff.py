"""Shared truncation and FIT stopping tolerance policies."""

from __future__ import annotations

import numpy as np


def dtype_auto_cutoff(dtype):
    """Return Pepsy's default truncation cutoff for ``dtype``.

    The policy is shared by the MPS, tree, PEPS, and Hamiltonian conversion
    APIs so ``cutoff="auto"`` has the same meaning everywhere:

    * ``float64``/``complex128``: ``1e-12``;
    * ``float32``/``complex64``: ``1e-6``;
    * 16-bit floating-point data: ``1e-3``.

    Backend dtype objects such as ``torch.float32`` are accepted in addition
    to NumPy dtypes. Their string form is sufficient for this precision
    classification when NumPy cannot construct a dtype directly.
    """

    dtype_name = str(dtype).strip().lower()
    # Torch compilation cannot trace NumPy's exception on torch.dtype.
    if not dtype_name.startswith("torch."):
        try:
            dtype_name = np.dtype(dtype).name.lower()
        except (TypeError, ValueError):
            pass

    if "16" in dtype_name:
        return 1.0e-3
    if "32" in dtype_name or "complex64" in dtype_name:
        return 1.0e-6
    return 1.0e-12


def resolve_fit_rtol(value, *, dtype=None):
    """Resolve an optimizer's FIT stopping tolerance without tensor work.

    ``"auto"`` uses the supplied backend dtype's string representation:
    16-bit data uses ``1e-3``, float32/complex64 uses ``1e-5``, and higher
    precision uses ``1e-9``. ``None`` disables tolerance stopping. Explicit
    values retain the optimizer API's float conversion and error messages.

    FIT convergence and singular-value truncation measure different errors,
    so these thresholds are separate from :func:`dtype_auto_cutoff`.
    """
    if value == "auto":
        dtype = str(dtype).lower()
        if "16" in dtype:
            return 1e-3
        if "32" in dtype or "complex64" in dtype:
            return 1e-5
        return 1e-9
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "fit_rtol must be 'auto', a non-negative number, or None."
        ) from exc
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(
            "fit_rtol must be 'auto', a non-negative number, or None."
        )
    return value
