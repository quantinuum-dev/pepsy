"""Small dependency-free helpers shared by the Torch VMC leaf modules."""

from __future__ import annotations

import os
import sysconfig
from numbers import Integral

from ..torch_types import FermionSiteEncoding, _check_positive_int, _require_torch

__all__ = [
    "_as_contraction_options",
    "_as_long_matrix",
    "_check_nonnegative_int",
    "_check_positive_int",
    "_count_spinful_particles",
    "_edge_value",
    "_iter_edges",
    "_make_torch_generator",
    "_model_device",
    "_proposal_log_probabilities",
    "_require_torch",
    "_run_cheap_torch_kernel",
    "_site_value",
    "_torch_finfo_tiny",
    "_validate_contraction",
]


def _as_long_matrix(configs, *, name="configs"):
    """Normalize a configuration batch to a two-dimensional long tensor."""
    torch = _require_torch()
    configs = torch.as_tensor(configs, dtype=torch.long)
    if configs.ndim == 1:
        configs = configs.reshape(1, -1)
    if configs.ndim != 2:
        raise ValueError(f"{name} must have shape (n_batch, n_sites).")
    return configs


def _check_nonnegative_int(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


_CONTRACTION_ALIASES = {
    "exact": "exact",
    "hotrg": "hotrg",
    "ctmrg": "ctmrg",
    "boundary": "boundary",
    "contract_boundary": "boundary",
    "mps": "boundary",
    "boundary_mps": "boundary",
    "contract-boundary": "boundary",
    "boundary-mps": "boundary",
}


def _validate_contraction(contraction, chi):
    key = str(contraction).replace("_", "-").lower()
    try:
        contraction = _CONTRACTION_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            "contraction must be 'exact', 'hotrg', 'ctmrg', or 'boundary'."
        ) from exc
    if contraction in {"hotrg", "ctmrg", "boundary"} and chi is None:
        raise ValueError(f"contraction={contraction!r} requires chi.")
    return contraction


def _as_contraction_options(contraction_opts):
    return {} if contraction_opts is None else dict(contraction_opts)


def _torch_finfo_tiny(dtype):
    torch = _require_torch()
    if dtype.is_complex:
        dtype = torch.empty((), dtype=dtype).real.dtype
    return torch.finfo(dtype).tiny


def _edge_value(value, edge):
    if isinstance(value, dict):
        i, j = edge
        if edge in value:
            return value[edge]
        if (j, i) in value:
            return value[(j, i)]
        return 0.0
    return value


def _site_value(value, site):
    if isinstance(value, dict):
        return value.get(site, 0.0)
    return value


def _iter_edges(graph):
    if hasattr(graph, "edges"):
        edges = graph.edges
        if callable(edges):
            edges = edges()
    else:
        edges = graph
    return tuple((int(i), int(j)) for i, j in edges)


def _make_torch_generator(seed, *, device=None):
    """Construct a reproducible torch generator for a target device."""
    torch = _require_torch()
    if seed is None:
        return None
    try:
        generator = torch.Generator(device=device)
    except (RuntimeError, TypeError, ValueError):
        generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def _count_spinful_particles(configs, *, encoding=None):
    """Return per-sample ``(n_up, n_down)`` counts."""
    encoding = FermionSiteEncoding.vmc_torch() if encoding is None else encoding
    configs = _as_long_matrix(configs)
    n_up, n_down = encoding.decode(configs)
    return n_up.sum(dim=-1), n_down.sum(dim=-1)


def _proposal_log_probabilities(omegas, *, device, allow_zero=False):
    """Decode ``PepsBpSampler`` mantissa/exponent proposal probabilities."""
    torch = _require_torch()
    if not isinstance(omegas, (tuple, list)) or len(omegas) != 2:
        raise ValueError(
            "proposal samples must expose omegas as (mantissas, exponents)."
        )
    mantissas = torch.as_tensor(omegas[0], dtype=torch.float64, device=device)
    exponents = torch.as_tensor(omegas[1], dtype=torch.float64, device=device)
    if mantissas.ndim != 1 or exponents.shape != mantissas.shape:
        raise ValueError("proposal mantissas and exponents must be one-dimensional.")
    if torch.any(mantissas < 0):
        raise ValueError("proposal probabilities cannot have negative mantissas.")
    if not allow_zero and torch.any(mantissas <= 0):
        raise ValueError("proposal probabilities must have positive mantissas.")
    positive = mantissas > 0
    log_prob = torch.where(
        positive,
        mantissas.log() + exponents * torch.log(
            torch.as_tensor(10.0, dtype=torch.float64, device=device)
        ),
        torch.full_like(mantissas, -torch.inf),
    )
    return log_prob


def _model_device(model, device=None):
    torch = _require_torch()
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        return torch.device("cpu")


_COMPILED_CHEAP_TORCH_KERNELS = {}
_FAILED_CHEAP_TORCH_KERNELS = set()


def _run_cheap_torch_kernel(name, fn, *args, compile_kernels=False):
    """Optionally compile pure tensor bookkeeping with an eager fallback."""
    if not compile_kernels or name in _FAILED_CHEAP_TORCH_KERNELS:
        return fn(*args)
    torch = _require_torch()
    compiled = _COMPILED_CHEAP_TORCH_KERNELS.get(name)
    if compiled is None:
        compile_fn = getattr(torch, "compile", None)
        include_dir = sysconfig.get_config_var("INCLUDEPY")
        has_python_headers = (
            include_dir is not None
            and os.path.isfile(os.path.join(include_dir, "Python.h"))
        )
        if not callable(compile_fn) or not has_python_headers:
            _FAILED_CHEAP_TORCH_KERNELS.add(name)
            return fn(*args)
        try:
            compiled = compile_fn(fn, dynamic=True)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            _FAILED_CHEAP_TORCH_KERNELS.add(name)
            return fn(*args)
        _COMPILED_CHEAP_TORCH_KERNELS[name] = compiled
    try:
        return compiled(*args)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        # The feature is opt-in. Sparse/device-specific compiler gaps should
        # never alter a VMC trajectory or make PEPS evaluation unavailable.
        _FAILED_CHEAP_TORCH_KERNELS.add(name)
        _COMPILED_CHEAP_TORCH_KERNELS.pop(name, None)
        return fn(*args)
