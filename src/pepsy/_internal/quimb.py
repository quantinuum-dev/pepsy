"""Small, execution-time capability probes for supported Quimb APIs.

Pepsy supports a range of Quimb releases, including releases where newer
features exist in a private dispatcher table before they are exported as
top-level functions.  Keep those probes in one place so optional features are
detected when used, without making import-time compatibility more fragile.
"""

from __future__ import annotations

import inspect
import threading
from functools import lru_cache
import math
import warnings
from collections.abc import Mapping
from copy import deepcopy
from numbers import Integral, Real

import quimb.tensor as qtn


_QUIMB_SEED_LOCK = threading.RLock()


@lru_cache(maxsize=2)
def _numpy_eig_split_supported(dtype):
    """Probe the public split API for the old single-precision Numba bug."""
    import numpy as np
    from numba.core.errors import TypingError

    tensor = qtn.Tensor(np.eye(2, 3, dtype=dtype), inds=("a", "b"))
    try:
        tensor.split(
            left_inds=("a",), method="svd:eig", cutoff=1e-6,
            max_bond=2, absorb=None,
        )
    except TypingError:
        return False
    return True


def quimb_safe_split_method(method, array):
    """Keep dtype and truncation policy when an upstream eig driver cannot compile."""
    import autoray as ar

    if method == "svd:eig" and ar.infer_backend(array) == "numpy":
        dtype = str(array.dtype)
        if dtype in {"float32", "complex64"} and not _numpy_eig_split_supported(dtype):
            warnings.warn(
                f"Quimb's NumPy {dtype} svd:eig driver cannot compile; using "
                "direct SVD with the same dtype, cutoff, and bond limit.",
                RuntimeWarning, stacklevel=2,
            )
            return "svd"
    return method


def quimb_src_backend_supported(method, like):
    """Legacy SRC creates NumPy noise; modern seeded SRC follows its input."""
    import autoray as ar

    method = str(method).strip().lower().replace("_", "-")
    if not method.startswith("src"):
        return True
    arrays = getattr(like, "arrays", None)
    if arrays is not None:
        like = arrays[0]
    return ar.infer_backend(like) == "numpy" or quimb_1d_compression_method_supports_seed(method)


def quimb_fit_guess_method(method, like):
    """Choose a supported disposable warm start without changing the FIT target."""
    if quimb_src_backend_supported(method, like):
        return method
    warnings.warn(
        "This Quimb build cannot create backend-native SRC noise; using a "
        "direct FIT warm start while preserving the exact target.",
        RuntimeWarning, stacklevel=2,
    )
    return "direct"


def run_seeded_quimb(random_seed, function, *args, quimb_method=None, **kwargs):
    """Seed a randomized compressor without leaking options into contractions.

    Modern compressors accept a backend-native seed directly. Older releases
    use Quimb's process-global generator, serialized across Pepsy callers.
    """
    method = kwargs.get("method") if quimb_method is None else quimb_method
    if args and not quimb_src_backend_supported(method, args[0]):
        raise NotImplementedError(
            f"Quimb compressor {method!r} needs a newer build for non-NumPy "
            "random arrays; use direct compression or upgrade Quimb."
        )
    if random_seed is None:
        return function(*args, **kwargs)
    if quimb_1d_compression_method_supports_seed(method):
        kwargs.setdefault("seed", int(random_seed))
        return function(*args, **kwargs)
    import quimb

    with _QUIMB_SEED_LOCK:
        quimb.seed_rand(int(random_seed))
        return function(*args, **kwargs)


_OPTIONAL_1D_METHODS = frozenset(
    {
        "sdc",
        "sdc-oversample",
        "sdcr",
        "sdcr-oversample",
        "src-first",
        "src-oversample",
        "srcmps",
        "srcmps-first",
        "srcmps-oversample",
        "zipup-first",
        "zipup-oversample",
    }
)
_SEEDED_METHODS = frozenset(
    {
        "src",
        "src-first",
        "src-oversample",
        "srcmps",
        "srcmps-first",
        "srcmps-oversample",
        "fit",
        "fit-oversample",
    }
)


def _signature_parameters(function):
    """Return ``(parameters, accepts_kwargs)`` for a callable if possible."""
    if function is None:
        return {}, False
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return {}, False
    parameters = signature.parameters
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    return parameters, accepts_kwargs


def quimb_callable_option_supported(function, option):
    """Return whether *function* explicitly supports ``option``."""
    parameters, _ = _signature_parameters(function)
    return option in parameters


def quimb_filter_options(function, options):
    """Filter options only when a callable has no ``**kwargs`` catch-all."""
    parameters, accepts_kwargs = _signature_parameters(function)
    if accepts_kwargs:
        return dict(options)
    return {key: value for key, value in options.items() if key in parameters}


def quimb_2d_options(function, options):
    """Translate the 2D compression keyword without changing route or policy."""
    options = dict(options)
    parameters, _ = _signature_parameters(function)
    policy_function = function
    if "method" not in parameters and "mode" not in parameters:
        owner = getattr(function, "__self__", None)
        policy_function = getattr(owner, "contract_boundary", function)
    modern = quimb_callable_option_supported(policy_function, "method")
    if modern and "mode" in options:
        mode = options.pop("mode")
        if "method" in options:
            if mode == "full-bond":
                options.setdefault("similarity_method", options.pop("method"))
            elif options["method"] != mode and mode is not None:
                raise ValueError("Conflicting boundary compression mode and method.")
        options.setdefault("method", mode)
    elif not modern and "method" in options and options.get("mode") != "full-bond":
        method = options.pop("method")
        if "mode" in options and options["mode"] != method:
            raise ValueError("Conflicting boundary compression mode and method.")
        options["mode"] = method
    if options.get("route") is not None and not quimb_callable_option_supported(function, "route"):
        if options["route"] != "boundary":
            raise NotImplementedError(
                "This Quimb build does not support the requested measurement route."
            )
        options.pop("route")
    elif options.get("route", False) is None:
        options.pop("route")
    return options


def call_quimb_2d(function, *args, **options):
    """Call a 2D boundary API with its installed keyword convention."""
    return function(*args, **quimb_2d_options(function, options))


def quimb_compression_options(method, options):
    """Validate explicitly requested intermediate/final 1D compression options.

    A catch-all ``**kwargs`` is not evidence of support: older compressors can
    leak unknown options into contraction or linear-algebra calls.
    """
    if options is None:
        return {}
    if not isinstance(options, Mapping):
        raise TypeError("compression_opts must be a mapping or None.")
    allowed = {
        "max_bond_oversample",
        "cutoff_oversample",
        "cutoff_mode_oversample",
        "compress_opts_final",
    }
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f"Unknown compression_opts: {sorted(unknown)}")
    result = deepcopy(dict(options))
    function = quimb_1d_compression_function(method)
    for key, value in result.items():
        if not quimb_callable_option_supported(function, key):
            raise NotImplementedError(
                f"Quimb compressor {method!r} does not explicitly support {key!r}."
            )
        if key == "max_bond_oversample":
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError("max_bond_oversample must be a positive integer.")
        elif key == "cutoff_oversample":
            if value == "auto":
                parameters, _ = _signature_parameters(function)
                if parameters[key].default != "auto":
                    raise ValueError(f"{method!r} requires a numeric cutoff_oversample.")
            if value != "auto" and (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError("cutoff_oversample must be finite and non-negative or 'auto'.")
        elif key == "cutoff_mode_oversample":
            if value not in {1, 2, 3, 4, 5, 6, "abs", "rel", "sum1", "sum2", "rsum1", "rsum2"}:
                raise ValueError("Invalid cutoff_mode_oversample.")
            if method == "sdcr-oversample" and value not in {1, 2, "abs", "rel"}:
                raise ValueError("SDCR intermediate compression requires an abs or rel cutoff.")
        elif not isinstance(value, Mapping):
            raise TypeError("compress_opts_final must be a mapping.")
        else:
            unknown_final = set(value) - {"method", "cutoff", "cutoff_mode"}
            if unknown_final:
                raise ValueError(
                    "compress_opts_final accepts method, cutoff, and cutoff_mode; "
                    "the final bond cap is controlled by chi/fit_max_bond."
                )
            final_method = value.get("method", "svd")
            if final_method not in {"svd", "svd:eig", "rsvd"}:
                raise ValueError("Final compression method must be svd, svd:eig, or rsvd.")
            if final_method == "rsvd" and value.get("cutoff_mode") not in {1, 2, "abs", "rel"}:
                raise ValueError("Final rsvd requires an explicit abs or rel cutoff_mode.")
            final_cutoff = value.get("cutoff", 0.0)
            if (
                isinstance(final_cutoff, bool)
                or not isinstance(final_cutoff, Real)
                or not math.isfinite(final_cutoff)
                or final_cutoff < 0
            ):
                raise ValueError("Final cutoff must be finite and non-negative.")
            if value.get("cutoff_mode", "rsum2") not in {
                1,
                2,
                3,
                4,
                5,
                6,
                "abs",
                "rel",
                "sum1",
                "sum2",
                "rsum1",
                "rsum2",
            }:
                raise ValueError("Invalid final cutoff_mode.")
    return result


def quimb_bp_class(method):
    """Return a Quimb BP class by its stable short method name."""
    from quimb.tensor import belief_propagation

    names = {
        "l1bp": "L1BP",
        "hv1bp": "HV1BP",
        "d1bp": "D1BP",
        "d2bp": "D2BP",
    }
    try:
        class_name = names[str(method).strip().lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown Quimb BP method: {method!r}") from exc
    try:
        return getattr(belief_propagation, class_name)
    except AttributeError as exc:
        raise NotImplementedError(
            f"The installed Quimb build does not provide belief-propagation class {class_name}."
        ) from exc


def quimb_bp_constructor_option_supported(method, option):
    """Return whether the selected BP class names *option* in ``__init__``."""
    return quimb_callable_option_supported(
        getattr(quimb_bp_class(method), "__init__", None), option
    )


def quimb_bp_constructor_options(method, options):
    """Forward only constructor-safe options for a Quimb BP class."""
    return quimb_filter_options(getattr(quimb_bp_class(method), "__init__", None), options)


def quimb_bp_run_options(bp, options):
    """Forward options supported by a concrete BP object's ``run`` method."""
    return quimb_filter_options(getattr(bp, "run", None), options)


def quimb_gloop_options(options):
    """Validate explicit generalized-loop options against Quimb's API."""
    function = getattr(qtn.TensorNetwork, "gen_gloops", None)
    parameters, accepts_kwargs = _signature_parameters(function)
    if function is None:
        raise NotImplementedError(
            "The installed Quimb build does not provide TensorNetwork.gen_gloops()."
        )

    if not accepts_kwargs:
        unsupported = sorted(set(options) - set(parameters))
        if unsupported:
            names = ", ".join(repr(name) for name in unsupported)
            raise NotImplementedError(
                "The installed Quimb build does not support generalized-loop "
                f"option(s): {names}. Upgrade Quimb to use these options."
            )
    return dict(options)


def quimb_process_loop_series_expansion_weights(weights, *, num_tensors, **options):
    """Call Quimb's loop-series weight processor across API revisions."""
    from quimb.tensor.belief_propagation.bp_common import (
        process_loop_series_expansion_weights,
    )

    parameters, accepts_kwargs = _signature_parameters(process_loop_series_expansion_weights)
    if accepts_kwargs or "num_tensors" in parameters:
        options = {"num_tensors": num_tensors, **options}
    return process_loop_series_expansion_weights(weights, **options)


def quimb_lattice_bond_map(*shape):
    """Return Quimb's lattice bond map when the installed build provides it."""
    lattice_bond_map = getattr(qtn, "LatticeBondMap", None)
    if lattice_bond_map is None:
        return None
    try:
        return lattice_bond_map(*shape)
    except (TypeError, ValueError):
        # A partially backported class should not make the existing open-cycle
        # fallback unusable.
        return None


def quimb_mpo_auto_swap_function(mpo):
    """Return the opt-in MPO auto-swap method, if available."""
    function = getattr(mpo, "gate_sandwich_with_auto_swap", None)
    return function if callable(function) else None


def quimb_1d_compression_function(method):
    """Return Quimb's concrete 1D compressor for *method*, if available."""
    method = str(method).strip().lower().replace("_", "-")
    function_name = f"tensor_network_1d_compress_{method.replace('-', '_')}"
    function = getattr(qtn, function_name, None)
    if callable(function):
        return function

    dispatcher = getattr(qtn, "tensor_network_1d_compress", None)
    methods = getattr(dispatcher, "__globals__", {}).get("_TN1D_COMPRESS_METHODS", {})
    return methods.get(method) if hasattr(methods, "get") else None


def quimb_1d_compression_method_available(method):
    """Return whether an optional Quimb 1D compressor is installed."""
    method = str(method).strip().lower().replace("_", "-")
    if method not in _OPTIONAL_1D_METHODS:
        return True
    return callable(quimb_1d_compression_function(method))


def quimb_1d_callable_compression_available():
    """Return whether Quimb's public 1D dispatcher accepts a callable."""
    dispatcher = getattr(qtn, "tensor_network_1d_compress", None)
    if not callable(dispatcher):
        return False
    dispatcher = inspect.unwrap(dispatcher)
    code = getattr(dispatcher, "__code__", None)
    return code is not None and "callable" in code.co_names


def require_quimb_1d_callable_compression():
    """Require callable 1D compression dispatch at execution time."""
    if quimb_1d_callable_compression_available():
        return
    raise NotImplementedError(
        "The installed Quimb build does not support custom callable 1D "
        "compression methods. Upgrade Quimb to use "
        "ctmrg_projector_region=(2, 3)."
    )


def quimb_1d_compression_cutoff_mode(method, cutoff_mode):
    """Return a cutoff mode safe for the selected Quimb compressor.

    ``sdcr`` uses randomized SVDs to form its successive environments. Newer
    Quimb releases reject cumulative cutoff modes for that randomized stage,
    while older releases accepted them. Keep the compatibility decision at
    Pepsy's boundary and preserve ``None`` so each installed Quimb release can
    select its own native default.
    """
    method = str(method).strip().lower().replace("_", "-")
    if method == "sdcr" and str(cutoff_mode).strip().lower() in {
        "sum1",
        "sum2",
        "rsum1",
        "rsum2",
    }:
        warnings.warn(
            f"Quimb sdcr uses cutoff_mode='rel' instead of {cutoff_mode!r}; "
            "use sdcr-oversample for an independent cumulative final cutoff.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "rel"
    return cutoff_mode


def require_quimb_1d_compression_method(method):
    """Require an optional Quimb compressor at execution time."""
    if quimb_1d_compression_method_available(method):
        return
    compressor_family = method.split("-", 1)[0]
    raise NotImplementedError(
        f"Quimb compression method {method!r} is not available in the installed "
        f"Quimb build. Install a Quimb build containing the {compressor_family} "
        "compressor. Existing compression modes remain available."
    )


def _quimb_ag_compression_function(method):
    """Return a concrete arbitrary-geometry compressor when discoverable."""
    method = str(method).strip().lower()
    function_name = f"tensor_network_ag_compress_{method.replace('-', '_')}"
    function = getattr(qtn, function_name, None)
    if callable(function):
        return function

    dispatcher = getattr(qtn, "tensor_network_1d_compress", None)
    ag_dispatcher = getattr(dispatcher, "__globals__", {}).get("tensor_network_ag_compress")
    methods = getattr(ag_dispatcher, "__globals__", {}).get("_TNAG_COMPRESS_METHODS", {})
    return methods.get(method) if hasattr(methods, "get") else None


def quimb_ctmrg_mode_available(mode):
    """Return whether the installed Quimb build provides a CTMRG mode."""
    mode = str(mode).strip().lower()
    if mode == "projector2d":
        owner = getattr(qtn, "TensorNetwork2D", None)
        return callable(getattr(owner, "_contract_boundary_projector", None))
    return callable(_quimb_ag_compression_function(mode))


def require_quimb_ctmrg_mode(mode):
    """Require a concrete Quimb CTMRG boundary mode at execution time."""
    if quimb_ctmrg_mode_available(mode):
        return
    raise NotImplementedError(
        f"Quimb CTMRG mode {mode!r} is not available in the installed Quimb "
        "build. Upgrade Quimb to use this Pepsy option."
    )


def quimb_ctmrg_projector_canonize_available(canonize):
    """Return whether Quimb implements a named projector-gauging policy."""
    canonize = str(canonize).strip().lower()
    function = _quimb_ag_compression_function("projector")
    if callable(function):
        function = inspect.unwrap(function)
    code = getattr(function, "__code__", None)
    return code is not None and canonize in code.co_consts


def require_quimb_ctmrg_projector_canonize(canonize):
    """Require a named Quimb projector-gauging policy at execution time."""
    if quimb_ctmrg_projector_canonize_available(canonize):
        return
    raise NotImplementedError(
        f"Quimb CTMRG projector canonicalization {canonize!r} is not "
        "available in the installed Quimb build. Upgrade Quimb to use this "
        "Pepsy option."
    )


def quimb_1d_compression_method_supports_seed(method):
    """Return whether Quimb's concrete compressor explicitly accepts ``seed``."""
    method = str(method).strip().lower().replace("_", "-")
    if method not in _SEEDED_METHODS:
        return False
    function = quimb_1d_compression_function(method)
    if function is None:
        return False
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return False
    return "seed" in parameters


def quimb_gate_option_supported(option, *, simple=False):
    """Return whether a Quimb gate entry point names *option* explicitly."""
    if simple:
        owner = getattr(qtn, "TensorNetworkGenVector", None)
        function = getattr(owner, "gate_simple_", None)
    else:
        function = getattr(qtn, "tensor_network_gate_inds", None)
    if function is None:
        return False
    # A named option is intentional here.  An old ``**kwargs`` catch-all does
    # not prove that the implementation understands the option.
    return quimb_callable_option_supported(function, option)


def require_quimb_gate_option(option, *, simple=False):
    """Require a named Quimb gate option before forwarding it."""
    if quimb_gate_option_supported(option, simple=simple):
        return
    entry_point = "gate_simple_" if simple else "tensor_network_gate_inds"
    raise NotImplementedError(
        f"The installed Quimb build does not support gate option {option!r} "
        f"on {entry_point}(). Upgrade Quimb to use this Pepsy option."
    )
