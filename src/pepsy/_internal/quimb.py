"""Small, execution-time capability probes for supported Quimb APIs.

Pepsy supports a range of Quimb releases, including releases where newer
features exist in a private dispatcher table before they are exported as
top-level functions.  Keep those probes in one place so optional features are
detected when used, without making import-time compatibility more fragile.
"""

from __future__ import annotations

import inspect

import quimb.tensor as qtn


_SDC_METHODS = frozenset({"sdc", "sdc-oversample"})
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


def quimb_1d_compression_function(method):
    """Return Quimb's concrete 1D compressor for *method*, if available."""
    method = str(method).strip().lower()
    function_name = f"tensor_network_1d_compress_{method.replace('-', '_')}"
    function = getattr(qtn, function_name, None)
    if callable(function):
        return function

    dispatcher = getattr(qtn, "tensor_network_1d_compress", None)
    methods = getattr(dispatcher, "__globals__", {}).get(
        "_TN1D_COMPRESS_METHODS", {}
    )
    return methods.get(method) if hasattr(methods, "get") else None


def quimb_1d_compression_method_available(method):
    """Return whether an optional Quimb 1D compressor is installed."""
    method = str(method).strip().lower()
    if method not in _SDC_METHODS:
        return True
    return callable(quimb_1d_compression_function(method))


def require_quimb_1d_compression_method(method):
    """Require an optional Quimb compressor at execution time."""
    if quimb_1d_compression_method_available(method):
        return
    raise NotImplementedError(
        f"Quimb compression method {method!r} is not available in the installed "
        "Quimb build. Install a Quimb build containing the sdc compressor "
        "(available in the v1.16 development line) to use this mode. Existing "
        "compression modes remain available."
    )


def quimb_1d_compression_method_supports_seed(method):
    """Return whether Quimb's concrete compressor explicitly accepts ``seed``."""
    method = str(method).strip().lower()
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
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return False
    # A named option is intentional here.  An old ``**kwargs`` catch-all does
    # not prove that the implementation understands the option.
    return option in parameters


def require_quimb_gate_option(option, *, simple=False):
    """Require a named Quimb gate option before forwarding it."""
    if quimb_gate_option_supported(option, simple=simple):
        return
    entry_point = "gate_simple_" if simple else "tensor_network_gate_inds"
    raise NotImplementedError(
        f"The installed Quimb build does not support gate option {option!r} "
        f"on {entry_point}(). Upgrade Quimb to use this Pepsy option."
    )
