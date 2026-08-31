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
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
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
            f"The installed Quimb build does not provide belief-propagation "
            f"class {class_name}."
        ) from exc


def quimb_bp_constructor_option_supported(method, option):
    """Return whether the selected BP class names *option* in ``__init__``."""
    return quimb_callable_option_supported(
        getattr(quimb_bp_class(method), "__init__", None), option
    )


def quimb_bp_constructor_options(method, options):
    """Forward only constructor-safe options for a Quimb BP class."""
    return quimb_filter_options(
        getattr(quimb_bp_class(method), "__init__", None), options
    )


def quimb_bp_run_options(bp, options):
    """Forward options supported by a concrete BP object's ``run`` method."""
    return quimb_filter_options(getattr(bp, "run", None), options)


def quimb_gloop_options(options):
    """Validate explicit generalized-loop options against Quimb's API."""
    function = getattr(qtn.TensorNetwork, "gen_gloops", None)
    parameters, accepts_kwargs = _signature_parameters(function)
    if function is None:
        raise NotImplementedError(
            "The installed Quimb build does not provide "
            "TensorNetwork.gen_gloops()."
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


def quimb_process_loop_series_expansion_weights(
    weights, *, num_tensors, **options
):
    """Call Quimb's loop-series weight processor across API revisions."""
    from quimb.tensor.belief_propagation.bp_common import (
        process_loop_series_expansion_weights,
    )

    parameters, accepts_kwargs = _signature_parameters(
        process_loop_series_expansion_weights
    )
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
