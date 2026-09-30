"""Quimb PEPS measurement dispatch, including degenerate 2D geometry."""

import quimb.tensor as qtn

from .._internal.quimb import call_quimb_2d, quimb_callable_option_supported


def compute_peps_local_expectation(
    tn, terms, *, normalized=True, return_all=False, route="boundary", **options
):
    """Preserve local normalization while avoiding 2D sweeps on a single line.

    A one-row or one-column PEPS is contracted as an exact operator-inserted
    network. This never builds the full state vector or a dense RDM and keeps
    native fermionic ordering and backend gradients. No boundary truncation is
    needed on this route; genuine 2D states keep their selected approximation.
    """
    if normalized not in (True, False, "return"):
        raise ValueError("normalized must be True, False, or 'return'.")
    if route not in (None, "boundary", "envs"):
        raise ValueError("route must be None, 'boundary', or 'envs'.")
    if route is None:
        cyclic = any(getattr(tn, name, lambda: False)() for name in ("is_cyclic_x", "is_cyclic_y"))
        route = "envs" if cyclic else "boundary"
    if route == "envs":
        if "contract_optimize" in options:
            options.setdefault("optimize", options.pop("contract_optimize"))
        for key in ("plaquette_envs", "plaquette_map"):
            if options.pop(key, None) is not None:
                raise ValueError("Precomputed boundary plaquettes require route='boundary'.")
        if options.pop("progbar", False):
            raise ValueError("route='envs' does not support progbar.")
    if options.get("method", False) is None:
        options.pop("method")
        if not quimb_callable_option_supported(tn.compute_local_expectation, "method"):
            options.setdefault("mode", "mps")
    if min(getattr(tn, "Lx", 2), getattr(tn, "Ly", 2)) != 1:
        result = call_quimb_2d(
            tn.compute_local_expectation,
            terms,
            normalized=normalized,
            return_all=return_all,
            route=route,
            **options,
        )
        # Legacy Quimb always returned (numerator, norm) pairs for return_all.
        # Keep Pepsy's result independent of the installed dispatcher revision.
        if (
            return_all
            and normalized != "return"
            and not quimb_callable_option_supported(tn.compute_local_expectation, "route")
        ):
            return {
                where: value / norm if normalized else value
                for where, (value, norm) in result.items()
            }
        return result

    # These options govern 2D environments, which are unnecessary on a line.
    options = dict(options)
    optimize = options.pop("contract_optimize", options.pop("optimize", "auto-hq"))
    for key in (
        "max_bond",
        "cutoff",
        "canonize",
        "mode",
        "method",
        "layer_tags",
        "autogroup",
        "equalize_norms",
        "first_contract",
        "second_dense",
        "progbar",
        "compress_opts",
    ):
        options.pop(key, None)
    for key in ("plaquette_envs", "plaquette_map"):
        if options.pop(key, None) is not None:
            raise ValueError(
                "Precomputed 2D plaquette environments cannot be used for a single-line PEPS."
            )
    if options:
        raise TypeError(f"Unsupported single-line measurement options: {sorted(options)}")

    bra = tn.H
    norm = (bra & tn).contract(all, optimize=optimize) if normalized else None
    values = {}
    for where, operator in terms.items():
        sites = (where,) if tn.has_site(where) else tuple(where)
        ket = qtn.tensor_network_gate_inds(
            tn,
            operator,
            [tn.site_ind(site) for site in sites],
            contract=False,
            inplace=False,
        )
        value = (bra & ket).contract(all, optimize=optimize)
        values[where] = (
            (value, norm) if normalized == "return" else value / norm if normalized else value
        )
    if return_all:
        return values
    if normalized == "return":
        return sum(value / norm for value, norm in values.values())
    return sum(values.values())
