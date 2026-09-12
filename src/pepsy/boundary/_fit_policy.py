"""Shared policy names and canonicalizers for PEPS boundary fitting.

This module deliberately stays private: the public boundary functions expose
the policy values, while the canonical forms and option sets are shared by the
boundary worker and the higher-level optimizers.
"""

_FIT_MODE_ALIASES = {
    "direct": "direct",
    "src": "src",
    "src-first": "src-first",
    "src-oversample": "src-oversample",
    "src-mps": "srcmps",
    "srcmps": "srcmps",
    "src-mps-first": "srcmps-first",
    "srcmps-first": "srcmps-first",
    "src-mps-oversample": "srcmps-oversample",
    "srcmps-oversample": "srcmps-oversample",
    "zipup": "zipup",
    "zipup-first": "zipup-first",
    "zipup-oversample": "zipup-oversample",
    "sdc": "sdc",
    "sdc-oversample": "sdc-oversample",
    "sdcr": "sdcr",
    "sdcr-oversample": "sdcr-oversample",
    "dm": "dm",
    "eff": "eff",
    "one-site": "eff",
    "dmrg": "eff",
    "dmrg1": "eff",
    "two-site": "two-site",
    "dmrg2": "dmrg2",
    "global": "global",
}

_FIT_INIT_STRATEGY_ALIASES = {
    "auto": "direct",
    "direct": "direct",
    "guess-direct": "guess-direct",
    "guess-src": "guess-src",
    "guess-sdc": "guess-sdc",
}

_FIT_QUIMB_MODES = frozenset(
    {
        "direct",
        "src",
        "src-first",
        "src-oversample",
        "srcmps",
        "srcmps-first",
        "srcmps-oversample",
        "zipup",
        "zipup-first",
        "zipup-oversample",
        "sdc",
        "sdc-oversample",
        "sdcr",
        "sdcr-oversample",
        "dm",
    }
)
_FIT_CUTOFF_MODES = frozenset(
    {"rel", "rsum2", "rsum1", "abs", "sum2", "sum1"}
)
_FIT_LAYER_MODES = frozenset({"joint", "sequential"})
_FIT_LAYER_ORDERS = frozenset({"input", "auto"})

# Options that describe the boundary fitting policy rather than a particular
# metric or optimizer backend. Keep this set shared so public wrappers don't
# drift as new FIT controls are added.
_FIT_POLICY_KEYS = frozenset(
    {
        "fit_mode",
        "fit_layer_mode",
        "fit_layer_order",
        "layer_tags",
        "fit_init_strategy",
        "fit_init_seed",
        "fit_block_size",
        "fit_adaptive_sweeps",
        "fit_max_bond",
        "fit_sweep_sequence",
        "fit_cutoff_mode",
        "fit_min_iter",
        "fit_rtol",
        "fit_patience",
        "fit_timing",
        "fit_timing_sync_device",
    }
)

# ``PepsOptimizer.boundary_kwargs`` is also accepted by standalone metric
# functions. These are the policy keys safe to pass into SweepOptimizer's
# constructor; metric-only controls such as ``method`` and ``mode_`` are
# intentionally excluded.
_SWEEP_BOUNDARY_INIT_KEYS = _FIT_POLICY_KEYS | frozenset(
    {
        "contraction_opt",
        "cutoff",
        "single_layer",
        "n_iter",
        "direction",
        "max_separation",
        "progress",
        "track_boundary_fidelity",
    }
)


def _canonical_fit_mode_selector(fit_mode):
    """Return the canonical boundary-FIT mode or fail clearly."""
    key = str(fit_mode).strip().lower().replace("_", "-")
    if key not in _FIT_MODE_ALIASES:
        raise ValueError(
            f"Unknown fit_mode={fit_mode!r}. Expected a supported Quimb "
            "compression mode ('direct', 'src', 'src-mps', 'zipup', "
            "'sdc', 'sdcr', or 'dm', including oversampling variants), "
            "a FIT mode ('eff', 'two-site', 'dmrg', 'dmrg2'), or 'global'."
        )
    return _FIT_MODE_ALIASES[key]


def _canonical_fit_init_strategy(strategy):
    """Return the supported disposable boundary-guess strategy."""
    key = str(strategy).strip().lower().replace("_", "-")
    try:
        return _FIT_INIT_STRATEGY_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown fit_init_strategy={strategy!r}. Expected 'direct', "
            "'auto', 'guess-direct', 'guess-src', or 'guess-sdc'."
        ) from exc


def _canonical_fit_cutoff_mode(mode):
    """Resolve the boundary cutoff-mode policy before calling Quimb."""
    if mode is None:
        return "rsum2"
    key = str(mode).strip().lower()
    if key == "auto":
        return "rsum2"
    if key not in _FIT_CUTOFF_MODES:
        allowed = ", ".join(sorted(_FIT_CUTOFF_MODES))
        raise ValueError(
            f"Unknown fit_cutoff_mode={mode!r}. Expected 'auto' or one of "
            f"{allowed}."
        )
    return key


def _canonical_fit_layer_mode(mode):
    """Normalize how direct compressors combine tagged tensor layers."""
    key = str(mode).strip().lower().replace("_", "-")
    aliases = {
        "joint": "joint",
        "combined": "joint",
        "all": "joint",
        "sequential": "sequential",
        "layerwise": "sequential",
        "layer-by-layer": "sequential",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(_FIT_LAYER_MODES))
        raise ValueError(
            f"Unknown fit_layer_mode={mode!r}. Expected one of {allowed}."
        ) from exc


def _canonical_fit_layer_order(order):
    """Normalize the explicit sequential-layer ordering policy."""
    key = str(order).strip().lower().replace("_", "-")
    aliases = {
        "input": "input",
        "given": "input",
        "specified": "input",
        "auto": "auto",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(_FIT_LAYER_ORDERS))
        raise ValueError(
            f"Unknown fit_layer_order={order!r}. Expected one of {allowed}."
        ) from exc
