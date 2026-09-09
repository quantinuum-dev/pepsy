"""Pure replay policy and the settings retained by optimizer copies.

Public optimizer attributes remain the configuration source. These helpers
resolve a snapshot when needed; they hold no mutable optimizer or tensor state.
"""

from numbers import Integral

from .ttn import _normalize_compression_mode


# Constructor settings with matching live attribute names. State, queue, RNG,
# aliases and dtype-dependent FIT tolerance have explicit handling at the copy
# boundary. Keep this list separate from transient replay/diagnostic state.
COPY_SETTINGS = (
    "n", "chi", "cutoff", "cutoff_mode", "mode", "compression_mode",
    "compression_seed", "fit_block_size", "fit_n_iter", "fit_adaptive_sweeps",
    "fit_two_site_transition_sweeps", "fit_min_iter", "fit_patience",
    "fit_init_strategy", "fit_init_rand_strength", "fit_init_seed",
    "fit_sweep_sequence", "fit_traversal", "fit_environment_strategy",
    "fit_single_node_fast_path", "fit_overlap_diagnostics", "fit_finite_check",
    "structure", "max_arity", "top_arity", "community_frac", "star_frac",
    "dtype", "threads", "subtree_workers", "layout_objective",
    "layout_weight_mode", "layout_time_decay", "layout_time_window",
    "track_truncation", "track_infidelity", "max_intermediate_bond",
    "max_operator_qubits", "max_subtree_nodes", "record_history", "profile",
    "profile_sync", "track_bond_diagnostics",
)

_ALGORITHM_MODES = frozenset({
    "dm", "sdc", "src", "zipup", "tree_mpo_direct", "tree_mpo_dm",
})


def normalize_mode(mode):
    """Normalize public and compatibility gate/sub-MPO selectors."""
    mode = str(mode).strip().lower().replace("-", "_")
    aliases = {
        "dem": "dm",
        "tree_mpo": "tree_mpo_direct",
        "treempo": "tree_mpo_direct",
        "treempo_direct": "tree_mpo_direct",
        "treempo_dm": "tree_mpo_dm",
        "treempo_dem": "tree_mpo_dm",
        "tree_mpo_dem": "tree_mpo_dm",
        "tree_mpo_svd": "tree_mpo_direct",
        "tree_mpo_eig": "tree_mpo_dm",
        "fit": "dmrg",
    }
    mode = aliases.get(mode, mode)
    if mode not in {
        "auto", "direct", "dm", "sdc", "src", "zipup", "mpo", "submpo",
        "tree_mpo_direct", "tree_mpo_dm", "dmrg", "dmrg1", "dmrg2", "dmrg3",
    }:
        raise ValueError(
            "mode must be one of 'auto', 'direct', 'dm', 'sdc', 'src', 'zipup', 'mpo', "
            "'submpo', 'dmrg', 'dmrg1', 'dmrg2', 'dmrg3', "
            "'tree_mpo_direct', or 'tree_mpo_dm'."
        )
    return mode


def compression_for_mode(mode, compression_mode):
    """Combined TreeMPO names and zipup own their compression method."""
    if mode not in {"zipup", "tree_mpo_direct", "tree_mpo_dm"}:
        return compression_mode
    expected = "dm" if mode == "tree_mpo_dm" else "direct"
    if compression_mode not in {expected, "direct"}:
        raise ValueError(
            f"mode={mode!r} requires compression_mode={expected!r}."
        )
    return expected


def resolve_replay_mode(mode, compression_mode):
    """Return route, compression method and named FIT schedule without mutation."""
    mode = normalize_mode(mode)
    compression_mode = _normalize_compression_mode(compression_mode)
    if mode in {"dm", "sdc", "src"}:
        if compression_mode not in {"direct", mode}:
            raise ValueError(
                f"mode={mode!r} cannot be combined with a different compression_mode."
            )
        compression_mode, mode = mode, "auto"
    compression_mode = compression_for_mode(mode, compression_mode)
    alias = mode if mode in {"dmrg1", "dmrg2", "dmrg3"} else None
    return "dmrg" if alias is not None else mode, compression_mode, alias


def resolve_replay_override(mode, compression_mode, *, current_mode,
                            current_compression, current_alias):
    """Resolve run overrides with the same neutral defaults as construction.

    Algorithm selectors own their method. Route-only selectors such as direct
    retain an independently configured compressor when it is not overridden.
    An omitted mode retains the named FIT schedule as well as the route.
    """
    requested = (current_alias or current_mode) if mode is None else normalize_mode(mode)
    if compression_mode is None:
        compression_mode = (
            "direct" if mode is not None and requested in _ALGORITHM_MODES
            else current_compression
        )
    return resolve_replay_mode(requested, compression_mode)


def replay_mode_name(mode, compression_mode, dmrg_alias):
    """Describe the live route/method without changing or validating settings."""
    mode = str(mode).strip().lower().replace("-", "_")
    if mode == "dmrg":
        return dmrg_alias or "dmrg"
    if mode in {"dmrg1", "dmrg2", "dmrg3"}:
        return mode
    if mode == "tree_mpo_dm":
        return "dm"
    if mode in {"auto", "direct", "mpo", "submpo", "tree_mpo_direct"}:
        compression_mode = str(compression_mode).strip().lower()
        if compression_mode in {"direct", "dm", "sdc", "src"}:
            return compression_mode
        return "direct"
    return mode


def _normalize_where(where):
    """Return integer labels without silently truncating caller supports."""
    if isinstance(where, Integral):
        where = (where,)
    else:
        try:
            where = tuple(where)
        except TypeError as exc:
            raise ValueError("gate support must contain integer qubit labels.") from exc
    if any(isinstance(site, bool) or not isinstance(site, Integral) for site in where):
        raise ValueError("gate support must contain integer qubit labels.")
    return tuple(int(site) for site in where)
