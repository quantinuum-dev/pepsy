"""Stateless Quimb compression adapters and disposable MPS guesses.

These helpers take explicit states and settings. Replay scheduling, FIT target
ownership, rollback, and canonical metadata remain on ``MpsOptimizer``.
"""

from __future__ import annotations

import quimb.tensor as qtn

from ..._internal.quimb import (
    quimb_1d_compression_cutoff_mode as _quimb_compression_cutoff_mode,
    require_quimb_1d_compression_method as _require_quimb_compression_method,
    run_seeded_quimb as _run_seeded_quimb,
)

__all__ = ["guess", "svd_guess"]

# Cutoff, seed, and interior-workaround groups describe separate policies;
# do not merge them or forward one family's options to another.
_MPO_COMPRESSION_METHODS = frozenset(
    {
        "direct",
        "dm",
        "zipup",
        "zipup-first",
        "zipup-oversample",
        "src",
        "src-first",
        "src-oversample",
        "srcmps",
        "srcmps-first",
        "srcmps-oversample",
        "sdc",
        "sdc-oversample",
        "sdcr",
        "sdcr-oversample",
        "fit",
        "fit-zipup",
        "fit-projector",
        "fit-oversample",
    }
)

_MPO_METHODS_IGNORE_CUTOFF_MODE = frozenset({"src", "srcmps"})

_MPO_METHODS_IGNORE_CUTOFF = frozenset({"src", "srcmps", "sdcr"})

_MPO_METHODS_USE_SEED = frozenset(
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

_MPO_METHODS_NEED_INTERIOR_WORKAROUND = frozenset(
    {
        "zipup-first",
        "zipup-oversample",
        "fit-zipup",
        "fit-projector",
    }
)


def _is_interior_submpo_span(p, where):
    """Return whether ``where`` omits one or more end sites of ``p``."""
    return min(where) > 0 or max(where) < int(p.L) - 1


def _apply_submpo_with_interior_workaround_impl(
    p,
    submpo,
    where,
    *,
    chi,
    method,
    cutoff,
    cutoff_mode,
    info=None,
    inplace_mpo=False,
    optimize=None,
    seed=None,
    compression_opts=None,
):
    """Apply selected Quimb methods without nested sub-MPO tag permutation.

    Quimb's oversampled zip-up and ``fit-{zipup,projector}`` wrappers call a
    second compression dispatcher internally. When the input is a partitioned
    interior sub-MPO, that nested call defaults to permuting full-chain MPS
    labels and can look for a missing ``I0`` tag. Keep the local sub-MPO
    partition, but reproduce the documented wrapper stages with
    ``permute_arrays=False`` at every level.
    """
    compression_opts = dict(compression_opts or {})
    si, sf = min(where), max(where)
    p.canonicalize_((si, sf), info=info)
    p.gate_with_op_lazy_(
        submpo,
        transpose=False,
        inplace_op=inplace_mpo,
    )
    site_tags = [p.site_tag(site) for site in range(si, sf + 1)]
    _, subp = p.partition(site_tags, which="any", inplace=True)

    cutoff_mode = _quimb_compression_cutoff_mode(method, cutoff_mode)
    common = {
        "site_tags": site_tags,
        "max_bond": chi,
        "cutoff": 0.0 if method in _MPO_METHODS_IGNORE_CUTOFF else cutoff,
        "permute_arrays": False,
        "inplace": True,
    }
    if cutoff_mode is not None:
        common["cutoff_mode"] = cutoff_mode
    if optimize is not None:
        common["optimize"] = optimize

    if method in {"zipup-first", "zipup-oversample"}:
        # Quimb's native default is max_bond_oversample = 2 * max_bond.
        qtn.tensor_network_1d_compress(
            subp,
            method="zipup",
            max_bond=compression_opts.get("max_bond_oversample", 2 * chi),
            cutoff=compression_opts.get("cutoff_oversample", cutoff),
            cutoff_mode=compression_opts.get("cutoff_mode_oversample", "rel"),
            site_tags=site_tags,
            canonize=True,
            sweep_reverse=True,
            permute_arrays=False,
            optimize=optimize or "auto-hq",
            inplace=True,
        )
        qtn.tensor_network_1d_compress(
            subp,
            method="direct",
            canonize=False,
            compress_opts=compression_opts.get("compress_opts_final"),
            **common,
        )
    else:
        # Quimb's fit-* wrappers use an isolated non-random guess, then an
        # eight-sweep one-site FIT with no fitting cutoff.
        guess_method = method.removeprefix("fit-")
        qtn.tensor_network_1d_compress(
            subp,
            method="fit",
            max_bond=chi,
            cutoff=0.0,
            bsz=1,
            max_iterations=8,
            tn_fit={
                "method": guess_method,
                "cutoff": cutoff,
                "canonize": guess_method != "projector",
                "permute_arrays": False,
            },
            **{
                key: value
                for key, value in common.items()
                if key not in {"max_bond", "cutoff", "cutoff_mode"}
            },
        )

    p |= subp
    if info is not None:
        info["cur_orthog"] = (si, si)
    return p


def _apply_submpo_with_interior_workaround(
    p,
    submpo,
    where,
    *,
    chi,
    method,
    cutoff,
    cutoff_mode,
    info=None,
    inplace_mpo=False,
    optimize=None,
    seed=None,
    compression_opts=None,
):
    """Apply selected Quimb methods with local tags and optional seeding."""
    seed = seed if method in _MPO_METHODS_USE_SEED else None
    return _run_seeded_quimb(
        seed,
        _apply_submpo_with_interior_workaround_impl,
        p,
        submpo,
        where,
        chi=chi,
        method=method,
        cutoff=cutoff,
        cutoff_mode=cutoff_mode,
        info=info,
        inplace_mpo=inplace_mpo,
        optimize=optimize,
        seed=seed,
        compression_opts=compression_opts,
    )


def _apply_dense_gate_with_method(
    p,
    gate,
    where,
    *,
    dims,
    chi,
    method,
    cutoff,
    cutoff_mode,
    info=None,
    inplace_mpo=True,
    optimize=None,
    seed=None,
    compression_opts=None,
):
    """Apply a dense gate using native Quimb or the interior workaround."""
    if dims is None:
        dims = tuple(p.phys_dim(site) for site in where)
    if not (
        method in _MPO_METHODS_NEED_INTERIOR_WORKAROUND
        and _is_interior_submpo_span(p, where)
    ):
        opts = {
            "dims": dims,
            "method": method,
            "max_bond": chi,
            "info": {} if info is None else info,
        }
        if cutoff is not None:
            opts["cutoff"] = (
                0.0 if method in _MPO_METHODS_IGNORE_CUTOFF else cutoff
            )
        cutoff_mode = _quimb_compression_cutoff_mode(method, cutoff_mode)
        if cutoff_mode is not None and method not in _MPO_METHODS_IGNORE_CUTOFF_MODE:
            opts["cutoff_mode"] = cutoff_mode
        if optimize is not None:
            opts["optimize"] = optimize
        if method == "fit-projector":
            # Simple-update gauging can divide by zero on exact product-state
            # bonds. The projector fit remains valid without that optional
            # pre-gauge and Quimb's own implementation supports this path.
            opts["canonize"] = False
        opts.update(compression_opts or {})
        quimb_seed = seed if method in _MPO_METHODS_USE_SEED else None
        return _run_seeded_quimb(quimb_seed, p.gate_nonlocal_, gate, where, **opts)

    submpo = qtn.MatrixProductOperator.from_dense(
        gate,
        dims=dims,
        sites=where,
        L=p.L,
    )
    return _apply_submpo_with_interior_workaround(
        p,
        submpo,
        where,
        chi=chi,
        method=method,
        cutoff=cutoff,
        cutoff_mode=cutoff_mode,
        info=info,
        inplace_mpo=inplace_mpo,
        optimize=optimize,
        seed=seed,
        compression_opts=compression_opts,
    )


def guess(
    p,
    gate,
    where,
    *,
    chi,
    method="zipup",
    dims=None,
    cutoff=0.0,
    cutoff_mode=None,
    info=None,
    inplace=False,
    seed=None,
):
    """Build a disposable compressed MPS guess for a non-local gate.

    This is deliberately a thin wrapper around Quimb's native operation. The
    exact target and the live MPS remain separate; by default only a deep copy
    is modified. ``seed`` is forwarded only to Quimb methods that support
    randomized initialization or sketching.
    """
    method = str(method).strip().lower()
    if method not in _MPO_COMPRESSION_METHODS:
        raise ValueError(f"Unknown compression guess method: {method}")
    _require_quimb_compression_method(method)
    guess = p if inplace else p.copy(deep=True)
    _apply_dense_gate_with_method(
        guess,
        gate,
        where,
        dims=dims,
        chi=chi,
        method=method,
        cutoff=cutoff,
        cutoff_mode=cutoff_mode,
        info=info,
        seed=seed,
    )
    return guess


def svd_guess(p, gate, where, *, chi, **kwargs):
    """Compatibility wrapper for ``guess(..., method="direct")``."""
    return guess(p, gate, where, chi=chi, method="direct", **kwargs)
