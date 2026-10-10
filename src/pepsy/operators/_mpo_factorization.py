"""Paired-factor controls for dense direct MPO gate application."""

import autoray as ar
from numbers import Integral


def projector_method(p, method):
    """Validate the supported representation before modifying any tensors."""
    if method != "direct" or p.cyclic:
        raise NotImplementedError("projector MPO factorization requires open MPOs and direct compression")
    if any(ar.infer_backend(a) not in {"numpy", "torch"} for a in p.arrays):
        raise TypeError("projector MPO factorization supports dense NumPy/Torch arrays")
    from ..backends import register_projector_split

    return register_projector_split()


def canonicalize_projector(p, where, info, split_method):
    """Canonicalize outside the update window through public paired splits."""
    si, sf = min(where), max(where)
    current = (info or {}).get("cur_orthog")
    if isinstance(current, Integral):
        current = (current, current)
    if not (isinstance(current, (tuple, list)) and len(current) == 2
            and all(isinstance(i, Integral) and 0 <= i < p.L for i in current)):
        current = (0, p.L - 1)
    left, right = min(current), max(current)
    # MPO.canonicalize does not accept split options. Use the public bond
    # operation so no hard-coded QR chart enters this composed derivative.
    # Preserve already canonical tensors. Repeatedly refactoring them creates
    # unnecessary, poorly conditioned charts in long gate streams.
    for i in range(left, si):
        p.canonize_between(p.site_tag(i), p.site_tag(i + 1), method=split_method)
    for i in range(right, sf, -1):
        p.canonize_between(p.site_tag(i), p.site_tag(i - 1), method=split_method)
    if info is not None:
        info["cur_orthog"] = (si, sf)


def compression_options(options, split_method):
    """Keep the requested cap at the split rather than Quimb's QR shortcut."""
    if {"canonize_opts", "compress_opts"} & options.keys():
        raise ValueError("projector MPO factorization owns canonize_opts and compress_opts")
    options = dict(options)
    options["canonize_opts"] = {"method": split_method}
    # With cutoff=0 and small outer dimensions, compress_between otherwise
    # skips the specified method and runs QR. Passing the cap to its nested
    # tensor split avoids that shortcut while retaining the actual bond cap.
    options["compress_opts"] = {
        "max_bond": None,
        "compress_opts": {"method": split_method, "max_bond": options.get("max_bond")},
        "reduce_opts": {"method": split_method},
    }
    return options
