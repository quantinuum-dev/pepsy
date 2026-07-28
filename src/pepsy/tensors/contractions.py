"""Tensor-network contraction and norm implementation."""

from __future__ import annotations

from typing import Any

import autoray as ar
import cotengra as ctg
import quimb.tensor as qtn

__all__ = [
    "build_contraction", "build_optimizer", "build_compressed_optimizer",
    "contract_hypercompressed_tn", "contract_hypercompressed_tn_batch",
    "tn_norm",
]

def _ensure_cotengrust():
    """Import cotengrust so cotengra can use accelerated pathfinders."""
    try:
        import cotengrust as ctgr  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - exercised by packaging/env failures
        raise ImportError(
            "Pepsy requires 'cotengrust' for accelerated cotengra path search. "
            "Install it with: pip install cotengrust"
        ) from exc
    return ctgr


def build_optimizer(
    progbar: bool = False,
    alpha: int = 64,
    max_time="rate:7e8",
    max_repeats: int = 2**8,
    parallel="auto",
    optlib: str = "cmaes",
    directory=False,
    hash_method: str = "b",
    overwrite=False,
    on_trial_error: str = "ignore",
    slicing_opts: dict | None = None,
    reconf_opts: dict | None = None,
    slicing_reconf_opts: dict | None = None,
):
    """Build a reusable cotengra contraction optimizer.

    ``cotengrust`` is imported up front so cotengra's ``accel="auto"``
    pathfinders can use the Rust implementations when constructing greedy,
    random-greedy, optimal, and reconfiguration paths.

    Parameters
    ----------
    progbar : bool, optional
        Whether to show optimizer progress.
    alpha : int, optional
        Weight for the combo objective.
    max_time : str | float | None, optional
        Search budget for the hyper-optimizer.
    max_repeats : int, optional
        Maximum number of optimization trials.
    parallel : bool | str, optional
        Parallel search setting passed to cotengra.
    optlib : str, optional
        Backend optimizer library.
    directory : None | bool | str, optional
        Cache directory for reusable contraction trees.
    hash_method : str, optional
        Hashing method for reusable contraction lookup.
    overwrite : bool | str, optional
        Cache overwrite behavior.
    on_trial_error : str, optional
        How to handle individual trial failures. Defaults to ``"ignore"``
        because one invalid candidate path does not make the reusable search
        itself invalid; pass ``"warn"`` or ``"raise"`` to inspect failures.
    slicing_opts : dict | None, optional
        Options passed to cotengra slicing heuristics.
    reconf_opts : dict | None, optional
        Options for subtree reconfiguration.
    slicing_reconf_opts : dict | None, optional
        Options for interleaved slicing and reconfiguration.
    """
    _ensure_cotengrust()

    # cotengra expects directory to be str, True, or None — not False.
    if directory is False:
        directory = None

    kwargs = dict(
        minimize=f"combo-{int(alpha)}",
        max_time=max_time,
        max_repeats=max_repeats,
        parallel=parallel,
        optlib=optlib,
        directory=directory,
        hash_method=hash_method,
        overwrite=overwrite,
        progbar=progbar,
        on_trial_error=on_trial_error,
    )

    if reconf_opts is not None:
        kwargs["reconf_opts"] = reconf_opts

    if slicing_opts is not None:
        kwargs["slicing_opts"] = slicing_opts

    if slicing_reconf_opts is not None:
        kwargs["slicing_reconf_opts"] = slicing_reconf_opts

    return ctg.ReusableHyperOptimizer(**kwargs)


def build_contraction(*args, **kwargs):
    """Build a reusable contraction optimizer.

    This is the short, backend-neutral alias for :func:`build_optimizer`.
    Numerical contractions use the array backend of the tensors supplied to
    Quimb; pair it with ``py.build_backend()`` and ``to_backend=`` to run the
    calculation on Torch CPU while keeping the existing ``build_optimizer``
    name available.
    """

    return build_optimizer(*args, **kwargs)


def build_compressed_optimizer(
    progbar=True,
    chi=4,
    directory=None,
    max_repeats=2**8,
    max_time="rate:1e7",
):
    """Build and return a reusable cotengra compressed optimizer.

    ``cotengrust`` is imported up front so cotengra can use accelerated
    contraction-ordering primitives in any supported compressed path searches.

    Parameters
    ----------
    directory : None, True, or str, optional
        Passed directly to cotengra. ``None`` disables caching; ``True``
        auto-generates a directory in the current working directory.
    """
    _ensure_cotengrust()

    copt = ctg.ReusableHyperCompressedOptimizer(
        chi,
        max_repeats=max_repeats,
        minimize="combo-compressed",
        progbar=progbar,
        max_time=max_time,
        directory=directory,
    )
    return copt


def contract_hypercompressed_tn(
    tn,
    copt=None,
    max_bond=None,
    *,
    chi=None,
    output_inds=None,
    tree_gauge_distance=4,
    progbar=False,
    cutoff=1.0e-12,
    equalize_norms=False,
    inplace=False,
    do_full_simplify=True,
    seq="R",
):
    """Contract a generic tensor network with compressed hyper-optimization.

    Parameters
    ----------
    tn : qtn.TensorNetwork
        Tensor network to compress-contract.
    copt : object, optional
        Reusable compressed cotengra optimizer. If ``None``, one is built
        with :func:`build_compressed_optimizer` using ``chi``.
    max_bond : int | None, optional
        Maximum retained bond dimension during compressed contraction.
        If ``None``, defaults to ``chi``.
    chi : int | None, optional
        Bond dimension used to build ``copt`` when ``copt`` is ``None``.
        Required if both ``copt`` and ``max_bond`` are missing.
    output_inds : sequence[str] | None, optional
        Output indices to preserve during contraction.
    tree_gauge_distance : int, optional
        Gauge distance passed to ``contract_compressed_``.
    progbar : bool, optional
        Whether to show progress during compressed contraction.
    cutoff : float, optional
        Truncation cutoff passed to ``contract_compressed_``.
    equalize_norms : bool | float, optional
        Norm equalization option passed to ``contract_compressed_``.
    inplace : bool, optional
        If ``True``, mutate ``tn`` directly. Otherwise, contract a copy.
    do_full_simplify : bool, optional
        Whether to run ``tn_out.full_simplify_(seq="R", split_method="svd")``
        before building the contraction tree. Enabled by default.
    seq : str, optional
        Simplification sequence passed to ``full_simplify_`` when
        ``do_full_simplify=True``.

    Returns
    -------
    qtn.TensorNetwork
        The compressed-contracted tensor network.
    """
    if max_bond is None:
        max_bond = chi

    if copt is None:
        if chi is None:
            raise ValueError(
                "When `copt` is not provided, please provide `chi` "
                "to build a compressed optimizer."
            )
        copt = build_compressed_optimizer(progbar=progbar, chi=chi)

    if max_bond is None:
        raise ValueError("Please provide `max_bond` (or `chi`) for compressed contraction.")

    tn_out = tn if inplace else tn.copy()
    if do_full_simplify:
        tn_out.full_simplify_(seq=seq, split_method="svd", inplace=True)
    tree = tn_out.contraction_tree(copt)
    tn_out.contract_compressed_(
        optimize=tree,
        output_inds=output_inds,
        max_bond=max_bond,
        tree_gauge_distance=tree_gauge_distance,
        equalize_norms=equalize_norms,
        cutoff=cutoff,
        progbar=progbar,
    )
    return tn_out


def contract_hypercompressed_tn_batch(
    tn,
    samples,
    *,
    copt=None,
    chi=None,
    max_bond=None,
    site_inds=None,
    tree=None,
    cutoff=0.0,
    tree_gauge_distance=6,
    equalize_norms=1.0,
    output_inds=(),
    vmap=True,
    strip_exponent=True,
    chunk_size=None,
    report_timing=True,
    progbar=False,
    mem_warn_gb=None,
    return_tree=False,
):
    """Batch-contract amplitudes ``<x|psi>`` for many configs with ONE fixed tree.

    Torch-only batched analogue of :func:`contract_hypercompressed_tn`, following
    the symmray *batch gpu amplitudes* pattern.  The tensor network ``tn`` (arrays
    on the torch backend) is packed once with ``quimb.tensor.pack``; each sample's
    physical index is selected by a **one-hot contraction** (a ``torch.vmap``-safe
    replacement for the dense ``isel``, which would call ``.item()`` on a traced
    tensor); and the compressed contraction tree is built **once** -- the cotengra
    hyper-optimization search is a single warm-up -- then reused for every sample,
    either fused with ``torch.vmap`` (``vmap=True``) or in a Python loop.

    The search runs only once and the resulting tree lives in host memory, so the
    per-sample cost is just the compressed contraction (no path re-search).  Pass
    ``return_tree=True`` once and feed the returned ``tree`` back in via ``tree=``
    to reuse the warm-up across separate calls.

    Parameters
    ----------
    tn : quimb.tensor.TensorNetwork
        State network with per-site physical indices (e.g. a PEPS), arrays on the
        torch backend.  Must expose ``sites``/``site_ind`` unless ``site_inds`` is
        given.
    samples : torch.Tensor
        Integer (``int64``) configs.  Shape ``(num_samples, L)`` (batch-major) or
        ``(L, num_samples)``; the ``L`` axis is matched to ``len(site_inds)`` and
        transposed to batch-major automatically.  Column ``i`` selects the
        physical value of ``site_inds[i]``.
    copt : object, optional
        Reusable compressed cotengra optimizer used to build the tree.  If
        ``None`` (and ``tree`` is not given) one is built from ``chi``.
    chi, max_bond : int, optional
        ``chi`` sizes ``copt``; ``max_bond`` caps the retained bond during
        contraction (defaults to ``chi``).
    site_inds : sequence[str], optional
        Physical index order matching the columns of ``samples``.  Defaults to
        ``[tn.site_ind(s) for s in tn.sites]``.
    tree : cotengra.ContractionTree, optional
        Pre-built contraction tree (from a previous ``return_tree=True`` call) to
        reuse, skipping the warm-up search.
    cutoff : float, optional
        Singular-value cutoff.  **Must be ``0.0`` when ``vmap=True``**: a positive
        cutoff makes the SVD truncation rank data-dependent, which ``torch.vmap``
        cannot trace.  Fixed-rank truncation to ``max_bond`` is used instead.
    tree_gauge_distance, equalize_norms, output_inds : optional
        Passed through to ``contract_compressed`` (see
        :func:`contract_hypercompressed_tn`).
    vmap : bool, optional
        If ``True`` (default) fuse the batch with ``torch.vmap``; otherwise loop.
    strip_exponent : bool, optional
        If ``True`` (default) return ``(mantissa, exponent)`` base-10 pairs
        (``amplitude = mantissa * 10 ** exponent``) for over/underflow stability.
    chunk_size : int | None, optional
        If given, split the batch into chunks of this many samples and process one
        chunk at a time (each chunk still fused by ``torch.vmap`` when
        ``vmap=True``).  Bounds peak memory for large batches; ``None`` (default)
        processes the whole batch at once.
    report_timing : bool, optional
        If ``True`` (default) print a one-line timing summary: the warm-up
        (tree-build) time and the cost of a single-sample contraction.  Set
        ``False`` to silence.
    progbar : bool, optional
        If ``True`` show a ``tqdm`` progress bar over the batch chunks (one
        update per chunk; set ``chunk_size`` for finer granularity).
    mem_warn_gb : float | None, optional
        Estimated peak-memory threshold (GB) above which a warning is emitted
        (the contraction still proceeds).  ``None`` (default) uses half of
        physical RAM.  The estimate is the compressed tree's peak intermediate
        size times the (chunked) batch size times the element byte-width.
    return_tree : bool, optional
        If ``True`` also return the (possibly newly built) contraction tree.

    Returns
    -------
    (mantissa, exponent) : tuple[torch.Tensor, torch.Tensor]
        Length-``num_samples`` tensors when ``strip_exponent=True``.
    amplitudes : torch.Tensor
        Length-``num_samples`` complex tensor when ``strip_exponent=False``.
    tree : optional
        Returned alongside the above when ``return_tree=True``.
    """
    import torch  # local import: the batch path is torch-only
    import time
    import os
    import warnings
    import math

    if not isinstance(samples, torch.Tensor):
        raise TypeError(
            "contract_hypercompressed_tn_batch expects `samples` as a torch "
            "int64 tensor (this batch path is torch-only)."
        )
    if vmap and float(cutoff) != 0.0:
        raise ValueError(
            "vmap=True requires cutoff=0.0: a positive cutoff makes the SVD "
            "truncation rank data-dependent (n_chi = count_nonzero(s > cutoff)), "
            "so the retained shape varies per sample, which torch.vmap cannot "
            "trace. Use cutoff=0.0 -- for a fixed max_bond it keeps exactly "
            "max_bond singular values (the most accurate choice for that bond) -- "
            "or pass vmap=False to loop (which supports an adaptive cutoff)."
        )

    if site_inds is None:
        site_inds = [tn.site_ind(s) for s in tn.sites]
    site_inds = list(site_inds)
    n_sites = len(site_inds)
    phys_dims = [int(tn.ind_size(ind)) for ind in site_inds]

    samples = samples.to(torch.int64)
    if samples.ndim != 2:
        raise ValueError(f"`samples` must be 2D, got shape {tuple(samples.shape)}.")
    # Accept (num_samples, L) or (L, num_samples): orient to batch-major.
    if samples.shape[1] != n_sites:
        if samples.shape[0] == n_sites:
            samples = samples.transpose(0, 1).contiguous()
        else:
            raise ValueError(
                f"`samples` shape {tuple(samples.shape)} does not match "
                f"L={n_sites} on either axis."
            )

    if max_bond is None:
        max_bond = chi
    if max_bond is None and tree is None:
        raise ValueError("Please provide `max_bond` (or `chi`) for compressed contraction.")

    params, skeleton = qtn.pack(tn)
    # Reference dtype from the packed leaves (all torch arrays share it).
    ref = next(iter(params.values())) if isinstance(params, dict) else params[0]
    dtype = ref.dtype
    ref_is_cuda = bool(getattr(ref, "is_cuda", False))

    def _sync():
        if ref_is_cuda:
            torch.cuda.synchronize()

    def _selected_tn(params, x):
        """Unpack and select each site's physical value via a one-hot contraction."""
        tnx = qtn.unpack(params, skeleton)
        for i, ind in enumerate(site_inds):
            one_hot = torch.nn.functional.one_hot(x[i], phys_dims[i]).to(dtype)
            tnx |= qtn.Tensor(one_hot, (ind,))
        return tnx

    # Warm-up: build the compressed contraction tree exactly once.  The one-hot
    # values do not change the network topology, so a tree built on a
    # representative config is valid for every sample.
    if tree is None:
        if copt is None:
            if chi is None:
                raise ValueError(
                    "When neither `copt` nor `tree` is provided, pass `chi` to "
                    "build a compressed optimizer."
                )
            copt = build_compressed_optimizer(progbar=False, chi=chi)
        x0 = torch.zeros(n_sites, dtype=torch.int64)
        _t0 = time.perf_counter()
        tree = _selected_tn(params, x0).contraction_tree(copt, output_inds=output_inds)
        warmup_time = time.perf_counter() - _t0
    else:
        warmup_time = None  # reused an existing tree

    def _amplitude(params, x):
        return _selected_tn(params, x).contract_compressed(
            optimize=tree,
            output_inds=output_inds,
            max_bond=max_bond,
            cutoff=cutoff,
            tree_gauge_distance=tree_gauge_distance,
            equalize_norms=equalize_norms,
            strip_exponent=strip_exponent,
        )

    n_samples = int(samples.shape[0])
    if n_samples == 0:
        empty = samples.new_zeros(0, dtype=dtype)
        out = (empty, empty.real.clone()) if strip_exponent else empty
        return (out, tree) if return_tree else out

    # Diagnostic: time a single-sample contraction (one extra call) so the
    # per-sample cost and the one-off warm-up are always visible.
    _sync()
    _t0 = time.perf_counter()
    _ = _amplitude(params, samples[0])
    _sync()
    one_sample_time = time.perf_counter() - _t0
    if report_timing:
        warm = "reused" if warmup_time is None else f"{warmup_time:.3f}s"
        print(
            f"[contract_hypercompressed_tn_batch] warm-up(tree)={warm} | "
            f"one-sample={one_sample_time * 1e3:.1f}ms | batch={n_samples}"
            + (f" | chunk_size={int(chunk_size)}" if chunk_size else "")
        )

    # Cost/memory estimate from the compressed contraction tree.  The vmapped
    # batch adds a leading dimension of size `chunk_eff` to every intermediate,
    # so peak memory ~= (per-sample peak intermediate elements) * chunk_eff *
    # bytes/element.  FLOPs are reported as log10 and memory as log2; a warning
    # (but not an abort) fires if the estimate exceeds `mem_warn_gb`.
    chunk_eff = n_samples if not chunk_size else min(int(chunk_size), n_samples)
    try:
        log10_flops = float(tree.total_flops(chi=max_bond, log=10))  # per sample
        peak_elems = float(tree.peak_size(chi=max_bond))
    except Exception:  # pragma: no cover - defensive: tree API drift
        log10_flops = float("nan")
        peak_elems = float("nan")
    bytes_per = 8 if dtype in (torch.complex64, torch.complex32) else 16
    peak_mem_bytes = peak_elems * chunk_eff * bytes_per
    log10_total_flops = log10_flops + (math.log10(n_samples) if n_samples > 0 else 0.0)
    log2_peak_mem = math.log2(peak_mem_bytes) if peak_mem_bytes > 0 else float("-inf")
    if mem_warn_gb is not None:
        mem_warn_bytes = float(mem_warn_gb) * 1e9
    else:
        try:
            mem_warn_bytes = 0.5 * os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError, AttributeError):
            mem_warn_bytes = float("inf")
    if report_timing:
        print(
            f"[contract_hypercompressed_tn_batch] log10(flops/sample)={log10_flops:.2f} "
            f"| log10(total_flops)={log10_total_flops:.2f} "
            f"| log2(peak_mem_bytes, chunk={chunk_eff})={log2_peak_mem:.2f} (chi={max_bond})"
        )
    if peak_mem_bytes > mem_warn_bytes:
        warnings.warn(
            "contract_hypercompressed_tn_batch estimated peak memory "
            f"log2(bytes)={log2_peak_mem:.1f} (~{peak_mem_bytes / 1e9:.1f} GB, "
            f"chunk={chunk_eff}, chi={max_bond}) exceeds "
            f"log2={math.log2(mem_warn_bytes):.1f} (~{mem_warn_bytes / 1e9:.1f} GB); "
            "reduce chunk_size or max_bond. Proceeding anyway.",
            stacklevel=2,
        )

    # Batch execution, optionally chunked (each chunk still vmap-fused) to bound
    # peak memory on large batches.
    cs = n_samples if not chunk_size else int(chunk_size)

    def _run_chunk(chunk):
        if vmap:
            return torch.vmap(_amplitude, in_dims=(None, 0))(params, chunk)
        if strip_exponent:
            pairs = [_amplitude(params, chunk[b]) for b in range(chunk.shape[0])]
            return (
                torch.stack([p[0] for p in pairs]),
                torch.stack([p[1] for p in pairs]),
            )
        return torch.stack([_amplitude(params, chunk[b]) for b in range(chunk.shape[0])])

    chunk_starts = range(0, n_samples, cs)
    if progbar:
        from tqdm.auto import tqdm  # pylint: disable=import-outside-toplevel

        chunk_starts = tqdm(chunk_starts, desc="amp batch", unit="chunk")
    chunk_outs = [_run_chunk(samples[i:i + cs]) for i in chunk_starts]
    if strip_exponent:
        out = (
            torch.cat([o[0] for o in chunk_outs]),
            torch.cat([o[1] for o in chunk_outs]),
        )
    else:
        out = torch.cat(chunk_outs)

    if return_tree:
        return out, tree
    return out


def tn_norm(
    psi,
    *,
    contraction_opt: Any | None = None,
    strip_exponent: bool = False,
    simplify: bool = False,
    simplify_seq: str = "R",
):
    """Compute the norm of a tensor network state.

    Parameters
    ----------
    psi : qtn.TensorNetwork
        State whose norm is computed.
    contraction_opt : object | None, optional
        Contraction optimizer. If ``None``, a default optimizer is built.
    strip_exponent : bool, default=False
        If ``True``, pass ``strip_exponent=True`` to the contraction, which
        returns ``(mantissa, exponent)`` instead of the scalar result.
    simplify : bool, default=False
        Whether to simplify the closed norm network before contraction.
    simplify_seq : str, optional
        Simplification sequence passed to ``full_simplify_`` when
        ``simplify=True``.

    Returns
    -------
    float | tuple[float, float]
        ``|<psi|psi>|`` when ``strip_exponent=False``, or
        ``(mantissa, exponent)`` when ``strip_exponent=True``.
    """
    if contraction_opt is None:
        contraction_opt = build_optimizer(progbar=False)
    norm_tn = psi.H & psi
    if simplify:
        norm_tn.full_simplify_(seq=simplify_seq, output_inds=())
    if not strip_exponent:
        return ar.do(
            "abs",
            norm_tn.contract(all, optimize=contraction_opt, output_inds=()),
        )

    return norm_tn.contract(
        all,
        optimize=contraction_opt,
        output_inds=(),
        strip_exponent=strip_exponent,
    )
