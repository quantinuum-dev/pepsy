"""Exact gate targets followed by protected ordinary-MPO compression."""

from dataclasses import dataclass
from numbers import Integral

import autoray as ar

from ...backends import infer_backend_converter_from_sample
from ...operators.mpo_dmt import (
    _compress_mpo_dmt, _gate_mpo_dmt, _gate_mpo_dmt_routed, _gate_mpo_dmt_single,
)
from ...operators.pauli_dmt_geometry import _normalize_regions
from ...operators.pauli_mpo import _array_state_token


def _state_token(p):
    return p.L, p.cyclic, p.exponent, p.site_tag_id, p.upper_ind_id, p.lower_ind_id


def _tensor_token(tensor):
    return tensor.inds, tensor.left_inds, _array_state_token(tensor.data)


@dataclass(frozen=True)
class _CanonicalCache:
    workspace: object
    state: object
    info: dict
    tensors: tuple
    arrays: tuple
    structure: tuple
    state_token: tuple
    options: tuple

    def matches(self, opt, options):
        if (self.state is not opt.p or self.info is not opt.info_c or self.options != options
                or opt.info_c.get("cur_orthog") != (self.workspace.center,) * 2
                or self.state_token != _state_token(opt.p)
                or len(self.tensors) != opt.p.num_tensors):
            return False
        for site, (old_tensor, old_array, structure) in enumerate(
                zip(self.tensors, self.arrays, self.structure)):
            tensor = opt.p[site]
            if (tensor is not old_tensor or tensor.data is not old_array
                    or _tensor_token(tensor) != structure):
                return False
        return True


def _cache_options(chi, cutoff, cutoff_mode, variant, regions, diagnostics, truncation,
                   max_refinement_sweeps, region_selection, constraint_mode,
                   constraint_weight, check_regions, preservation_check):
    return (chi, cutoff, cutoff_mode, variant, regions, diagnostics or constraint_mode == "relaxed",
            truncation, max_refinement_sweeps, region_selection, constraint_mode,
            constraint_weight, check_regions and preservation_check != "off", preservation_check)


def _remember(opt, workspace):
    tensors = tuple(opt.p[i] for i in range(opt.p.L))
    options = _cache_options(
        opt.chi, workspace.mpo_cutoff, workspace.cutoff_mode, workspace.variant,
        workspace.regions, workspace.diagnostics, workspace.truncation,
        workspace.max_refinement_sweeps, workspace.region_selection, workspace.constraint_mode,
        workspace.constraint_weight, workspace.check_regions, workspace.preservation_check)
    # Hold the objects themselves: bare ids can be recycled after a caller
    # replaces tensor data. Arrays are shared, not copied or densified.
    opt._dmt_canonical_cache = _CanonicalCache(
        workspace, opt.p, opt.info_c, tensors, tuple(t.data for t in tensors),
        tuple(_tensor_token(t) for t in tensors), _state_token(opt.p), options)


def validate_state(opt, regions, variant):
    if opt.p.cyclic or opt._has_symmray_data(opt.p):
        raise NotImplementedError("MPO DMT requires a dense open-boundary MPO")
    if variant not in {"density", "operator"}:
        raise ValueError("dmt_variant must be 'density' or 'operator'")
    if opt.ind_id_k != opt.p.upper_ind_id or opt.ind_id_b != opt.p.lower_ind_id:
        raise ValueError("DMT ind_id_k/ind_id_b must match the MPO physical index IDs")
    return _normalize_regions(regions, opt.p.L)


def validate_replay(opt, regions, variant, *, target_cutoff, submpo_method,
                    compression_seed, fit_fallback, layout):
    regions = validate_state(opt, regions, variant)
    if target_cutoff != 0.0 or submpo_method is not None or compression_seed is not None:
        raise ValueError("DMT requires target_cutoff=0 and no ordinary compressor overrides")
    if fit_fallback is not None:
        raise ValueError("DMT cannot fall back to a compressor that drops protected data")
    if layout is not None and layout is not False:
        raise ValueError("Install a layout before DMT replay and map regions to its storage positions")
    from .optimizer import MpoChannelEvent

    for payload, where in opt._stream_plan.entries:
        if isinstance(payload, MpoChannelEvent):
            raise NotImplementedError("DMT gate replay does not yet support channel-sum events")
        if len(where) not in {1, 2}:
            raise NotImplementedError("DMT gate replay supports one- and two-site gates")
        if (len(set(where)) != len(where)
                or any(isinstance(i, bool) or not isinstance(i, Integral)
                       or not 0 <= i < opt.p.L for i in where)):
            raise ValueError("DMT gate sites must be distinct integer MPO positions in range")
    return regions


def _promote(p, gates=()):
    """Align local gate constants using the shared native backend converter."""
    arrays = [t.data for t in p]
    backend = ar.infer_backend(arrays[0])
    if any(ar.infer_backend(a) != backend for a in arrays):
        raise TypeError("DMT requires one dense array backend")
    for gate in gates:
        if gate is not None and ar.infer_backend(gate) not in {"numpy", backend}:
            raise TypeError("DMT gates must use NumPy or the MPO's native backend")
    dtype = ar.get_common_dtype(*arrays, *(g for g in gates if g is not None))
    # NumPy astype copies even when its dtype is unchanged. Tensor wrappers
    # are already isolated; avoid full-chain array copies for uniform inputs.
    if any(ar.get_dtype_name(a) != dtype for a in arrays):
        p.apply_to_arrays(lambda a: a if ar.get_dtype_name(a) == dtype else ar.astype(a, dtype))
    convert = infer_backend_converter_from_sample(p[0].data)
    return tuple(None if g is None else convert(g) for g in gates)


def _compress(opt, candidate, *, cutoff, cutoff_mode, variant, regions, diagnostics,
              truncation="paper", max_refinement_sweeps=8, region_selection="smallest",
              constraint_mode="strict", constraint_weight=100., check_regions=True,
              preservation_check="raise"):
    # The kernel writes only after the selected verification policy accepts
    # the candidate. Its wrappers are separate from the committed operator.
    _promote(candidate)
    result = opt._timed_call(
        "dmt.compress", _compress_mpo_dmt, candidate, max_bond=opt.chi,
        cutoff=cutoff, cutoff_mode=cutoff_mode, variant=variant, regions=regions,
        inplace=True, return_report=diagnostics,
        truncation=truncation, max_refinement_sweeps=max_refinement_sweeps,
        region_selection=region_selection,
        constraint_mode=constraint_mode, constraint_weight=constraint_weight,
        check_regions=check_regions, preservation_check=preservation_check,
    )
    return result


def _commit(opt, candidate, *, center=None):
    # Validate metadata before any write to the live state. Public Quimb
    # removal/addition keeps tensor owners, index maps and tags consistent.
    opt._backend_info_for(candidate)
    opt._dmt_canonical_cache = None
    if opt.inplace:
        for tid in tuple(opt.p.tensor_map):
            opt.p.pop_tensor(tid)
        opt.p.add_tensor_network(candidate, virtual=False, check_collisions=False)
        opt.p.exponent = candidate.exponent
        if hasattr(candidate, "compression_report") or hasattr(opt.p, "compression_report"):
            opt.p.compression_report = getattr(candidate, "compression_report", None)
        for name in ("pepsy_first_degree", "pepsy_block_plan", "pepsy_pauli_mpo"):
            if hasattr(opt.p, name):
                setattr(opt.p, name, None)
    else:
        opt.p = candidate
    center = opt.p.L - 1 if center is None else center
    opt.info_c.clear()
    opt.info_c["cur_orthog"] = (center, center)
    if opt._replay_rank_cache is not None:
        opt._replay_rank_cache.clear()
    opt.backend_info()
    opt._dmt_compression_seen = True


def compress_current(opt, *, cutoff, cutoff_mode, variant, regions, return_report,
                     truncation="paper", max_refinement_sweeps=8, region_selection="smallest",
                     constraint_mode="strict", constraint_weight=100., check_regions=True,
                     preservation_check="raise"):
    regions = validate_state(opt, regions, variant)
    candidate, report, workspace = _compress(
        opt, opt.p.copy(), cutoff=cutoff, cutoff_mode=cutoff_mode,
        variant=variant, regions=regions, diagnostics=return_report,
        truncation=truncation, max_refinement_sweeps=max_refinement_sweeps,
        region_selection=region_selection,
        constraint_mode=constraint_mode, constraint_weight=constraint_weight,
        check_regions=check_regions, preservation_check=preservation_check,
    )
    _commit(opt, candidate)
    _remember(opt, workspace)
    opt._dmt_reports = [report] if report is not None else []
    opt._append_norm_proxy_sample(opt.p)
    return (opt.p, report) if return_report else opt.p


def replay(opt, gates, wheres, *, progbar, cutoff, cutoff_mode,
           fidelity_samples, finite_check, regions, variant, diagnostics,
           truncation="paper", max_refinement_sweeps=8, region_selection="smallest",
           constraint_mode="strict", constraint_weight=100., check_regions=False,
           preservation_check="raise", update="canonical"):
    from tqdm.auto import tqdm

    opt._dmt_reports = []
    options = _cache_options(
        opt.chi, cutoff, cutoff_mode, variant, regions, diagnostics, truncation,
        max_refinement_sweeps, region_selection, constraint_mode, constraint_weight,
        check_regions, preservation_check)
    cache = opt._dmt_canonical_cache
    workspace = (cache.workspace if update == "canonical" and cache is not None
                 and cache.matches(opt, options) else None)
    sample_steps = opt._sampling_steps(len(gates), fidelity_samples)
    with tqdm(total=len(gates), desc="dmt", disable=not progbar, leave=False) as bar:
        for step, (payload, where) in enumerate(zip(gates, wheres), 1):
            gate, bra_gate, where = opt._parse_gate_entry(payload, where)
            source = opt.p.copy()
            gate, bra_gate = _promote(source, (gate, bra_gate))
            next_workspace = None
            if (workspace is not None and update == "canonical"
                    and ar.get_dtype_name(source[0].data) == ar.get_dtype_name(workspace.cores[0])):
                if len(where) == 1:
                    candidate, report, next_workspace = opt._timed_call(
                        "dmt.single", _gate_mpo_dmt_single, source, workspace, gate, bra_gate, where,
                    )
                elif abs(where[0] - where[1]) == 1:
                    candidate, report, next_workspace = opt._timed_call(
                        "dmt.local", _gate_mpo_dmt, source, workspace, gate, bra_gate, where,
                    )
                else:
                    try:
                        candidate, report, next_workspace = opt._timed_call(
                            "dmt.routed", _gate_mpo_dmt_routed, source, workspace, gate, bra_gate, where,
                        )
                    except (ValueError, FloatingPointError) as exc:
                        # Temporary permutations can demand a larger protected
                        # budget or amplify roundoff. The unmodified source
                        # remains available for the full protected algorithm.
                        if isinstance(exc, ValueError) and "protected budget" not in str(exc):
                            raise
            if next_workspace is None:
                # Start with a full sweep to cap every bond. This is also the
                # reference route for cache misses, dtype changes and a
                # failed protected routing attempt. Targets stay untruncated.
                target = opt._timed_call(
                    "dmt.target", opt._build_mpo_target, source, gate, where,
                    bra_gate, cutoff=0.0, cutoff_mode="rel", gate_cutoff=0.0,
                )
                candidate, report, next_workspace = _compress(
                    opt, target, cutoff=cutoff, cutoff_mode=cutoff_mode,
                    variant=variant, regions=regions, diagnostics=diagnostics,
                    truncation=truncation, max_refinement_sweeps=max_refinement_sweeps,
                    region_selection=region_selection,
                    constraint_mode=constraint_mode, constraint_weight=constraint_weight,
                    check_regions=check_regions, preservation_check=preservation_check,
                )
            opt._check_finite(finite_check, candidate)
            _commit(opt, candidate, center=next_workspace.center)
            _remember(opt, next_workspace)
            workspace = next_workspace
            if report is not None:
                opt._dmt_reports.append(report)
            if step in sample_steps:
                opt._append_norm_proxy_sample(opt.p)
            bar.update()
