"""Long-range PEPS expectation helpers with BP-aware environment choices.

The boundary route delegates to Quimb's 2D PEPS environment contraction.  The
path-cluster route is useful when a two-site operator has widely separated
support: Quimb explicitly connects the sites, expands a graph-distance
neighbourhood, and contracts the resulting finite cluster. Native Symmray
clusters use a small adapter around Quimb's compressed-contraction API so
QR/SVD truncations retain their charge sectors and graded metadata.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import autoray as ar

from ._symmray import (
    is_symmray_array as _is_symmray_array,
    uses_symmray as _uses_symmray,
)

__all__ = [
    "compute_boundary_expectation",
    "compute_path_cluster_expectation",
    "compute_bp_path_expectation",
]


def _validate_terms(terms):
    if not isinstance(terms, Mapping):
        raise TypeError("terms must be a mapping from site support to operators")
    if not terms:
        raise ValueError("terms must contain at least one operator")
    return terms


def _term_sites(tn, where):
    """Normalize a Quimb local-term key to an ordered site tuple."""
    has_site = getattr(tn, "has_site", None)
    if callable(has_site) and has_site(where):
        return (where,)
    if isinstance(where, (str, bytes)):
        return (where,)
    try:
        sites = tuple(where)
    except TypeError:
        return (where,)
    if not sites:
        raise ValueError("a local expectation term must have at least one site")
    return sites


def _squeeze_native_singleton_bonds(tn):
    """Remove dimension-one bonds with Symmray's graded squeeze operation.

    Quimb's compressed-contraction implementation correctly detects such bonds
    but removes them with a generic reshape. For a fermionic Symmray array,
    ``squeeze`` additionally records an odd singlet as a dummy mode, which is
    necessary to retain its graded ordering. Doing this before the public
    ``contract_compressed`` call avoids the dense-only reshape path without
    changing Quimb's contraction or SVD algorithms.
    """
    for index in tuple(tn.inner_inds()):
        if tn.ind_size(index) != 1:
            continue
        for tid in tuple(tn.ind_map[index]):
            tensor = tn.tensor_map[tid]
            axis = tensor.inds.index(index)
            data = ar.do("squeeze", tensor.data, axis=axis)
            inds = tuple(ind for ind in tensor.inds if ind != index)
            left_inds = (
                None
                if tensor.left_inds is None
                else tuple(ind for ind in tensor.left_inds if ind != index)
            )
            tensor.modify(data=data, inds=inds, left_inds=left_inds)


def _align_native_cluster_bonds(tn):
    """Pad cluster bond sectors so both Symmray endpoints share one layout.

    A BP-to-SU conversion can legitimately leave a zero-weight charge sector
    absent from one endpoint tensor while the opposite endpoint retains it.
    Exact Symmray contractions handle that sparse support, but Quimb's
    compressed-contraction bookkeeping uses one dense bond size per index.
    Pad only the private cluster copy with zero blocks before compression.
    """
    for index in tuple(tn.inner_inds()):
        tids = tuple(tn.ind_map[index])
        if len(tids) != 2:
            continue
        left_tid, right_tid = tids
        left_tensor = tn.tensor_map[left_tid]
        right_tensor = tn.tensor_map[right_tid]
        if not (
            _is_symmray_array(left_tensor.data)
            and _is_symmray_array(right_tensor.data)
        ):
            continue

        left_axis = left_tensor.inds.index(index)
        right_axis = right_tensor.inds.index(index)
        left_index = left_tensor.data.indices[left_axis]
        right_index = right_tensor.data.indices[right_axis]
        left_map = dict(left_index.chargemap)
        right_map = dict(right_index.chargemap)
        for charge in left_map.keys() & right_map.keys():
            if left_map[charge] != right_map[charge]:
                raise ValueError(
                    "incompatible Symmray virtual charge dimensions on "
                    f"bond {index!r} for compressed path contraction"
                )
        shared_map = {**left_map, **right_map}
        if left_map == right_map:
            continue

        left_indices = list(left_tensor.data.indices)
        left_indices[left_axis] = left_index.copy_with(
            chargemap=shared_map,
            dual=left_index.dual,
        )
        left_data = left_tensor.data.copy_with(indices=tuple(left_indices))
        left_data.fill_missing_blocks()
        left_tensor.modify(data=left_data)

        right_indices = list(right_tensor.data.indices)
        right_indices[right_axis] = right_index.copy_with(
            chargemap=shared_map,
            dual=right_index.dual,
        )
        right_data = right_tensor.data.copy_with(indices=tuple(right_indices))
        right_data.fill_missing_blocks()
        right_tensor.modify(data=right_data)


def _native_compressed_path_expectation(
    tn,
    terms,
    *,
    max_distance,
    mode,
    fillin,
    grow_from,
    gauges,
    smudge,
    power,
    max_bond,
    normalized,
    optimize,
    return_all,
    contract_opts,
):
    """Contract native Symmray path clusters with Quimb's compression engine.

    Quimb's generic ``local_expectation`` route fuses the resulting RDM and
    contracts it as a dense matrix. Keep the RDM physical legs unfused instead:
    that retains their fermionic index metadata, lets the native operator
    supply the graded ordering, and still delegates every contraction and
    truncation to Quimb/Cotengra/Symmray.
    """
    if normalized not in (True, False):
        raise ValueError(
            "native Symmray compressed path clusters require normalized to "
            "be True or False"
        )

    options = dict(contract_opts)
    flatten = options.pop("flatten", True)
    reduce = options.pop("reduce", False)
    symmetrized = options.pop("symmetrized", "auto")
    rehearse = options.pop("rehearse", False)
    method = options.pop("method", "contract_compressed")
    if reduce:
        raise NotImplementedError(
            "reduce=True is not yet supported for native Symmray compressed "
            "path clusters"
        )
    if rehearse:
        raise NotImplementedError(
            "rehearse is not yet supported for native Symmray compressed "
            "path clusters"
        )
    if method != "contract_compressed":
        raise ValueError(
            "native Symmray compressed path clusters use "
            "method='contract_compressed'"
        )
    if symmetrized == "auto":
        symmetrized = not flatten

    user_post_contract = options.pop("callback_post_contract", None)
    user_post_compress = options.pop("callback_post_compress", None)

    def _post_contract(work_tn, tid):
        # A contraction can drop a zero-weight charge sector at one endpoint.
        # Re-pad it before a following compression inspects dense bond sizes.
        _align_native_cluster_bonds(work_tn)
        # A compression can also create a new singleton bond after the initial
        # cleanup. Remove it before Quimb's next generic squeeze pass.
        _squeeze_native_singleton_bonds(work_tn)
        if user_post_contract is not None:
            user_post_contract(work_tn, tid)

    def _post_compress(work_tn, tids):
        _align_native_cluster_bonds(work_tn)
        _squeeze_native_singleton_bonds(work_tn)
        if user_post_compress is not None:
            user_post_compress(work_tn, tids)

    options["callback_post_contract"] = _post_contract
    options["callback_post_compress"] = _post_compress

    expecs = {}
    for where, gate in terms.items():
        if not _is_symmray_array(gate):
            raise TypeError(
                "native Symmray compressed path clusters require native "
                "Symmray local operators"
            )

        sites = _term_sites(tn, where)
        cluster = tn.get_cluster(
            sites,
            gauges=gauges,
            max_distance=max_distance,
            mode=mode,
            fillin=fillin,
            grow_from=grow_from,
            smudge=smudge,
            power=power,
        ).copy()
        _align_native_cluster_bonds(cluster)
        ket_inds = tuple(map(cluster.site_ind, sites))
        bra_ind_id = "_pepsy_bra{}"
        bra_inds = tuple(map(bra_ind_id.format, sites))
        rdm = cluster.make_reduced_density_matrix(
            sites,
            bra_ind_id=bra_ind_id,
        )

        if flatten:
            for site in cluster.gen_site_coos():
                if site not in sites or flatten == "all":
                    tag = rdm.site_tag(site)
                    if tag in rdm.tag_map:
                        rdm ^= tag

        rdm.fuse_multibonds_()
        _squeeze_native_singleton_bonds(rdm)
        rho_tensor = rdm.contract_compressed(
            optimize,
            max_bond=max_bond,
            output_inds=ket_inds + bra_inds,
            **options,
        )
        rho = rho_tensor.data
        if normalized:
            norm = ar.do("trace", rho_tensor.to_dense(ket_inds, bra_inds))
            rho = rho / norm
        if symmetrized:
            rho = (rho + ar.do("dag", rho)) / 2

        nsites = len(sites)
        if ar.do("ndim", gate) != 2 * nsites:
            gate = ar.do("reshape", gate, ar.do("shape", rho))
        expecs[where] = ar.do(
            "tensordot",
            rho,
            gate,
            axes=(
                tuple(range(2 * nsites)),
                tuple(range(nsites, 2 * nsites)) + tuple(range(nsites)),
            ),
        )

    if return_all:
        return expecs
    return sum(expecs.values())


def compute_boundary_expectation(
    tn,
    terms,
    *,
    max_bond=None,
    cutoff=1.0e-10,
    canonize=True,
    mode="mps",
    layer_tags=("KET", "BRA"),
    normalized=True,
    autogroup=True,
    contract_optimize="auto-hq",
    return_all=False,
    plaquette_envs=None,
    plaquette_map=None,
    **plaquette_env_options,
):
    """Compute batched PEPS expectations using Quimb's boundary environment.

    ``terms`` accepts the same one- and two-site support mapping as Quimb's
    ``TensorNetwork2DVector.compute_local_expectation``.  In particular, a
    key such as ``((x0, y0), (x1, y1))`` remains a connected long-range
    operator support; it is not replaced by a product of endpoint estimates.

    Parameters are forwarded to Quimb's PEPS boundary implementation.  The
    returned value is the sum of locally normalized terms unless
    ``return_all=True``.
    """
    _validate_terms(terms)
    if not hasattr(tn, "compute_local_expectation"):
        raise TypeError(
            "tn must provide Quimb's compute_local_expectation method"
        )

    return tn.compute_local_expectation(
        terms,
        max_bond=max_bond,
        cutoff=cutoff,
        canonize=canonize,
        mode=mode,
        layer_tags=layer_tags,
        normalized=normalized,
        autogroup=autogroup,
        contract_optimize=contract_optimize,
        return_all=return_all,
        plaquette_envs=plaquette_envs,
        plaquette_map=plaquette_map,
        **plaquette_env_options,
    )


def compute_path_cluster_expectation(
    tn,
    terms,
    *,
    max_distance=0,
    mode="graphdistance",
    fillin=True,
    grow_from="all",
    gauges=None,
    smudge=1.0e-12,
    power=1.0,
    max_bond=None,
    normalized=True,
    optimize="auto-hq",
    return_all=False,
    **contract_opts,
):
    """Compute expectations on connected, distance-expanded PEPS clusters.

    For a two-site term, Quimb first adds a graph path between the sites, then
    expands that path by ``max_distance``.  ``fillin=True`` adds lattice corner
    tensors.  If ``gauges`` is supplied, it must contain simple-update/SU-style
    bond vectors used to close the cluster boundary; D2BP matrix messages must
    not be passed here directly.

    Native Symmray clusters use a Pepsy adapter around Quimb's public
    compressed-contraction API when ``max_bond`` is supplied. The adapter
    preserves the unfused fermionic RDM legs and removes singleton virtual
    bonds with Symmray's graded squeeze operation before Quimb performs its
    QR/SVD truncations. Thus ``optimize`` accepts Quimb/Cotengra contraction
    paths for both exact and compressed native clusters.
    """
    _validate_terms(terms)
    if not hasattr(tn, "compute_local_expectation_cluster"):
        raise TypeError(
            "tn must provide Quimb's compute_local_expectation_cluster method"
        )
    if max_distance < 0:
        raise ValueError("max_distance must be nonnegative")
    if _uses_symmray(tn) and max_bond is not None:
        return _native_compressed_path_expectation(
            tn,
            terms,
            max_distance=max_distance,
            mode=mode,
            fillin=fillin,
            grow_from=grow_from,
            gauges=gauges,
            smudge=smudge,
            power=power,
            max_bond=max_bond,
            normalized=normalized,
            optimize=optimize,
            return_all=return_all,
            contract_opts=contract_opts,
        )

    return tn.compute_local_expectation_cluster(
        terms,
        max_distance=max_distance,
        mode=mode,
        fillin=fillin,
        grow_from=grow_from,
        gauges=gauges,
        smudge=smudge,
        power=power,
        max_bond=max_bond,
        normalized=normalized,
        optimize=optimize,
        return_all=return_all,
        **contract_opts,
    )


def compute_bp_path_expectation(
    tn,
    terms,
    *,
    max_distance=0,
    mode="graphdistance",
    fillin=True,
    max_bond=None,
    normalized=True,
    optimize="auto-hq",
    return_all=False,
    bp_options: dict[str, Any] | None = None,
    conversion_options: dict[str, Any] | None = None,
    require_converged=True,
    **contract_opts,
):
    """Compute path-cluster expectations using a D2BP-derived SU closure.

    This is the safe BP route for fermionic Symmray PEPS.  D2BP is run on the
    physical network, then Pepsy's tested BP-to-SU bridge converts its native
    positive-semidefinite matrix messages into SU-style bond vectors.  The
    converted core and vectors are passed to Quimb's connected path-cluster
    expectation routine; D2BP matrices are never passed as SU gauges.

    ``bp_options`` is forwarded to :func:`pepsy.bp.gauge_all` under its
    ``bp_options`` argument, for example ``{"run_opts": {"diis": False}}``.
    ``conversion_options`` is forwarded to the BP-to-SU conversion and
    defaults to a small regularization of singular message eigenvalues.

    Native Symmray path clusters support ``max_bond`` through the same graded
    compressed path as :func:`compute_path_cluster_expectation`. Set
    ``require_converged=False`` only for diagnostic experiments with an
    unconverged BP fixed point.
    """
    _validate_terms(terms)
    from .gauges import gauge_all

    bp_options = {} if bp_options is None else dict(bp_options)
    conversion_options = (
        {"smudge": 1.0e-12}
        if conversion_options is None
        else dict(conversion_options)
    )
    bridge = gauge_all(
        tn,
        start="bp",
        target="su",
        norm="2norm",
        bp_options=bp_options,
        conversion_options=conversion_options,
    )

    if require_converged and (
        bridge.bp_result is None or not bridge.bp_result.converged
    ):
        diagnostic = (
            None
            if bridge.bp_result is None
            else bridge.bp_result.max_mdiff
        )
        raise RuntimeError(
            "D2BP did not converge before the path-cluster expectation was "
            f"requested (max_mdiff={diagnostic!r}); pass "
            "require_converged=False for a diagnostic estimate"
        )

    return compute_path_cluster_expectation(
        bridge.core,
        terms,
        max_distance=max_distance,
        mode=mode,
        fillin=fillin,
        gauges=bridge.gauges,
        max_bond=max_bond,
        normalized=normalized,
        optimize=optimize,
        return_all=return_all,
        **contract_opts,
    )
