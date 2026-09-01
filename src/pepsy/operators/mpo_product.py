"""Joint ordered MPO cluster products.

This module implements the one-dimensional construction of Vanhecke,
Vanderstraeten, and Verstraete, arXiv:1912.10512, for one or more ordered
exponential factors. A finite connected interval or graph cluster is
exponentiated locally, its already represented connected partitions are
subtracted, and the residual is put into one MPO topology.

For factors ``A``, ``B``, and ``C``, the local target is formed as
``exp(A_S) @ exp(B_S) @ exp(C_S)`` before connected residual assembly. The
graph path preserves genuine non-nearest-neighbour clusters and products of
crossing or nested clusters without materializing independent full-lattice
MPO factors.

The local work scales with the requested cluster size, not with the global
Hilbert space dimension.  The result is an ordinary finite open
:class:`FirstDegreeMPO`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from numbers import Integral
import warnings

import autoray as ar
import numpy as np

from .mpo_semantic import (
    FirstDegreeMPO,
    MPOLocalOperatorTerm,
    MPOParameter,
    MPOProductTerm,
    _as_backend,
    _backend_name,
    _backend_reference,
    _fixed_rank_svd,
    _multiply_scalar,
    _resolve_compression_cutoff,
    _resolve_compression_cutoff_mode,
    _resolve_exp_step,
    _normalize_exp_compress_opts,
    _ensure_pepsy_mpo_boundary,
    _native_sector_summary,
    _normalize_sector_aware_request,
    _resolve_sector_aware,
    _scatter_add_2d,
    _term_from_input,
)
from .diagnostics import OperatorReportInfo

__all__ = [
    "MPOClusterFactor",
    "MPOClusterExpansionReport",
    "MPOClusterProductExpansion",
    "CompiledMPOClusterProduct",
    "MPOGraphClusterProductExpansion",
    "MPOClusterBasisExpansion",
    "MPOGraphClusterBasisExpansion",
    "CompiledMPOClusterExp",
    "ClusterBasisExpansion",
    "ClusterExpansionBasis",
    "ClusterExpBasis",
    "MPOClusterExpansion",
    "compress_mpo_product",
    "exp_mpo_cluster",
]


def _kron(left, right):
    """Kronecker product using only backend tensor operations."""

    product = ar.do("tensordot", left, right, axes=0)
    product = ar.do("transpose", product, (0, 2, 1, 3))
    return ar.do(
        "reshape",
        product,
        (
            int(left.shape[0]) * int(right.shape[0]),
            int(left.shape[1]) * int(right.shape[1]),
        ),
    )


def _kron_all(matrices):
    result = matrices[0]
    for matrix in matrices[1:]:
        result = _kron(result, matrix)
    return result


def _embed_matrix_on_positions(matrix, positions, nsites, phys_dim):
    """Embed a dense operator on selected sites of a local cluster.

    ``matrix`` uses the selected sites in the order given by ``positions``.
    The tensor-product construction keeps the matrix on its native backend,
    which is important for Torch and JAX autodiff paths.
    """

    positions = tuple(int(position) for position in positions)
    if not positions:
        return _identity(phys_dim**nsites, like=matrix)
    if len(set(positions)) != len(positions):
        raise ValueError("operator embedding positions must be distinct.")
    if any(position < 0 or position >= nsites for position in positions):
        raise ValueError("operator embedding position is outside the cluster.")
    support_size = len(positions)
    tensor = ar.do("reshape", matrix, (phys_dim,) * (2 * support_size))
    missing = tuple(position for position in range(nsites) if position not in positions)
    factors = [tensor]
    identity = _identity(phys_dim, like=matrix)
    factors.extend(identity for _ in missing)
    embedded = factors[0]
    for factor in factors[1:]:
        embedded = ar.do("tensordot", embedded, factor, axes=0)

    # The outer product is ordered as support outputs, support inputs, then
    # one output/input pair for each missing site. Reorder to all outputs
    # followed by all inputs before flattening back to a matrix.
    missing_axis = {
        site: 2 * support_size + 2 * index
        for index, site in enumerate(missing)
    }
    output_axes = []
    input_axes = []
    support_index = {site: index for index, site in enumerate(positions)}
    for site in range(nsites):
        if site in support_index:
            output_axes.append(support_index[site])
        else:
            output_axes.append(missing_axis[site])
    for site in range(nsites):
        if site in support_index:
            input_axes.append(support_size + support_index[site])
        else:
            input_axes.append(missing_axis[site] + 1)
    embedded = ar.do("transpose", embedded, tuple(output_axes + input_axes))
    return ar.do("reshape", embedded, (phys_dim**nsites, phys_dim**nsites))


def _set_partitions(items):
    """Yield each set partition once, preserving the input ordering."""

    items = tuple(items)
    if not items:
        yield ()
        return
    first, rest = items[0], items[1:]
    for partition in _set_partitions(rest):
        yield ((first,),) + partition
        for index in range(len(partition)):
            block = partition[index]
            updated = tuple(sorted((first,) + block))
            yield partition[:index] + (updated,) + partition[index + 1 :]


def _identity(dim, *, like):
    return ar.do("eye", int(dim), like=like)


def _as_quimb_mpo(mpo):
    """Return a Quimb MPO while preserving the input tensor backend."""

    if hasattr(mpo, "tensors") and hasattr(mpo, "gate_upper_with_op_lazy"):
        return mpo
    converter = getattr(mpo, "to_mpo", None)
    if converter is None:
        raise TypeError(
            "MPO product compression expects a Quimb MPO or an object "
            "with a to_mpo() method."
        )
    result = converter()
    if not hasattr(result, "tensors") or not hasattr(
        result,
        "gate_upper_with_op_lazy",
    ):
        raise TypeError("to_mpo() did not return a compatible Quimb MPO.")
    return result


def _mpo_product_bond_sizes(mpo):
    """Get ordinary MPO bond sizes without converting tensor data."""

    try:
        return tuple(int(size) for size in mpo.bond_sizes())
    except (AttributeError, TypeError, ValueError):
        return tuple(
            int(mpo.bond_size(site, site + 1))
            for site in range(int(mpo.L) - 1)
        )


def _mpo_product_reference(*mpos):
    for mpo in mpos:
        for tensor in mpo.tensors:
            return tensor.data
    raise ValueError("MPO product compression received an empty MPO.")


def _has_symmray_data(*mpos):
    return any(
        hasattr(tensor.data, "blocks") and hasattr(tensor.data, "indices")
        for mpo in mpos
        for tensor in mpo.tensors
    )


def _attach_mpo_product_metadata(mpo, metadata):
    """Attach copy-safe compression metadata to a Quimb result."""

    setattr(mpo, "pepsy_mpo_product_metadata", dict(metadata))
    # Quimb tensor networks intentionally have a small core object model and
    # do not expose a universal metadata field.  The Pepsy-prefixed attribute
    # above is therefore the stable boundary for ordinary MPO results.
    return mpo


def compress_mpo_product(
    A,
    B,
    *,
    chi=None,
    method="auto",
    cutoff="auto",
    cutoff_mode="auto",
    sector_aware="auto",
    guess_method="auto",
    guess_seed=None,
):
    """Compress the ordered MPO product ``A @ B`` into one ordinary MPO.

    The product is first represented lazily as ``B.gate_upper_with_op_lazy(A)``
    and only then materialized or compressed.  Thus the intermediate virtual
    bond structure is never expanded into a dense global operator.  With
    ``chi=None`` the lazy target is materialized exactly and no numerical
    truncation is performed.

    Parameters
    ----------
    A, B : Quimb MPO or Pepsy semantic MPO
        Open-boundary MPOs with matching site count and physical dimensions.
        The returned operator represents ``A @ B``.
    chi : int or None, optional
        Final MPO bond cap. ``None`` means exact materialization without
        compression.
    method : str, default="auto"
        Numerical compression method. Supported methods are ``"auto"``,
        ``"direct"``/``"svd"``, ``"dm"``, ``"sdc"``, ``"src"``,
        ``"fit"``, ``"dmrg"``, ``"dmrg2"``, and ``"dmrg3"``. The DMRG
        names use Pepsy's native :class:`FIT` solver with one-, two-, or
        three-site updates; the other names dispatch to Quimb's 1D
        compressor.
    cutoff, cutoff_mode : optional
        Numerical truncation controls. ``"auto"`` resolves the cutoff from
        the input dtype and resolves the mode to ``"rsum2"``. ``cutoff`` is
        ignored for the exact ``chi=None`` materialization.
    sector_aware : {True, False, "auto"}, default="auto"
        Preserve and report native Symmray charge sectors during compression.
        ``True`` rejects a dense product boundary instead of silently falling
        back to ordinary compression.
    guess_method : {"auto", "direct", "dm", "sdc", "sdc-oversample", "src", "src-oversample"}, default="auto"
        Initial rank-``chi`` approximation for the DMRG/FIT methods. ``auto``
        selects deterministic SDC, which is compatible with native Symmray
        sectors. ``src`` and ``src-oversample`` are opt-in randomized warm
        starts for dense MPOs only.
    guess_seed : optional
        Random seed forwarded to an SRC warm start. It has no effect for
        deterministic guess methods and is ignored when ``method`` is not a
        DMRG/FIT method.

    Notes
    -----
    The lazy target keeps backend arrays untouched, so NumPy, Torch, CuPy,
    JAX, and native Symmray data remain at their respective Quimb/Pepsy
    boundaries.  Numerical ``chi`` compression is deliberately separate from
    analytical MPO construction and from symbolic history compression.
    """
    # Import Quimb only when this public compression operation is requested.
    # The rest of the cluster-product module remains usable without importing
    # the optional 1D compression implementation at module import time.
    import quimb.tensor as qtn  # pylint: disable=import-outside-toplevel

    A = _as_quimb_mpo(A)
    B = _as_quimb_mpo(B)
    if int(A.L) != int(B.L):
        raise ValueError(
            f"MPO products require equal lengths, got {A.L} and {B.L}."
        )
    if not isinstance(method, str):
        raise TypeError("method must be a string.")
    requested_method = method
    method = method.strip().lower()
    aliases = {
        "svd": "direct",
        "dmrg1": "dmrg",
        "fit-1": "dmrg",
        "fit-2": "dmrg2",
        "fit-3": "dmrg3",
    }
    method = aliases.get(method, method)
    allowed = {
        "auto",
        "direct",
        "dm",
        "sdc",
        "sdc-oversample",
        "src",
        "src-oversample",
        "fit",
        "dmrg",
        "dmrg2",
        "dmrg3",
    }
    if method not in allowed:
        raise ValueError(
            "unknown MPO product compression method "
            f"{requested_method!r}; expected one of "
            + ", ".join(sorted(allowed))
            + "."
        )
    if chi is not None:
        if not isinstance(chi, Integral) or int(chi) < 1:
            raise ValueError("chi must be a positive integer or None.")
        chi = int(chi)

    if guess_method is None:
        guess_method = "auto"
    if not isinstance(guess_method, str):
        raise TypeError("guess_method must be a string or None.")
    requested_guess_method = guess_method
    guess_method = guess_method.strip().lower()
    guess_aliases = {"svd": "direct", "srcmps": "src"}
    guess_method = guess_aliases.get(guess_method, guess_method)
    allowed_guess_methods = {
        "auto",
        "direct",
        "dm",
        "sdc",
        "sdc-oversample",
        "src",
        "src-oversample",
    }
    if guess_method not in allowed_guess_methods:
        raise ValueError(
            "unknown MPO product DMRG guess method "
            f"{requested_guess_method!r}; expected one of "
            + ", ".join(sorted(allowed_guess_methods))
            + "."
        )
    resolved_guess_method = None
    fit_environment_reuse_count = None

    reference = _mpo_product_reference(A, B)
    resolved_cutoff = _resolve_compression_cutoff(cutoff, reference)
    resolved_cutoff_mode = _resolve_compression_cutoff_mode(cutoff_mode)
    if method == "auto":
        if chi is None:
            resolved_method = "direct"
        elif _has_symmray_data(A, B):
            # Keep native block structure inside FIT.  Generic randomized or
            # successive compressors are useful for dense arrays but should
            # not be selected automatically for symmetry-aware inputs.
            resolved_method = "dmrg2"
        else:
            raw_bonds = _mpo_product_bond_sizes(A)
            raw_bonds_b = _mpo_product_bond_sizes(B)
            raw_max_bond = max(
                (left * right for left, right in zip(raw_bonds, raw_bonds_b)),
                default=1,
            )
            # A mild product is cheaper and usually sufficiently stable with
            # deterministic SDC.  Stronger truncation uses a short variational
            # two-site refinement, initialized from the same lazy target.
            resolved_method = (
                "sdc" if raw_max_bond <= 2 * chi else "dmrg2"
            )
    else:
        resolved_method = method

    target = B.copy().gate_upper_with_op_lazy(A.copy())
    sector_aware_request = _normalize_sector_aware_request(sector_aware)
    target_sector_summary = _native_sector_summary(target)
    sector_aware = _resolve_sector_aware(
        sector_aware_request,
        target_sector_summary,
    )
    if chi is None:
        # max_bond=None and cutoff=0 are the explicit exact-materialization
        # contract.  The caller's cutoff is validated and recorded, but never
        # used to discard a singular value on this branch.
        result = qtn.tensor_network_1d_compress(
            target,
            max_bond=None,
            cutoff=0.0,
            cutoff_mode=resolved_cutoff_mode,
            method="direct",
            inplace=False,
        )
    elif resolved_method in {"dmrg", "dmrg2", "dmrg3", "fit"}:
        from pepsy.fitting import FIT  # pylint: disable=import-outside-toplevel

        block_size = {
            "dmrg": 1,
            "fit": 2,
            "dmrg2": 2,
            "dmrg3": 3,
        }[resolved_method]
        resolved_guess_method = (
            "sdc" if guess_method == "auto" else guess_method
        )
        if (
            target_sector_summary is not None
            and resolved_guess_method.startswith("src")
        ):
            raise NotImplementedError(
                "SRC warm starts are not currently sector-aware for native "
                "Symmray MPOs; use guess_method='sdc' or 'direct'."
            )
        guess_kwargs = {
            "max_bond": chi,
            "method": resolved_guess_method,
            "inplace": False,
        }
        if resolved_guess_method.startswith("src"):
            # SRC is rank-controlled and its base implementation does not
            # accept cutoff_mode on all supported Quimb versions.
            guess_kwargs["cutoff"] = 0.0
            if guess_seed is not None:
                guess_kwargs["seed"] = guess_seed
        else:
            guess_kwargs["cutoff"] = resolved_cutoff
            guess_kwargs["cutoff_mode"] = resolved_cutoff_mode
        # The warm start is disposable. The exact lazy target remains the
        # variational objective passed to FIT, and FIT.run_eff is the only
        # DMRG refinement entry point so its cached sweep environments are
        # reused across the full-chain sweeps.
        guess = qtn.tensor_network_1d_compress(
            target.copy(),
            **guess_kwargs,
        )
        fitter = FIT(
            target,
            p=guess,
            cutoffs=resolved_cutoff,
            copy_target=False,
            inplace=False,
        )
        fitter.run_eff(
            n_iter=4,
            block_size=block_size,
            max_bond=chi,
            cutoff=resolved_cutoff,
            cutoff_mode=resolved_cutoff_mode,
            adaptive_block_sweeps=(
                2 if block_size in {2, 3} else None
            ),
        )
        fit_environment_reuse_count = int(
            getattr(fitter, "_sweep_environment_reuse_count", 0)
        )
        result = fitter.p
    else:
        if target_sector_summary is not None and resolved_method.startswith("src"):
            raise NotImplementedError(
                "SRC compression is not currently available for native "
                "Symmray MPOs because its randomized path is not sector-aware."
            )
        kwargs = {
            "max_bond": chi,
            "method": resolved_method,
            "inplace": False,
        }
        if not resolved_method.startswith("src"):
            kwargs["cutoff"] = resolved_cutoff
            kwargs["cutoff_mode"] = resolved_cutoff_mode
        else:
            # Quimb's SRC method is rank-controlled and ignores non-zero
            # cutoffs. Passing zero explicitly suppresses its advisory warning
            # while the requested/resolved cutoff remains in metadata.
            kwargs["cutoff"] = 0.0
        result = qtn.tensor_network_1d_compress(target, **kwargs)

    final_sector_summary = _native_sector_summary(result)
    if sector_aware and final_sector_summary is None:
        raise RuntimeError(
            "sector-aware MPO product compression lost native Symmray "
            "sector structure."
        )
    # ``tensor_network_1d_compress`` may return a new object and therefore
    # drop arbitrary attributes from the Pepsy source MPO. Retain the local
    # physical charge map so the Pepsy Quimb MPO boundary can restore
    # computational-basis order after a later ``to_dense()`` call.
    source_metadata = next(
        (
            candidate
            for candidate in (A, B)
            if getattr(candidate, "pepsy_mpo_symmetry", None) is not None
        ),
        None,
    )
    if source_metadata is not None:
        result = _ensure_pepsy_mpo_boundary(result)
        result.pepsy_mpo_symmetry = source_metadata.pepsy_mpo_symmetry
        result.pepsy_mpo_physical_charges = (
            source_metadata.pepsy_mpo_physical_charges
        )
        result.pepsy_mpo_physical_dimension = (
            source_metadata.pepsy_mpo_physical_dimension
        )
    final_bonds = _mpo_product_bond_sizes(result)
    metadata = {
        "operation": "compress_mpo_product",
        "ordered_product": "A @ B",
        "requested_method": requested_method,
        "method": resolved_method,
        "chi": chi,
        "cutoff": cutoff,
        "cutoff_resolved": resolved_cutoff,
        "cutoff_mode": cutoff_mode,
        "cutoff_mode_resolved": resolved_cutoff_mode,
        "initial_bond_dimensions_A": _mpo_product_bond_sizes(A),
        "initial_bond_dimensions_B": _mpo_product_bond_sizes(B),
        "final_bond_dimensions": final_bonds,
        "lazy_target": True,
        "exact": chi is None,
        "backend": type(reference).__name__,
        "sector_aware": sector_aware,
        "sector_aware_requested": sector_aware_request,
        "guess_method": (
            resolved_guess_method
            if chi is not None
            and resolved_method in {"dmrg", "dmrg2", "dmrg3", "fit"}
            else None
        ),
        "guess_method_requested": (
            requested_guess_method
            if chi is not None
            and resolved_method in {"dmrg", "dmrg2", "dmrg3", "fit"}
            else None
        ),
        "guess_seed": (
            guess_seed
            if chi is not None
            and resolved_method in {"dmrg", "dmrg2", "dmrg3", "fit"}
            else None
        ),
        "fit_solver": (
            "FIT.run_eff"
            if chi is not None
            and resolved_method in {"dmrg", "dmrg2", "dmrg3", "fit"}
            else None
        ),
        "fit_environment_reuse_count": fit_environment_reuse_count,
        "initial_sector_dimensions": (
            ()
            if target_sector_summary is None
            else target_sector_summary["bond_sector_dimensions"]
        ),
        "final_sector_dimensions": (
            ()
            if final_sector_summary is None
            else final_sector_summary["bond_sector_dimensions"]
        ),
        "initial_sector_block_counts": (
            ()
            if target_sector_summary is None
            else target_sector_summary["site_block_counts"]
        ),
        "final_sector_block_counts": (
            ()
            if final_sector_summary is None
            else final_sector_summary["site_block_counts"]
        ),
    }
    return _attach_mpo_product_metadata(result, metadata)


def _matrix_exponential(matrix):
    """Evaluate a small local matrix exponential on its native backend."""

    backend = _backend_name(matrix)
    if backend == "torch":
        import torch  # pylint: disable=import-outside-toplevel

        return torch.matrix_exp(matrix)
    if backend == "jax":
        import jax.scipy.linalg as jsl  # pylint: disable=import-outside-toplevel

        return jsl.expm(matrix)
    if backend == "cupy":
        from cupyx.scipy.linalg import expm  # pylint: disable=import-outside-toplevel

        return expm(matrix)
    try:
        from scipy.linalg import expm  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "MPO cluster expansion requires scipy for NumPy local matrix "
            "exponentials; install pepsy[solvers]."
        ) from exc
    return expm(np.asarray(matrix))


def _resolve(value, parameters, to_backend=None):
    if isinstance(value, MPOParameter):
        if parameters is None:
            raise ValueError(
                "parameters are required to resolve an MPOParameter in a "
                "cluster expansion."
            )
        value = value.resolve(parameters)
    if callable(value):
        value = value(parameters)
    return _cluster_to_backend(value, to_backend)


def _cluster_to_backend(value, to_backend):
    """Convert a host scalar at the cluster backend boundary."""

    if (
        to_backend is not None
        and _backend_name(value) in {"builtins", "numpy"}
    ):
        try:
            return to_backend(value)
        except (TypeError, ValueError):
            # A real-valued backend converter may intentionally reject a
            # complex scalar such as ``-1j * dt``. Let backend arithmetic
            # promote the scalar alongside the converted tensor blocks; do
            # not turn this common real-operator/complex-step case into an
            # avoidable API failure.
            if np.iscomplexobj(value):
                return value
            raise
    return value


def _select_rank(singular_values, cutoff):
    """Select a fixed structural rank from a local Schmidt spectrum."""

    size = int(singular_values.shape[0])
    if cutoff is None or cutoff == 0.0:
        return size
    values = singular_values
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    try:
        values = np.asarray(values)
    except Exception:  # JAX tracers cannot cross the NumPy boundary.
        # A traced backend needs a static structural rank. The caller can
        # still provide ``max_bond`` to select a smaller fixed rank without
        # making the autodiff graph depend on singular-value comparisons.
        return size
    scale = float(np.max(np.abs(values), initial=0.0))
    if scale == 0.0:
        return 1
    return max(1, int(np.count_nonzero(np.abs(values) > float(cutoff) * scale)))


def _is_zero_operator(operator):
    """Detect an exact structural zero without retaining backend graph data."""

    values = operator
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    try:
        return bool(np.count_nonzero(np.asarray(values)) == 0)
    except (TypeError, ValueError):  # pragma: no cover - tracer guard
        return False


def _jax_stable_factorization(matrix):
    """Return a trace-safe full factorization for a JAX operator block.

    JAX's SVD VJP is undefined at repeated or zero singular values. A small
    deterministic probe chooses a differentiable basis, while the right
    factor is projected from the *unperturbed* matrix. With the full reduced
    rank this remains an exact factorization; ``max_bond`` may subsequently
    select a fixed differentiable truncation.
    """

    rows, columns = int(matrix.shape[0]), int(matrix.shape[1])
    indices = np.arange(1, rows * columns + 1, dtype=float).reshape(rows, columns)
    probe = np.sin(indices) + 0.37 * np.cos(indices * 1.61803398875)
    probe /= max(float(np.linalg.norm(probe)), 1.0)
    probe = _as_backend(probe, like=matrix)
    absolute = ar.do("abs", matrix)
    scale = ar.do(
        "sqrt",
        ar.do("sum", ar.do("multiply", absolute, absolute)),
    )
    perturbation = _multiply_scalar(1.0e-5 * (1.0 + scale), probe)
    left, _singular_values, _right = _fixed_rank_svd(
        ar.do("add", matrix, perturbation)
    )
    right = ar.do("matmul", left.T.conj(), matrix)
    return left, right


def _graph_lattice_from_input(graph, L):
    """Normalize an arbitrary graph input to chain-labelled sites."""

    from .pepo_dense import ClusterLattice  # pylint: disable=import-outside-toplevel

    if isinstance(graph, ClusterLattice):
        sites = tuple(graph.sites)
        edges = tuple(graph.edges)
        name = graph.name
    elif isinstance(graph, Mapping):
        sites = graph.get("sites")
        edges = graph.get("edges")
        name = graph.get("name", "graph")
        if sites is None or edges is None:
            raise ValueError("graph mappings require 'sites' and 'edges'.")
        sites = tuple(sites)
        edges = tuple(edges)
    elif isinstance(graph, (tuple, list)) and len(graph) == 2:
        sites, edges = tuple(graph[0]), tuple(graph[1])
        name = "graph"
    else:
        raise TypeError(
            "graph must be a ClusterLattice, a (sites, edges) pair, or a "
            "mapping with 'sites' and 'edges'."
        )
    if set(sites) != set(range(L)):
        raise ValueError(
            "MPO graph clusters require graph sites labelled by every chain "
            f"site 0..{L - 1}; map coordinate labels through MPOBasis first."
        )
    return ClusterLattice.from_edges(tuple(range(L)), edges, name=name)


def _normalize_graph_assembly(value):
    """Normalize the graph-cluster collection assembly policy."""

    if not isinstance(value, str):
        raise TypeError(
            "graph_assembly must be 'auto', 'exact', or 'bounded'."
        )
    value = value.strip().lower().replace("-", "_")
    if value not in {"auto", "exact", "bounded"}:
        raise ValueError(
            "graph_assembly must be 'auto', 'exact', or 'bounded'."
        )
    return value


def _validate_graph_collection_order(value):
    """Validate the number of non-single graph residuals per collection."""

    if value is None:
        return None
    if (
        not isinstance(value, Integral)
        or isinstance(value, bool)
        or int(value) < 1
    ):
        raise ValueError(
            "max_collection_order must be a positive integer or None."
        )
    return int(value)


def _validate_graph_collection_budget(value):
    """Validate the hard cap used before graph collection materialization."""

    if value is None:
        return None
    if (
        not isinstance(value, Integral)
        or isinstance(value, bool)
        or int(value) < 1
    ):
        raise ValueError(
            "collection_budget must be a positive integer or None."
        )
    return int(value)


def _graph_lattice_for_basis(graph, basis):
    """Map coordinate-labelled graph sites to a basis' MPO chain."""

    from .pepo_dense import ClusterLattice  # pylint: disable=import-outside-toplevel

    if graph is None:
        if basis.lattice_shape is not None and len(basis.lattice_shape) == 2:
            graph = ClusterLattice.square(*basis.lattice_shape)
            chain_to_coordinate = basis.chain_to_lattice or {}
            extra_edges = set()
            for term in basis.terms:
                sites = tuple(term.sites)
                for left, right in zip(sites, sites[1:]):
                    left = chain_to_coordinate.get(left, left)
                    right = chain_to_coordinate.get(right, right)
                    extra_edges.add(frozenset((left, right)))
            if extra_edges:
                existing = {frozenset(edge) for edge in graph.edges}
                graph = ClusterLattice.from_edges(
                    graph.sites,
                    graph.edges
                    + tuple(
                        tuple(edge)
                        for edge in sorted(extra_edges - existing, key=repr)
                    ),
                    name="square+terms",
                )
        else:
            edges = set()
            for term in basis.terms:
                sites = tuple(term.sites)
                for left, right in zip(sites, sites[1:]):
                    edges.add(tuple(sorted((left, right))))
            if not edges and basis.L > 1:
                edges.update((site, site + 1) for site in range(basis.L - 1))
            return ClusterLattice.from_edges(
                tuple(range(basis.L)), tuple(sorted(edges)), name="inferred"
            )

    if isinstance(graph, ClusterLattice):
        sites = tuple(graph.sites)
        edges = tuple(graph.edges)
        name = graph.name
    elif isinstance(graph, Mapping):
        sites = tuple(graph["sites"])
        edges = tuple(graph["edges"])
        name = graph.get("name", "graph")
    elif isinstance(graph, (tuple, list)) and len(graph) == 2:
        sites, edges = tuple(graph[0]), tuple(graph[1])
        name = "graph"
    else:
        raise TypeError(
            "graph must be a ClusterLattice, a (sites, edges) pair, or a "
            "mapping with 'sites' and 'edges'."
        )
    coordinate_to_chain = basis.lattice_to_chain or {}
    mapped_sites = tuple(coordinate_to_chain.get(site, site) for site in sites)
    mapped_edges = tuple(
        (
            coordinate_to_chain.get(source, source),
            coordinate_to_chain.get(target, target),
        )
        for source, target in edges
    )
    return _graph_lattice_from_input(
        ClusterLattice.from_edges(mapped_sites, mapped_edges, name=name),
        basis.L,
    )


def _graph_lattice_from_spec(graph, *, shape, length, basis, cyclic=False):
    """Resolve the compact graph/shape/cyclic facade arguments."""

    from .pepo_dense import ClusterLattice  # pylint: disable=import-outside-toplevel

    if not isinstance(graph, str):
        if isinstance(cyclic, (bool, np.bool_)):
            cyclic_requested = bool(cyclic)
        else:
            cyclic_requested = True
        if cyclic_requested:
            raise ValueError(
                "cyclic is only a shorthand for graph='chain' or "
                "graph='square'; an explicit graph already defines its "
                "periodic edges."
            )
        if basis is None:
            return _graph_lattice_from_input(graph, length)
        return _graph_lattice_for_basis(graph, basis)

    graph_name = graph.strip().lower().replace("-", "_").replace(" ", "_")
    if graph_name == "square":
        lattice_shape = shape
        if lattice_shape is None and basis is not None:
            lattice_shape = basis.lattice_shape
        if isinstance(lattice_shape, Integral) or lattice_shape is None:
            raise ValueError(
                "graph='square' requires a two-dimensional shape=(lx, ly)."
            )
        try:
            lattice_shape = tuple(lattice_shape)
        except TypeError as exc:
            raise TypeError(
                "graph='square' requires a two-dimensional shape=(lx, ly)."
            ) from exc
        if len(lattice_shape) != 2:
            raise ValueError(
                "graph='square' requires a two-dimensional shape=(lx, ly)."
            )
        lattice = ClusterLattice.square(
            lattice_shape[0],
            lattice_shape[1],
            cyclic=cyclic,
        )
    elif graph_name in {"chain", "line"}:
        if not isinstance(cyclic, (bool, np.bool_)):
            raise TypeError(
                "cyclic must be a boolean when graph='chain'."
            )
        chain_length = length if basis is None else basis.L
        edges = [(site, site + 1) for site in range(chain_length - 1)]
        if bool(cyclic) and chain_length > 1:
            edges.append((chain_length - 1, 0))
        lattice = ClusterLattice.from_edges(
            range(chain_length),
            edges,
            name="ring" if cyclic else "chain",
        )
    else:
        raise ValueError(
            "graph must be 'chain', 'square', an explicit ClusterLattice, "
            "a (sites, edges) pair, or a graph mapping."
        )

    if basis is None:
        return _graph_lattice_from_input(lattice, length)
    return _graph_lattice_for_basis(lattice, basis)


def _operator_schmidt(operator, nsites, phys_dim, cutoff, max_bond=None):
    """Return an exact-or-cutoff operator TT decomposition."""

    if _is_zero_operator(operator):
        zero = ar.do("zeros", (1, 1, phys_dim, phys_dim), like=operator)
        return [zero] * nsites
    if nsites == 1:
        return [ar.do("reshape", operator, (1, 1, phys_dim, phys_dim))]

    axes = tuple(index for site in range(nsites) for index in (site, nsites + site))
    tensor = ar.do("reshape", operator, (phys_dim,) * (2 * nsites))
    tensor = ar.do("transpose", tensor, axes)
    tensor = ar.do("reshape", tensor, (phys_dim * phys_dim,) * nsites)

    cores = []
    local_size = phys_dim * phys_dim
    carry = ar.do("reshape", tensor, (1, *([local_size] * nsites)))
    left_rank = 1
    for site in range(nsites - 1):
        remaining = int(np.prod(carry.shape[2:]))
        matrix = ar.do(
            "reshape",
            carry,
            (left_rank * phys_dim * phys_dim, remaining),
        )
        if (
            _backend_name(operator) == "jax"
            and max_bond is None
            and cutoff in (None, 0.0)
        ):
            # A no-truncation JAX path does not need singular vectors. Use an
            # exact identity factorization on the smaller side of the split;
            # unlike the stabilized SVD this introduces no perturbation into
            # the represented operator while preserving autodiff.
            rows, columns = int(matrix.shape[0]), int(matrix.shape[1])
            if rows <= columns:
                rank = rows
                u = _identity(rank, like=matrix)
                carry = matrix
            else:
                rank = columns
                u = matrix
                carry = _identity(rank, like=matrix)
            core = ar.do(
                "reshape",
                u,
                (left_rank, phys_dim, phys_dim, rank),
            )
            cores.append(ar.do("transpose", core, (0, 3, 1, 2)))
            carry = ar.do(
                "reshape",
                carry,
                (rank, *([local_size] * (nsites - site - 1))),
            )
            left_rank = rank
            continue
        if _backend_name(operator) == "jax":
            u, vh = _jax_stable_factorization(matrix)
            rank = int(u.shape[1])
        else:
            u, singular_values, vh = _fixed_rank_svd(matrix)
            rank = _select_rank(singular_values, cutoff)
        if max_bond is not None:
            rank = min(rank, int(max_bond))
        u = u[:, :rank]
        vh = vh[:rank, :]
        core = ar.do(
            "reshape",
            u,
            (left_rank, phys_dim, phys_dim, rank),
        )
        cores.append(ar.do("transpose", core, (0, 3, 1, 2)))
        if _backend_name(operator) == "jax":
            carry = vh
        else:
            carry = ar.do(
                "multiply",
                ar.do("reshape", singular_values[:rank], (rank, 1)),
                vh,
            )
        carry = ar.do(
            "reshape",
            carry,
            (rank, *([local_size] * (nsites - site - 1))),
        )
        left_rank = rank

    cores.append(ar.do("reshape", carry, (left_rank, 1, phys_dim, phys_dim)))
    return cores


@dataclass(frozen=True)
class MPOClusterFactor:
    """One factor ``exp(coefficient * sum(terms))`` in a local product."""

    terms: tuple
    coefficient: object = 1.0

    def __post_init__(self):
        terms = tuple(_term_from_input(term) for term in self.terms)
        if not terms:
            raise ValueError("MPOClusterFactor requires at least one local term.")
        object.__setattr__(self, "terms", terms)

    @classmethod
    def from_mpo_basis(cls, basis, *, coefficient=1.0):
        """Create a factor from an existing parameterized ``MPOBasis``."""

        from .mpo_basis import MPOBasis  # pylint: disable=import-outside-toplevel

        if not isinstance(basis, MPOBasis):
            raise TypeError("basis must be an MPOBasis.")
        return cls(basis.terms, coefficient=coefficient)


@dataclass(frozen=True)
class MPOClusterExpansionReport:
    """Diagnostics for one cluster-basis MPO construction."""

    cluster_size: int
    factor_count: int
    interval_count: int
    residual_ranks: tuple[tuple[tuple[int, int], tuple[int, ...]], ...]
    initial_bond_dimensions: tuple[int, ...]
    cutoff: float | None
    local_svd_truncated: bool
    max_bond: int | None = None
    cluster_mode: str = "interval"
    graph_cluster_count: int = 0
    graph_loop_counts: tuple[int, ...] = ()
    graph_assembly: str = "direct"
    graph_collection_order: int | None = None
    graph_collection_count: int = 0
    graph_collection_budget: int | None = None
    graph_collection_truncated: bool = False
    graph_frontier_width: int = 0

    @property
    def api_info(self):
        """Return the stable cross-family report summary."""
        return OperatorReportInfo(
            family="mpo",
            algorithm="cluster_expansion",
            representation="semantic_mpo",
            order=self.cluster_size,
            factor_count=self.factor_count,
            truncated=self.local_svd_truncated,
        )


class CompiledMPOClusterProduct:
    """Reusable callable for a compiled MPO cluster-expansion topology.

    The wrapper stores only topology-owned data. Every call creates fresh
    backend tensors, so changing Torch/JAX parameters never returns stale
    values or retains an obsolete autodiff graph.
    """

    def __init__(self, basis):
        if not isinstance(basis, MPOClusterProductExpansion):
            raise TypeError("basis must be an MPOClusterProductExpansion.")
        self.basis = basis

    @property
    def cache_info(self):
        """Return topology and evaluation diagnostics."""

        return self.basis.cache_info

    def exp(self, step=1.0, *, parameters=None):
        """Evaluate the compiled ordered cluster product."""

        return self.basis.exp(step, parameters=parameters)

    evaluate = exp
    __call__ = exp


class MPOClusterProductExpansion:
    """Build a finite 1D MPO cluster expansion from local exponential factors.

    ``factors`` are applied in the order supplied, so ``[A, B, C]`` denotes
    ``exp(A) @ exp(B) @ exp(C)``.  Each factor is an
    :class:`MPOClusterFactor`, a sequence of local terms, or a mapping with
    ``terms`` and optional ``coefficient`` keys.  Use
    :meth:`from_local_terms` for the common single-Hamiltonian case.

    The local exact products are formed only on contiguous intervals up to
    ``cluster_size``.  Connected residuals are then assembled as disjoint
    interval paths, so the returned MPO remains size-extensive instead of
    becoming a global Taylor or dense-matrix construction. ``cutoff`` selects
    local operator-Schmidt ranks and ``max_bond`` applies an explicit fixed
    cap. Use :meth:`compile_exp` for repeated ordered products; it caches only
    interval/factor schedules and never numerical autodiff values.

    For graph inputs, ``graph_assembly`` controls the additional collection
    expansion caused by crossing or nested graph clusters in the MPO ordering.
    The default ``"auto"`` policy is exact for small plans and falls back to a
    reported one-cluster approximation when its finite budget is exceeded.
    """

    def __init__(
        self,
        L,
        factors,
        *,
        phys_dim=None,
        cluster_size=2,
        cutoff=1.0e-12,
        max_bond=None,
        graph=None,
        to_backend=None,
        graph_assembly="auto",
        max_collection_order=None,
        collection_budget=128,
    ):
        if not isinstance(L, Integral) or isinstance(L, bool) or int(L) < 1:
            raise ValueError("L must be a positive integer.")
        if to_backend is not None and not callable(to_backend):
            raise TypeError("to_backend must be callable or None.")
        graph_assembly = _normalize_graph_assembly(graph_assembly)
        max_collection_order = _validate_graph_collection_order(
            max_collection_order
        )
        collection_budget = _validate_graph_collection_budget(collection_budget)
        self.L = int(L)
        if not isinstance(cluster_size, Integral) or isinstance(cluster_size, bool):
            raise TypeError("cluster_size must be a positive integer.")
        self.cluster_size = min(self.L, int(cluster_size))
        if self.cluster_size < 1:
            raise ValueError("cluster_size must be positive.")
        if cutoff is not None and float(cutoff) < 0.0:
            raise ValueError("cutoff must be non-negative or None.")
        self.cutoff = None if cutoff is None else float(cutoff)
        if max_bond is not None:
            if (
                not isinstance(max_bond, Integral)
                or isinstance(max_bond, bool)
                or int(max_bond) < 1
            ):
                raise ValueError("max_bond must be a positive integer or None.")
            max_bond = int(max_bond)
        self.max_bond = max_bond
        self.to_backend = to_backend
        self.graph_assembly = graph_assembly
        self.max_collection_order = max_collection_order
        self.collection_budget = collection_budget
        self.graph = None if graph is None else _graph_lattice_from_input(graph, self.L)
        self.cluster_mode = "graph" if self.graph is not None else "interval"
        if self.graph is None and max_collection_order is not None:
            raise ValueError(
                "max_collection_order is only valid for graph cluster assembly."
            )
        if graph_assembly == "exact" and max_collection_order is not None:
            raise ValueError(
                "max_collection_order cannot be combined with "
                "graph_assembly='exact'."
            )
        self._graph_auto_warned = False
        self.factors = tuple(self._normalize_factor(factor) for factor in factors)
        if not self.factors:
            raise ValueError("at least one MPO cluster factor is required.")
        if phys_dim is None:
            first = self.factors[0].terms[0]
            phys_dim = (
                first.phys_dim
                if isinstance(first, MPOLocalOperatorTerm)
                else int(first.operators[0].shape[0])
            )
        if not isinstance(phys_dim, Integral) or isinstance(phys_dim, bool):
            raise TypeError("phys_dim must be a positive integer.")
        self.phys_dim = int(phys_dim)
        if self.phys_dim < 1:
            raise ValueError("phys_dim must be positive.")
        for factor in self.factors:
            for term in factor.terms:
                if any(site < 0 or site >= self.L for site in term.sites):
                    raise ValueError("a cluster term site is outside the chain.")
                if (
                    isinstance(term, MPOProductTerm)
                    and term.string_operators is not None
                ):
                    gap_count = sum(
                        right - left - 1
                        for left, right in zip(term.sites, term.sites[1:])
                    )
                    if len(term.string_operators) != gap_count:
                        raise ValueError(
                            "string_operators must have one operator for each "
                            f"gap, got {len(term.string_operators)} for "
                            f"{gap_count} gaps."
                        )
                term_dim = (
                    term.phys_dim
                    if isinstance(term, MPOLocalOperatorTerm)
                    else int(term.operators[0].shape[0])
                )
                if term_dim != self.phys_dim:
                    raise ValueError("all local terms must use the same phys_dim.")
                if self.graph is None:
                    span = max(term.sites) - min(term.sites) + 1
                    if span > self.cluster_size:
                        raise ValueError(
                            "cluster_size must include the full span of every "
                            "local term."
                        )
                    if (
                        isinstance(term, MPOLocalOperatorTerm)
                        and tuple(term.sites)
                        != tuple(range(min(term.sites), max(term.sites) + 1))
                    ):
                        raise ValueError(
                            "non-factorized local operator terms must have "
                            "contiguous support."
                        )

        if self.graph is None:
            self._intervals = tuple(
                (start, start + length - 1)
                for length in range(1, self.cluster_size + 1)
                for start in range(self.L - length + 1)
            )
            # Compile topology once. Backend-connected matrices are rebuilt
            # per evaluation; immutable NumPy embeddings are copied into a
            # safe cache because they are the common optimization-loop case.
            self._interval_factor_terms = {
                interval: tuple(
                    tuple(
                        term
                        for term in factor.terms
                        if all(
                            interval[0] <= site <= interval[1]
                            for site in term.sites
                        )
                    )
                    for factor in self.factors
                )
                for interval in self._intervals
            }
            self._interval_static_matrices = {
                interval: tuple(
                    tuple(
                        self._maybe_static_term_matrix(term, *interval)
                        for term in terms
                    )
                    for terms in factor_terms
                )
                for interval, factor_terms in self._interval_factor_terms.items()
            }
            self._static_matrix_count = sum(
                matrix is not None
                for factor_matrices in self._interval_static_matrices.values()
                for matrices in factor_matrices
                for matrix in matrices
            )
            self._interval_splits = {
                (start, end): tuple(
                    ((start, split), (split + 1, end))
                    for split in range(start, end)
                )
                for start, end in self._intervals
            }
            self._graph_clusters = ()
            self._graph_factor_terms = {}
            self._graph_static_matrices = {}
            self._graph_partitions = {}
            self._graph_loop_counts = ()
        else:
            if self.cluster_size > len(self.graph.sites):
                self.cluster_size = len(self.graph.sites)
            self._intervals = ()
            self._interval_factor_terms = {}
            self._interval_static_matrices = {}
            self._interval_splits = {}
            self._graph_adjacency = {
                site: set(neighbor for neighbor, _ in links)
                for site, links in self.graph.adjacency.items()
            }
            shapes = self.graph.connected_cluster_shapes(self.cluster_size)
            self._graph_clusters = tuple(
                tuple(sorted(shape.sites))
                for shape in shapes
            )
            self._graph_loop_counts = tuple(int(shape.loops) for shape in shapes)
            self._graph_factor_terms = {
                cluster: tuple(
                    tuple(
                        term
                        for term in factor.terms
                        if set(term.sites).issubset(cluster)
                    )
                    for factor in self.factors
                )
                for cluster in self._graph_clusters
            }
            missing_terms = [
                term
                for factor in self.factors
                for term in factor.terms
                if not any(set(term.sites).issubset(cluster) for cluster in self._graph_clusters)
            ]
            if missing_terms:
                raise ValueError(
                    "cluster_size is too small to contain every graph interaction "
                    "support."
                )
            self._graph_static_matrices = {
                cluster: tuple(
                    tuple(
                        self._maybe_static_graph_term_matrix(term, cluster)
                        for term in terms
                    )
                    for terms in factor_terms
                )
                for cluster, factor_terms in self._graph_factor_terms.items()
            }
            self._static_matrix_count = sum(
                matrix is not None
                for factor_matrices in self._graph_static_matrices.values()
                for matrices in factor_matrices
                for matrix in matrices
            )
            self._graph_partitions = {
                cluster: self._connected_partitions(cluster)
                for cluster in self._graph_clusters
            }
        self._build_count = 0
        self._last_report = None

    @staticmethod
    def _normalize_factor(factor):
        if isinstance(factor, MPOClusterFactor):
            return factor
        if isinstance(factor, Mapping):
            terms = factor.get("terms")
            if terms is None:
                raise ValueError("cluster factor mappings require 'terms'.")
            return MPOClusterFactor(terms, factor.get("coefficient", 1.0))
        if hasattr(factor, "terms") and hasattr(factor, "L"):
            return MPOClusterFactor(factor.terms)
        if isinstance(factor, MPOClusterProductExpansion):
            raise TypeError("nested MPOClusterProductExpansion factors are not supported.")
        return MPOClusterFactor(tuple(factor))

    @classmethod
    def from_local_terms(cls, L, terms, **kwargs):
        """Construct a single-factor ``exp(step * sum(local terms))`` basis."""

        return cls(L, (MPOClusterFactor(tuple(terms)),), **kwargs)

    @classmethod
    def from_factors(cls, L, factors, **kwargs):
        """Construct an ordered ``exp(A) exp(B) ...`` cluster basis."""

        return cls(L, factors, **kwargs)

    @classmethod
    def from_graph(cls, L, graph, factors, **kwargs):
        """Construct a graph-aware ordered cluster basis.

        Graph sites must be labelled by MPO chain positions. Use
        :meth:`MPOBasis.compile_graph_cluster_expansion` when the graph is
        specified in two-dimensional coordinate labels.
        """

        return cls(L, factors, graph=graph, **kwargs)

    @classmethod
    def from_mpo_bases(cls, bases, *, coefficients=None, **kwargs):
        """Construct ordered factors from reusable :class:`MPOBasis` objects.

        ``coefficients`` supplies the scalar multiplying each basis generator;
        the local term coefficients inside each basis remain intact. This is
        the convenient repeated-evaluation API for
        ``exp(A) @ exp(B) @ exp(C)`` when each factor already has a compiled
        MPO term basis.
        """

        from .mpo_basis import MPOBasis  # pylint: disable=import-outside-toplevel

        bases = tuple(bases)
        if not bases:
            raise ValueError("bases must contain at least one MPOBasis.")
        if not all(isinstance(basis, MPOBasis) for basis in bases):
            raise TypeError("bases must contain only MPOBasis objects.")
        reference = bases[0]
        if any(basis.L != reference.L for basis in bases[1:]):
            raise ValueError("all MPO bases must have matching chain lengths.")
        if any(basis.phys_dim != reference.phys_dim for basis in bases[1:]):
            raise ValueError("all MPO bases must have matching physical dimensions.")
        if coefficients is None:
            coefficients = (1.0,) * len(bases)
        else:
            coefficients = tuple(coefficients)
            if len(coefficients) != len(bases):
                raise ValueError("coefficients must align with bases.")
        factors = tuple(
            MPOClusterFactor.from_mpo_basis(basis, coefficient=coefficient)
            for basis, coefficient in zip(bases, coefficients)
        )
        if kwargs.get("graph") is not None:
            kwargs["graph"] = _graph_lattice_for_basis(kwargs["graph"], bases[0])
        return cls.from_factors(bases[0].L, factors, **kwargs)

    def compile_exp(self):
        """Return a reusable callable over this topology-only expansion."""

        return CompiledMPOClusterProduct(self)

    @property
    def cache_info(self):
        """Return topology-only compilation and evaluation diagnostics."""

        return {
            "compiled": True,
            "builds": self._build_count,
            "interval_count": len(self._intervals),
            "cluster_mode": self.cluster_mode,
            "graph_cluster_count": len(self._graph_clusters),
            "graph_edge_count": 0 if self.graph is None else len(self.graph.edges),
            "cluster_size": self.cluster_size,
            "factor_count": len(self.factors),
            "max_bond": self.max_bond,
            "cutoff": self.cutoff,
            "graph_assembly": self.graph_assembly,
            "max_collection_order": self.max_collection_order,
            "collection_budget": self.collection_budget,
            "graph_frontier_width": (
                0 if self.graph is None else self._graph_frontier_width()
            ),
            "static_matrix_count": self._static_matrix_count,
        }

    @classmethod
    def from_mpo_basis(cls, basis, **kwargs):
        """Reuse the local terms and coefficient references of an ``MPOBasis``."""

        from .mpo_basis import MPOBasis  # pylint: disable=import-outside-toplevel

        if not isinstance(basis, MPOBasis):
            raise TypeError("basis must be an MPOBasis.")
        kwargs.setdefault("phys_dim", basis.phys_dim)
        if "graph" in kwargs and kwargs["graph"] is not None:
            kwargs["graph"] = _graph_lattice_for_basis(kwargs["graph"], basis)
        return cls.from_local_terms(basis.L, basis.terms, **kwargs)

    @property
    def last_report(self):
        """Return diagnostics from the most recent :meth:`exp` call."""

        return self._last_report

    def _term_matrix(self, term, start, end):
        sites = tuple(range(start, end + 1))
        reference = None
        if isinstance(term, MPOProductTerm):
            factors = {site: operator for site, operator in zip(term.sites, term.operators)}
            string_factors = {}
            if term.string_operators is not None:
                string_index = 0
                for left, right in zip(term.sites, term.sites[1:]):
                    for gap_site in range(left + 1, right):
                        string_factors[gap_site] = term.string_operators[string_index]
                        string_index += 1
            reference = _backend_reference(
                (*term.operators, *(term.string_operators or ()))
            )
            matrices = []
            for site in sites:
                operator = factors.get(site)
                if operator is None:
                    operator = string_factors.get(site)
                if operator is None:
                    operator = np.eye(self.phys_dim)
                matrices.append(operator)
            matrices = [
                _as_backend(matrix, like=reference)
                for matrix in matrices
            ]
            return _kron_all(matrices)

        if not isinstance(term, MPOLocalOperatorTerm):
            raise TypeError("unsupported cluster term type.")
        operator = term.operator
        support_start = min(term.sites)
        support_end = max(term.sites)
        left = np.eye(self.phys_dim ** (support_start - start))
        right = np.eye(self.phys_dim ** (end - support_end))
        left = _as_backend(left, like=operator)
        right = _as_backend(right, like=operator)
        return _kron_all((_kron(left, operator), right))

    def _graph_term_matrix(self, term, cluster):
        """Embed one local term into a graph cluster's site ordering."""

        if isinstance(term, MPOProductTerm):
            if term.string_operators is not None:
                raise NotImplementedError(
                    "graph cluster expansion does not yet support explicit "
                    "fermionic string operators; include their path in the graph."
                )
            operator = _kron_all(
                tuple(
                    _as_backend(value, like=term.operators[0])
                    for value in term.operators
                )
            )
        elif isinstance(term, MPOLocalOperatorTerm):
            operator = term.operator
        else:
            raise TypeError("unsupported graph cluster term type.")
        positions = tuple(cluster.index(site) for site in term.sites)
        return _embed_matrix_on_positions(
            operator,
            positions,
            len(cluster),
            self.phys_dim,
        )

    def _maybe_static_graph_term_matrix(self, term, cluster):
        """Cache a NumPy graph embedding when it carries no autodiff values."""

        operators = (
            (term.operator,)
            if isinstance(term, MPOLocalOperatorTerm)
            else term.operators
        )
        if not all(_backend_name(operator) in {"builtins", "numpy"} for operator in operators):
            return None
        try:
            matrix = self._graph_term_matrix(term, cluster)
        except (TypeError, ValueError, NotImplementedError):
            return None
        if _backend_name(matrix) not in {"builtins", "numpy"}:
            return None
        return np.array(matrix, copy=True)

    def _connected_partitions(self, cluster):
        """Return proper partitions whose blocks are graph-connected."""

        connected = []
        for partition in _set_partitions(cluster):
            if len(partition) == 1:
                continue
            if all(self._is_graph_connected(block) for block in partition):
                connected.append(tuple(tuple(block) for block in partition))
        return tuple(connected)

    def _is_graph_connected(self, sites):
        sites = set(sites)
        if len(sites) <= 1:
            return True
        reached = {next(iter(sites))}
        frontier = tuple(reached)
        while frontier:
            frontier = tuple(
                neighbor
                for site in frontier
                for neighbor in self._graph_adjacency[site]
                if neighbor in sites and neighbor not in reached
            )
            reached.update(frontier)
        return reached == sites

    def _maybe_static_term_matrix(self, term, start, end):
        """Cache a copied NumPy embedding when it cannot carry autodiff data."""

        operators = (
            (term.operator,)
            if isinstance(term, MPOLocalOperatorTerm)
            else term.operators
        )
        if not all(_backend_name(operator) in {"builtins", "numpy"} for operator in operators):
            return None
        try:
            matrix = self._term_matrix(term, start, end)
        except (TypeError, ValueError):
            return None
        if _backend_name(matrix) not in {"builtins", "numpy"}:
            return None
        return np.array(matrix, copy=True)

    def _local_exponential(self, start, end, step, parameters):
        dimension = self.phys_dim ** (end - start + 1)
        references = [step]
        interval_factors = self._interval_factor_terms[(start, end)]
        static_factors = self._interval_static_matrices[(start, end)]
        active = tuple(
            index for index, terms in enumerate(interval_factors) if terms
        )
        if not active:
            return _identity(dimension, like=_backend_reference(references))
        for index in active:
            factor = self.factors[index]
            terms = interval_factors[index]
            references.append(
                _resolve(factor.coefficient, parameters, self.to_backend)
            )
            references.extend(
                _resolve(term.coefficient, parameters, self.to_backend)
                for term in terms
            )
            references.extend(
                term.operator if isinstance(term, MPOLocalOperatorTerm)
                else term.operators[0]
                for term in terms
            )
        reference = _backend_reference(references)
        local_exponentials = []
        for index in active:
            factor = self.factors[index]
            terms = interval_factors[index]
            static_terms = static_factors[index]
            generator = ar.do("zeros", (dimension, dimension), like=reference)
            for term, static in zip(terms, static_terms):
                local = static if static is not None else self._term_matrix(term, start, end)
                local = _as_backend(local, like=reference)
                generator = ar.do(
                    "add",
                    generator,
                    _multiply_scalar(
                        _resolve(term.coefficient, parameters, self.to_backend),
                        local,
                    ),
                )
            exponent = _multiply_scalar(
                _resolve(factor.coefficient, parameters, self.to_backend),
                _multiply_scalar(step, generator),
            )
            local_exponentials.append(_matrix_exponential(exponent))
        if len(local_exponentials) == 1:
            return local_exponentials[0]
        total = _identity(dimension, like=reference)
        for local_exponential in local_exponentials:
            total = ar.do("matmul", total, local_exponential)
        return total

    def _graph_local_exponential(self, cluster, step, parameters):
        """Evaluate the ordered local product on one graph cluster."""

        dimension = self.phys_dim ** len(cluster)
        references = [step]
        factor_terms = self._graph_factor_terms[cluster]
        active = tuple(index for index, terms in enumerate(factor_terms) if terms)
        if not active:
            return _identity(dimension, like=_backend_reference(references))
        for index in active:
            factor = self.factors[index]
            terms = factor_terms[index]
            references.append(
                _resolve(factor.coefficient, parameters, self.to_backend)
            )
            references.extend(
                _resolve(term.coefficient, parameters, self.to_backend)
                for term in terms
            )
            references.extend(
                term.operator if isinstance(term, MPOLocalOperatorTerm)
                else term.operators[0]
                for term in terms
            )
        reference = _backend_reference(references)
        local_exponentials = []
        for index in active:
            factor = self.factors[index]
            terms = factor_terms[index]
            static_terms = self._graph_static_matrices[cluster][index]
            generator = ar.do("zeros", (dimension, dimension), like=reference)
            for term, static in zip(terms, static_terms):
                local = static if static is not None else self._graph_term_matrix(term, cluster)
                local = _as_backend(local, like=reference)
                generator = ar.do(
                    "add",
                    generator,
                    _multiply_scalar(
                        _resolve(term.coefficient, parameters, self.to_backend),
                        local,
                    ),
                )
            exponent = _multiply_scalar(
                _resolve(factor.coefficient, parameters, self.to_backend),
                _multiply_scalar(step, generator),
            )
            local_exponentials.append(_matrix_exponential(exponent))
        if len(local_exponentials) == 1:
            return local_exponentials[0]
        total = _identity(dimension, like=reference)
        for local_exponential in local_exponentials:
            total = ar.do("matmul", total, local_exponential)
        return total

    def _graph_residuals(self, step, parameters):
        """Compute connected residuals using graph-connected partitions."""

        products = {}
        residuals = {}
        for cluster in self._graph_clusters:
            products[cluster] = self._graph_local_exponential(cluster, step, parameters)
            residual = products[cluster]
            for partition in self._graph_partitions[cluster]:
                contribution = _identity(
                    self.phys_dim ** len(cluster),
                    like=residual,
                )
                for block in partition:
                    positions = tuple(cluster.index(site) for site in block)
                    embedded = _embed_matrix_on_positions(
                        residuals[block],
                        positions,
                        len(cluster),
                        self.phys_dim,
                    )
                    contribution = ar.do("matmul", contribution, embedded)
                residual = ar.do("subtract", residual, contribution)
            residuals[cluster] = residual
        return residuals

    def _graph_span_cores(
        self,
        cluster,
        residual,
        *,
        residuals=None,
        include_background=False,
    ):
        """Embed a graph residual without densifying its MPO chain span."""

        cluster = tuple(cluster)
        cluster_cores = _operator_schmidt(
            residual,
            len(cluster),
            self.phys_dim,
            self.cutoff,
            self.max_bond,
        )

        def gap_core(left_rank, right_rank, operator):
            if left_rank != right_rank:
                raise ValueError(
                    "graph residual factorization has incompatible virtual "
                    "ranks across a chain gap."
                )
            operator = _as_backend(operator, like=residual)
            residual_dtype = getattr(residual, "dtype", None)
            if (
                residual_dtype is not None
                and getattr(operator, "dtype", None) != residual_dtype
            ):
                operator = ar.do("astype", operator, residual_dtype)
            rank = int(left_rank)
            array = ar.do(
                "zeros",
                (rank, rank, self.phys_dim, self.phys_dim),
                like=residual,
            )
            values = ar.do("stack", (operator,) * rank, axis=0)
            return _scatter_add_2d(
                array,
                np.arange(rank, dtype=int),
                np.arange(rank, dtype=int),
                values,
            )

        span_cores = []
        for index, site in enumerate(cluster):
            if index:
                left_rank = int(cluster_cores[index - 1].shape[1])
                right_rank = int(cluster_cores[index].shape[0])
                for gap_site in range(cluster[index - 1] + 1, site):
                    if include_background:
                        if residuals is None:
                            raise ValueError(
                                "residuals are required when graph-span "
                                "background cores are requested."
                            )
                        operator = _as_backend(
                            residuals[(gap_site,)],
                            like=residual,
                        )
                    else:
                        operator = _identity(self.phys_dim, like=residual)
                    span_cores.append(gap_core(left_rank, right_rank, operator))
            span_cores.append(cluster_cores[index])
        return tuple(span_cores)

    def _residuals(self, step, parameters):
        if self.graph is not None:
            return self._graph_residuals(step, parameters)
        products = {}
        residuals = {}
        for interval in self._intervals:
            start, end = interval
            products[interval] = self._local_exponential(start, end, step, parameters)
            residual = products[interval]
            for left_interval, right_interval in self._interval_splits[interval]:
                residual = ar.do(
                    "subtract",
                    residual,
                    _kron(residuals[left_interval], products[right_interval]),
                )
            residuals[interval] = residual
        return residuals

    def _assemble_graph(self, residuals):
        """Assemble graph residuals into a finite open-chain MPO."""

        cores = {}
        for cluster, residual in residuals.items():
            if len(cluster) == 1:
                continue
            cores[cluster] = self._graph_span_cores(
                cluster,
                residual,
                residuals=residuals,
                include_background=True,
            )

        state_lists = [[("rail",)]]
        for cut in range(1, self.L):
            states = [("rail",)]
            for cluster, cluster_cores in cores.items():
                start, end = min(cluster), max(cluster)
                if start < cut <= end:
                    for position in range(int(cluster_cores[cut - start - 1].shape[1])):
                        states.append((cluster, position))
            state_lists.append(states)
        state_lists.append([("rail",)])

        reference = _backend_reference(tuple(residuals.values()))
        arrays = []
        for site in range(self.L):
            left_states = state_lists[site]
            right_states = state_lists[site + 1]
            array = ar.do(
                "zeros",
                (len(left_states), len(right_states), self.phys_dim, self.phys_dim),
                like=reference,
            )
            left_index = {state: index for index, state in enumerate(left_states)}
            right_index = {state: index for index, state in enumerate(right_states)}
            rows = [left_index[("rail",)]]
            columns = [right_index[("rail",)]]
            values = [residuals[(site,)]]
            for cluster, cluster_cores in cores.items():
                start, end = min(cluster), max(cluster)
                if site < start or site > end:
                    continue
                local = cluster_cores[site - start]
                if site == start:
                    for right in range(local.shape[1]):
                        rows.append(left_index[("rail",)])
                        columns.append(right_index[(cluster, right)])
                        values.append(local[0, right])
                elif site == end:
                    for left in range(local.shape[0]):
                        rows.append(left_index[(cluster, left)])
                        columns.append(right_index[("rail",)])
                        values.append(local[left, 0])
                else:
                    for left in range(local.shape[0]):
                        for right in range(local.shape[1]):
                            rows.append(left_index[(cluster, left)])
                            columns.append(right_index[(cluster, right)])
                            values.append(local[left, right])
            values = ar.do("stack", tuple(values), axis=0)
            array = _scatter_add_2d(
                array,
                np.asarray(rows, dtype=int),
                np.asarray(columns, dtype=int),
                values,
            )
            arrays.append(array)
        return tuple(arrays), state_lists, cores

    def _graph_needs_collection_assembly(self):
        """Whether disjoint graph clusters have overlapping chain spans."""

        clusters = tuple(
            cluster for cluster in self._graph_clusters if len(cluster) > 1
        )
        for index, left in enumerate(clusters):
            left_sites = set(left)
            left_start, left_end = min(left), max(left)
            for right in clusters[index + 1 :]:
                if left_sites.intersection(right):
                    continue
                right_start, right_end = min(right), max(right)
                if max(left_start, right_start) <= min(left_end, right_end):
                    # A single path channel cannot represent a product of
                    # disjoint graph clusters whose chain intervals cross or
                    # nest.  The collection assembler below gives every
                    # compatible cluster collection its own tensor-product
                    # virtual path.
                    return True
        return False

    def _bounded_graph_cluster_collections(
        self,
        *,
        max_collection_order=None,
        budget=None,
    ):
        """Enumerate graph collections up to explicit safety limits.

        The returned boolean is true when another collection would have been
        emitted after ``budget`` was reached. This lets the ``auto`` policy
        inspect a plan without ever constructing the complete collection list.
        """

        clusters = tuple(
            cluster for cluster in self._graph_clusters if len(cluster) > 1
        )
        collections = []
        truncated = False

        def visit(start, occupied, chosen):
            nonlocal truncated
            for index in range(start, len(clusters)):
                cluster = clusters[index]
                if occupied.intersection(cluster):
                    continue
                updated = chosen + (cluster,)
                if (
                    max_collection_order is not None
                    and len(updated) > max_collection_order
                ):
                    continue
                if budget is not None and len(collections) >= budget:
                    truncated = True
                    return
                collections.append(updated)
                if (
                    max_collection_order is None
                    or len(updated) < max_collection_order
                ):
                    visit(index + 1, occupied.union(cluster), updated)
                if truncated:
                    return

        visit(0, set(), ())
        return tuple(collections), truncated

    def _graph_cluster_collections(self):
        """Enumerate all non-empty disjoint graph-cluster collections.

        This unbounded compatibility helper is retained for diagnostics. The
        public graph assembly path uses the bounded planner below instead of
        calling it implicitly.
        """

        collections, _truncated = self._bounded_graph_cluster_collections()
        return collections

    def _graph_frontier_width(self):
        """Return the graph-cluster cutwidth in the MPO ordering."""

        clusters = tuple(
            cluster for cluster in self._graph_clusters if len(cluster) > 1
        )
        return max(
            (
                sum(min(cluster) < cut <= max(cluster) for cluster in clusters)
                for cut in range(1, self.L)
            ),
            default=0,
        )

    def _graph_collection_plan(self):
        """Choose a safe exact or bounded graph assembly plan.

        Exact collection assembly is useful for small custom graphs, but its
        collection count is a hard scalability boundary for 2D MPO orderings.
        ``auto`` probes only up to the configured budget and falls back to the
        explicit one-cluster approximation when that boundary is crossed.
        """

        if self.graph is None or not self._graph_needs_collection_assembly():
            return {
                "strategy": "direct",
                "collections": (),
                "collection_order": 1,
                "collection_count": 0,
                "collection_truncated": False,
            }

        if self.graph_assembly == "auto" and self.collection_budget is None:
            raise ValueError(
                "graph_assembly='auto' requires a finite collection_budget; "
                "use graph_assembly='exact' or 'bounded' when disabling "
                "the safety limit explicitly."
            )

        if self.graph_assembly == "bounded":
            collection_order = self.max_collection_order or 1
            if collection_order == 1:
                return {
                    "strategy": "bounded",
                    "collections": (),
                    "collection_order": 1,
                    "collection_count": 0,
                    "collection_truncated": True,
                }
            collections, truncated = self._bounded_graph_cluster_collections(
                max_collection_order=collection_order,
                budget=self.collection_budget,
            )
            if truncated:
                raise ValueError(
                    "bounded graph MPO assembly exceeded collection_budget="
                    f"{self.collection_budget}; reduce max_collection_order "
                    "or increase collection_budget explicitly."
                )
            return {
                "strategy": "bounded",
                "collections": collections,
                "collection_order": collection_order,
                "collection_count": len(collections),
                "collection_truncated": True,
            }

        collections, truncated = self._bounded_graph_cluster_collections(
            budget=self.collection_budget,
        )
        if self.graph_assembly == "exact":
            if truncated:
                raise ValueError(
                    "exact graph MPO assembly exceeds collection_budget="
                    f"{self.collection_budget}; use graph_assembly='bounded' "
                    "for a controlled approximation or set "
                    "collection_budget=None explicitly."
                )
            return {
                "strategy": "exact",
                "collections": collections,
                "collection_order": None,
                "collection_count": len(collections),
                "collection_truncated": False,
            }

        if not truncated:
            return {
                "strategy": "exact",
                "collections": collections,
                "collection_order": None,
                "collection_count": len(collections),
                "collection_truncated": False,
            }

        if not self._graph_auto_warned:
            warnings.warn(
                "graph MPO cluster collection assembly exceeded "
                f"collection_budget={self.collection_budget}; using the "
                "bounded one-cluster approximation. Pass "
                "graph_assembly='exact' to request the full collection plan "
                "or graph_assembly='bounded' with max_collection_order to "
                "choose the approximation explicitly.",
                RuntimeWarning,
                stacklevel=3,
            )
            self._graph_auto_warned = True
        return {
            "strategy": "bounded",
            "collections": (),
            "collection_order": 1,
            "collection_count": 0,
            "collection_truncated": True,
        }

    @staticmethod
    def _multiply_mpo_cores(left, right):
        """Multiply two local MPO cores without scalar backend indexing."""

        product = ar.do("tensordot", left, right, axes=([3], [2]))
        product = ar.do("transpose", product, (0, 3, 1, 4, 2, 5))
        return ar.do(
            "reshape",
            product,
            (
                int(left.shape[0]) * int(right.shape[0]),
                int(left.shape[1]) * int(right.shape[1]),
                int(left.shape[2]),
                int(right.shape[3]),
            ),
        )

    def _assemble_graph_collections(self, residuals, *, collections=None):
        """Assemble crossing/nested graph-cluster products exactly.

        The ordinary graph assembly is a direct sum of one cluster path at a
        time.  That is sufficient while disjoint graph clusters have disjoint
        chain spans, but it drops products such as ``K_(0,3) K_(1,2)`` when
        long-range edges are nested in the MPO ordering.  Here each compatible
        collection receives a tensor-product path, so those partition terms
        remain local in the graph expansion without forming the global dense
        operator.
        """

        pure_cores = {}
        for cluster, residual in residuals.items():
            if len(cluster) == 1:
                continue
            pure_cores[cluster] = self._graph_span_cores(
                cluster,
                residual,
            )

        if collections is None:
            collections = self._graph_cluster_collections()
        collection_cores = []
        for collection in collections:
            occupied = {
                site
                for cluster in collection
                for site in cluster
            }
            local_cores = []
            for site in range(self.L):
                factors = []
                for cluster in collection:
                    if min(cluster) <= site <= max(cluster):
                        factors.append(pure_cores[cluster][site - min(cluster)])
                if site not in occupied:
                    factors.append(
                        ar.do(
                            "reshape",
                            residuals[(site,)],
                            (1, 1, self.phys_dim, self.phys_dim),
                        )
                    )
                local = factors[0]
                for factor in factors[1:]:
                    local = self._multiply_mpo_cores(local, factor)
                local_cores.append(local)
            collection_cores.append((collection, tuple(local_cores)))

        state_lists = [[("rail",)]]
        for cut in range(1, self.L):
            states = [("rail",)]
            for collection_index, (_collection, local_cores) in enumerate(
                collection_cores
            ):
                for position in range(int(local_cores[cut - 1].shape[1])):
                    states.append((collection_index, position))
            state_lists.append(states)
        state_lists.append([("rail",)])

        reference = _backend_reference(tuple(residuals.values()))
        arrays = []
        for site in range(self.L):
            left_states = state_lists[site]
            right_states = state_lists[site + 1]
            array = ar.do(
                "zeros",
                (len(left_states), len(right_states), self.phys_dim, self.phys_dim),
                like=reference,
            )
            left_index = {state: index for index, state in enumerate(left_states)}
            right_index = {state: index for index, state in enumerate(right_states)}
            rows = [np.asarray([left_index[("rail",)]], dtype=int)]
            columns = [np.asarray([right_index[("rail",)]], dtype=int)]
            values = [
                ar.do(
                    "reshape",
                    residuals[(site,)],
                    (1, self.phys_dim, self.phys_dim),
                )
            ]
            for collection_index, (_collection, local_cores) in enumerate(
                collection_cores
            ):
                local = local_cores[site]
                if site == 0:
                    right_count = int(local.shape[1])
                    rows.append(
                        np.full(right_count, left_index[("rail",)], dtype=int)
                    )
                    columns.append(
                        np.asarray(
                            [
                                right_index[(collection_index, right)]
                                for right in range(right_count)
                            ],
                            dtype=int,
                        )
                    )
                    values.append(local[0])
                elif site == self.L - 1:
                    left_count = int(local.shape[0])
                    rows.append(
                        np.asarray(
                            [
                                left_index[(collection_index, left)]
                                for left in range(left_count)
                            ],
                            dtype=int,
                        )
                    )
                    columns.append(
                        np.full(left_count, right_index[("rail",)], dtype=int)
                    )
                    values.append(local[:, 0])
                else:
                    left_count = int(local.shape[0])
                    right_count = int(local.shape[1])
                    rows.append(
                        np.asarray(
                            [
                                left_index[(collection_index, left)]
                                for left in range(left_count)
                                for _right in range(right_count)
                            ],
                            dtype=int,
                        )
                    )
                    columns.append(
                        np.asarray(
                            [
                                right_index[(collection_index, right)]
                                for _left in range(left_count)
                                for right in range(right_count)
                            ],
                            dtype=int,
                        )
                    )
                    values.append(
                        ar.do(
                            "reshape",
                            local,
                            (left_count * right_count, self.phys_dim, self.phys_dim),
                        )
                    )
            arrays.append(
                _scatter_add_2d(
                    array,
                    np.concatenate(rows),
                    np.concatenate(columns),
                    ar.do("concatenate", tuple(values), axis=0),
                )
            )
        return tuple(arrays), state_lists, pure_cores

    def _assemble(self, residuals):
        cores = {}
        for (start, end), residual in residuals.items():
            length = end - start + 1
            if length > 1:
                cores[(start, end)] = _operator_schmidt(
                    residual,
                    length,
                    self.phys_dim,
                    self.cutoff,
                    self.max_bond,
                )

        state_lists = [[("rail",)]]
        for cut in range(1, self.L):
            states = [("rail",)]
            for key, cluster_cores in cores.items():
                start, end = key
                if start < cut <= end:
                    for position in range(int(cluster_cores[cut - start - 1].shape[1])):
                        states.append((key, position))
            state_lists.append(states)
        state_lists.append([("rail",)])

        reference = _backend_reference(tuple(residuals.values()))
        arrays = []
        for site in range(self.L):
            left_states = state_lists[site]
            right_states = state_lists[site + 1]
            array = ar.do(
                "zeros",
                (len(left_states), len(right_states), self.phys_dim, self.phys_dim),
                like=reference,
            )
            left_index = {state: index for index, state in enumerate(left_states)}
            right_index = {state: index for index, state in enumerate(right_states)}
            rows = []
            columns = []
            values = []
            rows.append(left_index[("rail",)])
            columns.append(right_index[("rail",)])
            values.append(residuals[(site, site)])
            for key, cluster_cores in cores.items():
                start, end = key
                if site < start or site > end:
                    continue
                local = cluster_cores[site - start]
                if site == start:
                    for right in range(local.shape[1]):
                        rows.append(left_index[("rail",)])
                        columns.append(right_index[(key, right)])
                        values.append(local[0, right])
                elif site == end:
                    for left in range(local.shape[0]):
                        rows.append(left_index[(key, left)])
                        columns.append(right_index[("rail",)])
                        values.append(local[left, 0])
                else:
                    for left in range(local.shape[0]):
                        for right in range(local.shape[1]):
                            rows.append(left_index[(key, left)])
                            columns.append(right_index[(key, right)])
                            values.append(local[left, right])
            values = ar.do("stack", tuple(values), axis=0)
            array = _scatter_add_2d(
                array,
                np.asarray(rows, dtype=int),
                np.asarray(columns, dtype=int),
                values,
            )
            arrays.append(array)
        return tuple(arrays), state_lists, cores

    def exp(self, step=1.0, *, parameters=None):
        """Build the cluster expansion for the supplied exponential step."""

        self._build_count += 1
        step = _cluster_to_backend(step, self.to_backend)
        residuals = self._residuals(step, parameters)
        graph_plan = {
            "strategy": "direct",
            "collections": (),
            "collection_order": None,
            "collection_count": 0,
            "collection_truncated": False,
        }
        if self.graph is None:
            arrays, state_lists, cores = self._assemble(residuals)
        else:
            graph_plan = self._graph_collection_plan()
            if graph_plan["strategy"] == "direct":
                arrays, state_lists, cores = self._assemble_graph(residuals)
            elif graph_plan["collections"]:
                arrays, state_lists, cores = self._assemble_graph_collections(
                    residuals,
                    collections=graph_plan["collections"],
                )
            else:
                arrays, state_lists, cores = self._assemble_graph(residuals)
        residual_ranks = tuple(
            (interval, tuple(int(core.shape[1]) for core in cores[interval]))
            for interval in sorted(cores)
        )
        bond_dimensions = tuple(len(states) for states in state_lists[1:-1])
        report = MPOClusterExpansionReport(
            cluster_size=self.cluster_size,
            factor_count=len(self.factors),
            interval_count=len(residuals),
            residual_ranks=residual_ranks,
            initial_bond_dimensions=bond_dimensions,
            cutoff=self.cutoff,
            local_svd_truncated=(
                self.cutoff not in (None, 0.0) or self.max_bond is not None
            ),
            max_bond=self.max_bond,
            cluster_mode=self.cluster_mode,
            graph_cluster_count=len(self._graph_clusters),
            graph_loop_counts=self._graph_loop_counts,
            graph_assembly=(
                "interval"
                if self.graph is None
                else graph_plan["strategy"]
            ),
            graph_collection_order=(
                None
                if self.graph is None
                else graph_plan["collection_order"]
            ),
            graph_collection_count=(
                0
                if self.graph is None
                else graph_plan["collection_count"]
            ),
            graph_collection_budget=(
                None
                if self.graph is None
                else self.collection_budget
            ),
            graph_collection_truncated=(
                False
                if self.graph is None
                else graph_plan["collection_truncated"]
            ),
            graph_frontier_width=(
                0 if self.graph is None else self._graph_frontier_width()
            ),
        )
        self._last_report = report
        return FirstDegreeMPO(
            arrays,
            degree=self.cluster_size,
            metadata={
                "operation": "cluster_expansion",
                "cluster_size": self.cluster_size,
                "factor_count": len(self.factors),
                "cluster_report": report,
                "history_valid": False,
            },
        )

    def residuals(self, step=1.0, *, parameters=None):
        """Return connected residual matrices keyed by intervals or graph sites."""

        return self._residuals(step, parameters)


class MPOGraphClusterProductExpansion(MPOClusterProductExpansion):
    """Graph-aware MPO cluster expansion with chain-compatible output.

    ``graph`` uses the same :class:`ClusterLattice` contract as the PEPO graph
    expansion, but its site labels must be MPO chain positions. Graph
    clusters are locally exponentiated using the supplied ordered factors,
    recursively reduced over connected graph partitions, and then embedded
    into the chain with identity sites before operator-Schmidt factorization.
    """

    def __init__(self, L, factors, *, graph, **kwargs):
        super().__init__(L, factors, graph=graph, **kwargs)


# Compatibility names retain the historical ``BasisExpansion`` vocabulary.
MPOClusterBasisExpansion = MPOClusterProductExpansion
CompiledMPOClusterExp = CompiledMPOClusterProduct
MPOGraphClusterBasisExpansion = MPOGraphClusterProductExpansion
ClusterBasisExpansion = MPOClusterProductExpansion
ClusterExpansionBasis = MPOClusterProductExpansion
ClusterExpBasis = MPOClusterProductExpansion
MPOClusterExpansion = MPOClusterProductExpansion


def _cluster_factor_from_source(
    source,
    *,
    shape,
    mapper,
    map_mode,
    phys_dim,
    to_backend,
):
    """Normalize one term-centric ordered-product factor."""

    from .mpo_basis import (  # pylint: disable=import-outside-toplevel
        MPOBasis,
        _convert_term_to_backend,
    )

    if isinstance(source, MPOClusterFactor):
        if to_backend is None:
            return source, None
        return (
            MPOClusterFactor(
                tuple(
                    _convert_term_to_backend(term, to_backend)
                    for term in source.terms
                ),
                coefficient=source.coefficient,
            ),
            None,
        )

    if isinstance(source, MPOBasis):
        if to_backend is not None:
            return (
                MPOClusterFactor(
                    tuple(
                        _convert_term_to_backend(term, to_backend)
                        for term in source.terms
                    )
                ),
                source,
            )
        return MPOClusterFactor.from_mpo_basis(source), source

    factor_coefficient = 1.0
    if isinstance(source, Mapping):
        terms = source.get("terms")
        if terms is None:
            raise ValueError(
                "cluster factor mappings require a 'terms' entry."
            )
        factor_coefficient = source.get("coefficient", 1.0)
    else:
        terms = source

    basis = MPOBasis.from_terms(
        terms,
        shape=shape,
        mapper=mapper,
        map_mode=map_mode,
        phys_dim=phys_dim,
        to_backend=to_backend,
    )
    return (
        MPOClusterFactor.from_mpo_basis(
            basis,
            coefficient=factor_coefficient,
        ),
        basis,
    )


def _cluster_factor_reference(factors):
    """Return one local operator for dtype-aware cutoff resolution."""

    for factor in factors:
        for term in factor.terms:
            if isinstance(term, MPOLocalOperatorTerm):
                return term.operator
            return term.operators[0]
    return None


def exp_mpo_cluster(
    terms=None,
    step=None,
    *,
    shape=None,
    mapper=None,
    map_mode="snake",
    parameters=None,
    coefficients=None,
    dt=None,
    phys_dim=None,
    cluster_size=2,
    graph=None,
    cyclic=False,
    factors=None,
    max_bond=None,
    cutoff=1.0e-12,
    graph_assembly="auto",
    max_collection_order=None,
    collection_budget=128,
    chi=None,
    cutoff_mode="rel",
    compression=None,
    differentiable=False,
    sector_aware="auto",
    symmetry=None,
    physical_charges=None,
    fermionic=False,
    physical_space=None,
    to_backend=None,
    return_semantic=False,
    return_report=False,
    form=None,
    create_bond=False,
    compress_opts=None,
    progress=False,
):
    """Build a term-centric connected-cluster MPO.

    This is the cluster-family counterpart to :func:`exp_mpo`. The shared
    term parser handles chain and regular-lattice locations, coefficient
    parameters, custom one-dimensional maps, and backend conversion. The
    cluster-specific ``cluster_size`` counts connected spatial sites; when
    ``graph`` is supplied it is a graph-site cutoff rather than a chain-span
    cutoff.

    Parameters
    ----------
    terms : iterable or mapping, optional
        One local Hamiltonian factor in the same forms accepted by
        :func:`exp_mpo`. Required unless ``factors`` is supplied.
    step, dt : scalar, optional
        The scalar in ``exp(step * H)``. ``dt`` is the compatibility spelling
        used by the rest of the MPO API; pass only one of them.
    shape, mapper, map_mode, parameters, coefficients, phys_dim : optional
        Shared term-centric parsing and coefficient controls. ``coefficients``
        is mutually exclusive with ``parameters`` and overrides the parsed
        term coefficient slots, matching :func:`exp_mpo`.
    cluster_size : int, default=2
        Largest connected interval or graph cluster retained.
    graph : optional
        ``"chain"`` or ``"square"`` for the common geometries, or a
        :class:`ClusterLattice`, ``(sites, edges)`` pair, or mapping. For
        coordinate-labelled graphs, supply ``shape`` or a lattice-aware
        basis through the term parser so coordinates can be mapped to the MPO
        chain. Omitting ``graph`` selects the ordinary open-chain interval
        path.
    cyclic : bool or tuple of bool, default=False
        Periodic-edge shorthand for ``graph="chain"`` or ``graph="square"``.
        A boolean makes a square lattice periodic in both directions; a
        ``(cyclic_x, cyclic_y)`` tuple selects square directions separately.
        Explicit graph objects already define their edges and cannot be
        combined with a non-default ``cyclic`` value.
    factors : iterable, optional
        Ordered factors for ``exp(A) @ exp(B) @ ...``. Each factor may be an
        ``MPOClusterFactor``, an ``MPOBasis``, a term iterable, or a mapping
        with ``terms`` and optional factor ``coefficient``. When supplied,
        ``terms`` and ``coefficients`` must be omitted.
    max_bond, cutoff : optional
        Analytical local operator-Schmidt controls. ``max_bond`` caps each
        residual factorization; it is not the final MPO bond cap. ``cutoff``
        is a relative local singular-value cutoff. ``"auto"`` is resolved
        from the local operator dtype.
    graph_assembly : {"auto", "exact", "bounded"}, default="auto"
        Assembly policy for crossing or nested graph clusters. ``"auto"``
        keeps exact collection assembly below ``collection_budget`` and
        otherwise uses the bounded one-cluster approximation. ``"exact"``
        raises instead of exceeding the budget. ``"bounded"`` uses
        ``max_collection_order``.
    max_collection_order : int, optional
        Maximum number of non-single graph residuals in one assembled
        collection when ``graph_assembly="bounded"``. The default is one,
        which retains every individual graph residual and omits products of
        multiple graph residuals.
    collection_budget : int or None, default=128
        Hard limit on graph-cluster collections inspected or materialized.
        Set ``None`` only when an explicitly unbounded exact plan is intended.
    chi, cutoff_mode, compression, differentiable, sector_aware, form,
    create_bond, compress_opts : optional
        Optional final numerical MPO compression, using the same semantic
        boundary as :func:`exp_mpo`. ``chi`` is separate from ``max_bond``.
        With ``return_semantic=True``, use ``compression="fixed_rank"`` or
        ``differentiable=True`` to retain a semantic result.
    symmetry, physical_charges, fermionic, physical_space : optional
        Native Symmray and graded history metadata are intentionally rejected
        here. The analytical cluster assembler currently produces ordinary
        dense-backend MPO blocks; use the higher-order ``exp_mpo`` path for
        native block-sparse history construction.
    to_backend : callable, optional
        Converter applied to parsed local operators, the exponential step,
        and resolved scalar coefficients before local exponentials, residual
        SVDs, MPO assembly, and the final Quimb boundary. Torch/JAX autodiff
        values therefore remain on the requested backend.
    return_semantic : bool, default=False
        Return the semantic :class:`FirstDegreeMPO`; otherwise return a
        Quimb MPO, matching :func:`exp_mpo`.
    return_report : bool, default=False
        Return ``(result, cluster_report)``. If final ``chi`` compression is
        requested, its numerical report is attached to the result while the
        returned report remains the analytical cluster report.
    progress : bool, default=False
        Show one construction stage and, when requested, one compression
        stage.

    Notes
    -----
    ``order``, ``mode``, ``history_storage``, and ``extension_budget`` are not
    accepted because they belong to the separate higher-order MPO history
    family. Here ``cluster_size`` is the spatial expansion control.

    Graph collection assembly is a second, independent approximation axis.
    It matters only when disjoint graph clusters overlap in the MPO chain
    ordering. ``graph_assembly="bounded"`` with
    ``max_collection_order=1`` is the fast graph-MPO mode; use a graph-native
    PEPO when the full 2D connected expansion is required at scale.
    """

    if not isinstance(progress, bool):
        raise TypeError("progress must be a boolean.")
    if symmetry is not None or physical_charges is not None or fermionic:
        raise NotImplementedError(
            "native symmetry and graded MPO metadata are not currently "
            "supported by exp_mpo_cluster; use exp_mpo for that path."
        )
    if physical_space is not None:
        raise NotImplementedError(
            "physical_space is not currently supported by exp_mpo_cluster."
        )
    if not isinstance(differentiable, bool):
        raise TypeError("differentiable must be a boolean.")
    sector_aware = _normalize_sector_aware_request(sector_aware)
    step = _resolve_exp_step(step, dt)
    step = _cluster_to_backend(step, to_backend)

    from .mpo_basis import (  # pylint: disable=import-outside-toplevel
        MPOBasis,
        _apply_to_backend,
    )

    if factors is not None and terms is not None:
        raise ValueError("pass either terms or factors, not both.")
    if factors is not None and coefficients is not None:
        raise ValueError(
            "coefficients are only supported for the single-factor terms "
            "interface; put coefficients in each factor's terms instead."
        )

    reference_basis = None
    if factors is None:
        if terms is None:
            raise TypeError("exp_mpo_cluster requires terms or factors.")
        basis = MPOBasis.from_terms(
            terms,
            shape=shape,
            mapper=mapper,
            map_mode=map_mode,
            phys_dim=phys_dim,
            to_backend=to_backend,
        )
        reference_basis = basis
        if coefficients is None:
            cluster_factors = (
                MPOClusterFactor.from_mpo_basis(basis),
            )
        else:
            values = basis._coefficient_values(  # pylint: disable=protected-access
                parameters,
                coefficients,
            )
            cluster_factors = (
                MPOClusterFactor(
                    tuple(
                        replace(term, coefficient=value)
                        for term, value in zip(basis.terms, values)
                    )
                ),
            )
            parameters = None
    else:
        if isinstance(factors, (MPOClusterFactor, MPOBasis, Mapping)):
            factor_sources = (factors,)
        else:
            factor_sources = tuple(factors)
        if not factor_sources:
            raise ValueError("factors must contain at least one factor.")
        cluster_factors = []
        for source in factor_sources:
            factor, basis = _cluster_factor_from_source(
                source,
                shape=shape,
                mapper=mapper,
                map_mode=map_mode,
                phys_dim=phys_dim,
                to_backend=to_backend,
            )
            cluster_factors.append(factor)
            if reference_basis is None and basis is not None:
                reference_basis = basis
        cluster_factors = tuple(cluster_factors)

    reference = _cluster_factor_reference(cluster_factors)
    if isinstance(cutoff, str):
        local_cutoff = _resolve_compression_cutoff(cutoff, reference)
    else:
        local_cutoff = cutoff

    if reference_basis is not None:
        length = reference_basis.L
        if shape is not None and isinstance(shape, Integral) and int(shape) != length:
            raise ValueError(
                f"shape={shape} does not match the parsed MPO length {length}."
            )
    else:
        if shape is not None:
            if not isinstance(shape, Integral) or isinstance(shape, bool):
                raise ValueError(
                    "shape must be an integer chain length when factors do not "
                    "contain a term-centric MPOBasis."
                )
            length = int(shape)
        else:
            length = max(
                site
                for factor in cluster_factors
                for term in factor.terms
                for site in term.sites
            ) + 1

    if graph is None:
        if isinstance(cyclic, (bool, np.bool_)):
            cyclic_requested = bool(cyclic)
        else:
            cyclic_requested = True
        if cyclic_requested:
            raise ValueError(
                "cyclic requires graph='chain', graph='square', or an "
                "explicit graph with its periodic edges."
            )
        expansion = MPOClusterProductExpansion.from_factors(
            length,
            cluster_factors,
            phys_dim=phys_dim,
            cluster_size=cluster_size,
            cutoff=local_cutoff,
            max_bond=max_bond,
            to_backend=to_backend,
            graph_assembly=graph_assembly,
            max_collection_order=max_collection_order,
            collection_budget=collection_budget,
        )
    else:
        normalized_graph = _graph_lattice_from_spec(
            graph,
            shape=shape,
            length=length,
            basis=reference_basis,
            cyclic=cyclic,
        )
        expansion = MPOGraphClusterProductExpansion.from_factors(
            length,
            cluster_factors,
            graph=normalized_graph,
            phys_dim=phys_dim,
            cluster_size=cluster_size,
            cutoff=local_cutoff,
            max_bond=max_bond,
            to_backend=to_backend,
            graph_assembly=graph_assembly,
            max_collection_order=max_collection_order,
            collection_budget=collection_budget,
        )

    if chi is not None:
        if not isinstance(chi, Integral) or isinstance(chi, bool) or int(chi) < 1:
            raise ValueError("chi must be a positive integer or None.")
        chi = int(chi)
    compression_options = _normalize_exp_compress_opts(
        compress_opts,
        form=form,
        create_bond=create_bond,
    )
    if chi is None and (
        compression is not None
        or differentiable
        or compression_options
        or sector_aware is True
    ):
        raise ValueError(
            "compression options require chi; omit them for an uncompressed MPO."
        )
    if return_semantic and chi is not None and not differentiable and compression != "fixed_rank":
        raise ValueError(
            "return_semantic=True with chi requires compression='fixed_rank' "
            "or differentiable=True; numerical Quimb compression returns an "
            "ordinary MPO."
        )

    progress_bar = None
    timings = {}
    import time  # pylint: disable=import-outside-toplevel

    construction_start = time.perf_counter()
    if progress:
        from tqdm.auto import tqdm  # pylint: disable=import-outside-toplevel

        progress_bar = tqdm(
            total=2 if chi is not None else 1,
            desc="exp_mpo_cluster",
            unit="stage",
            leave=True,
            dynamic_ncols=True,
        )
    try:
        stage_start = time.perf_counter()
        semantic = expansion.exp(step, parameters=parameters)
        timings["cluster"] = time.perf_counter() - stage_start
        cluster_report = expansion.last_report
        if progress_bar is not None:
            progress_bar.set_description(
                "exp_mpo_cluster | cluster "
                f"({cluster_report.graph_assembly})"
            )
            progress_bar.update(1)

        numerical_report = None
        if chi is not None:
            stage_start = time.perf_counter()
            numerical_cutoff = 0.0 if cutoff is None else cutoff
            compressed = semantic.compress_to_bond(
                chi,
                cutoff=numerical_cutoff,
                cutoff_mode=cutoff_mode,
                compression=compression,
                differentiable=differentiable,
                return_report=True,
                sector_aware=sector_aware,
                **compression_options,
            )
            result, numerical_report = compressed
            timings["chi_compression"] = time.perf_counter() - stage_start
            if progress_bar is not None:
                progress_bar.set_description(
                    f"exp_mpo_cluster | chi-compress (chi={chi})"
                )
                progress_bar.update(1)
        else:
            result = semantic

        result_metadata = getattr(result, "metadata", None)
        if isinstance(result_metadata, dict):
            result_metadata["exp_mpo_cluster"] = True
            result_metadata["cluster_report"] = cluster_report
            result_metadata["cluster_mode"] = expansion.cluster_mode
            result_metadata["graph_requested"] = graph is not None
            if graph is not None:
                result_metadata["graph_assembly"] = cluster_report.graph_assembly
                result_metadata["graph_collection_count"] = (
                    cluster_report.graph_collection_count
                )
                result_metadata["graph_collection_truncated"] = (
                    cluster_report.graph_collection_truncated
                )
            if numerical_report is not None:
                result_metadata["numerical_compression_report"] = numerical_report
            if progress:
                result_metadata["progress"] = True
                result_metadata["timings"] = dict(timings)
                result_metadata["order_seconds"] = (
                    time.perf_counter() - construction_start
                )

        if return_semantic:
            output = result
        elif hasattr(result, "to_mpo"):
            output = result.to_mpo()
        else:
            output = result
        _apply_to_backend(output, to_backend)
        output.pepsy_cluster_report = cluster_report
        cluster_metadata = {
            "cluster_mode": expansion.cluster_mode,
            "graph_requested": graph is not None,
            "cluster_size": expansion.cluster_size,
            "factor_count": len(expansion.factors),
            "cutoff": expansion.cutoff,
            "max_bond": expansion.max_bond,
            "graph_assembly": expansion.graph_assembly,
            "max_collection_order": expansion.max_collection_order,
            "collection_budget": expansion.collection_budget,
            "selected_graph_assembly": cluster_report.graph_assembly,
            "graph_collection_count": cluster_report.graph_collection_count,
            "graph_collection_truncated": cluster_report.graph_collection_truncated,
            "graph_frontier_width": cluster_report.graph_frontier_width,
            "chi": chi,
            "compression": (
                None
                if chi is None
                else compression or ("fixed_rank" if differentiable else "quimb")
            ),
            "differentiable": bool(differentiable),
            "numerical_compression_report": numerical_report,
        }
        if progress:
            cluster_metadata["progress"] = True
            cluster_metadata["timings"] = dict(timings)
            cluster_metadata["order_seconds"] = (
                time.perf_counter() - construction_start
            )
        output.pepsy_cluster_metadata = cluster_metadata
        return (output, cluster_report) if return_report else output
    finally:
        if progress_bar is not None:
            progress_bar.close()
