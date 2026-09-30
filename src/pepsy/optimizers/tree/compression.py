"""Successive compression of layered, acyclic tensor networks.

SRC contracts product-noise sketches of the complementary branches. SDC
contracts deterministic low-rank factors of those branches instead, while
SDCR forms those factors with randomized SVDs. A second pass constructs nested
output isometries and projects the *original* target onto them. These methods
never truncate an already materialized target tree.
"""

import math
import warnings
from contextlib import contextmanager
from functools import lru_cache
from types import MappingProxyType

import autoray as ar
import quimb.tensor as qtn

from ..._internal.random import backend_random_array
from ..._internal.quimb import quimb_callable_option_supported


_SUCCESSIVE_COMPRESSION_MODES = frozenset({
    "src", "src_oversample", "sdc", "sdc_oversample", "sdcr",
    "sdcr_oversample",
})

@contextmanager
def _high_precision_matmul(like):
    """Use accurate accumulation for JAX complex64 compression contractions."""
    if ar.infer_backend(like) != "jax":
        yield
        return
    try:
        import jax  # pragma: no cover - imported only for JAX-backed calls
    except ImportError:  # pragma: no cover - backend check normally prevents this
        yield
        return
    with jax.default_matmul_precision("highest"):
        yield


def _tensor_contract(*tensors, **kwargs):
    """Contract through Quimb, with scoped JAX accumulation precision."""
    if not tensors:
        return qtn.tensor_contract(*tensors, **kwargs)
    with _high_precision_matmul(tensors[0].data):
        return qtn.tensor_contract(*tensors, **kwargs)


def _oversample_bond(max_bond, max_bond_oversample=None):
    """Return an explicit or Quimb-default intermediate oversampling rank.

    Quimb interprets floating point ``max_bond_oversample`` values as
    multipliers and integral values as explicit ranks. Keep that convention
    shared by successive and zipup oversampling modes. Zipup supplies its
    distinct default multiplier explicitly.
    """
    if max_bond is None:
        raise ValueError(
            "oversampled compression requires max_bond."
        )
    max_bond = int(max_bond)
    if max_bond < 1:
        raise ValueError("max_bond must be positive")
    if max_bond_oversample is None:
        return max(round(1.5 * max_bond), max_bond + 10)
    if isinstance(max_bond_oversample, bool):
        raise TypeError("max_bond_oversample must be a positive rank or multiplier")
    if isinstance(max_bond_oversample, float):
        if not math.isfinite(max_bond_oversample) or max_bond_oversample <= 0.0:
            raise ValueError("max_bond_oversample must be positive")
        max_bond_oversample = round(max_bond * max_bond_oversample)
    else:
        max_bond_oversample = int(max_bond_oversample)
    if max_bond_oversample < 1:
        raise ValueError("max_bond_oversample must be positive")
    return max_bond_oversample


def _src_oversample_bond(max_bond, max_bond_oversample=None):
    """Backward-compatible name for the common SRC oversample policy."""
    return _oversample_bond(max_bond, max_bond_oversample)


def _oversample_order(order, hub):
    """Choose a reversed first sweep for path-shaped regions.

    Quimb's oversampled successive methods use the opposite path direction for
    the initial pass and the direct rounding pass. A branched tree has no
    unique opposite endpoint, so retain the caller's valid inward order there
    and let the tree-native direct rounder handle the branches.
    """
    order = tuple(order)
    if order and all(
        left[1] == right[0] for left, right in zip(order, order[1:])
    ):
        reversed_order = tuple((v, u) for u, v in reversed(order))
        return reversed_order, reversed_order[-1][1]
    return order, hub


def _src_oversample_order(order, hub):
    """Backward-compatible name for the shared oversample ordering policy."""
    return _oversample_order(order, hub)


@lru_cache(maxsize=128)
def _successive_environment_plan(order, hub):
    """Cache immutable geometry only, never tensors, dimensions, or RNG state.

    Order and hub jointly specify both the active tree and sweep direction.
    Numerical messages and their mutable last-use counters belong to one
    compression invocation; sharing a plan cannot share either of those.
    """
    nodes = {hub, *(u for edge in order for u in edge)}
    remaining = set(nodes)
    neighbors = {u: [] for u in nodes}
    for u, v in order:
        if u == v or u not in remaining or v not in remaining:
            raise ValueError("order must peel each tree node once toward the hub")
        neighbors[u].append(v)
        neighbors[v].append(u)
        remaining.remove(u)
    if remaining != {hub}:
        raise ValueError("order must describe one connected tree ending at the hub")
    required = set()
    todo = [(v, u) for u, v in order]
    while todo:
        u, v = todo.pop()
        if (u, v) in required:
            continue
        required.add((u, v))
        todo.extend((w, u) for w in neighbors[u] if w != v)
    schedule = [edge for edge in order if edge in required]
    schedule.extend((v, u) for u, v in reversed(order) if (v, u) in required)
    uses = {edge: 0 for edge in required}
    for u, v in schedule:
        for w in neighbors[u]:
            if w != v:
                uses[w, u] += 1
    for u, v in order:
        uses[v, u] += 1
    return (
        MappingProxyType({u: tuple(vs) for u, vs in neighbors.items()}),
        tuple(schedule),
        MappingProxyType(uses),
    )


def successive_tree_compress(local, order, hub, *, method, max_bond,
                             cutoff=0.0, cutoff_mode="rsum2", seed=None,
                             sample_bond=None):
    """Return one tensor per node and edge records, without mutating inputs.

    ``local`` groups target layers by node; ``order`` peels children toward
    ``hub``. Exterior state bonds count as local output indices, so callers
    must first make the exterior isometric toward the active region.
    On a path these are the SRC/SDC/SDCR low-rank-environment sweeps.
    Branching uses a directed environment for each cut and nested child
    projections.
    """
    if method not in {"src", "sdc", "sdcr"}:
        raise ValueError(
            "successive compression requires 'src', 'sdc', or 'sdcr'"
        )
    if sample_bond is not None and method not in {"src", "sdc", "sdcr"}:
        raise ValueError("sample_bond is only valid for successive methods")
    if method == "sdcr" and max_bond is None:
        raise ValueError("max_bond must be given for tree SDCR")
    if any(ar.infer_backend(t.data) not in {"numpy", "torch", "jax", "cupy"}
           for ts in local.values() for t in ts):
        raise NotImplementedError(
            f"tree {method} environments support dense tree tensors only; "
            "use direct or zipup for native symmetry tensors"
        )
    if method == "src" and cutoff != 0.0:
        warnings.warn(
            "cutoff is ignored for tree SRC; use max_bond instead",
            UserWarning,
            stacklevel=2,
        )
    order = tuple(order)
    neighbors, schedule, consumers = _successive_environment_plan(order, hub)
    if neighbors.keys() != local.keys():
        raise ValueError("local tensors must exactly cover the planned tree nodes")
    indices = {u: {ix for t in ts for ix in t.inds} for u, ts in local.items()}
    sizes = {ix: t.ind_size(ix) for ts in local.values() for t in ts for ix in t.inds}
    bonds = {}
    for u, v in order:
        bonds[u, v] = bonds[v, u] = tuple(sorted(indices[u] & indices[v]))
    # Every invocation owns fresh counters and numerical caches, including
    # repeated targets and failed retries. Array edits, new ranks, changed
    # seeds/backends, and gates therefore cannot reuse stale environments.
    uses = dict(consumers)
    outer = {}
    for u, ts in local.items():
        inner = {ix for v in neighbors[u] for ix in bonds[u, v]}
        counts = {}
        for t in ts:
            for ix in t.inds:
                counts[ix] = counts.get(ix, 0) + 1
        outer[u] = tuple(ix for ix, count in counts.items() if count == 1 and ix not in inner)
    if max_bond is None:
        # Uncapped SRC samples enough columns to span every original cut.
        rank = max(
            (math.prod(sizes[ix] for ix in bix) for bix in bonds.values()),
            default=1,
        )
    else:
        rank = int(max_bond)
        if rank < 1:
            raise ValueError("max_bond must be positive")
    if sample_bond is not None:
        rank = int(sample_bond)
        if rank < 1:
            raise ValueError("sample_bond must be positive")
    batch = qtn.rand_uuid()
    noise = {}
    if method == "src":
        rng = (
            ar.get_namespace(like=next(iter(local.values()))[0].data)
            .random.default_rng(seed)
            if seed is not None else None
        )
        for u in dict.fromkeys(u for u, _ in schedule):
            if outer[u]:
                data = backend_random_array(
                    (rank, *(sizes[ix] for ix in outer[u])),
                    like=local[u][0].data,
                    dtype=ar.get_dtype_name(local[u][0].data),
                    rng=rng,
                )
                noise[u] = qtn.Tensor(data, inds=(batch, *outer[u]))
    environments = {}
    latent = {}

    def release(key):
        uses[key] -= 1
        if uses[key] == 0:
            del environments[key]
            del latent[key]

    def environment(u, v):
        key = u, v
        if key in environments:
            return environments[key]
        ts = list(local[u])
        left = list(outer[u])
        for w in neighbors[u]:
            if w != v:
                ts.append(environments[w, u])
                left.extend(latent[w, u])
        if method == "src":
            if u in noise:
                ts.append(noise[u])
            if not any(batch in t.inds for t in ts):
                # A physically empty complementary component is a scalar
                # boundary vector, replicated across the sample index.
                ts.append(qtn.Tensor(ar.do("ones", (rank,), like=ts[0].data,
                                          dtype=ar.get_dtype_name(ts[0].data)),
                                     inds=(batch,)))
            message = _tensor_contract(
                *ts,
                output_inds=(batch, *bonds[key]),
                drop_tags=True,
            )
            # Uniform rescaling leaves the sampled range unchanged and avoids
            # exponential scale drift along large complementary components.
            scale = ar.do("max", ar.do("abs", message.data))
            message.modify(data=message.data / ar.do("where", scale > 0, scale, 1.0))
            latent[key] = (batch,)
        else:
            tensor = _tensor_contract(
                *ts,
                output_inds=(*left, *bonds[key]),
                drop_tags=True,
            )
            split_method = "svd:rand" if method == "sdcr" else "svd"
            # Quimb's randomized SVD path is deliberately a rank-only
            # environment sketch. New Quimb releases reject cumulative
            # cutoff modes (``rsum*`` / ``sum*``) for randomized SVD, while
            # the surrounding Pepsy tree compression still reports and
            # applies its configured final cutoff independently.
            environment_cutoff = 0.0 if method == "sdcr" else cutoff
            environment_cutoff_mode = "rel" if method == "sdcr" else cutoff_mode
            split_opts = {
                "left_inds": left,
                "right_inds": bonds[key],
                "method": split_method,
                "absorb": "right",
                "max_bond": max_bond,
                "cutoff": environment_cutoff,
                "cutoff_mode": environment_cutoff_mode,
                "get": "tensors",
            }
            if method == "sdcr":
                if ar.infer_backend(tensor.data) == "torch":
                    from quimb.tensor.decomp import svd_rand_truncated

                    if not quimb_callable_option_supported(svd_rand_truncated, "noise_dist"):
                        raise NotImplementedError(
                            "Torch SDCR requires Quimb's dtype-aware random split driver; "
                            "upgrade Quimb or use SDC/direct compression."
                        )
                # Match Quimb's SDCR defaults: the environment is a static
                # randomized sketch, without oversampling or power iteration.
                # Pass the seed through unchanged so seeded paths have the
                # same per-split contract as Quimb's public SDCR wrapper.
                split_opts.update(
                    num_iterations=0,
                    oversample=0,
                    seed=seed,
                )
            _, message = tensor.split(
                # Deterministic truncated SVD forms the SDC environment
                # directly. SDCR uses the same environment construction with
                # Quimb's static randomized SVD driver instead.
                **split_opts,
            )
            latent[key] = tuple(ix for ix in message.inds if ix not in bonds[key])
        environments[key] = message
        for w in neighbors[u]:
            if w != v:
                release((w, u))
        return message

    # Build fixed complementary environments before making any projections.
    for u, v in schedule:
        environment(u, v)
    noise.clear()
    pending = {u: list(ts) for u, ts in local.items()}
    result = {}
    records = []
    for u, v in order:
        env = environments[v, u]
        if method == "src":
            # More columns than the original cut dimension can only add
            # arbitrary null-space Q columns, needlessly growing exact bonds.
            cut_rank = math.prod(sizes[ix] for ix in bonds[u, v])
            if cut_rank < rank:
                env = env.isel({batch: slice(0, cut_rank)})
        sample = _tensor_contract(*pending[u], env, drop_tags=True)
        left = tuple(ix for ix in sample.inds if ix not in latent[v, u])
        q, _ = sample.split(left_inds=left, right_inds=latent[v, u],
                            method="qr", absorb="lorthog", get="tensors")
        release((v, u))
        new_bond, = (ix for ix in q.inds if ix not in sample.inds)
        message = _tensor_contract(
            *pending.pop(u),
            q.H,
            output_inds=(new_bond, *bonds[u, v]),
            drop_tags=True,
        )
        result[u] = q
        pending[v].append(message)
        records.append((u, v, math.prod(sizes[ix] for ix in bonds[u, v]),
                        q.ind_size(new_bond), new_bond))
    result[hub] = _tensor_contract(*pending[hub], drop_tags=True)
    # As in Quimb SRC, structural tags belong to the final local tensor,
    # never to sketches accumulating an entire complementary component.
    for u, tensor in result.items():
        tensor.modify(tags=tuple(dict.fromkeys(tag for t in local[u] for tag in t.tags)))
    return result, records
