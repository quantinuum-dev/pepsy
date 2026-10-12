"""Bounded dense GPU gate/SVD batches for adjacent swap-mode trajectories.

Only the tensor update is prepared here. Each optimizer still executes its
normal replay validation, norm ledger, normalization and metadata handling.
"""

import autoray as ar
import numpy as np


def _retained_ranks(s, cutoff, cutoff_mode, chi):
    """Compute independent ranks, transferring only the small rank vector."""
    size = s.shape[-1]
    limit = size if chi is None else min(size, chi)
    if cutoff == 0:
        return [limit] * s.shape[0]
    if cutoff_mode in {"abs", "rel"}:
        threshold = cutoff if cutoff_mode == "abs" else cutoff * s[:, :1]
        ranks = ar.do("sum", s > threshold, axis=-1)
    else:
        weights = s ** (2 if cutoff_mode in {"sum2", "rsum2"} else 1)
        total = ar.do("sum", weights, axis=-1, keepdims=True)
        threshold = total * (1 - cutoff) if cutoff_mode.startswith("r") else total - cutoff
        ranks = ar.do("sum", ar.do("cumsum", weights, axis=-1) < threshold, axis=-1) + 1
    return np.clip(ar.to_numpy(ranks), 1, limit).astype(int).tolist()


def try_run_gate_batch(nodes, entries, run_kwargs, run_node):
    """Run a compatible one-gate segment, or return False before any update."""
    from .optimizer import MpsOptimizer, _normalize_gate_queue
    from ._streams import _prepare_gate_stream
    from ..noise import _stream_on_optimizer_backend

    if len(nodes) < 2 or len(entries) != 1:
        return False
    if run_kwargs.get("layout") is not None or any(run_kwargs.get(key) for key in (
        "non_unitary", "_trajectory_non_unitary", "use_layout_finder",
        "normalize_every", "normalize_final", "finite_check", "timing",
        "quality_check_every",
    )):
        return False
    first = nodes[0].optimizer
    if not all(type(node.optimizer) is MpsOptimizer and not node.leakage_state.leaked
               for node in nodes):
        return False
    if any((run_kwargs.get("mode") or node.optimizer.mode) != "swap"
           or node.optimizer.p.cyclic
           or node.optimizer.p.isfermionic()
           or node.optimizer._persistent_layout_plan is not None
           for node in nodes):
        return False
    backend = ar.infer_backend(first.p[0].data)
    if backend not in {"torch", "cupy"}:
        return False
    device = str(first.p[0].data.device)
    if backend == "torch" and not device.startswith("cuda"):
        return False
    plan = _prepare_gate_stream(
        list(_stream_on_optimizer_backend(entries, first)),
        to_backend=first._symbolic_gate_to_backend,
        backend_sample=first._state_backend_like(),
    )
    gates, wheres, kinds = _normalize_gate_queue(plan.entries)
    if len(gates) != 1 or kinds != ["gate"]:
        return False
    where = tuple(wheres[0])
    if len(where) != 2 or where[1] != where[0] + 1:
        return False
    cutoff = first._resolve_cutoff(run_kwargs.get("cutoff", "auto"))
    cutoff_mode = first._resolve_cutoff_mode(run_kwargs.get("cutoff_mode", "auto"))
    if cutoff_mode not in {"abs", "rel", "sum1", "sum2", "rsum1", "rsum2"}:
        return False
    dtype = str(first.p[0].data.dtype)
    for node in nodes:
        opt = node.optimizer
        if opt.chi != first.chi or any(
            ar.infer_backend(t.data) != backend or str(t.data.dtype) != dtype
            or str(t.data.device) != device for t in opt.p.tensors
        ):
            return False

    # Canonicalization is still per parent. Group by the actual matrix shape
    # after moving the center; rank changes never force zero padding.
    groups = {}
    i, j = where
    for node in nodes:
        opt, p = node.optimizer, node.optimizer.p
        opt._start_unitary_norm_tracking(p)
        opt.canonize_mps(p, where)
        bond = p.bond(i, j)
        li = tuple(k for k in p[i].inds if k not in (p.site_ind(i), bond))
        ri = tuple(k for k in p[j].inds if k not in (p.site_ind(j), bond))
        left = p[i].transpose(*li, p.site_ind(i), bond)
        right = p[j].transpose(bond, p.site_ind(j), *ri)
        d1, d2 = p.phys_dim(i), p.phys_dim(j)
        a = ar.do("reshape", left.data, (-1, d1, p.bond_size(i, j)))
        b = ar.do("reshape", right.data, (p.bond_size(i, j), d2, -1))
        groups.setdefault((a.shape, b.shape), []).append((node, a, b, left, right))

    for (ashape, bshape), group in groups.items():
        l, d1, _ = ashape
        _, d2, r = bshape
        # Conservative workspace bound, including stacked inputs and SVD
        # factors. Large blocks retain the normal unbatched implementation.
        block_bytes = l * d1 * d2 * r * (16 if "128" in dtype else 8)
        batch_size = min(32, (32 << 20) // max(1, 16 * block_bytes))
        for start in range(0, len(group), max(1, batch_size)):
            chunk = group[start:start + max(1, batch_size)]
            if len(chunk) < 2:
                run_node(chunk[0][0])
                continue
            a = ar.do("stack", [item[1] for item in chunk])
            b = ar.do("stack", [item[2] for item in chunk])
            theta = ar.do("einsum", "blim,bmjr->blijr", a, b)
            gate = ar.do("reshape", gates[0], (d1, d2, d1, d2))
            theta = ar.do("einsum", "pqij,blijr->blpqr", gate, theta)
            matrix = ar.do("reshape", theta, (len(chunk), l * d1, d2 * r))
            u, s, vh = ar.do("linalg.svd", matrix)
            ranks = _retained_ranks(s, cutoff, cutoff_mode, first.chi)
            for index, ((node, _a, _b, left, right), rank) in enumerate(zip(chunk, ranks)):
                opt = node.optimizer
                # A retained parent must not pin the full batched U storage.
                # Backend copy remains differentiable for Torch callers.
                ud = ar.do("reshape", opt._copy_mix_tensor_data(u[index, :, :rank]),
                           (*left.shape[:-1], rank))
                vd = ar.do("reshape", s[index, :rank, None] * vh[index, :rank, :],
                           (rank, *right.shape[1:]))
                # Replay consumes this once, at the exact adjacent gate. Its
                # normal ledger and canonical-center updates remain in charge.
                opt._trajectory_prepared_gate = (where, left.inds, ud, right.inds, vd)
                try:
                    run_node(node)
                    info = getattr(opt, "_trajectory_diagnostics", None)
                    if info is None:
                        info = opt._trajectory_diagnostics = {}
                    info["max_gate_parent_batch"] = max(
                        info.get("max_gate_parent_batch", 1), len(chunk))
                finally:
                    del opt._trajectory_prepared_gate
    return True
