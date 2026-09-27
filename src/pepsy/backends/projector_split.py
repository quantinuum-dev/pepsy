"""Isometric factors of a retained matrix subspace for boundary contraction.

This is a composed-factor derivative, not an SVD-vector derivative. Losses
must be invariant under (Q, B) -> (Q W, W^H B) for unitary W. Numerical null
directions are removed at matrix-size times machine precision. Rank-changing
points and a closed kept/discarded spectral gap are not smooth charts.
"""

import autoray as ar
import numpy as np

_METHOD = "pepsy:projector"
_REGISTERED = False


def _retained_rank(u, s, vh, *, cutoff, cutoff_mode, max_bond):
    """Use the installed Quimb cutoff policy, then remove numerical nulls.

    Keep this private Quimb compatibility boundary in one place. It is the
    same trimming API used by Pepsy's existing raw Torch split drivers.
    """
    import quimb.tensor.decomp as qd

    if not hasattr(qd, "_trim_and_renorm_svd_result"):
        raise RuntimeError("projector splitting requires Quimb's SVD trimming API")
    ur, _, _ = qd._trim_and_renorm_svd_result(
        u, s, vh, cutoff=cutoff, cutoff_mode=qd._CUTOFF_MODE_MAP[cutoff_mode],
        max_bond=max_bond, absorb=qd.get_U_s_VH, renorm=0,
        info=None, xp=qd.get_namespace(s),
    )
    if ar.infer_backend(s) == "torch":
        import torch

        eps = torch.finfo(s.dtype).eps
        rank = int((s > max(u.shape[0], vh.shape[1]) * eps * s[0]).sum())
    else:
        eps = np.finfo(s.dtype).eps
        rank = int(np.count_nonzero(s > max(u.shape[0], vh.shape[1]) * eps * s[0]))
    return min(ur.shape[1], rank)


def projector_split(x, cutoff=-1.0, cutoff_mode="rsum2", max_bond=-1,
                    absorb="right", renorm=0, info=None):
    """Return Q,B or their transposed LQ counterpart; Q is isometric.

    Only left/right absorption is supported. The boundary algorithm uses
    both factors together. Native symmetry, other backends, renormalization,
    singular values as independent outputs, and higher derivatives are outside
    this first-order dense NumPy/Torch route.
    """
    import quimb.tensor.decomp as qd

    if renorm:
        raise ValueError("projector splitting does not support renorm")
    absorb_i = qd._ABSORB_MAP[absorb]
    if absorb_i not in (qd.get_U_sVH, qd.get_Us_VH):
        raise ValueError("projector splitting requires absorb='left' or 'right'")
    if x.ndim != 2 or min(x.shape) == 0:
        raise ValueError("projector splitting requires a nonempty matrix")
    backend = ar.infer_backend(x)
    transposed = absorb_i == qd.get_Us_VH
    if transposed:
        x = ar.do("transpose", x)
    if backend == "torch":
        from .linalg_torch_projector import torch_projector_split

        q, b = torch_projector_split(x, cutoff, cutoff_mode, max_bond)
    elif backend == "numpy":
        u, s, vh = np.linalg.svd(x, full_matrices=False)
        rank = _retained_rank(u, s, vh, cutoff=cutoff,
                              cutoff_mode=cutoff_mode, max_bond=max_bond)
        if rank:
            q, b = u[:, :rank], s[:rank, None] * vh[:rank]
        else:
            q, b = np.eye(x.shape[0], 1, dtype=x.dtype), np.zeros((1, x.shape[1]), dtype=x.dtype)
    else:
        raise TypeError("projector splitting currently supports dense NumPy/Torch arrays")
    if info is not None:
        info["retained_rank"] = q.shape[1]
    if transposed:
        return ar.do("transpose", b), None, ar.do("transpose", q)
    return q, None, b


def register_projector_split():
    """Install a package-owned Quimb method without replacing any driver."""
    global _REGISTERED  # pylint: disable=global-statement
    if not _REGISTERED:
        import quimb.tensor.decomp as qd

        if not callable(getattr(qd, "register_split_driver", None)):
            raise RuntimeError("projector splitting requires Quimb split-driver registration")
        qd.register_split_driver(_METHOD, default_absorb=qd.get_U_sVH)(projector_split)
        _REGISTERED = True
    return _METHOD
