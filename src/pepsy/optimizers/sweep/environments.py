"""Sweep boundary-environment providers."""

from __future__ import annotations

import inspect

from ...backends.convert import resolve_backend_sample_data_from_tn

__all__ = [
    "QuimbMpsBoundaryStore",
    "canonical_boundary_engine_selector",
    "normalize_boundary_engine",
    "uses_symmray_arrays",
]


_DEFAULT_LAYER_TAGS = ("KET", "BRA")
_BOUNDARY_ENGINE_ALIASES = {
    "auto": "auto",
    "dmrg": "dmrg",
    "fit": "dmrg",
    "pepsy": "dmrg",
    "pepsy-dmrg": "dmrg",
    "mps": "quimb-mps",
    "quimb": "quimb-mps",
    "quimb-mps": "quimb-mps",
    "boundary": "quimb-mps",
    "boundary-mps": "quimb-mps",
    "contract-boundary": "quimb-mps",
    "contract-boundary-mps": "quimb-mps",
}


def _call_with_accepted_kwargs(fn, **kwargs):
    """Call ``fn`` with only the keyword arguments it accepts."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(**kwargs)

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
        return fn(**kwargs)

    accepted = {key: val for key, val in kwargs.items() if key in sig.parameters}
    return fn(**accepted)


def _is_symmray_array_data(data):
    """Return whether ``data`` looks like a Symmray array."""
    return type(data).__module__.split(".", 1)[0] == "symmray"


def uses_symmray_arrays(*states):
    """Return whether any tensor-network-like object stores Symmray arrays."""
    for state in states:
        if state is None:
            continue

        sample_data = resolve_backend_sample_data_from_tn(state)
        if sample_data is not None and _is_symmray_array_data(sample_data):
            return True

        tensor_map = getattr(state, "tensor_map", None)
        tensors = tensor_map.values() if tensor_map else ()
        for tensor in tensors:
            data = getattr(tensor, "data", None)
            if data is not None and _is_symmray_array_data(data):
                return True

    return False


def canonical_boundary_engine_selector(engine):
    """Canonicalize a boundary engine selector without resolving ``"auto"``."""
    key = "auto" if engine is None else str(engine).strip().lower().replace("_", "-")
    if key not in _BOUNDARY_ENGINE_ALIASES:
        raise ValueError(
            "Unknown boundary_engine="
            f"{engine!r}. Expected 'auto', 'dmrg', or 'quimb-mps'."
        )
    return _BOUNDARY_ENGINE_ALIASES[key]


def normalize_boundary_engine(engine, *states, boundaries_supplied=False):
    """Normalize a boundary engine selector."""
    engine_norm = canonical_boundary_engine_selector(engine)
    if engine_norm == "auto":
        if boundaries_supplied:
            return "dmrg"
        return "quimb-mps" if uses_symmray_arrays(*states) else "dmrg"

    if boundaries_supplied and engine_norm != "dmrg":
        raise ValueError(
            "boundary_engine='quimb-mps' builds its own environment stores; "
            "do not pass bdy/bdy_overlap."
        )

    return engine_norm


class QuimbMpsBoundaryStore:
    """Reusable Quimb MPS row/column environment store for sweeps.

    The store adapts Quimb environment keys such as ``("ymin", j)`` and
    ``("xmax", i)`` into Pepsy's legacy ``mps_b`` keys so the current local
    sweep objective can attach environments without caring which engine built
    them.
    """

    def __init__(
        self,
        *,
        chi,
        cutoff=1.0e-12,
        canonize=True,
        mode="mps",
        layer_tags=_DEFAULT_LAYER_TAGS,
        dense=False,
        compress_opts=None,
        equalize_norms=False,
        **contract_boundary_opts,
    ):
        if chi is None:
            raise ValueError("chi is required for Quimb MPS boundary stores.")
        if int(chi) < 1:
            raise ValueError("chi must be >= 1")

        self._chi_target = int(chi)
        self.cutoff = cutoff
        self.canonize = canonize
        self.mode = mode
        self.layer_tags = tuple(layer_tags) if layer_tags is not None else None
        self.dense = bool(dense)
        self.compress_opts = dict(compress_opts or {})
        self.equalize_norms = equalize_norms
        self.contract_boundary_opts = dict(contract_boundary_opts)
        self.mps_b = {}
        self.envs = {}
        self.update_count = 0

    @property
    def chi(self):
        """Return the requested maximum Quimb environment bond dimension."""
        return int(self._chi_target)

    @chi.setter
    def chi(self, value):
        value = int(value)
        if value < 1:
            raise ValueError("chi must be >= 1")
        self._chi_target = value

    @property
    def norm(self):
        """Return a cheap average norm diagnostic over stored environments."""
        if not self.mps_b:
            return 1.0

        total = None
        count = 0
        for env in self.mps_b.values():
            norm_fn = getattr(env, "norm", None)
            if not callable(norm_fn):
                continue
            value = norm_fn()
            total = value if total is None else total + value
            count += 1
        if total is None or count == 0:
            return 1.0
        return total / count

    def expand_bnd(self, chi, rand_strength=0.0, inplace=True):
        """Retune future Quimb environment builds to ``chi``."""
        del rand_strength
        self.chi = chi
        return self if inplace else self.copy()

    def copy(self):
        """Return a shallow copy of this environment store."""
        other = type(self)(
            chi=self.chi,
            cutoff=self.cutoff,
            canonize=self.canonize,
            mode=self.mode,
            layer_tags=self.layer_tags,
            dense=self.dense,
            compress_opts=self.compress_opts,
            equalize_norms=self.equalize_norms,
            **self.contract_boundary_opts,
        )
        other.mps_b = dict(self.mps_b)
        other.envs = dict(self.envs)
        other.update_count = int(self.update_count)
        return other

    def normalize(self):
        """Normalize stored environment networks when they support it."""
        for env in self.mps_b.values():
            normalize_fn = getattr(env, "normalize", None)
            if callable(normalize_fn):
                try:
                    normalize_fn()
                except TypeError:
                    normalize_fn(inplace=True)
        return self

    def _compute_kwargs(self, progress=False):
        opts = dict(self.contract_boundary_opts)
        if "progress" in opts and "progbar" not in opts:
            opts["progbar"] = opts.pop("progress")
        opts.setdefault("progbar", bool(progress))
        opts.update(
            {
                "max_bond": self.chi,
                "cutoff": self.cutoff,
                "canonize": self.canonize,
                "mode": self.mode,
                "dense": self.dense,
                "equalize_norms": self.equalize_norms,
            }
        )
        if self.layer_tags is not None:
            opts["layer_tags"] = list(self.layer_tags)
        if self.compress_opts:
            opts["compress_opts"] = dict(self.compress_opts)
        return opts

    def update_axis(self, tn, axis, *, progress=False):
        """Recompute Quimb environments for one sweep axis."""
        if axis == "y":
            compute_fn = getattr(tn, "compute_y_environments", None)
        elif axis == "x":
            compute_fn = getattr(tn, "compute_x_environments", None)
        else:
            raise ValueError("axis must be 'x' or 'y'")
        if not callable(compute_fn):
            raise TypeError(
                "Quimb MPS boundary engine requires a TensorNetwork2D with "
                f"compute_{axis}_environments()."
            )

        envs = _call_with_accepted_kwargs(
            compute_fn,
            **self._compute_kwargs(progress=progress),
        )
        self.envs.update(envs)
        self._sync_axis_mps_b(tn, axis, envs)
        self.update_count += 1
        return self

    def _sync_axis_mps_b(self, tn, axis, envs):
        if axis == "y":
            ly = int(getattr(tn, "Ly", 0))
            if ly < 1:
                raise ValueError("Cannot infer Ly for Quimb y environments.")
            for j in range(ly):
                if j > 0 and ("ymin", j) in envs:
                    self.mps_b[f"Y{j - 1}_l"] = envs["ymin", j]
                if j < ly - 1 and ("ymax", j) in envs:
                    self.mps_b[f"Y{ly - 2 - j}_r"] = envs["ymax", j]
            return

        lx = int(getattr(tn, "Lx", 0))
        if lx < 1:
            raise ValueError("Cannot infer Lx for Quimb x environments.")
        for i in range(lx):
            if i > 0 and ("xmin", i) in envs:
                self.mps_b[f"X{i - 1}_l"] = envs["xmin", i]
            if i < lx - 1 and ("xmax", i) in envs:
                self.mps_b[f"X{lx - 2 - i}_r"] = envs["xmax", i]
