"""Sweep boundary-environment providers."""

from __future__ import annotations

import inspect

from ...backends.convert import resolve_backend_sample_data_from_tn

__all__ = [
    "QuimbMpsBoundaryStore",
    "canonical_boundary_engine_selector",
    "normalize_boundary_engine",
    "symmray_array_backends",
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


def symmray_array_backends(*states):
    """Return the array backends used by Symmray tensors in ``states``."""
    backends = set()
    for state in states:
        if state is None:
            continue
        tensor_map = getattr(state, "tensor_map", None)
        tensors = tensor_map.values() if tensor_map else ()
        for tensor in tensors:
            data = getattr(tensor, "data", None)
            if not _is_symmray_array_data(data):
                continue
            backend = getattr(data, "backend", None)
            if backend is not None:
                backends.add(str(backend).lower())
    return frozenset(backends)


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
        # Quimb's compute_*_environments forwards kwargs into lower-level
        # boundary contraction internals whose progress kwarg name differs
        # across versions. Avoid forwarding either spelling here to keep the
        # boundary refresh path version-agnostic.
        opts.pop("progress", None)
        opts.pop("progbar", None)
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
        """Recompute Quimb environments for one sweep axis.

        The returned environment dictionary is passed back to Quimb so that
        the store remains a real cache, rather than only retaining the result
        after a full rebuild.
        """
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
            envs={},
            **self._compute_kwargs(progress=progress),
        )
        axis_labels = {f"{axis}min", f"{axis}max"}
        self.envs = {
            key: value
            for key, value in self.envs.items()
            if not (isinstance(key, tuple) and key[0] in axis_labels)
        }
        self.envs.update(envs)
        self.mps_b = {
            key: value
            for key, value in self.mps_b.items()
            if not str(key).startswith(axis.upper())
        }
        self._sync_axis_mps_b(tn, axis, envs)
        self.update_count += 1
        return self

    def start_sweep(self, tn, axis, update_side, *, progress=False):
        """Prepare one half-sweep with one static and one moving boundary.

        Quimb computes the opposite-side environments once. The selected
        side is then advanced one row or column at a time with
        :meth:`advance_sweep`, after the local tensor update has been applied.
        """
        if update_side not in {"left", "right"}:
            raise ValueError("update_side must be 'left' or 'right'")
        if axis not in {"x", "y"}:
            raise ValueError("axis must be 'x' or 'y'")

        # The public Quimb names are ``compute_ymin_environments`` etc.
        compute_fn = getattr(tn, f"compute_{axis}{'max' if update_side == 'left' else 'min'}_environments")
        envs = _call_with_accepted_kwargs(
            compute_fn,
            envs={},
            **self._compute_kwargs(progress=progress),
        )

        axis_labels = {f"{axis}min", f"{axis}max"}
        self.envs = {
            key: value
            for key, value in self.envs.items()
            if not (isinstance(key, tuple) and key[0] in axis_labels)
        }
        self.envs.update(envs)
        self.mps_b = {
            key: value
            for key, value in self.mps_b.items()
            if not str(key).startswith(axis.upper())
        }
        self._sync_axis_mps_b(tn, axis, envs)
        self._sweep_axis = axis
        self._sweep_update_side = update_side
        self._sweep_moving_env = None
        self.update_count += 1
        return self

    def _extend_moving_boundary(self, tn, index, *, axis, update_side):
        """Add one updated row or column to the moving boundary MPS."""
        axis_tag = axis.upper()
        column = tn.select(f"{axis_tag}{index}").copy()
        if update_side == "left":
            if self._sweep_moving_env is None:
                return column
            env = self._sweep_moving_env.copy()
            env.retag_(
                {
                    tag: f"{axis_tag}0"
                    for tag in tuple(env.tags)
                    if isinstance(tag, str) and tag.startswith(axis_tag)
                }
            )
            column.retag_({f"{axis_tag}{index}": f"{axis_tag}1"})
            first, second = env, column
            direction = f"{axis}min"
        else:
            if self._sweep_moving_env is None:
                return column
            env = self._sweep_moving_env.copy()
            env.retag_(
                {
                    tag: f"{axis_tag}1"
                    for tag in tuple(env.tags)
                    if isinstance(tag, str) and tag.startswith(axis_tag)
                }
            )
            column.retag_({f"{axis_tag}{index}": f"{axis_tag}0"})
            first, second = column, env
            direction = f"{axis}max"

        def _rebase_plane_tags(network, coordinate):
            """Give a compressed boundary one logical 2D plane coordinate.

            ``contract_boundary_from_*`` identifies sites using ``I{x,y}``
            tags, not the ``X*`` / ``Y*`` tags alone. A moving environment
            carries every absorbed plane's original site tags, so it must be
            rebased before combining it with one new physical plane.
            """
            tag_map = {}
            for x in range(int(tn.Lx)):
                for y in range(int(tn.Ly)):
                    old_tag = tn.site_tag(x, y)
                    if old_tag not in network.tags:
                        continue
                    new_tag = (
                        tn.site_tag(x, coordinate)
                        if axis == "y"
                        else tn.site_tag(coordinate, y)
                    )
                    tag_map[old_tag] = new_tag
            if tag_map:
                network.retag_(tag_map)

        # The two-plane temporary network always has coordinates 0 and 1.
        # Rebase both the compressed environment and incoming physical plane
        # so Quimb actually contracts the pair rather than merely returning
        # the original boundary side unchanged.
        _rebase_plane_tags(first, 0)
        _rebase_plane_tags(second, 1)

        if axis == "x":
            pair = (first | second).view_like(tn, Lx=2, Ly=tn.Ly)
            xrange, yrange = (0, 1), (0, tn.Ly - 1)
        else:
            pair = (first | second).view_like(tn, Lx=tn.Lx, Ly=2)
            xrange, yrange = (0, tn.Lx - 1), (0, 1)

        opts = self._compute_kwargs()
        opts.pop("dense", None)
        opts.pop("envs", None)
        getattr(pair, f"contract_boundary_from_{direction}_")(
            xrange=xrange,
            yrange=yrange,
            **opts,
        )
        # Quimb keeps the boundary on the side it contracted *from*. For a
        # ymin/xmin move that is the zero-labelled row/column, whereas a
        # ymax/xmax return move leaves the boundary on the one-labelled side.
        # Returning ``axis_tag0`` unconditionally made every backward cached
        # step retain the uncontracted new slice instead of the compressed
        # moving environment.
        retained_index = 0 if update_side == "left" else 1
        return pair.select(f"{axis_tag}{retained_index}").copy()

    def advance_sweep(self, tn, index, *, axis=None, update_side=None):
        """Advance the moving boundary by one updated row or column."""
        axis = self._sweep_axis if axis is None else axis
        update_side = (
            self._sweep_update_side if update_side is None else update_side
        )
        n = int(getattr(tn, "Ly" if axis == "y" else "Lx"))
        if update_side == "left":
            next_index = index + 1
            if next_index >= n:
                return self
            boundary_index = next_index
            boundary_label = f"{axis}min"
        else:
            next_index = index - 1
            if next_index < 0:
                return self
            boundary_index = next_index
            boundary_label = f"{axis}max"

        self._sweep_moving_env = self._extend_moving_boundary(
            tn,
            index,
            axis=axis,
            update_side=update_side,
        )
        env_key = boundary_label, boundary_index
        self.envs[env_key] = self._sweep_moving_env
        self._sync_axis_mps_b(tn, axis, {env_key: self._sweep_moving_env})
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
