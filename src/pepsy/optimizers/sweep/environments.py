"""Sweep boundary-environment providers."""

from __future__ import annotations

import inspect
import math
from functools import lru_cache

from ...backends.convert import resolve_backend_sample_data_from_tn
from ..._internal.cutoff import dtype_auto_cutoff

__all__ = [
    "QuimbMpsBoundaryStore",
    "canonical_boundary_engine_selector",
    "normalize_boundary_engine",
    "symmray_array_backends",
    "uses_symmray_arrays",
]


_DEFAULT_LAYER_TAGS = ("KET", "BRA")
_QUIMB_CUTOFF_MODES = frozenset({"abs", "rel", "sum1", "sum2", "rsum1", "rsum2"})
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


@lru_cache(maxsize=128)
def _accepted_keyword_names(fn):
    """Return accepted keyword names for a stable callable target."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
        return None
    return frozenset(sig.parameters)


def _call_with_accepted_kwargs(fn, **kwargs):
    """Call ``fn`` with only the keyword arguments it accepts.

    Quimb exposes a few ``partialmethod`` variants for directional boundary
    contractions. Signature inspection is consequently needed for version
    compatibility, but the callable target is stable across sweep steps, so
    cache the inspection whenever it is safe to do so.
    """
    target = getattr(fn, "__func__", None)
    if target is None:
        target = getattr(fn, "func", None)
    if target is None or not callable(target):
        target = fn
    try:
        accepted = _accepted_keyword_names(target)
    except TypeError:
        # Callable instances can be unhashable and therefore cannot be used
        # as an lru-cache key. Fall back to one uncached inspection.
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return fn(**kwargs)
        if any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in sig.parameters.values()
        ):
            return fn(**kwargs)
        accepted = frozenset(sig.parameters)

    if accepted is None:
        return fn(**kwargs)
    accepted_kwargs = {
        key: val for key, val in kwargs.items() if key in accepted
    }
    return fn(**accepted_kwargs)


def _canonical_axis(axis):
    """Normalize a Quimb boundary axis selector."""
    axis = str(axis).strip().lower()
    if axis not in {"x", "y"}:
        raise ValueError("axis must be 'x' or 'y'")
    return axis


def _normalize_cutoff(cutoff):
    """Validate a Quimb boundary cutoff while preserving ``"auto"``."""
    if isinstance(cutoff, str):
        if cutoff.strip().lower() == "auto":
            return "auto"
        raise ValueError("cutoff must be 'auto' or a non-negative number.")
    if isinstance(cutoff, bool):
        raise TypeError("cutoff must be 'auto' or a non-negative number.")
    try:
        cutoff = float(cutoff)
    except (TypeError, ValueError) as exc:
        raise ValueError("cutoff must be 'auto' or a non-negative number.") from exc
    if not math.isfinite(cutoff) or cutoff < 0.0:
        raise ValueError("cutoff must be 'auto' or a non-negative number.")
    return cutoff


def _canonical_cutoff_mode(cutoff_mode):
    """Normalize Quimb's singular-value cutoff-mode spelling."""
    if cutoff_mode is None:
        return None
    if not isinstance(cutoff_mode, str):
        raise TypeError(
            "cutoff_mode must be None, 'auto', or a Quimb cutoff-mode string."
        )
    cutoff_mode = cutoff_mode.strip().lower()
    if cutoff_mode == "auto":
        return "rsum2"
    if cutoff_mode not in _QUIMB_CUTOFF_MODES:
        allowed = ", ".join(sorted(_QUIMB_CUTOFF_MODES))
        raise ValueError(
            f"Unknown cutoff_mode={cutoff_mode!r}. Expected 'auto' or one of {allowed}."
        )
    return cutoff_mode


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


class QuimbMpsBoundaryStore:  # pylint: disable=protected-access,too-many-instance-attributes
    """Reusable Quimb MPS row/column environment store for sweeps.

    ``envs`` is the authoritative native Quimb cache. ``mps_b`` is a thin
    compatibility index over the same environment objects, adapting keys such
    as ``("ymin", j)`` and ``("xmax", i)`` to Pepsy's legacy ``Y*_l`` and
    ``X*_r`` names. The store never creates a second copy of the boundary MPS
    data for that compatibility view.

    ``update_axis`` performs a complete refresh of one axis. The optimizer's
    half-sweep path uses ``start_sweep`` followed by ``advance_sweep`` instead:
    one side is computed once and the moving side is extended one plane at a
    time. ``expand_bnd`` changes the requested future ``chi``; it does not pad
    an existing compressed environment, because padding cannot restore the
    information discarded by its previous compression.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        chi,
        cutoff=1.0e-12,
        cutoff_mode=None,
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
        self.cutoff = _normalize_cutoff(cutoff)
        self.cutoff_mode = _canonical_cutoff_mode(cutoff_mode)
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
        self._resolved_cutoff = None
        self._sweep_axis = None
        self._sweep_update_side = None
        self._sweep_moving_env = None
        # Plane-tag rebasing is structural: it can be reused across all
        # boundary steps for networks with the same lattice tags.
        self._plane_tag_maps = {}

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
        """Return a cheap average norm diagnostic over cached environments."""
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
        """Set the bond cap used by future Quimb environment builds.

        Unlike :class:`pepsy.boundary.states.BdyMPS`, this method does not
        expand already-compressed environments. Call ``update_axis`` or start
        a new half-sweep after changing ``chi`` to rebuild with the new cap.
        ``rand_strength`` is accepted for compatibility with ``BdyMPS`` and
        has no meaning for Quimb's deterministic boundary contraction.
        """
        del rand_strength
        self.chi = chi
        return self if inplace else self.copy()

    def copy(self):  # pylint: disable=protected-access
        """Return a shallow copy with independent cache dictionaries.

        The native environment objects remain shared, matching Quimb's usual
        lightweight tensor-network copy semantics.
        """
        other = type(self)(
            chi=self.chi,
            cutoff=self.cutoff,
            cutoff_mode=self.cutoff_mode,
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
        other._resolved_cutoff = self._resolved_cutoff
        other._sweep_axis = self._sweep_axis
        other._sweep_update_side = self._sweep_update_side
        other._sweep_moving_env = self._sweep_moving_env
        other._plane_tag_maps = dict(self._plane_tag_maps)
        return other

    def _clear_side(self, axis, side):
        """Drop one native/legacy boundary side in place."""
        for key in tuple(self.envs):
            if isinstance(key, tuple) and key and key[0] == side:
                del self.envs[key]

        axis_tag = axis.upper()
        suffix = "_l" if side == f"{axis}min" else "_r"
        for key in tuple(self.mps_b):
            if (
                isinstance(key, str)
                and key.startswith(axis_tag)
                and key.endswith(suffix)
            ):
                del self.mps_b[key]

    def clear(self, axis=None):
        """Clear cached environments, optionally for one lattice axis.

        The dictionaries are cleared in place so callers holding references to
        ``envs`` or ``mps_b`` continue to observe the store's current cache.
        """
        if axis is None:
            self.envs.clear()
            self.mps_b.clear()
            self._sweep_axis = None
            self._sweep_update_side = None
            self._sweep_moving_env = None
            return self

        axis = _canonical_axis(axis)
        self._clear_side(axis, f"{axis}min")
        self._clear_side(axis, f"{axis}max")
        if self._sweep_axis == axis:
            self._sweep_axis = None
            self._sweep_update_side = None
            self._sweep_moving_env = None
        return self

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

    def _resolve_cutoff(self, tn=None):
        """Resolve the requested cutoff for the current network dtype."""
        if self.cutoff != "auto":
            return self.cutoff
        if tn is None and self._resolved_cutoff is not None:
            return self._resolved_cutoff

        sample_data = resolve_backend_sample_data_from_tn(tn)
        dtype = getattr(sample_data, "dtype", None)
        # An empty test/dummy network has no dtype to inspect. Match the
        # existing high-precision default in that case rather than forwarding
        # the string ``"auto"`` into Quimb.
        self._resolved_cutoff = float(
            dtype_auto_cutoff("float64" if dtype is None else dtype)
        )
        return self._resolved_cutoff

    def _compute_kwargs(self, *, tn=None, progress=False):
        del progress
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
                "cutoff": self._resolve_cutoff(tn),
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
        if self.cutoff_mode is not None:
            opts.setdefault("compress_opts", {})
            opts["compress_opts"].setdefault("cutoff_mode", self.cutoff_mode)
        return opts

    def update_axis(self, tn, axis, *, progress=False):
        """Recompute Quimb environments for one sweep axis.

        This is an explicit full refresh of the selected axis. Half-sweeps
        should use :meth:`start_sweep` and :meth:`advance_sweep` to reuse one
        static environment and update the other side incrementally.
        """
        axis = _canonical_axis(axis)
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
            **self._compute_kwargs(tn=tn, progress=progress),
        )
        self.clear(axis)
        self.envs.update(envs)
        self._sync_axis_mps_b(tn, axis, envs)
        self.update_count += 1
        return self

    def _has_complete_side(self, tn, axis, side):
        """Return whether cached environments cover all useful cuts."""
        size_name = "Ly" if axis == "y" else "Lx"
        size = int(getattr(tn, size_name, 0))
        if side == f"{axis}min":
            required = range(1, size)
        else:
            required = range(0, max(size - 1, 0))
        return all((side, index) in self.envs for index in required)

    def start_sweep(
        self,
        tn,
        axis,
        update_side,
        *,
        progress=False,
        reuse_static=False,
    ):
        """Prepare one half-sweep with one static and one moving boundary.

        Quimb computes the opposite-side environments once. The selected
        side is then advanced one row or column at a time with
        :meth:`advance_sweep`, after the local tensor update has been applied.
        When ``reuse_static=True``, a complete opposite-side cache from the
        preceding half-sweep is reused and only the moving side is cleared.
        This is safe when the preceding half-sweep updated every plane, as in
        :class:`pepsy.optimizers.sweep.optimizer.SweepOptimizer`.
        """
        if update_side not in {"left", "right"}:
            raise ValueError("update_side must be 'left' or 'right'")
        axis = _canonical_axis(axis)

        static_side = f"{axis}{'max' if update_side == 'left' else 'min'}"
        moving_side = f"{axis}{'min' if update_side == 'left' else 'max'}"
        if reuse_static and self._has_complete_side(tn, axis, static_side):
            self._resolve_cutoff(tn)
            self._clear_side(axis, moving_side)
            static_envs = {
                key: value
                for key, value in self.envs.items()
                if isinstance(key, tuple) and key and key[0] == static_side
            }
            self._sync_axis_mps_b(tn, axis, static_envs)
            self._sweep_axis = axis
            self._sweep_update_side = update_side
            self._sweep_moving_env = None
            self.update_count += 1
            return self

        # The public Quimb names are ``compute_ymin_environments`` etc.
        compute_fn = getattr(
            tn,
            f"compute_{axis}{'max' if update_side == 'left' else 'min'}_environments",
            None,
        )
        if not callable(compute_fn):
            raise TypeError(
                "Quimb MPS boundary engine requires a TensorNetwork2D with "
                f"compute_{axis}{'max' if update_side == 'left' else 'min'}_environments()."
            )
        envs = _call_with_accepted_kwargs(
            compute_fn,
            envs={},
            **self._compute_kwargs(tn=tn, progress=progress),
        )

        self.clear(axis)
        self.envs.update(envs)
        self._sync_axis_mps_b(tn, axis, envs)
        self._sweep_axis = axis
        self._sweep_update_side = update_side
        self._sweep_moving_env = None
        self.update_count += 1
        return self

    def _get_plane_tag_maps(self, tn, axis):
        """Return cached maps from original site tags to two-plane tags."""
        axis = _canonical_axis(axis)
        lx = int(tn.Lx)
        ly = int(tn.Ly)
        site_tags = tuple(
            tn.site_tag(x, y)
            for x in range(lx)
            for y in range(ly)
        )
        cache_key = axis, lx, ly, site_tags
        maps = self._plane_tag_maps.get(cache_key)
        if maps is not None:
            return maps

        map_zero = {}
        map_one = {}
        for x in range(lx):
            for y in range(ly):
                old_tag = tn.site_tag(x, y)
                if axis == "y":
                    map_zero[old_tag] = tn.site_tag(x, 0)
                    map_one[old_tag] = tn.site_tag(x, 1)
                else:
                    map_zero[old_tag] = tn.site_tag(0, y)
                    map_one[old_tag] = tn.site_tag(1, y)

        maps = map_zero, map_one
        self._plane_tag_maps[cache_key] = maps
        return maps

    def _extend_moving_boundary(  # pylint: disable=too-many-locals
        self, tn, index, *, axis, update_side
    ):
        """Add one updated row or column to the moving boundary MPS."""
        axis = _canonical_axis(axis)
        axis_tag = axis.upper()
        plane = tn.select(f"{axis_tag}{index}").copy()
        if update_side == "left":
            if self._sweep_moving_env is None:
                return plane
            env = self._sweep_moving_env.copy()
            env.retag_(
                {
                    tag: f"{axis_tag}0"
                    for tag in tuple(env.tags)
                    if isinstance(tag, str) and tag.startswith(axis_tag)
                }
            )
            plane.retag_({f"{axis_tag}{index}": f"{axis_tag}1"})
            first, second = env, plane
            direction = f"{axis}min"
        else:
            if self._sweep_moving_env is None:
                return plane
            env = self._sweep_moving_env.copy()
            env.retag_(
                {
                    tag: f"{axis_tag}1"
                    for tag in tuple(env.tags)
                    if isinstance(tag, str) and tag.startswith(axis_tag)
                }
            )
            plane.retag_({f"{axis_tag}{index}": f"{axis_tag}0"})
            first, second = plane, env
            direction = f"{axis}max"

        plane_tag_maps = self._get_plane_tag_maps(tn, axis)

        def _rebase_plane_tags(network, coordinate):
            """Give a compressed boundary one logical 2D plane coordinate.

            ``contract_boundary_from_*`` identifies sites using ``I{x,y}``
            tags, not the ``X*`` / ``Y*`` tags alone. A moving environment
            carries every absorbed plane's original site tags, so it must be
            rebased before combining it with one new physical plane.
            """
            tag_map = {
                old_tag: new_tag
                for old_tag, new_tag in plane_tag_maps[coordinate].items()
                if old_tag in network.tags
            }
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
        axis = _canonical_axis(axis)
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
        axis = _canonical_axis(axis)
        size_name = "Ly" if axis == "y" else "Lx"
        size = int(getattr(tn, size_name, 0))
        if size < 1:
            raise ValueError(f"Cannot infer {size_name} for Quimb {axis} environments.")

        axis_tag = axis.upper()
        for key, value in envs.items():
            if not isinstance(key, tuple) or len(key) != 2:
                continue
            side, index = key
            if side == f"{axis}min" and 0 < index < size:
                legacy_key = f"{axis_tag}{index - 1}_l"
            elif side == f"{axis}max" and 0 <= index < size - 1:
                legacy_key = f"{axis_tag}{size - 2 - index}_r"
            else:
                continue
            self.mps_b[legacy_key] = value
