"""Graded/Symmray Torch contraction helpers.

This module owns the optional native Symmray graded-contraction projector.
"""

from __future__ import annotations

from dataclasses import dataclass

import autoray as ar
import numpy as np

from ..torch_types import _require_torch

__all__ = [
    "_GradedTorchPair",
    "_GradedTorchProjector",
    "_graded_torch_compile_pair",
    "_graded_torch_contraction_mask",
    "_graded_torch_dense",
    "_graded_torch_embed_dense",
    "_graded_torch_index_map",
    "_graded_torch_pad",
    "_graded_torch_prepare_pair",
    "_graded_torch_sign_mask",
    "_graded_torch_unit_probe",
    "_is_symmray_data",
    "_find_symmray_tensors",
]


def _is_symmray_data(data):
    cls = type(data)
    return cls.__module__.split(".", 1)[0] == "symmray"


def _find_symmray_tensors(tn):
    tensor_map = getattr(tn, "tensor_map", {})
    return [
        tensor_id
        for tensor_id, tensor in tensor_map.items()
        if _is_symmray_data(getattr(tensor, "data", None))
    ]


def _graded_torch_index_map(index):
    """Return the dense charge label at every position of ``index``."""
    return tuple(
        charge
        for charge, size in index.chargemap.items()
        for _ in range(int(size))
    )


def _graded_torch_embed_dense(array, labels, full_maps):
    """Embed a sparse result into the fixed dense index layout."""
    shape = tuple(len(full_maps[label]) for label in labels)
    if getattr(array, "num_blocks", 0):
        dense = np.asarray(ar.to_numpy(array.to_dense()))
    else:
        dense = np.zeros(shape, dtype=float)

    selectors = []
    for index, label in zip(array.indices, labels):
        positions = []
        for charge in _graded_torch_index_map(index):
            positions.append(full_maps[label].index(charge))
        selectors.append(positions)

    if not labels:
        return dense.reshape(())

    full = np.zeros(shape, dtype=dense.dtype)
    if all(selectors):
        full[np.ix_(*selectors)] = dense
    return full


def _graded_torch_pad(array, labels, full_maps):
    """Pad a Symmray result while retaining its graded metadata."""
    dense = _graded_torch_embed_dense(array, labels, full_maps)
    return type(array).from_dense(
        dense,
        [full_maps[label] for label in labels],
        duals=array.duals,
        charge=array.charge,
        symmetry=array.symmetry,
        invalid_sectors="ignore",
        dummy_modes=(),
    )


def _graded_torch_dense(array, labels, full_maps):
    """Get a padded dense probe, including the empty-array case."""
    return _graded_torch_embed_dense(array, labels, full_maps)


def _graded_torch_sign_mask(left, right):
    """Extract a +/-1 phase mask from two equal-layout dense arrays."""
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        raise ValueError(
            "Symmray graded projector produced incompatible dense shapes: "
            f"{left.shape} and {right.shape}."
        )
    mask = np.ones_like(left, dtype=float)
    nonzero = np.abs(right) > 1.0e-12
    if np.any(nonzero):
        ratio = np.real(left[nonzero] / right[nonzero])
        if np.max(np.abs(np.abs(ratio) - 1.0)) > 1.0e-7:
            raise ValueError(
                "Could not compile a fixed graded Torch phase mask: "
                "the Symmray phase was not a +/-1 sector phase; "
                f"observed ratios {np.unique(ratio)!r}."
            )
        mask[nonzero] = ratio
    return mask


def _graded_torch_unit_probe(
    array_cls,
    reference,
    labels,
    charge,
    full_maps,
    duals,
    rng,
):
    """Construct a random symmetric probe for static phase compilation."""
    shape = tuple(len(full_maps[label]) for label in labels)
    dense = rng.normal(size=shape)
    return array_cls.from_dense(
        dense,
        [full_maps[label] for label in labels],
        duals=duals,
        charge=charge,
        symmetry=reference.symmetry,
        invalid_sectors="ignore",
        dummy_modes=(),
    )


def _graded_torch_prepare_pair(a, b, axes):
    """Use Symmray's public graded operation with a narrow compatibility shim.

    Symmray currently exposes the complete preparation logic through the
    internal helper used by its public ``tensordot`` implementation. Keeping
    this call in one adapter makes the dependency/version assumption explicit,
    rather than reproducing the upstream Koszul-phase rules here.
    """
    prepare = getattr(a, "_prepare_for_tensordot_fermionic", None)
    if prepare is None:
        raise TypeError(
            "The installed Symmray version does not expose the graded "
            "tensordot preparation needed by graded_torch."
        )
    return prepare(b, axes)


def _graded_torch_contraction_mask(a, b, axes, perm_b):
    """Keep only equal-charge positions on contracted dense axes."""
    mask = np.ones(tuple(b.shape[axis] for axis in perm_b), dtype=float)
    for axis_a, axis_b in zip(*axes):
        charges_a = _graded_torch_index_map(a.indices[axis_a])
        charges_b = _graded_torch_index_map(b.indices[axis_b])
        if len(charges_a) != len(charges_b):
            raise ValueError(
                "Symmray graded projector found mismatched contracted index "
                f"sizes {len(charges_a)} and {len(charges_b)}."
            )
        valid = np.asarray(
            [charge_a == charge_b for charge_a, charge_b in zip(charges_a, charges_b)],
            dtype=float,
        )
        axis_b_new = perm_b.index(axis_b)
        shape = [1] * len(perm_b)
        shape[axis_b_new] = len(valid)
        mask *= valid.reshape(shape)
    return mask


def _graded_torch_compile_pair(
    a,
    b,
    labels_a,
    labels_b,
    axes,
    full_maps,
    rng,
):
    """Compile dense input/output masks for one graded contraction."""
    left_axes = tuple(k for k in range(a.ndim) if k not in axes[0])
    right_axes = tuple(k for k in range(b.ndim) if k not in axes[1])
    perm_a = (*left_axes, *axes[0])
    perm_b = (*axes[1], *right_axes)
    aa, bb, new_axes_a, new_axes_b = _graded_torch_prepare_pair(a, b, axes)

    a_dense = a.to_dense()
    b_dense = b.to_dense()
    mask_a = _graded_torch_sign_mask(
        aa.to_dense(), np.transpose(a_dense, perm_a)
    )
    mask_b = _graded_torch_sign_mask(
        bb.to_dense(), np.transpose(b_dense, perm_b)
    )
    mask_b = mask_b * _graded_torch_contraction_mask(a, b, axes, perm_b)
    raw = np.tensordot(
        np.transpose(a_dense, perm_a) * mask_a,
        np.transpose(b_dense, perm_b) * mask_b,
        axes=(new_axes_a, new_axes_b),
    )

    c = a.tensordot(b, axes=axes, preserve_array=True)
    labels_c = tuple(
        labels_a[k] for k in left_axes
    ) + tuple(labels_b[k] for k in right_axes)
    c_padded = _graded_torch_pad(c, labels_c, full_maps)
    c_dense = _graded_torch_dense(c_padded, labels_c, full_maps)
    try:
        mask_c = _graded_torch_sign_mask(c_dense, raw)
    except ValueError as exc:
        raise ValueError(
            "Failed compiling graded Torch pair "
            f"{labels_a!r} x {labels_b!r}, axes={axes!r}, "
            f"charges={a.charge!r},{b.charge!r}, duals={a.duals!r},{b.duals!r}."
        ) from exc
    return c_padded, mask_a, mask_b, mask_c


@dataclass(frozen=True)
class _GradedTorchPair:
    """One static dense contraction in the graded Torch projector."""

    left: int
    right: int
    axes_a: tuple[int, ...]
    axes_b: tuple[int, ...]
    perm_a: tuple[int, ...]
    perm_b: tuple[int, ...]
    ncon: int
    charge_masks_a: tuple[np.ndarray, ...]
    charge_masks_b: tuple[np.ndarray, ...]
    output_masks: tuple[np.ndarray, ...]
    output_charge_ids: np.ndarray
    n_charge_b: int
    left_sites: tuple[int, ...]
    right_sites: tuple[int, ...]


class _GradedTorchProjector:
    """Compile a fixed-shape, graded dense projector for native Symmray PEPS.

    The dense values remain differentiable Torch tensors. Symmray is used at
    construction time to compile charge-sector transitions and the exact
    fermionic phase masks for the chosen contraction tree. This avoids the
    dynamic sparse ``isel`` path, whose scalar charge lookup is not vmap-safe.
    """

    def __init__(self, tn, sites, *, contraction_opts=None):
        import quimb.tensor as qtn

        self.sites = tuple(sites)
        self.n_sites = len(self.sites)
        self.full_maps = {}
        self.array_cls = None
        self.reference = None
        self.leaf_labels = []
        self.leaf_duals = []
        self.physical_axes = []
        self.leaf_charge_values = []
        self.leaf_charge_ids = []
        self.leaf_dummy_modes = []
        self.leaf_dummy_parities = []
        self._mode_objects = {}

        for site in self.sites:
            tensor = tn[site]
            data = getattr(tensor, "data", None)
            if not _is_symmray_data(data) or not getattr(data, "fermionic", False):
                raise TypeError(
                    "graded_torch requires native Symmray fermionic arrays."
                )
            if not hasattr(data, "_prepare_for_tensordot_fermionic"):
                raise TypeError(
                    "graded_torch requires a Symmray sparse fermionic backend."
                )
            if self.array_cls is None:
                self.array_cls = type(data)
                self.reference = data
                if str(data.symmetry).upper() != "U1U1":
                    raise NotImplementedError(
                        "graded_torch currently targets native U1U1 fermionic "
                        "arrays; use the established native path for other "
                        "Symmray symmetries."
                    )

            physical_ind = tn.site_ind(site)
            try:
                physical_axis = tuple(tensor.inds).index(physical_ind)
            except ValueError as exc:
                raise ValueError(
                    f"Could not locate physical index for PEPS site {site!r}."
                ) from exc
            self.physical_axes.append(physical_axis)
            labels = tuple(
                ind for axis, ind in enumerate(tensor.inds)
                if axis != physical_axis
            )
            self.leaf_labels.append(labels)
            virtual_indices = tuple(
                index for axis, index in enumerate(data.indices)
                if axis != physical_axis
            )
            self.leaf_duals.append(tuple(index.dual for index in virtual_indices))
            for label, index in zip(labels, virtual_indices):
                index_map = list(_graded_torch_index_map(index))
                old_map = self.full_maps.setdefault(label, index_map)
                if old_map != index_map:
                    raise ValueError(
                        f"Symmray index {label!r} has inconsistent charge order."
                    )

            charge_values = []
            dummy_modes = []
            dummy_parities = []
            physical_dim = int(data.shape[physical_axis])
            for physical_value in range(physical_dim):
                selected = data.isel(physical_axis, physical_value)
                charge = self._charge_key(selected.charge)
                if charge not in charge_values:
                    charge_values.append(charge)
                modes = tuple(selected.dummy_modes)
                if any(mode.dual for mode in modes):
                    raise NotImplementedError(
                        "graded_torch currently supports ket PEPS with "
                        "non-dual dummy fermion modes only."
                    )
                mode_keys = tuple(self._register_mode(mode) for mode in modes)
                dummy_modes.append(mode_keys)
                dummy_parities.append(sum(mode.parity for mode in modes) % 2)
            self.leaf_charge_values.append(tuple(charge_values))
            self.leaf_charge_ids.append(
                tuple(charge_values.index(self._charge_key(data.isel(
                    physical_axis, value
                ).charge)) for value in range(physical_dim))
            )
            self.leaf_dummy_modes.append(tuple(dummy_modes))
            self.leaf_dummy_parities.append(tuple(dummy_parities))

        self._mode_order = self._build_mode_order()
        self._dummy_inversions = self._build_dummy_inversions()

        path_tensors = [
            qtn.Tensor(
                np.zeros(tuple(len(self.full_maps[label]) for label in labels)),
                inds=labels,
            )
            for labels in self.leaf_labels
        ]
        optimize = "auto-hq"
        if contraction_opts:
            final_opts = contraction_opts.get("final_contract_opts") or {}
            optimize = final_opts.get("optimize", optimize)
        path = qtn.TensorNetwork(path_tensors).contract(
            all, get="path", optimize=optimize
        )

        rng = np.random.default_rng(928371)
        active_labels = list(self.leaf_labels)
        active_duals = list(self.leaf_duals)
        active_charges = [list(values) for values in self.leaf_charge_values]
        active_sites = [(site,) for site in range(self.n_sites)]
        self.pairs = []

        for left, right in path:
            labels_a = active_labels[left]
            labels_b = active_labels[right]
            left_sites = active_sites[left]
            right_sites = active_sites[right]
            axes_a = tuple(
                axis for axis, label in enumerate(labels_a)
                if label in labels_b
            )
            axes_b = tuple(labels_b.index(labels_a[axis]) for axis in axes_a)
            if len(axes_a) != len(axes_b):
                raise ValueError("Invalid contraction path for graded_torch.")
            charge_masks_a = []
            charge_masks_b = []
            output_masks = []
            output_charges = []
            for charge_a in active_charges[left]:
                for charge_b in active_charges[right]:
                    a = _graded_torch_unit_probe(
                        self.array_cls,
                        self.reference,
                        labels_a,
                        charge_a,
                        self.full_maps,
                        active_duals[left],
                        rng,
                    )
                    b = _graded_torch_unit_probe(
                        self.array_cls,
                        self.reference,
                        labels_b,
                        charge_b,
                        self.full_maps,
                        active_duals[right],
                        rng,
                    )
                    if not getattr(a, "num_blocks", 0) or not getattr(
                        b, "num_blocks", 0
                    ):
                        labels_c = tuple(
                            labels_a[axis]
                            for axis in range(len(labels_a))
                            if axis not in axes_a
                        ) + tuple(
                            labels_b[axis]
                            for axis in range(len(labels_b))
                            if axis not in axes_b
                        )
                        duals_c = tuple(
                            active_duals[left][axis]
                            for axis in range(len(labels_a))
                            if axis not in axes_a
                        ) + tuple(
                            active_duals[right][axis]
                            for axis in range(len(labels_b))
                            if axis not in axes_b
                        )
                        charge_c = self._charge_key(
                            self.reference.symmetry.combine(charge_a, charge_b)
                        )
                        shape_c = tuple(
                            len(self.full_maps[label]) for label in labels_c
                        )
                        zero = np.zeros(shape_c, dtype=float)
                        c = self.array_cls.from_dense(
                            zero,
                            [self.full_maps[label] for label in labels_c],
                            duals=duals_c,
                            charge=charge_c,
                            symmetry=self.reference.symmetry,
                            invalid_sectors="ignore",
                            dummy_modes=(),
                        )
                        mask_a = np.ones(a.shape, dtype=float)
                        mask_b = np.ones(b.shape, dtype=float)
                        mask_c = np.ones(shape_c, dtype=float)
                    else:
                        c, mask_a, mask_b, mask_c = _graded_torch_compile_pair(
                            a,
                            b,
                            labels_a,
                            labels_b,
                            (axes_a, axes_b),
                            self.full_maps,
                            rng,
                        )
                        charge_c = self._charge_key(c.charge)
                    charge_masks_a.append(mask_a)
                    charge_masks_b.append(mask_b)
                    output_masks.append(mask_c)
                    output_charges.append(charge_c)

            n_charge_b = len(active_charges[right])
            charge_ids_c = tuple(dict.fromkeys(output_charges))
            output_charge_ids = np.asarray(
                [charge_ids_c.index(charge) for charge in output_charges],
                dtype=np.int64,
            ).reshape(len(active_charges[left]), n_charge_b)
            labels_c = tuple(
                labels_a[axis] for axis in range(len(labels_a))
                if axis not in axes_a
            ) + tuple(
                labels_b[axis] for axis in range(len(labels_b))
                if axis not in axes_b
            )
            active_labels[left] = labels_c
            active_duals[left] = tuple(
                active_duals[left][axis] for axis in range(len(labels_a))
                if axis not in axes_a
            ) + tuple(
                active_duals[right][axis] for axis in range(len(labels_b))
                if axis not in axes_b
            )
            active_charges[left] = list(charge_ids_c)
            active_sites[left] = active_sites[left] + active_sites[right]
            active_labels.pop(right)
            active_duals.pop(right)
            active_charges.pop(right)
            active_sites.pop(right)

            self.pairs.append(
                _GradedTorchPair(
                    left=left,
                    right=right,
                    axes_a=axes_a,
                    axes_b=axes_b,
                    perm_a=tuple(
                        axis for axis in range(len(labels_a)) if axis not in axes_a
                    ) + axes_a,
                    perm_b=axes_b + tuple(
                        axis for axis in range(len(labels_b)) if axis not in axes_b
                    ),
                    ncon=len(axes_a),
                    charge_masks_a=tuple(charge_masks_a),
                    charge_masks_b=tuple(charge_masks_b),
                    output_masks=tuple(output_masks),
                    output_charge_ids=output_charge_ids,
                    n_charge_b=n_charge_b,
                    left_sites=left_sites,
                    right_sites=right_sites,
                )
            )

        self._torch_cache = {}

    @staticmethod
    def _charge_key(charge):
        if isinstance(charge, np.ndarray):
            charge = charge.tolist()
        if isinstance(charge, (list, tuple)):
            return tuple(int(value) for value in charge)
        return int(charge)

    def _register_mode(self, mode):
        key = (repr(mode.label), bool(mode.dual), int(mode.parity))
        self._mode_objects.setdefault(key, mode)
        return key

    def _build_mode_order(self):
        keys = list(self._mode_objects)
        try:
            keys.sort(key=lambda key: self._mode_objects[key])
        except TypeError as exc:
            raise NotImplementedError(
                "graded_torch could not order Symmray dummy fermion modes."
            ) from exc
        return {key: position for position, key in enumerate(keys)}

    def _build_dummy_inversions(self):
        inversions = []
        for left in range(self.n_sites):
            row = []
            for right in range(self.n_sites):
                table = np.zeros(
                    (
                        len(self.leaf_dummy_modes[left]),
                        len(self.leaf_dummy_modes[right]),
                    ),
                    dtype=np.int64,
                )
                for code_a, modes_a in enumerate(self.leaf_dummy_modes[left]):
                    for code_b, modes_b in enumerate(self.leaf_dummy_modes[right]):
                        table[code_a, code_b] = sum(
                            self._mode_order[mode_b] < self._mode_order[mode_a]
                            for mode_a in modes_a
                            for mode_b in modes_b
                        ) % 2
                row.append(table)
            inversions.append(row)
        return inversions

    def _torch_tables(self, reference):
        torch = _require_torch()
        key = (reference.device, reference.dtype)
        cached = self._torch_cache.get(key)
        if cached is not None:
            return cached
        charge_luts = tuple(
            torch.as_tensor(ids, dtype=torch.long, device=reference.device)
            for ids in self.leaf_charge_ids
        )
        parity_luts = tuple(
            torch.as_tensor(values, dtype=torch.long, device=reference.device)
            for values in self.leaf_dummy_parities
        )
        pairs = []
        for pair in self.pairs:
            pairs.append(
                (
                    torch.stack([
                        torch.as_tensor(mask, dtype=reference.dtype, device=reference.device)
                        for mask in pair.charge_masks_a
                    ]),
                    torch.stack([
                        torch.as_tensor(mask, dtype=reference.dtype, device=reference.device)
                        for mask in pair.charge_masks_b
                    ]),
                    torch.stack([
                        torch.as_tensor(mask, dtype=reference.dtype, device=reference.device)
                        for mask in pair.output_masks
                    ]),
                    torch.as_tensor(
                        pair.output_charge_ids.reshape(-1),
                        dtype=torch.long,
                        device=reference.device,
                    ),
                )
            )
        cached = charge_luts, parity_luts, pairs
        self._torch_cache[key] = cached
        return cached

    def _dummy_phase(self, config, left_sites, right_sites, parity_luts):
        torch = _require_torch()
        def lookup_1d(table, index):
            return torch.index_select(table, 0, index.reshape(1)).squeeze(0)

        left_parity = sum(
            lookup_1d(parity_luts[site], config[site]) for site in left_sites
        ) % 2
        right_parity = sum(
            lookup_1d(parity_luts[site], config[site]) for site in right_sites
        ) % 2
        exponent = left_parity * right_parity
        for site_a in left_sites:
            for site_b in right_sites:
                table = torch.as_tensor(
                    self._dummy_inversions[site_a][site_b],
                    dtype=torch.long,
                    device=config.device,
                )
                row = lookup_1d(table, config[site_a])
                exponent = exponent + lookup_1d(row, config[site_b])
        return torch.where(
            exponent % 2 == 0,
            torch.ones((), dtype=parity_luts[0].dtype, device=config.device),
            torch.full((), -1, dtype=parity_luts[0].dtype, device=config.device),
        )

    def __call__(self, dense_leaves, configs, *, use_vmap=True):
        torch = _require_torch()
        reference = dense_leaves[0]
        charge_luts, parity_luts, pairs = self._torch_tables(reference)

        def select(data, axis, value):
            return torch.index_select(data, axis, value.reshape(1)).squeeze(axis)

        def lookup_1d(table, index):
            return torch.index_select(table, 0, index.reshape(1)).squeeze(0)

        def evaluate(config):
            arrays = [
                select(data, axis, config[site])
                for site, (data, axis) in enumerate(
                    zip(dense_leaves, self.physical_axes)
                )
            ]
            charge_ids = [
                lookup_1d(charge_luts[site], config[site])
                for site in range(self.n_sites)
            ]
            for pair, (mask_a, mask_b, mask_c, output_ids) in zip(
                self.pairs, pairs
            ):
                lookup = charge_ids[pair.left] * pair.n_charge_b + charge_ids[
                    pair.right
                ]
                a = arrays[pair.left].permute(pair.perm_a) * lookup_1d(
                    mask_a, lookup
                )
                b = arrays[pair.right].permute(pair.perm_b) * lookup_1d(
                    mask_b, lookup
                )
                raw = torch.tensordot(
                    a,
                    b,
                    dims=(
                        tuple(range(a.ndim - pair.ncon, a.ndim)),
                        tuple(range(pair.ncon)),
                    ),
                )
                phase = self._dummy_phase(
                    config,
                    pair.left_sites,
                    pair.right_sites,
                    parity_luts,
                ).to(dtype=raw.dtype)
                arrays[pair.left] = raw * lookup_1d(mask_c, lookup) * phase
                charge_ids[pair.left] = lookup_1d(output_ids, lookup)
                arrays.pop(pair.right)
                charge_ids.pop(pair.right)
            return arrays[0]

        if use_vmap:
            return torch.vmap(evaluate)(configs)
        return torch.stack([evaluate(config) for config in configs])
