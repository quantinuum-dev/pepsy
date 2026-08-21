"""Infinite/unit-cell MPO representation with explicit finite boundaries."""

from __future__ import annotations

from numbers import Integral

import autoray as ar
import numpy as np

from .mpo import FirstDegreeMPO
from .mpo_automaton import _as_backend, _backend_reference
from .mpo_space import MPOPhysicalSpace

__all__ = ["InfiniteMPO"]


class InfiniteMPO:
    """An MPO tensor unit cell whose virtual bonds close periodically.

    This type deliberately has no implicit finite boundary convention.
    :meth:`finite_window` requires left/right boundary vectors whenever the
    chosen seam has dimension greater than one. This prevents an infinite
    transfer object from being silently interpreted as either an open MPO or
    a virtual trace.
    """

    def __init__(
        self,
        arrays,
        *,
        physical_space=None,
        degree=1,
        upper_ind_id="k{}",
        lower_ind_id="b{}",
        site_tag_id="I{}",
        metadata=None,
    ):
        arrays = tuple(arrays)
        if not arrays:
            raise ValueError("an infinite MPO unit cell must contain tensors.")
        shapes = tuple(tuple(getattr(array, "shape", ())) for array in arrays)
        if any(len(shape) != 4 for shape in shapes):
            raise ValueError(
                "infinite MPO tensors must have shape (left, right, upper, lower)."
            )
        if any(shape[2] != shape[3] for shape in shapes):
            raise ValueError("infinite MPO physical dimensions must be square.")
        phys_dim = int(shapes[0][2])
        if any(shape[2:] != (phys_dim, phys_dim) for shape in shapes):
            raise ValueError("all unit-cell tensors must use one physical dimension.")
        for site, (left, right) in enumerate(zip(shapes, shapes[1:] + shapes[:1])):
            if left[1] != right[0]:
                raise ValueError(
                    f"unit-cell virtual bond {site} has dimensions "
                    f"{left[1]} and {right[0]}."
                )
        if not isinstance(degree, Integral) or isinstance(degree, bool) or int(degree) < 0:
            raise ValueError("degree must be a non-negative integer.")
        if physical_space is None:
            physical_space = MPOPhysicalSpace(phys_dim)
        if not isinstance(physical_space, MPOPhysicalSpace):
            raise TypeError("physical_space must be an MPOPhysicalSpace or None.")
        if physical_space.phys_dim != phys_dim:
            raise ValueError(
                f"physical_space has phys_dim={physical_space.phys_dim}, "
                f"but unit-cell tensors use {phys_dim}."
            )

        self._arrays = arrays
        self.physical_space = physical_space
        self.degree = int(degree)
        self.upper_ind_id = str(upper_ind_id)
        self.lower_ind_id = str(lower_ind_id)
        self.site_tag_id = str(site_tag_id)
        self.metadata = dict(metadata or {})

    @classmethod
    def from_finite_cell(cls, mpo, **kwargs):
        """Treat one finite semantic MPO as a periodically repeated cell.

        Open boundary dimensions are accepted only because their singleton
        seam is a valid cyclic bond. Repetition represents a product of
        independent cells at that seam; it is not an inferred infinite
        Hamiltonian automaton.
        """

        if not isinstance(mpo, FirstDegreeMPO):
            raise TypeError("mpo must be a FirstDegreeMPO.")
        options = {
            "physical_space": mpo.physical_space,
            "degree": mpo.degree,
            "upper_ind_id": mpo.upper_ind_id,
            "lower_ind_id": mpo.lower_ind_id,
            "site_tag_id": mpo.site_tag_id,
            "metadata": {**mpo.metadata, "geometry": "infinite_unit_cell"},
        }
        options.update(kwargs)
        return cls(mpo.arrays, **options)

    @property
    def arrays(self):
        """Read-only tuple view of unit-cell tensors."""

        return self._arrays

    @property
    def unit_cell_length(self):
        """Number of physical sites in one periodic unit cell."""

        return len(self._arrays)

    @property
    def phys_dim(self):
        """Local physical dimension."""

        return self.physical_space.phys_dim

    @property
    def bond_dimensions(self):
        """Right virtual dimensions around the periodic cell."""

        return tuple(int(array.shape[1]) for array in self._arrays)

    def copy(self):
        """Return a structural copy sharing backend tensor storage."""

        return type(self)(
            self._arrays,
            physical_space=self.physical_space,
            degree=self.degree,
            upper_ind_id=self.upper_ind_id,
            lower_ind_id=self.lower_ind_id,
            site_tag_id=self.site_tag_id,
            metadata=self.metadata,
        )

    def shift(self, offset=1):
        """Return the same infinite MPO with a rotated unit-cell origin."""

        if not isinstance(offset, Integral) or isinstance(offset, bool):
            raise TypeError("offset must be an integer.")
        offset = int(offset) % self.unit_cell_length
        arrays = self._arrays[offset:] + self._arrays[:offset]
        return type(self)(
            arrays,
            physical_space=self.physical_space,
            degree=self.degree,
            upper_ind_id=self.upper_ind_id,
            lower_ind_id=self.lower_ind_id,
            site_tag_id=self.site_tag_id,
            metadata={**self.metadata, "unit_cell_shift": offset},
        )

    def repeat_cell(self, cells):
        """Return an equivalent enlarged unit cell containing ``cells`` copies."""

        if not isinstance(cells, Integral) or isinstance(cells, bool) or int(cells) < 1:
            raise ValueError("cells must be a positive integer.")
        return type(self)(
            self._arrays * int(cells),
            physical_space=self.physical_space,
            degree=self.degree,
            upper_ind_id=self.upper_ind_id,
            lower_ind_id=self.lower_ind_id,
            site_tag_id=self.site_tag_id,
            metadata={**self.metadata, "repeated_unit_cells": int(cells)},
        )

    @staticmethod
    def _boundary_vector(value, dimension, *, name, like):
        if value is None:
            if dimension != 1:
                raise ValueError(
                    f"{name} is required for a seam of dimension {dimension}."
                )
            value = np.ones(1)
        shape = tuple(getattr(value, "shape", np.shape(value)))
        if shape != (dimension,):
            raise ValueError(f"{name} must have shape ({dimension},), got {shape}.")
        return _as_backend(value, like=like)

    def finite_window(
        self,
        cells=1,
        *,
        start=0,
        left_boundary=None,
        right_boundary=None,
    ):
        """Cut out an exact open finite window using explicit seam vectors."""

        if not isinstance(cells, Integral) or isinstance(cells, bool) or int(cells) < 1:
            raise ValueError("cells must be a positive integer.")
        if not isinstance(start, Integral) or isinstance(start, bool):
            raise TypeError("start must be an integer.")
        shifted = self.shift(int(start))
        arrays = list(shifted.arrays * int(cells))
        reference = _backend_reference((arrays[0], arrays[-1]))
        left = self._boundary_vector(
            left_boundary,
            int(arrays[0].shape[0]),
            name="left_boundary",
            like=reference,
        )
        right = self._boundary_vector(
            right_boundary,
            int(arrays[-1].shape[1]),
            name="right_boundary",
            like=reference,
        )

        first = ar.do("tensordot", left, arrays[0], axes=([0], [0]))
        arrays[0] = ar.do(
            "reshape",
            first,
            (1, first.shape[0], first.shape[1], first.shape[2]),
        )
        last = ar.do("tensordot", arrays[-1], right, axes=([1], [0]))
        arrays[-1] = ar.do(
            "reshape",
            last,
            (last.shape[0], 1, last.shape[1], last.shape[2]),
        )
        return FirstDegreeMPO(
            arrays,
            degree=self.degree,
            physical_space=self.physical_space,
            upper_ind_id=self.upper_ind_id,
            lower_ind_id=self.lower_ind_id,
            site_tag_id=self.site_tag_id,
            metadata={
                **self.metadata,
                "geometry": "finite_window",
                "unit_cells": int(cells),
                "unit_cell_start": int(start) % self.unit_cell_length,
            },
        )

    def to_mpo(self, cells=1, **boundary_options):
        """Compile one explicitly bounded finite window to a Quimb MPO."""

        return self.finite_window(cells, **boundary_options).to_mpo()
