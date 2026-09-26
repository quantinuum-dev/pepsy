"""Shared sampling maps, array conversion, and fermionic code ordering."""

from __future__ import annotations

import autoray as ar
import numpy as np

__all__ = []


def _validate_one_d_to_two_d(
    one_d_to_two_d: dict[int, tuple[int, int]],
    *,
    expected_L: int | None = None,
) -> int:
    """Validate a complete 0..L-1 site map and return ``L``."""
    if not one_d_to_two_d:
        raise ValueError("one_d_to_two_d must contain at least one site.")
    L = int(expected_L) if expected_L is not None else len(one_d_to_two_d)
    if L < 1:
        raise ValueError("expected_L must be >= 1.")
    expected = set(range(L))
    got = set(one_d_to_two_d)
    if got != expected:
        raise ValueError(
            "one_d_to_two_d keys must be exactly consecutive site indices "
            f"0..{L - 1}; got {sorted(got)!r}."
        )
    for site, coord in one_d_to_two_d.items():
        if not (
            isinstance(coord, tuple)
            and len(coord) == 2
            and all(isinstance(value, int) for value in coord)
        ):
            raise TypeError(
                "one_d_to_two_d values must be (x, y) integer tuples; "
                f"site {site!r} maps to {coord!r}."
            )
    return L


def _mps_array_backend(array):
    module = type(array).__module__.split(".", 1)[0]
    if module == "torch":
        return "torch"
    if module == "cupy":
        return "cupy"
    if isinstance(array, np.ndarray):
        return "numpy"
    if hasattr(array, "blocks"):
        return "symmray"
    return "unknown"


def _backend_array_to_numpy(array):
    if hasattr(array, "to_dense"):
        array = array.to_dense()
    return np.asarray(ar.to_numpy(array))


def _fermion_code_order(encoding, *, default=(0, 1, 2, 3)):
    """Return physical codes ordered by ``(n_up, n_down)`` bits."""
    if encoding is None:
        codes = {
            "empty": int(default[0]),
            "down": int(default[1]),
            "up": int(default[2]),
            "double": int(default[3]),
        }
    else:
        try:
            codes = {
                "empty": int(encoding.empty),
                "double": int(encoding.double),
                "up": int(encoding.up),
                "down": int(encoding.down),
            }
        except (AttributeError, TypeError, ValueError) as exc:
            raise TypeError(
                "encoding must expose integer empty, double, up, and down "
                "codes, for example FermionSiteEncoding."
            ) from exc
    values = (
        codes["empty"],
        codes["down"],
        codes["up"],
        codes["double"],
    )
    if sorted(values) != [0, 1, 2, 3]:
        raise ValueError(
            "A spinful fermion encoding must contain exactly the physical "
            "codes 0, 1, 2, and 3."
        )
    return values


def _infer_fermion_code_order(tn, sites):
    """Infer a four-state code order from PEPS symmetry metadata when possible."""
    default = (0, 1, 2, 3)
    if not sites:
        return default
    tensor = tn[sites[0]]
    data = getattr(tensor, "data", None)
    symmetry = str(getattr(data, "symmetry", "")).upper()
    if symmetry == "Z2":
        # The two even states precede the two odd states in Symmray's
        # charge-collapsed physical index.
        return (0, 3, 2, 1)
    if symmetry not in {"U1U1", "Z2Z2"}:
        return default
    try:
        site_ind = tn.site_ind(sites[0])
        axis = tensor.inds.index(site_ind)
        charges = tuple(data.indices[axis].chargemap)
        position = {tuple(charge): i for i, charge in enumerate(charges)}
        required = ((0, 0), (0, 1), (1, 0), (1, 1))
        if all(charge in position for charge in required):
            return tuple(position[charge] for charge in required)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        pass
    return default
