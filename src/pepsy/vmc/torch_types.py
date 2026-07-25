"""Small immutable configuration types shared by the Torch VMC kernels."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

__all__ = ["FermionSiteEncoding", "SpinlessSiteEncoding", "TorchSquareLattice"]


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "pepsy.vmc.torch requires optional dependency 'torch'. "
            "Install it with `pip install pepsy[torch]` or `pip install torch`."
        ) from exc
    return torch


def _check_positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


@dataclass(frozen=True)
class FermionSiteEncoding:
    """Four-state spinful-fermion on-site encoding."""

    empty: int = 0
    double: int = 1
    up: int = 2
    down: int = 3

    def __post_init__(self):
        values = (self.empty, self.double, self.up, self.down)
        if len(set(values)) != 4 or any(v < 0 for v in values):
            raise ValueError("Fermion site codes must be unique non-negative ints.")

    @classmethod
    def symmray(cls):
        """Return Pepsy/Symmray's spinful physical-index convention."""
        return cls(empty=0, double=1, up=2, down=3)

    @classmethod
    def vmc_torch(cls):
        """Return the convention used by the native Torch VMC path."""
        return cls(empty=0, double=3, up=2, down=1)

    @classmethod
    def from_fermion(cls, fermion, *, physical_charges=None):
        """Return the PEPS physical-index encoding for a native Fermion."""
        if not bool(getattr(fermion, "spinful", False)):
            raise ValueError("FermionSiteEncoding.from_fermion requires spinful=True.")
        if physical_charges:
            try:
                return cls.from_physical_charges(physical_charges)
            except ValueError:
                pass
        return cls.vmc_torch()

    @classmethod
    def from_physical_charges(cls, physical_charges):
        """Return an encoding from four resolved ``(n_up, n_down)`` charges."""
        lookup = {}
        for code, charge in enumerate(tuple(physical_charges)):
            if not isinstance(charge, tuple) or len(charge) != 2:
                raise ValueError("Physical charges must be two-component tuples.")
            charge = tuple(int(value) for value in charge)
            if charge in lookup:
                raise ValueError("PEPS physical charges must be unique.")
            lookup[charge] = code
        required = {(0, 0), (0, 1), (1, 0), (1, 1)}
        if set(lookup) != required:
            raise ValueError(
                "Physical charges must contain exactly the four spinful states."
            )
        return cls(
            empty=lookup[(0, 0)],
            down=lookup[(0, 1)],
            up=lookup[(1, 0)],
            double=lookup[(1, 1)],
        )

    @property
    def max_code(self):
        return max(self.empty, self.double, self.up, self.down)

    def validate(self, configs):
        """Raise if ``configs`` contains a code outside this encoding."""
        torch = _require_torch()
        valid = (
            (configs == self.empty)
            | (configs == self.double)
            | (configs == self.up)
            | (configs == self.down)
        )
        if not torch.all(valid):
            bad = torch.unique(configs[~valid]).detach().cpu().tolist()
            raise ValueError(f"Unknown fermion site code(s): {bad!r}.")

    def decode(self, configs):
        """Return ``(n_up, n_down)`` tensors for encoded site configs."""
        torch = _require_torch()
        configs = torch.as_tensor(configs, dtype=torch.long)
        self.validate(configs)
        lookup = torch.zeros(
            (self.max_code + 1, 2), dtype=torch.long, device=configs.device
        )
        lookup[self.up, 0] = 1
        lookup[self.down, 1] = 1
        lookup[self.double, 0] = 1
        lookup[self.double, 1] = 1
        occ = lookup[configs]
        return occ[..., 0], occ[..., 1]

    def encode(self, n_up, n_down):
        """Encode ``(n_up, n_down)`` occupation tensors as site states."""
        torch = _require_torch()
        n_up = torch.as_tensor(n_up)
        n_down = torch.as_tensor(n_down, device=n_up.device)
        code = torch.full_like(n_up.long(), self.empty)
        code = torch.where((n_up == 1) & (n_down == 0), self.up, code)
        code = torch.where((n_up == 0) & (n_down == 1), self.down, code)
        code = torch.where((n_up == 1) & (n_down == 1), self.double, code)
        return code


@dataclass(frozen=True)
class SpinlessSiteEncoding:
    """Binary local encoding for spinless fermions."""

    empty: int = 0
    occupied: int = 1

    def __post_init__(self):
        if self.empty == self.occupied or min(self.empty, self.occupied) < 0:
            raise ValueError("Spinless site codes must be distinct and non-negative.")

    @classmethod
    def from_physical_charges(cls, physical_charges):
        """Infer the binary code order from an ordered charge map."""
        charges = tuple(physical_charges)
        if set(charges) != {0, 1}:
            raise ValueError("Spinless physical charges must be exactly {0, 1}.")
        return cls(empty=charges.index(0), occupied=charges.index(1))

    @property
    def max_code(self):
        return max(self.empty, self.occupied)

    def validate(self, configs):
        torch = _require_torch()
        valid = (configs == self.empty) | (configs == self.occupied)
        if not torch.all(valid):
            bad = torch.unique(configs[~valid]).detach().cpu().tolist()
            raise ValueError(f"Unknown spinless fermion site code(s): {bad!r}.")

    def decode(self, configs):
        """Return binary occupation tensors."""
        torch = _require_torch()
        configs = torch.as_tensor(configs, dtype=torch.long)
        self.validate(configs)
        return (configs == self.occupied).long()

    def encode(self, occupied):
        torch = _require_torch()
        occupied = torch.as_tensor(occupied, dtype=torch.long)
        return torch.where(
            occupied == 1,
            torch.as_tensor(self.occupied, device=occupied.device),
            torch.as_tensor(self.empty, device=occupied.device),
        )


@dataclass(frozen=True)
class TorchSquareLattice:
    """Square-lattice nearest-neighbor graph with grouped row/column edges."""

    Lx: int
    Ly: int
    pbc: bool | tuple[bool, bool] = False

    def __post_init__(self):
        Lx = _check_positive_int("Lx", self.Lx)
        Ly = _check_positive_int("Ly", self.Ly)
        if isinstance(self.pbc, bool):
            pbc_x = pbc_y = self.pbc
        else:
            pbc_x, pbc_y = self.pbc

        row_edges = {i: [] for i in range(Lx)}
        for i in range(Lx):
            for j in range(Ly - 1):
                row_edges[i].append((i * Ly + j, i * Ly + j + 1))
            if pbc_y and Ly > 2:
                row_edges[i].append((i * Ly + Ly - 1, i * Ly))

        col_edges = {j: [] for j in range(Ly)}
        for j in range(Ly):
            for i in range(Lx - 1):
                col_edges[j].append((i * Ly + j, (i + 1) * Ly + j))
            if pbc_x and Lx > 2:
                col_edges[j].append(((Lx - 1) * Ly + j, j))

        object.__setattr__(self, "Lx", Lx)
        object.__setattr__(self, "Ly", Ly)
        object.__setattr__(self, "row_edges", {k: tuple(v) for k, v in row_edges.items()})
        object.__setattr__(self, "col_edges", {k: tuple(v) for k, v in col_edges.items()})

    @property
    def n_sites(self):
        return self.Lx * self.Ly

    @property
    def edges(self):
        edges = []
        for group in self.row_edges.values():
            edges.extend(group)
        for group in self.col_edges.values():
            edges.extend(group)
        return tuple(edges)
