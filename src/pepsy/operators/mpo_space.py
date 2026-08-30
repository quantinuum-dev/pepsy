"""Physical-sector and braiding semantics for MPO construction."""

from dataclasses import dataclass

from .._internal.validation import is_strict_integer

__all__ = ["MPOBraiding", "MPOPhysicalSpace"]


@dataclass(frozen=True)
class MPOBraiding:
    """Define the exchange phase used while canonicalizing MPO factors.

    ``kind="bosonic"`` assigns unit phase to every exchange.
    ``kind="fermionic"`` assigns ``-1`` exactly when two odd factors cross.
    Operator parities are integers modulo two and remain separate from charge
    sectors because a conserved symmetry does not by itself define grading.
    """

    kind: str = "bosonic"

    def __post_init__(self):
        kind = str(self.kind).strip().lower().replace("-", "_")
        aliases = {"none": "bosonic", "graded": "fermionic"}
        kind = aliases.get(kind, kind)
        if kind not in {"bosonic", "fermionic"}:
            raise ValueError("MPO braiding kind must be 'bosonic' or 'fermionic'.")
        object.__setattr__(self, "kind", kind)

    @classmethod
    def resolve(cls, value=None, *, fermionic=False):
        """Normalize a braiding object/string or select a metadata default."""

        if value is None:
            return cls("fermionic" if fermionic else "bosonic")
        if isinstance(value, cls):
            if fermionic and value.kind != "fermionic":
                raise ValueError("fermionic metadata requires fermionic braiding.")
            return value
        resolved = cls(value)
        if fermionic and resolved.kind != "fermionic":
            raise ValueError("fermionic metadata requires fermionic braiding.")
        return resolved

    @property
    def fermionic(self):
        """Whether exchanges use the odd-odd fermionic sign rule."""

        return self.kind == "fermionic"

    def normalize_parities(self, parities, *, size):
        """Validate and normalize one parity per operator factor."""

        if parities is None:
            if self.fermionic:
                raise ValueError(
                    "fermionic factor canonicalization requires explicit parities."
                )
            return (0,) * size
        try:
            parities = tuple(parities)
        except TypeError as exc:
            raise TypeError("parities must be an iterable of integers modulo two.") from exc
        if len(parities) != size or not all(
            is_strict_integer(parity) for parity in parities
        ):
            raise TypeError("parities must contain one integer per operator.")
        return tuple(int(parity) % 2 for parity in parities)

    def canonical_order(self, sites, parities=None):
        """Return stable site order, normalized parities, and exchange phase."""

        parities = self.normalize_parities(parities, size=len(sites))
        order = tuple(sorted(range(len(sites)), key=sites.__getitem__))
        phase = 1
        if self.fermionic:
            for left in range(len(sites)):
                for right in range(left + 1, len(sites)):
                    if sites[left] > sites[right] and parities[left] and parities[right]:
                        phase = -phase
        return order, tuple(parities[index] for index in order), phase


@dataclass(frozen=True)
class MPOPhysicalSpace:
    """Intrinsic local dimension, sectors, and braiding for an MPO.

    The object is backend-neutral. ``physical_charges`` are already-normalized
    sector labels in dense-basis order; backend adapters remain responsible
    for turning those labels into concrete index maps.
    """

    phys_dim: int
    symmetry: str | None = None
    physical_charges: tuple | None = None
    fermionic: bool = False
    braiding: MPOBraiding | str | None = None

    def __post_init__(self):
        if not is_strict_integer(self.phys_dim) or int(self.phys_dim) < 1:
            raise TypeError("phys_dim must be a positive integer.")
        object.__setattr__(self, "phys_dim", int(self.phys_dim))
        object.__setattr__(self, "fermionic", bool(self.fermionic))
        object.__setattr__(
            self,
            "braiding",
            MPOBraiding.resolve(self.braiding, fermionic=self.fermionic),
        )
        charges = self.physical_charges
        if charges is not None:
            charges = tuple(charges)
            if len(charges) != self.phys_dim:
                raise ValueError(
                    "physical_charges must contain one charge per local basis state."
                )
            object.__setattr__(self, "physical_charges", charges)
        if self.symmetry is None:
            if charges is not None:
                raise ValueError("physical_charges requires symmetry metadata.")
            if self.fermionic:
                raise ValueError("fermionic=True requires symmetry metadata.")
        elif charges is None:
            raise ValueError("symmetry requires physical_charges.")
