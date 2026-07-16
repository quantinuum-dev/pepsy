"""qMERA block schedules and reverse-lightcone metadata."""

from __future__ import annotations

from dataclasses import dataclass

from .geometry import QMeraGeometry

__all__ = [
    "QMeraBlockSpec",
    "QMeraGatePlacement",
    "QMeraLayerSpec",
    "QMeraSchedule",
    "build_qmera_schedule",
]


def _normalize_stage(stage):
    key = str(stage).strip().lower().replace("_", "-")
    if key in {"dis", "disentangler", "disentanglers"}:
        return "disentangler"
    if key in {"iso", "isometry", "isometries"}:
        return "isometry"
    raise ValueError("block kind must be 'disentangler' or 'isometry'.")


def _normalize_structure(structure):
    key = str(structure).strip().lower().replace("_", "-")
    if key != "brickwall":
        raise NotImplementedError("only brickwall qMERA schedules are implemented.")
    return key


def _tag_token(value):
    chars = []
    for char in str(value).upper().replace("-", "_"):
        chars.append(char if char.isalnum() or char == "_" else "_")
    return "".join(chars).strip("_") or "X"


@dataclass(frozen=True)
class QMeraBlockSpec:
    """Local qMERA block layout for one operation family."""

    kind: str
    block_size: int = 2
    circuit_depth: int = 1
    structure: str = "brickwall"
    gate_family: str = "rxx"

    def __post_init__(self):
        kind = _normalize_stage(self.kind)
        block_size = int(self.block_size)
        circuit_depth = int(self.circuit_depth)
        if block_size < 2:
            raise ValueError("block_size must be >= 2.")
        if circuit_depth < 0:
            raise ValueError("circuit_depth must be >= 0.")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "block_size", block_size)
        object.__setattr__(self, "circuit_depth", circuit_depth)
        object.__setattr__(self, "structure", _normalize_structure(self.structure))
        object.__setattr__(self, "gate_family", str(self.gate_family))


@dataclass(frozen=True)
class QMeraGatePlacement:
    """One parametrized gate placement in a qMERA schedule."""

    gate_id: str
    param_key: str
    where: tuple[int, ...]
    scale: int
    stage: str
    round: int
    block: int
    gate_family: str
    tags: tuple[str, ...]

    @property
    def arity(self):
        """Number of register sites acted on by this gate."""
        return len(self.where)


@dataclass(frozen=True)
class QMeraLayerSpec:
    """One MERA scale with disentangler and isometry stages."""

    scale: int
    input_sites: tuple[int, ...]
    output_sites: tuple[int, ...]
    disentanglers: tuple[QMeraGatePlacement, ...]
    isometries: tuple[QMeraGatePlacement, ...]

    @property
    def placements(self):
        """All gate placements in execution order for this layer."""
        return (*self.disentanglers, *self.isometries)


@dataclass(frozen=True)
class QMeraSchedule:
    """Static qMERA schedule with stable gate ids, parameter keys, and tags."""

    geometry: QMeraGeometry
    layers: tuple[QMeraLayerSpec, ...]
    disentangler: QMeraBlockSpec
    isometry: QMeraBlockSpec
    top_sites: tuple[int, ...]

    @property
    def placements(self):
        """All gate placements in execution order."""
        return tuple(placement for layer in self.layers for placement in layer.placements)

    @property
    def param_keys(self):
        """Parameter keys in deterministic gate execution order."""
        return tuple(placement.param_key for placement in self.placements)

    @property
    def num_gates(self):
        """Number of scheduled parametrized gates."""
        return len(self.placements)

    def placements_by_id(self):
        """Return a mapping from gate id to placement."""
        return {placement.gate_id: placement for placement in self.placements}

    def reverse_lightcone_placements(self, where):
        """Return scheduled gates in the reverse lightcone of ``where``."""
        support = set(self.geometry.to_register_where(where))
        selected = []
        for placement in reversed(self.placements):
            if support.intersection(placement.where):
                selected.append(placement)
                support.update(placement.where)
        selected.reverse()
        return tuple(selected)

    def reverse_lightcone_tags(self, where):
        """Return physical and gate tags in the reverse lightcone of ``where``."""
        tags = []
        seen = set()

        def add(tag):
            if tag not in seen:
                seen.add(tag)
                tags.append(tag)

        for site in self.geometry.to_register_where(where):
            add(f"I{site}")
        for placement in self.reverse_lightcone_placements(where):
            for tag in placement.tags:
                add(tag)
        return tuple(tags)


def _block_ranges(active, block_size):
    for start in range(0, len(active), block_size):
        block = tuple(active[start : start + block_size])
        if len(block) >= 2:
            yield start // block_size, block


def _pairs_for_round(block, round_index, *, periodic=False):
    if len(block) == 2:
        return ((block[0], block[1]),)
    start = round_index % 2
    pairs = [
        (block[idx], block[idx + 1])
        for idx in range(start, len(block) - 1, 2)
    ]
    if periodic and start == 1 and len(block) > 2:
        pairs.append((block[-1], block[0]))
    return tuple(pairs)


def _placement_tags(gate_id, *, scale, stage, round_index, block, gate_family):
    stage_tag = "DISENTANGLER" if stage == "disentangler" else "ISOMETRY"
    return (
        f"GATE_{gate_id}",
        f"LAYER{scale}",
        stage_tag,
        f"ROUND{round_index}",
        f"BLOCK{block}",
        f"FAMILY_{_tag_token(gate_family)}",
    )


def _stage_placements(
    active,
    *,
    scale,
    stage_spec,
    counter_start,
    boundary,
):
    placements = []
    counter = counter_start
    stage = stage_spec.kind
    short = "DIS" if stage == "disentangler" else "ISO"
    periodic = boundary == "periodic"
    for round_index in range(stage_spec.circuit_depth):
        for block_index, block in _block_ranges(active, stage_spec.block_size):
            for pair in _pairs_for_round(block, round_index, periodic=periodic):
                gate_id = f"L{scale}_{short}_{counter:04d}"
                placements.append(
                    QMeraGatePlacement(
                        gate_id=gate_id,
                        param_key=gate_id,
                        where=tuple(pair),
                        scale=scale,
                        stage=stage,
                        round=round_index,
                        block=block_index,
                        gate_family=stage_spec.gate_family,
                        tags=_placement_tags(
                            gate_id,
                            scale=scale,
                            stage=stage,
                            round_index=round_index,
                            block=block_index,
                            gate_family=stage_spec.gate_family,
                        ),
                    )
                )
                counter += 1
    return tuple(placements), counter


def _coarse_grain(active, block_size):
    outputs = []
    for start in range(0, len(active), block_size):
        block = tuple(active[start : start + block_size])
        if block:
            outputs.append(block[0])
    return tuple(outputs)


def build_qmera_schedule(
    geometry,
    *,
    disentangler=None,
    isometry=None,
    max_layers=None,
    top_size=1,
):
    """Build a deterministic brickwall qMERA schedule."""
    geometry = geometry if isinstance(geometry, QMeraGeometry) else QMeraGeometry(geometry)
    disentangler = (
        disentangler
        if isinstance(disentangler, QMeraBlockSpec)
        else QMeraBlockSpec(kind="disentangler", **dict(disentangler or {}))
    )
    isometry = (
        isometry
        if isinstance(isometry, QMeraBlockSpec)
        else QMeraBlockSpec(kind="isometry", **dict(isometry or {}))
    )
    if disentangler.kind != "disentangler":
        raise ValueError("disentangler spec must have kind='disentangler'.")
    if isometry.kind != "isometry":
        raise ValueError("isometry spec must have kind='isometry'.")

    top_size = int(top_size)
    if top_size < 1:
        raise ValueError("top_size must be >= 1.")
    max_layers = None if max_layers is None else int(max_layers)
    if max_layers is not None and max_layers < 0:
        raise ValueError("max_layers must be >= 0.")

    active = geometry.register_sites
    layers = []
    gate_counter = 0
    scale = 0
    while len(active) > top_size and (max_layers is None or scale < max_layers):
        dis, gate_counter = _stage_placements(
            active,
            scale=scale,
            stage_spec=disentangler,
            counter_start=gate_counter,
            boundary=geometry.boundary,
        )
        iso, gate_counter = _stage_placements(
            active,
            scale=scale,
            stage_spec=isometry,
            counter_start=gate_counter,
            boundary=geometry.boundary,
        )
        output_sites = _coarse_grain(active, isometry.block_size)
        if output_sites == active:
            break
        layers.append(
            QMeraLayerSpec(
                scale=scale,
                input_sites=active,
                output_sites=output_sites,
                disentanglers=dis,
                isometries=iso,
            )
        )
        active = output_sites
        scale += 1

    return QMeraSchedule(
        geometry=geometry,
        layers=tuple(layers),
        disentangler=disentangler,
        isometry=isometry,
        top_sites=active,
    )
