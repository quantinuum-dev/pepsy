"""Explicit 4x4 PBC spinful Fermi--Hubbard qMERA RG schedule.

The first layer has four 2x2 covering blocks and square disentanglers on all
horizontal and vertical block interfaces, including the periodic wraps. The
second layer combines the four coarse sites into one 2x2 block.
"""

import pepsy as py
from pepsy.optimizers.qmera import (
    QMeraBuilder,
    QMeraDisentanglerSpec,
    QMeraGeometry,
    QMeraIsometrySpec,
    QMeraSymmrayFermionBackend,
    QMeraUnitarySpec,
    symmray_fermion_gate_registry,
)


def main():
    geometry = QMeraGeometry(
        shape=(4, 4),
        boundary="periodic",
        site_modes=("up", "down"),
        mode_order="mode-major",
    )
    backend = QMeraSymmrayFermionBackend(
        symmetry="U1U1",
        site_modes=("up", "down"),
        mode_order="mode-major",
    )
    unitary = QMeraUnitarySpec(
        gate_family="symmray-hubbard",
        family="fermion",
        arity_kind="mode",
        symmetry="U1U1",
        preserves_parity=True,
        metadata={"model": "fermi-hubbard", "term": "hopping"},
    )
    builder = QMeraBuilder(
        geometry=geometry,
        gate_registry=symmray_fermion_gate_registry(backend=backend),
        disentangler=QMeraDisentanglerSpec(
            block_shape=(2, 2),
            unitary=unitary,
            placement="boundary-square",
            circuit_depth=2,
            periodic_wrap=True,
        ),
        isometry=QMeraIsometrySpec(
            block_shape=(2, 2),
            unitary=unitary,
            circuit_depth=2,
            implementation="unitary-completion",
        ),
        max_layers=2,
    )
    schedule = builder.build_schedule()

    # Fermion fixes the local U1U1 convention only; physical couplings remain
    # explicit at term construction so every simulation path sees the same model.
    fermion = py.Fermion(spinful=True, symmetry="U1U1")
    terms = builder.fermion_terms(fermion, t=0.2, U=4.0, mu=0.1)

    print("RG register sizes:", [len(layer.input_sites) for layer in schedule.layers], "->", len(schedule.top_sites))
    print("first-layer isometry blocks:", len(schedule.layers[0].isometry_blocks))
    print("first-layer square disentanglers:", len(schedule.layers[0].disentangler_blocks))
    print("Fermi--Hubbard terms:", len(terms))


if __name__ == "__main__":
    main()
