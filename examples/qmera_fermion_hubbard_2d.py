"""Native Symmray 2D spinful Fermi--Hubbard qMERA example.

This example keeps the physical lattice two-dimensional while using an
explicit register mode for each ``(site, spin)`` pair. The native
``U1U1`` path conserves up and down particle number independently.
"""

import pepsy as py
from pepsy.optimizers.mera import (
    QMeraBuilder,
    QMeraGeometry,
    QMeraSymmrayFermionBackend,
    symmray_fermion_gate_registry,
)


def main():
    try:
        import symmray  # noqa: F401  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - example dependency
        raise SystemExit("Install Symmray to run this example.") from exc

    geometry = QMeraGeometry(
        shape=(2, 2),
        site_modes=("up", "down"),
        mode_order="mode-major",
    )
    backend = QMeraSymmrayFermionBackend(
        symmetry="U1U1",
        site_modes=("up", "down"),
        mode_order="mode-major",
    )
    registry = symmray_fermion_gate_registry(backend=backend)

    def product_state_factory(schedule, sites, **kwargs):
        # One particle per site in a checkerboard spin pattern.
        occupations = {
            site: int(
                (sum(schedule.geometry.to_site(site)) % 2 == 0)
                == (schedule.geometry.to_mode(site)[1] == "up")
            )
            for site in sites
        }
        return backend.product_state(
            schedule,
            sites,
            occupations=occupations,
            **kwargs,
        )

    builder = QMeraBuilder(
        geometry=geometry,
        gate_registry=registry,
        gate_family="symmray-fsim",
        disentangler={
            "block_size": 2,
            "circuit_depth": 1,
            "gate_family": "symmray-fsim",
        },
        isometry={
            "block_size": (2, 2),
            "circuit_depth": 1,
            "gate_family": "symmray-fsim",
        },
        max_layers=1,
        seed=7,
        param_scale=0.02,
        product_state_factory=product_state_factory,
    )
    fermion = py.Fermion(
        spinful=True,
        symmetry="U1U1",
        t=0.2,
        U=4.0,
        mu=0.1,
    )
    schedule = builder.build_schedule()
    parameters = builder.initialize_parameters(schedule)
    terms = builder.fermion_terms(fermion)

    # Symmray operators are already native; preserve them with
    # convert_terms=False. The cache reuses paths for repeated local cones.
    path_cache = builder.contraction_path_cache(max_repeats=8)
    lightcone_energy = builder.parametric_loss(
        parameters,
        terms,
        schedule=schedule,
        convert_terms=False,
        energy_per_site=False,
        path_cache=path_cache,
    )
    direct_energy = builder.direct_parametric_loss(
        parameters,
        terms,
        schedule=schedule,
        convert_terms=False,
        energy_per_site=False,
        path_cache=path_cache,
    )
    print("2D U1U1 qMERA lightcone energy:", lightcone_energy)
    print("2D U1U1 qMERA direct energy:   ", direct_energy)
    print("cached cone paths:", path_cache.num_cached_paths)


if __name__ == "__main__":
    main()
