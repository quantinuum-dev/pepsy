"""Native Z2 Majorana/pairing 2D qMERA example.

The convention uses one complex spinless mode per physical site:
``gamma_x = c + c^dag`` and ``gamma_y = -i (c - c^dag)``. Individual
Majoranas are parity odd, while bilinears and pairing gates are Z2 neutral.
"""

import pepsy as py
from pepsy.optimizers.mera import (
    QMeraBuilder,
    QMeraGeometry,
    QMeraSymmrayFermionBackend,
    symmray_majorana_gate_registry,
)


def main():
    try:
        import symmray  # noqa: F401  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - example dependency
        raise SystemExit("Install Symmray to run this example.") from exc

    geometry = QMeraGeometry(
        shape=(2, 2),
        site_modes=("mode",),
        mode_order="mode-major",
    )
    backend = QMeraSymmrayFermionBackend(
        symmetry="Z2",
        site_modes=("mode",),
    )
    registry = symmray_majorana_gate_registry(backend=backend)

    def product_state_factory(schedule, sites, **kwargs):
        occupations = {
            site: sum(schedule.geometry.to_site(site)) % 2
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
        gate_family="symmray-majorana",
        disentangler={
            "block_size": 2,
            "circuit_depth": 1,
            "gate_family": "symmray-majorana",
        },
        isometry={
            "block_size": (2, 2),
            "circuit_depth": 1,
            "gate_family": "symmray-majorana",
        },
        max_layers=1,
        seed=11,
        param_scale=0.02,
        product_state_factory=product_state_factory,
    )
    fermion = py.Fermion(spinful=False, symmetry="Z2")
    schedule = builder.build_schedule()
    parameters = builder.initialize_parameters(schedule)
    terms = builder.majorana_terms(
        fermion,
        coupling=0.4,
        pairing=0.2,
        phase=0.1,
    )

    # Generic Majorana pairing is not a U1U1 charge-conserving operation, so
    # this example deliberately uses the native parity-preserving Z2 route.
    lightcone_energy = builder.parametric_loss(
        parameters,
        terms,
        schedule=schedule,
        convert_terms=False,
        energy_per_site=False,
    )
    direct_energy = builder.direct_parametric_loss(
        parameters,
        terms,
        schedule=schedule,
        convert_terms=False,
        energy_per_site=False,
    )
    print("2D Z2 Majorana lightcone energy:", lightcone_energy)
    print("2D Z2 Majorana direct energy:   ", direct_energy)


if __name__ == "__main__":
    main()
