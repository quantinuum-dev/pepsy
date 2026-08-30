"""Evaluate a sparse fixed-channel square-lattice PEPO exponential."""

from __future__ import annotations

from pepsy.operators import PauliPEPOBasis


def main() -> None:
    basis = PauliPEPOBasis.compile(
        2,
        2,
        [("onsite", "X"), ("edge", "ZZ")],
        order=2,
    )

    active = basis.compile_exp().exp(
        -0.02,
        coefficients=(0.3, 1.0),
        materialize=False,
    )
    pepo = active.to_pepo()

    print(f"active sectors: {active.active_block_count}")
    print(f"materialized type: {type(pepo).__name__}")
    assert active.active_block_count > 0
    assert pepo.Lx == 2 and pepo.Ly == 2


if __name__ == "__main__":
    main()
