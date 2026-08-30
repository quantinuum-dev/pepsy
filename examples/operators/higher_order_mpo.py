"""Build and materialize a small higher-order MPO exponential."""

from __future__ import annotations

from pepsy.operators import MPOBasis


def main() -> None:
    basis = MPOBasis.from_pauli_terms(
        4,
        [((site, site + 1), "ZZ", 0.7) for site in range(3)]
        + [((site,), "X", 0.2) for site in range(4)],
    )

    compiled = basis.compile_exp(order=2, mode="base")
    semantic = compiled.exp(-0.02)
    mpo = semantic.to_mpo()

    print(f"semantic type: {type(semantic).__name__}")
    print(f"MPO bond dimensions: {semantic.bond_dimensions}")
    print(f"materialized type: {type(mpo).__name__}")
    assert len(mpo.tensors) == 4


if __name__ == "__main__":
    main()
