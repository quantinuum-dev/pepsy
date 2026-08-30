"""Build an ordered PEPO product ``exp(A) @ exp(B)``."""

from __future__ import annotations

from pepsy.operators import PauliPEPOBasis, PEPOClusterProductExpansion


def main() -> None:
    basis_a = PauliPEPOBasis.compile(2, 2, [("onsite", "X")], order=2)
    basis_b = PauliPEPOBasis.compile(2, 2, [("edge", "ZZ")], order=2)

    product = PEPOClusterProductExpansion.from_bases(
        (basis_a, basis_b),
        coefficients=(0.2, -0.3),
    )
    compiled = product.compile_exp()
    pepo = compiled.exp(0.02)

    print(f"factor count: {product.cache_info['factor_count']}")
    print(f"materialized type: {type(pepo).__name__}")
    assert product.cache_info["factor_count"] == 2
    assert pepo.Lx == 2 and pepo.Ly == 2


if __name__ == "__main__":
    main()
