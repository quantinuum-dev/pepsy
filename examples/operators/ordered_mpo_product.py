"""Build a joint ordered MPO product ``exp(A) @ exp(B)``."""

from __future__ import annotations

import numpy as np

from pepsy.operators import MPOBasis, MPOClusterProductExpansion


def main() -> None:
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.diag([1.0, -1.0])
    basis_a = MPOBasis.from_local_terms(3, [((0, 1), (x, x))])
    basis_b = MPOBasis.from_local_terms(3, [((1, 2), (z, z))])

    product = MPOClusterProductExpansion.from_mpo_bases(
        (basis_a, basis_b),
        coefficients=(0.2, -0.3),
        cluster_size=3,
        cutoff=0.0,
    )
    mpo = product.compile_exp().exp(0.02)

    print(f"factor count: {product.cache_info['factor_count']}")
    print(f"materialized type: {type(mpo.to_mpo()).__name__}")
    assert product.cache_info["factor_count"] == 2
    assert mpo.metadata["history_valid"] is False


if __name__ == "__main__":
    main()
