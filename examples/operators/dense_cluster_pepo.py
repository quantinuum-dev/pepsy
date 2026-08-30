"""Build a small dense connected-cluster PEPO."""

from __future__ import annotations

import numpy as np

from pepsy.operators import ClusterExpansionPlan


def main() -> None:
    z = np.diag([1.0, -1.0])
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    plan = ClusterExpansionPlan(
        2,
        2,
        np.kron(z, z),
        0.4 * x,
        order=2,
    )

    active = plan.build(beta=0.03, materialize=False)
    pepo = active.to_pepo()

    print(f"active sectors: {active.active_block_count}")
    print(f"materialized type: {type(pepo).__name__}")
    assert active.active_block_count > 0
    assert pepo.Lx == 2 and pepo.Ly == 2


if __name__ == "__main__":
    main()
