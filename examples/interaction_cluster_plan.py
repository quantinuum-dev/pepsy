"""Share NN+NNN topology and exact bookkeeping between MPO and graph PEPO."""

import numpy as np

from pepsy.operators import (
    ClusterPlan,
    MPOClusterFactor,
    MPOClusterProductExpansion,
    PEPOClusterProductExpansion,
)


def main():
    terms = [
        {"sites": (i, i + distance), "paulis": "ZZ", "coefficient": coupling}
        for distance, coupling in ((1, 0.3), (2, 0.1))
        for i in range(4 - distance)
    ]
    terms += [{"sites": (i,), "paulis": "X", "coefficient": 0.2} for i in range(4)]
    plan = ClusterPlan.from_terms(terms, sites=4, cluster_size=3)
    factors = [MPOClusterFactor(terms)]
    mpo = MPOClusterProductExpansion.from_plan(plan, factors, cutoff=0)
    pepo = PEPOClusterProductExpansion.from_plan(plan, factors)
    a = mpo.trace_exp(-0.05j, normalized=True)
    b = pepo.trace_exp(-0.05j, normalized=True)
    np.testing.assert_allclose(a, b, atol=1e-13)
    print("Finite connected placements:", plan.counts)
    print("Complete collections including background:", plan.collection_count())
    print("Normalized trace:", a)
    print("Shared topology:", mpo.cluster_plan is pepo.cluster_plan)


if __name__ == "__main__":
    main()
