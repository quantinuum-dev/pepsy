"""Build a square PEPO and compare the generic graph representation."""

import numpy as np

from pepsy.operators import ClusterPlan, MPOParameter, PauliPEPOBasis


def main():
    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    terms = [((i,), "X", MPOParameter("h")) for i in range(4)]
    terms += [(edge, "ZZ", MPOParameter("J")) for edge in edges]
    plan = ClusterPlan.from_supports(
        4, [term[0] for term in terms], shape=(2, 2), cluster_size=3,
    )
    square = PauliPEPOBasis.from_plan(plan, terms, factorization="fixed").compile_exp()
    graph = PauliPEPOBasis.from_plan(plan, terms, layout="graph")
    params = {"h": .3, "J": .7}
    step = -.05j
    pepo = square.exp(step, params, materialize=True)
    error = np.max(np.abs(pepo.to_dense() - graph.exp(step, params).to_dense()))
    np.testing.assert_allclose(pepo.to_dense(), graph.exp(step, params).to_dense(), atol=1e-12)
    print("Layout:", square.cache_info["representation"])
    print("PEPO shape:", (pepo.Lx, pepo.Ly))
    print("Cluster counts:", plan.counts)
    print("Maximum compact bond dimension:", pepo.max_bond())
    print("Maximum matrix difference from graph construction:", error)
    print("Normalized partition trace:", square.trace_exp(step, params, normalized=True))


if __name__ == "__main__":
    main()
