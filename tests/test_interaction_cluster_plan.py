"""Independent finite combinatorics and cross-representation numerical checks."""

from dataclasses import FrozenInstanceError
from itertools import combinations, product

import numpy as np
import pytest
from scipy.linalg import expm

from pepsy.operators import (
    ClusterLattice,
    ClusterPlan,
    GraphClusterExpansionPlan,
    MPOClusterFactor,
    MPOClusterProductExpansion,
    MPOProductTerm,
)

pytestmark = [pytest.mark.integration, pytest.mark.operators]
X = np.array([[0.0, 1.0], [1.0, 0.0]])
Z = np.diag([1.0, -1.0])


def connected(sites, edges):
    remaining = set(sites)
    reached = {remaining.pop()}
    while remaining:
        frontier = {
            b for a, b in (*edges, *((b, a) for a, b in edges)) if a in reached and b in remaining
        }
        if not frontier:
            return False
        reached |= frontier
        remaining -= frontier
    return True


def partitions(sites):
    if not sites:
        yield ()
        return
    first, *rest = sites
    for partition in partitions(rest):
        yield ((first,), *partition)
        for i, block in enumerate(partition):
            yield (*partition[:i], (first, *block), *partition[i + 1 :])


@pytest.mark.parametrize(
    "edges",
    [
        (),
        ((0, 1), (1, 2), (2, 3)),
        ((0, 1), (1, 2), (2, 3), (0, 2), (1, 3)),
        ((0, 1), (1, 2), (2, 0), (2, 3)),
    ],
)
@pytest.mark.parametrize("cutoff", [1, 2, 3, 4])
def test_inventory_and_exact_counts_against_exhaustive_sets(edges, cutoff):
    plan = ClusterPlan.from_supports(4, edges, cluster_size=cutoff)
    expected = tuple(
        s for k in range(1, cutoff + 1) for s in combinations(range(4), k) if connected(s, edges)
    )
    assert plan.index_clusters == expected
    all_partitions = tuple(partitions(list(range(4))))
    total = sum(
        all(len(block) <= cutoff and connected(block, edges) for block in p) for p in all_partitions
    )
    assert plan.collection_count() == total
    assert plan.collection_count(include_background=False) == total - 1
    for cluster in expected:
        proper = [
            p
            for p in partitions(list(cluster))
            if len(p) > 1 and all(connected(block, edges) for block in p)
        ]
        assert plan.partition_count(cluster) == len(proper)
        assert plan.partition_count(cluster, proper=False) == len(proper) + 1
        canonical = lambda p: frozenset(frozenset(block) for block in p)
        assert set(map(canonical, plan.partitions(cluster))) == set(map(canonical, proper))


def test_hyperedges_isolated_sites_duplicates_order_and_zero_terms():
    terms = [{"sites": (2, 0, 1), "coefficient": 0.0}] * 2
    plan = ClusterPlan.from_terms(terms, sites=4, cluster_size=3)
    assert plan.supports == ((2, 0, 1),) * 2
    assert plan.counts == {1: 4, 2: 3, 3: 1}
    assert plan.lattice.edges[0] == (2, 0)
    assert plan.graph_shapes[4].edges[0][:2] in ((1, 0), (0, 1))
    assert plan.collection_count() == 5  # Bell(3), plus the isolated singleton.
    with pytest.raises(FrozenInstanceError):
        plan.cluster_size = 1
    with pytest.raises(TypeError, match="located"):
        ClusterPlan.from_terms([("edge", "ZZ")])


def test_budgets_fail_without_returning_approximate_counts():
    edges = tuple(combinations(range(5), 2))
    with pytest.raises(ValueError, match="max_clusters"):
        ClusterPlan.from_supports(5, edges, cluster_size=5, max_clusters=12).index_clusters
    plan = ClusterPlan.from_supports(5, edges, cluster_size=5)
    with pytest.raises(ValueError, match="max_partitions"):
        plan.partitions(range(5), max_partitions=10)
    with pytest.raises(ValueError, match="state_budget"):
        plan.collection_count(state_budget=2)
    assert plan.collection_count() == 52


def test_topology_cache_shared_but_edge_direction_and_labels_retained():
    a = ClusterPlan(ClusterLattice(("a", "b", "c"), (("a", "b"), ("c", "b"))), cluster_size=3)
    b = ClusterPlan(ClusterLattice(3, ((1, 0), (1, 2))), cluster_size=3)
    assert a.index_clusters is b.index_clusters
    assert a.graph_shapes is a.graph_shapes
    assert a.graph_shapes[-1].edges == ((0, 1, 0), (2, 1, 1))
    assert b.graph_shapes[-1].edges == ((1, 0, 0), (1, 2, 1))
    assert a.reuse(()).info["representatives"] == 3
    assert sum(a.reuse(()).multiplicities.values()) == 6
    assert a.cache_info()["inventory"]["hits"] > 0


def labelled(plan, *, independent=False, anisotropic=False):
    return (
        tuple(
            (f"h{i}" if independent else "h", ((site, "X"),)) for i, site in enumerate(plan.sites)
        )
        + tuple(
            ("J", ((a, "X" if anisotropic and i == 0 else "Z"), (b, "Z")))
            for i, (a, b) in enumerate(plan.lattice.edges)
        ),
    )


def test_square_symmetries_require_geometry_and_binding_identity():
    plan = ClusterPlan(ClusterLattice.square(3, 3), cluster_size=3)
    assert len(plan.spatial_symmetries()) == 8
    assert len(plan.spatial_symmetries(labelled(plan))) == 8
    assert len(plan.spatial_symmetries(labelled(plan, independent=True))) == 1
    assert len(plan.spatial_symmetries(labelled(plan, anisotropic=True))) < 8
    assert plan.reuse(labelled(plan)).info["reused"] > 0
    assert plan.reuse(labelled(plan, independent=True)).info["reused"] == 0
    assert plan.reuse().info["reused"] == 0


def test_triangular_periodic_generators_and_obc_no_wrap_inference():
    sites = tuple(product(range(3), repeat=2))
    edges = tuple(
        (s, ((s[0] + dx) % 3, (s[1] + dy) % 3))
        for s in sites
        for dx, dy in ((1, 0), (0, 1), (1, -1))
    )
    torus = ClusterPlan.from_supports(sites, edges, cluster_size=3, cyclic=True)
    rotation = tuple(((x + y) % 3, -x % 3) for x, y in sites)
    shift = tuple(((x + 1) % 3, y) for x, y in sites)
    assert rotation in torus.spatial_symmetries()
    assert shift in torus.spatial_symmetries()
    obc_edges = tuple(
        (s, (s[0] + dx, s[1] + dy))
        for s in sites
        for dx, dy in ((1, 0), (0, 1), (1, -1))
        if (s[0] + dx, s[1] + dy) in sites
    )
    obc = ClusterPlan.from_supports(sites, obc_edges, cluster_size=3)
    assert shift not in obc.spatial_symmetries()
    # Metadata never invents missing periodic interactions.
    marked = ClusterPlan.from_supports(sites, obc_edges, cluster_size=3, cyclic=True)
    assert marked.index_clusters == obc.index_clusters
    assert shift not in marked.spatial_symmetries()


def embed(sites, ops, nsites):
    result = np.array([[1.0]])
    for site in range(nsites):
        result = np.kron(result, ops[sites.index(site)] if site in sites else np.eye(2))
    return result


def test_shared_mpo_graph_pepo_full_size_matches_dense_nnn_hamiltonian():
    edges = ((0, 1), (1, 2), (0, 2))
    terms = tuple(MPOProductTerm(edge, (Z, Z), 0.31) for edge in edges)
    terms += tuple(MPOProductTerm((i,), (X,), 0.17) for i in range(3))
    plan = ClusterPlan.from_terms(terms, sites=3, cluster_size=3)
    mpo = MPOClusterProductExpansion.from_plan(
        plan, [MPOClusterFactor(terms)], cutoff=0, graph_assembly="exact"
    )
    pepo = GraphClusterExpansionPlan.from_plan(plan, 0.31 * np.kron(Z, Z), 0.17 * X)
    assert mpo.cluster_plan is pepo.cluster_plan is plan
    h = sum(term.coefficient * embed(term.sites, term.operators, 3) for term in terms)
    for step in (0.1, -0.07j):
        expected = expm(step * h)
        np.testing.assert_allclose(mpo.exp(step, materialize=True).to_dense(), expected, atol=2e-12)
        np.testing.assert_allclose(pepo.exp(step).to_dense(), expected, atol=2e-12)
        np.testing.assert_allclose(mpo.trace_exp(step), np.trace(expected), atol=2e-12)


def test_inference_uses_all_ordered_factors_and_retains_isolates():
    a = MPOClusterFactor([MPOProductTerm((0, 2), (X, Z), 0.2)])
    b = MPOClusterFactor([MPOProductTerm((1, 2), (Z, X), -0.3)])
    expansion = MPOClusterProductExpansion(
        4, (f for f in (a, b)), graph="interactions", cluster_size=3
    )
    assert expansion.cluster_plan.counts == {1: 4, 2: 2, 3: 1}
    expected = expm(0.1 * 0.2 * embed((0, 2), (X, Z), 4)) @ expm(
        -0.1 * 0.3 * embed((1, 2), (Z, X), 4)
    )
    np.testing.assert_allclose(
        expansion.exp(0.1, materialize=True).to_dense(), expected, atol=2e-12
    )


def test_arbitrary_plan_site_labels_map_to_mpo_indices():
    plan = ClusterPlan.from_supports(["left", "right"], [("left", "right")], cluster_size=2)
    term = MPOProductTerm((0, 1), (X, Z), 0.2)
    expansion = MPOClusterProductExpansion.from_plan(plan, [MPOClusterFactor([term])])
    np.testing.assert_allclose(
        expansion.exp(0.1, materialize=True).to_dense(), expm(0.02 * np.kron(X, Z)), atol=1e-12
    )
    with pytest.raises(ValueError, match="match"):
        MPOClusterProductExpansion.from_plan(plan, [MPOClusterFactor([term])], cluster_size=1)


def test_explicit_reuse_family_never_chooses_an_excluded_representative():
    plan = ClusterPlan.from_supports(4, [(0, 1), (1, 2), (2, 3)], cluster_size=3)
    family = ((1,), (2,), (3,), (1, 2), (2, 3))
    reused = plan.reuse((), clusters=family)
    assert set(reused.entries) == set(family)
    assert set(reused.multiplicities) <= set(family)
    assert reused.info["representatives"] == 2


@pytest.mark.parametrize("site", [-1, 2])
def test_plan_factory_rejects_out_of_range_term_indices(site):
    plan = ClusterPlan.from_supports(2, [(0, 1)], cluster_size=2)
    with pytest.raises(ValueError, match="outside the chain"):
        MPOClusterProductExpansion.from_plan(
            plan, [MPOClusterFactor([MPOProductTerm((site,), (X,), .2)])]
        )
