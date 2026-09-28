"""Independent finite-square expansion and coefficient binding regressions."""

from itertools import combinations

import numpy as np
import pytest
from scipy.linalg import expm

from pepsy.operators import (
    ClusterLattice, MPOBasis, MPOClusterFactor, MPOClusterProductExpansion,
    MPOParameter, MPOProductTerm, PEPOClusterProductExpansion,
    PauliPEPOBasis, PauliPEPOTerm, exp_mpo_cluster,
)

pytestmark = [pytest.mark.integration, pytest.mark.operators]
X = np.array([[0., 1.], [1., 0.]])
Z = np.diag([1., -1.])
EDGES = ((0, 1), (0, 2), (1, 3), (2, 3))


@pytest.mark.parametrize("binding", ["sequence", "default", "callable"])
@pytest.mark.parametrize("graph", [None, "chain"])
def test_fixed_binding_values_gradients_and_callable_count(binding, graph):
    torch = pytest.importorskip("torch")
    calls = []
    current = [None]

    def coefficient(_parameters):
        calls.append(1)
        return current[0]

    default = torch.tensor(.2, dtype=torch.float64, requires_grad=True)
    term_coefficient = (coefficient if binding == "callable" else
                        MPOParameter(0, default=default) if binding == "default" else
                        MPOParameter(0))
    basis = MPOBasis.from_local_terms(2, [((0, 1), (Z, Z), term_coefficient)])
    if graph is None:
        compiled = basis.compile_cluster_expansion(cluster_size=2, factorization="fixed", cutoff=0.)
    else:
        compiled = basis.compile_graph_cluster_expansion(
            graph=graph, cluster_size=2, factorization="fixed", cutoff=0., assembly="recursive"
        )
    for value in (0., .2):
        with torch.no_grad():
            default.fill_(value)
        j = default if binding == "default" else torch.tensor(
            value, dtype=torch.float64, requires_grad=True
        )
        current[0] = j
        parameters = [j] if binding == "sequence" else None
        counts = []
        for step in (-.1j, torch.tensor(-.1j, dtype=torch.complex128)):
            calls.clear()
            actual = compiled(step, parameters=parameters).to_mpo().to_dense()
            counts.append(len(calls))
            expected = torch.matrix_exp(-.1j * j * torch.as_tensor(np.kron(Z, Z)))
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
            (a,) = torch.autograd.grad(actual.imag[0, 0], (j,), retain_graph=True)
            (b,) = torch.autograd.grad(expected.imag[0, 0], (j,))
            torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
        assert counts[0] == counts[1]


def test_fixed_empty_support_preserves_float32():
    torch = pytest.importorskip("torch")
    j = torch.tensor(.2, dtype=torch.float32, requires_grad=True)
    basis = MPOBasis.from_local_terms(2, [((0, 1), (Z.astype('float32'),) * 2, MPOParameter(0))])
    result = basis.compile_graph_cluster_expansion(
        graph="chain", cluster_size=2, factorization="fixed", cutoff=0.
    )(.1, parameters=[j]).to_mpo().to_dense()
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, torch.matrix_exp(.1 * j * torch.tensor(np.kron(Z, Z), dtype=j.dtype)))


def _partitions(sites):
    """Enumerate set partitions independently of the production graph planner."""
    if not sites:
        yield ()
        return
    first, *rest = sites
    for size in range(len(rest) + 1):
        for chosen in combinations(rest, size):
            block = (first, *chosen)
            remaining = tuple(site for site in rest if site not in chosen)
            for partition in _partitions(remaining):
                yield (block, *partition)


def _embed(matrix, support, sites):
    """Dense basis-index embedding; deliberately independent of tensor helpers."""
    result = np.zeros((2**len(sites),) * 2, dtype=complex)
    for row in range(len(result)):
        rb = dict(zip(sites, np.binary_repr(row, width=len(sites))))
        for col in range(len(result)):
            cb = dict(zip(sites, np.binary_repr(col, width=len(sites))))
            if all(rb[i] == cb[i] for i in sites if i not in support):
                a = int(''.join(rb[i] for i in support), 2)
                b = int(''.join(cb[i] for i in support), 2)
                result[row, col] = matrix[a, b]
    return result


def _square_reference(order, fields, bonds, step, *, extra_factors=()):
    sites = tuple(range(4))
    residuals = {}
    for size in range(1, order + 1):
        for support in combinations(sites, size):
            reached = {support[0]}
            for _ in support:
                reached |= {b for a, b in EDGES + tuple((b, a) for a, b in EDGES)
                            if a in reached and b in support}
            if len(reached) != size:
                continue
            h = sum(fields[i] * _embed(X, (i,), support) for i in support)
            h += sum((bonds[e] * _embed(np.kron(Z, Z), e, support)
                      for e in EDGES if set(e) <= set(support)), start=np.zeros_like(h))
            residual = expm(step * h)
            for onsite, pair, extra_fields, extra_bonds in extra_factors:
                extra_h = sum(
                    extra_fields[i] * _embed(onsite, (i,), support)
                    for i in support
                )
                extra_h += sum(
                    (extra_bonds[e] * _embed(pair, e, support)
                     for e in EDGES if set(e) <= set(support)),
                    start=np.zeros_like(h),
                )
                residual = residual @ expm(step * extra_h)
            for partition in _partitions(support):
                if len(partition) == 1 or not all(block in residuals for block in partition):
                    continue
                contribution = np.eye(2**size, dtype=complex)
                for block in partition:
                    contribution = contribution @ _embed(residuals[block], block, support)
                residual -= contribution
            residuals[support] = residual
    result = np.zeros((16, 16), dtype=complex)
    for partition in _partitions(sites):
        if all(block in residuals for block in partition):
            contribution = np.eye(16, dtype=complex)
            for block in partition:
                contribution = contribution @ _embed(residuals[block], block, sites)
            result += contribution
    return result


@pytest.mark.parametrize("order", [2, 3, 4])
@pytest.mark.parametrize("uniform", [True, False])
def test_square_mpo_and_pepo_match_independent_partition_sum(order, uniform):
    fields = [.2 if uniform else .1 + i / 10 for i in range(4)]
    bonds = {edge: .3 if uniform else .15 + i / 10 for i, edge in enumerate(EDGES)}
    step = -.08j
    expected = _square_reference(order, fields, bonds, step)
    mpo_terms = [((i,), (X,), fields[i]) for i in range(4)]
    mpo_terms += [(edge, (Z, Z), bonds[edge]) for edge in EDGES]
    sites = ((0, 0), (0, 1), (1, 0), (1, 1))
    pepo_terms = [PauliPEPOTerm("onsite", "X", fields[i], where=sites[i]) for i in range(4)]
    pepo_terms += [PauliPEPOTerm("edge", "ZZ", bonds[e], where=tuple(sites[i] for i in e)) for e in EDGES]
    for reuse in (False, True):
        mpo = exp_mpo_cluster(
            mpo_terms, step, shape=4, graph=ClusterLattice.from_edges(range(4), EDGES),
            cluster_size=order, cutoff=0., factorization="fixed", assembly="recursive",
            spatial_reuse=reuse,
        )
        pepo = PauliPEPOBasis(
            2, 2, pepo_terms, order=order, factorization="fixed", spatial_reuse=reuse,
        ).compile_exp()(step).to_pepo()
        np.testing.assert_allclose(mpo.to_dense(), expected, atol=2e-12)
        np.testing.assert_allclose(pepo.to_dense(), expected, atol=2e-12)


@pytest.mark.parametrize("order", [2, 3, 4])
def test_joint_square_mpo_and_pepo_match_independent_partition_sum(order, monkeypatch):
    from pepsy.operators import pepo_basis

    sites = ((0, 0), (0, 1), (1, 0), (1, 1))
    fields_a = [0.2] * 4
    bonds_a = dict.fromkeys(EDGES, 0.4)
    fields_b = [-0.15] * 4
    bonds_b = dict.fromkeys(EDGES, 0.25)
    step = -0.037j
    expected = _square_reference(
        order, fields_a, bonds_a, step,
        extra_factors=((Z, np.kron(X, X), fields_b, bonds_b),),
    )
    pepo_a = [
        PauliPEPOTerm("onsite", "X", fields_a[i], where=sites[i])
        for i in range(4)
    ] + [
        PauliPEPOTerm("edge", "ZZ", bonds_a[e], where=tuple(sites[i] for i in e))
        for e in EDGES
    ]
    pepo_b = [
        PauliPEPOTerm("onsite", "Z", fields_b[i], where=sites[i])
        for i in range(4)
    ] + [
        PauliPEPOTerm("edge", "XX", bonds_b[e], where=tuple(sites[i] for i in e))
        for e in EDGES
    ]
    mpo_a = [
        MPOProductTerm((i,), (X,), fields_a[i]) for i in range(4)
    ] + [
        MPOProductTerm(e, (Z, Z), bonds_a[e]) for e in EDGES
    ]
    mpo_b = [
        MPOProductTerm((i,), (Z,), fields_b[i]) for i in range(4)
    ] + [
        MPOProductTerm(e, (X, X), bonds_b[e]) for e in EDGES
    ]

    def unexpected_uniform_components(*_args, **_kwargs):
        pytest.fail("located joint PEPO evaluated unused uniform component maps")

    monkeypatch.setattr(
        PauliPEPOBasis, "_hamiltonian_components", unexpected_uniform_components
    )
    original_contract = pepo_basis._contract_active_support_backend
    lower_calls = []

    def count_contract(*args, **kwargs):
        lower_calls.append(1)
        return original_contract(*args, **kwargs)

    monkeypatch.setattr(
        pepo_basis, "_contract_active_support_backend", count_contract
    )
    counts = []
    for reuse in (False, True):
        pepo = PEPOClusterProductExpansion.from_bases((
            PauliPEPOBasis.compile(
                2, 2, pepo_a, order=order, factorization="fixed", spatial_reuse=reuse
            ),
            PauliPEPOBasis.compile(
                2, 2, pepo_b, order=order, factorization="fixed", spatial_reuse=reuse
            ),
        )).compile_exp()
        lower_calls.clear()
        actual_pepo = pepo(step).to_dense()
        counts.append(len(lower_calls))
        mpo = MPOClusterProductExpansion(
            4, (MPOClusterFactor(mpo_a), MPOClusterFactor(mpo_b)),
            graph=ClusterLattice(tuple(range(4)), EDGES),
            cluster_size=order, graph_assembly="exact", assembly="recursive",
            assembly_state_budget=1000, assembly_chi=None,
            factorization="fixed", cutoff=0.0, spatial_reuse=reuse,
        ).compile_exp()
        actual_mpo = mpo(step).to_mpo().to_dense()
        if reuse:
            assert mpo.cache_info["spatial_plan"]["reused"] > 0
            assert any(
                plan["reused"]
                for plan in pepo.cache_info["factor_cache_info"][0]["spatial_plans"]
            )
        np.testing.assert_allclose(actual_pepo, expected, atol=2e-12)
        np.testing.assert_allclose(actual_mpo, expected, atol=2e-12)
    if order == 4:
        assert counts == [9, 3]


def test_joint_order_four_pepo_reuse_preserves_torch_gradients():
    torch = pytest.importorskip("torch")
    sites = ((0, 0), (0, 1), (1, 0), (1, 1))
    terms_a = [
        PauliPEPOTerm("onsite", "X", MPOParameter("h"), where=site)
        for site in sites
    ] + [
        PauliPEPOTerm("edge", "ZZ", 0.4, where=tuple(sites[i] for i in edge))
        for edge in EDGES
    ]
    terms_b = [
        PauliPEPOTerm("onsite", "Z", -0.15, where=site)
        for site in sites
    ] + [
        PauliPEPOTerm("edge", "XX", 0.25, where=tuple(sites[i] for i in edge))
        for edge in EDGES
    ]
    compiled = [
        PEPOClusterProductExpansion.from_bases((
            PauliPEPOBasis.compile(
                2, 2, terms_a, order=4, factorization="fixed",
                spatial_reuse=reuse,
            ),
            PauliPEPOBasis.compile(
                2, 2, terms_b, order=4, factorization="fixed",
                spatial_reuse=reuse,
            ),
        )).compile_exp()
        for reuse in (False, True)
    ]
    universe = tuple(range(4))
    hx = torch.as_tensor(sum(
        (_embed(X, (i,), universe) for i in universe),
        start=np.zeros((16, 16), dtype=complex),
    ))
    hz = torch.as_tensor(sum(
        (_embed(Z, (i,), universe) for i in universe),
        start=np.zeros((16, 16), dtype=complex),
    ))
    hzz = torch.as_tensor(sum(
        (_embed(np.kron(Z, Z), edge, universe) for edge in EDGES),
        start=np.zeros((16, 16), dtype=complex),
    ))
    hxx = torch.as_tensor(sum(
        (_embed(np.kron(X, X), edge, universe) for edge in EDGES),
        start=np.zeros((16, 16), dtype=complex),
    ))
    weights = torch.linspace(0.0, 1.0, 256, dtype=torch.float64).reshape(16, 16)
    for h_value, time_value in ((0.0, 0.03), (0.2, 0.0), (0.2, 0.03)):
        h = torch.tensor(h_value, dtype=torch.float64, requires_grad=True)
        time = torch.tensor(time_value, dtype=torch.float64, requires_grad=True)
        outputs = [
            product(-1j * time, parameters={"h": h}).to_dense()
            for product in compiled
        ]
        expected = (
            torch.matrix_exp(-1j * time * (h * hx + 0.4 * hzz))
            @ torch.matrix_exp(-1j * time * (-0.15 * hz + 0.25 * hxx))
        )
        gradients = [
            torch.autograd.grad((output.real * weights).sum(), (h, time))
            for output in (*outputs, expected)
        ]
        for output in outputs:
            torch.testing.assert_close(output, expected, atol=2e-11, rtol=2e-11)
        for actual in gradients[:2]:
            for derivative, reference in zip(actual, gradients[2]):
                torch.testing.assert_close(
                    derivative, reference, atol=2e-9, rtol=2e-9
                )


def test_joint_periodic_directed_pepo_matches_dense_with_translation():
    y = np.array([[0.0, -1j], [1j, 0.0]])
    paulis = {"X": X, "Y": y, "Z": Z}
    sites = tuple((i, j) for i in range(2) for j in range(2))
    site_index = {site: index for index, site in enumerate(sites)}
    terms_a = [PauliPEPOTerm("onsite", "X", 0.2, where=s) for s in sites]
    terms_b = [PauliPEPOTerm("onsite", "Z", -0.15, where=s) for s in sites]
    for site in sites:
        up = ((site[0] + 1) % 2, site[1])
        right = (site[0], (site[1] + 1) % 2)
        terms_a.extend((
            PauliPEPOTerm("edge", "XY", 0.05, where=(site, up), direction="u"),
            PauliPEPOTerm("edge", "ZZ", 0.1, where=(site, right), direction="r"),
        ))
        terms_b.extend((
            PauliPEPOTerm("edge", "YX", 0.07, where=(site, up), direction="u"),
            PauliPEPOTerm("edge", "XX", -0.08, where=(site, right), direction="r"),
        ))

    def dense_hamiltonian(terms):
        result = np.zeros((16, 16), dtype=complex)
        for term in terms:
            if term.support == "onsite":
                support = (site_index[term.where],)
                operator = paulis[term.paulis[0]]
            else:
                support = tuple(site_index[site] for site in term.where)
                operator = np.kron(
                    paulis[term.paulis[0]], paulis[term.paulis[1]]
                )
            result += term.coefficient * _embed(operator, support, tuple(range(4)))
        return result

    step = -0.017j
    expected = expm(step * dense_hamiltonian(terms_a)) @ expm(
        step * dense_hamiltonian(terms_b)
    )
    shift = {site: ((site[0] + 1) % 2, site[1]) for site in sites}
    counts = []
    for reuse, declared in ((False, ()), (True, ()), (True, (shift,))):
        a = PauliPEPOBasis.compile(
            2, 2, terms_a, cyclic=True, order=4,
            factorization="fixed", spatial_reuse=reuse,
            spatial_symmetries=declared,
        )
        b = PauliPEPOBasis.compile(
            2, 2, terms_b, cyclic=True, order=4,
            factorization="fixed", spatial_reuse=reuse,
            spatial_symmetries=declared,
        )
        actual = PEPOClusterProductExpansion.from_bases((a, b)).compile_exp()(
            step
        ).to_dense()
        np.testing.assert_allclose(actual, expected, atol=2e-12)
        counts.append(a.cache_info["last_local_targets_evaluated"])
        assert a.cache_info["declared_spatial_symmetry_count"] == len(declared)
    assert counts[0] == 13
    assert counts[1] < counts[0]
    assert counts[1] == counts[2]


def test_fixed_alignment_keeps_host_complex_factor():
    torch = pytest.importorskip("torch")
    j = torch.tensor(.2, dtype=torch.float32, requires_grad=True)
    basis = MPOBasis.from_local_terms(2, [
        ((0,), (X.astype('float32'),), MPOParameter(0)),
        ((1,), (Z.astype('float32'),), 1j),
    ])
    result = basis.compile_cluster_expansion(
        cluster_size=1, factorization="fixed", cutoff=0.
    )(.1, parameters=[j]).to_mpo().to_dense()
    expected = torch.kron(
        torch.matrix_exp(.1 * j * torch.tensor(X, dtype=j.dtype)).to(torch.complex64),
        torch.matrix_exp(.1j * torch.tensor(Z, dtype=j.dtype)),
    )
    assert result.dtype == torch.complex64
    torch.testing.assert_close(result, expected)
    (a,) = torch.autograd.grad(result.imag.sum(), (j,), retain_graph=True)
    (b,) = torch.autograd.grad(expected.imag.sum(), (j,))
    torch.testing.assert_close(a, b)


def test_fixed_positional_jax_gradient_through_target_alignment():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    basis = MPOBasis.from_local_terms(2, [((0, 1), (Z, Z), MPOParameter(0))])
    compiled = basis.compile_graph_cluster_expansion(
        graph="chain", cluster_size=2, factorization="fixed", cutoff=0., assembly="recursive"
    )

    def loss(j):
        return compiled(-.1j, parameters=[j]).to_mpo().to_dense()[0, 0].imag

    for value in (0., .2):
        actual, gradient = jax.value_and_grad(loss)(jnp.asarray(value))
        np.testing.assert_allclose(actual, -np.sin(.1 * value), atol=2e-7)
        np.testing.assert_allclose(gradient, -.1 * np.cos(.1 * value), atol=2e-7)
