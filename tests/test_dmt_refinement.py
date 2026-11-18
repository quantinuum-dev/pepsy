"""Fixed-border objectives and independently contracted physical regions."""

from itertools import product

import numpy as np
import pytest

from pepsy.operators import compress_mpo_dmt, square_lattice_dmt_regions
from pepsy.operators._pauli_dmt import _PauliDMTWorkspace, _region_pattern, _split_scaled
from pepsy.operators.pauli_mpo import _host_array
from pepsy.optimizers import MpoOptimizer, PauliDMT
from test_mpo_dmt_replay import dense, embed, marginal, mixture
from test_pauli_dmt import _density
from test_pauli_dmt_large import product_mixture


def split(m, lv, rv, cap, *, sweeps=None, variant='density'):
    l, r, report = _split_scaled(
        m, lv, rv, max_bond=cap, cutoff=0., variant=variant, absorb='right',
        bond=0, diagnostics=True, truncation='paper' if sweeps is None else 'optimized',
        max_refinement_sweeps=sweeps or 0,
    )
    a = _host_array(l @ r)
    target = _host_array(m)
    np.testing.assert_allclose(_host_array(lv).conj().T @ (a - target), 0., atol=2e-11)
    np.testing.assert_allclose((a - target) @ _host_array(rv), 0., atol=2e-11)
    assert l.shape[1] <= cap
    np.testing.assert_allclose(_host_array(l).conj().T@_host_array(l), np.eye(l.shape[1]), atol=2e-12)
    assert np.linalg.svd(a, compute_uv=False)[cap:].max(initial=0.) < 2e-11
    return a, report


def test_closed_form_beats_conservative_budget_and_handles_smaller_cap():
    m, v = np.diag([1., .3, .1]), np.eye(3)[:, :1]
    paper, _ = split(m, v, v, 2)
    best, report = split(m, v, v, 2, sweeps=0)
    np.testing.assert_allclose(best, np.diag([1., .3, 0.]), atol=1e-14)
    assert np.linalg.norm(best - m)**2 < np.linalg.norm(paper - m)**2 / 9
    assert report.truncation_method == 'schur'
    # The old p+q bound is conservative, not an infeasibility theorem.
    smaller, _ = split(m, v, v, 1, sweeps=0)
    np.testing.assert_allclose(smaller, np.diag([1., 0., 0.]), atol=1e-14)


@pytest.mark.parametrize('p,q,compatible', [(2, 2, True), (2, 3, False), (2, 2, False)])
def test_complex_fixed_border_and_monotone_refinement(p, q, compatible):
    rng = np.random.default_rng(50 + q)
    m = rng.normal(size=(12, 12)) + 1j * rng.normal(size=(12, 12))
    m[0, 0] += 10
    if not compatible and p == q:
        m[:2, :2] = [[10., 0.], [0., 0.]]
    # Independent complex rotations exercise both covectors and right vectors.
    ul = np.linalg.qr(rng.normal(size=m.shape) + 1j*rng.normal(size=m.shape))[0]
    ur = np.linalg.qr(rng.normal(size=m.shape) + 1j*rng.normal(size=m.shape))[0]
    lv, rv = ul[:, :p], ur[:, :q]
    original = ul @ m @ ur.conj().T
    paper, _ = split(original, lv, rv, 7)
    seed, _ = split(original, lv, rv, 7, sweeps=0)
    best, report = split(original, lv, rv, 7, sweeps=8)
    assert np.linalg.norm(best-original) <= np.linalg.norm(seed-original) + 2e-12
    assert np.linalg.norm(seed-original) <= np.linalg.norm(paper-original) + 2e-12
    assert np.linalg.norm(best-original) < .95*np.linalg.norm(paper-original)
    assert ('schur' in report.truncation_method) == compatible
    if compatible:
        pblock, b, c, d = m[:p, :q], m[:p, q:], m[p:, :q], m[p:, q:]
        base = c @ np.linalg.solve(pblock, b)
        u, s, vh = np.linalg.svd(d-base, full_matrices=False)
        expected = m.copy()
        expected[p:, q:] = base + (u[:, :7-p]*s[:7-p]) @ vh[:7-p]
        np.testing.assert_allclose(best, ul @ expected @ ur.conj().T, atol=2e-11)


def test_rectangular_compatible_corner_and_incompatible_pseudoinverse_rejection():
    rng = np.random.default_rng(9)
    p = np.array([[2., 0., 0.], [0., 0., 0.]])
    b = np.vstack([rng.normal(size=6), np.zeros(6)])
    c = np.column_stack([rng.normal(size=7), np.zeros((7, 2))])
    m = np.block([[p, b], [c, rng.normal(size=(7, 6))]])
    lv, rv = np.eye(9)[:, :2], np.eye(9)[:, :3]
    _, report = split(m, lv, rv, 4, sweeps=0, variant='operator')
    assert report.truncation_method == 'schur'
    m[1, 4] = 1.
    # Incompatible border: a naive pseudoinverse would corrupt this entry.
    _, report = split(m, lv, rv, 6, sweeps=4, variant='operator')
    assert 'schur' not in report.truncation_method


def test_unresolved_corner_uses_safe_seed_and_zero_corner_can_share_rank():
    rng = np.random.default_rng(7)
    m = rng.normal(size=(8,8))
    m[:2,:2] = [[10.,0.], [0.,1e-12]]
    v = np.eye(8)[:, :2]
    _, report = split(m, v, v, 5, sweeps=2)
    assert 'schur' not in report.truncation_method
    # A zero corner with zero borders is compatible even for a traceless
    # operator. Its protected directions need not consume any actual rank.
    m = np.diag([0.,3.,2.])
    v = np.eye(3)[:, :1]
    a, report = split(m, v, v, 1, sweeps=0, variant='operator')
    np.testing.assert_allclose(a, np.diag([0.,3.,0.]), atol=1e-13)
    assert report.truncation_method == 'schur'


def test_cover_is_complete_minimal_and_deterministic():
    regions = [(0, 3, 4), (1, 3, 4), (0, 1, 3)]
    left, right = _region_pattern(regions, 2, 2, selection='min_cover')
    assert (left, right) == _region_pattern(reversed(regions), 2, 2, selection='min_cover')
    # Enumerate every observable on each region, including identity factors.
    for region in regions:
        for labels in product(range(2), repeat=len(region)):
            ls = tuple((s, p) for s, p in zip(region, labels) if s <= 2 and p)
            rs = tuple((s, p) for s, p in zip(region, labels) if s > 2 and p)
            assert ls in left or rs in right
    # For this binary example the graph is small enough for exhaustive cover.
    all_l, all_r, edges = set(), set(), set()
    for region in regions:
        for labels in product(range(2), repeat=len(region)):
            l = tuple((s,p) for s,p in zip(region,labels) if s <= 2 and p)
            r = tuple((s,p) for s,p in zip(region,labels) if s > 2 and p)
            if l and r:
                all_l.add(l); all_r.add(r); edges.add((l,r))
    vertices = [('l',s) for s in sorted(all_l)] + [('r',s) for s in sorted(all_r)]
    optimum = len(vertices)
    for bits in product([False, True], repeat=len(vertices)):
        chosen = {v for v, bit in zip(vertices, bits) if bit}
        if all(('l',l) in chosen or ('r',r) in chosen for l,r in edges):
            optimum = min(optimum, len(chosen))
    assert len(left) + len(right) - 2 == optimum


@pytest.mark.parametrize('n', [3, 6, 8])
def test_square_cover_never_uses_more_strings(n):
    regions = square_lattice_dmt_regions(n, n)
    old, new = [], []
    for bond in range(n*n-1):
        old.append(sum(map(len, _region_pattern(regions, bond))))
        new.append(sum(map(len, _region_pattern(regions, bond, selection='min_cover'))))
    assert all(a <= b for a,b in zip(new, old))
    assert sum(new) < sum(old)


@pytest.mark.parametrize('ordinary', [False, True])
def test_full_sweep_preserves_noncontiguous_2d_marginals(ordinary):
    rho = _density()
    regions = square_lattice_dmt_regions(3, 2)
    original = rho.to_mpo() if ordinary else rho
    before = dense(original) if ordinary else original.to_dense()
    options = dict(max_bond=20, regions=regions, truncation='optimized',
                   region_selection='min_cover', max_refinement_sweeps=3, return_report=True)
    out, report = compress_mpo_dmt(original, **options) if ordinary else original.compress_dmt(**options)
    after = dense(out) if ordinary else out.to_dense()
    assert np.linalg.norm(after-before) > 1e-7
    assert max(report.final_bond_dimensions) <= 20
    for region in regions:
        np.testing.assert_allclose(marginal(after, 6, region), marginal(before, 6, region), atol=2e-12)


def test_each_square_split_improves_same_target_and_preserves_regions():
    rho, _, _ = product_mixture(9, rank=40, strength=.5)
    regions = square_lattice_dmt_regions(3, 3)
    workspace = _PauliDMTWorkspace(rho, max_bond=30, regions=regions,
                                  truncation='optimized', region_selection='min_cover',
                                  max_refinement_sweeps=3)
    for bond in range(8):
        from pepsy.operators import PauliMPO
        target = PauliMPO.from_pauli_cores(workspace.cores).to_dense()
        paper = _PauliDMTWorkspace(
            PauliMPO.from_pauli_cores(workspace.cores), max_bond=30, regions=regions,
            region_selection='min_cover',
        )
        paper.split_pair(bond)
        paper_matrix = PauliMPO.from_pauli_cores(paper.cores).to_dense()
        workspace.split_pair(bond)
        candidate = PauliMPO.from_pauli_cores(workspace.cores).to_dense()
        assert np.linalg.norm(candidate-target) <= np.linalg.norm(paper_matrix-target) + 2e-12
        for region in regions:
            np.testing.assert_allclose(marginal(candidate, 9, region), marginal(target, 9, region), atol=2e-12)


@pytest.mark.parametrize('backend,dtype', [('numpy','complex128'), ('torch','complex128'),
                                         ('jax','complex128'), ('numpy','complex64')])
def test_native_backend_preservation(backend, dtype):
    rng = np.random.default_rng(8)
    m = (rng.normal(size=(9,9)) + 1j*rng.normal(size=(9,9))).astype(dtype)
    m[0,0] += 10
    v = np.eye(9, dtype=dtype)[:, :2]
    if backend == 'torch':
        torch = pytest.importorskip('torch')
        m, v = torch.tensor(m), torch.tensor(v)
    elif backend == 'jax':
        jax = pytest.importorskip('jax')
        jax.config.update('jax_enable_x64', True)
        m, v = jax.numpy.asarray(m), jax.numpy.asarray(v)
    l, r, _ = _split_scaled(m, v, v, max_bond=6, cutoff=0., variant='density',
                            absorb='left', bond=0, diagnostics=True, truncation='optimized',
                            max_refinement_sweeps=2)
    assert l.dtype == m.dtype and r.dtype == m.dtype
    np.testing.assert_allclose(_host_array(r)@_host_array(r).conj().T,
                               np.eye(r.shape[0]), atol=2e-6 if dtype=='complex64' else 2e-12)
    if backend != 'numpy':
        assert l.device == m.device and r.device == m.device
    defect = _host_array(l@r-m)
    np.testing.assert_allclose(_host_array(v).conj().T@defect, 0., atol=2e-4 if dtype=='complex64' else 2e-11)


def test_optimizer_replay_and_persistent_pauli_driver():
    mpo, _, _ = mixture(6, rank=24)
    gate = np.diag(np.exp(1j*np.array([.1,.2,-.3,.1])))
    queue = [(gate,(2,3)), (gate,(1,4))]
    reference, checked = dense(mpo), []

    def check(candidate):
        nonlocal reference
        payload, where = queue[len(checked)]
        full = embed(payload.T, where, 6)
        target = full @ reference @ full.conj().T
        reference = dense(candidate)
        for i in range(4):
            sites = tuple(range(i,i+3))
            np.testing.assert_allclose(marginal(reference,6,sites), marginal(target,6,sites), atol=2e-12)
        checked.append(1)
        return True

    opt = MpoOptimizer(mpo, queue, chi=20, mode='dmt', contraction_opt='greedy')
    with pytest.warns(RuntimeWarning, match='finite_check'):
        opt.run(dmt_truncation='optimized', dmt_region_selection='min_cover',
                dmt_max_refinement_sweeps=2, dmt_diagnostics=True, fidelity_samples=0,
                finite_check=check)
    assert len(checked) == 2
    assert opt.p.max_bond() <= 20
    _, report = opt.compress_dmt(truncation='optimized', region_selection='min_cover', return_report=True)
    assert report.max_constraint_error < 1e-12
    pauli = _density()
    target = pauli.apply_gate(gate, (2,3), mode='conjugate')
    driver = PauliDMT(pauli, max_bond=12, truncation='optimized', max_refinement_sweeps=2)
    driver.apply_gate(gate, (2,3))
    for i in range(4):
        sites = tuple(range(i,i+3))
        np.testing.assert_allclose(driver.state.partial_trace(sites, keep=True).to_dense(),
                                   target.partial_trace(sites, keep=True).to_dense(), atol=2e-12)


def test_torch_refined_protected_observable_gradient():
    torch = pytest.importorskip('torch')
    from pepsy.backends import TorchLinalgConfig

    rng = np.random.default_rng(2)
    matrix = rng.normal(size=(10,10)) + 1j*rng.normal(size=(10,10))
    matrix[:2,:2] = [[10.,0.], [0.,0.]]  # incompatible corner exercises ALS
    m = torch.tensor(matrix, dtype=torch.complex128, requires_grad=True)
    v = torch.eye(10, dtype=torch.complex128)[:, :2]
    with TorchLinalgConfig(stabilized=True).activated():
        l, r, _ = _split_scaled(m, v, v, max_bond=6, cutoff=0., variant='density',
                                absorb='right', bond=0, diagnostics=True,
                                truncation='optimized', max_refinement_sweeps=2)
        (l@r)[0].real.sum().backward()
    expected = np.zeros((10,10), dtype=complex)
    expected[0] = 1.
    np.testing.assert_allclose(m.grad.detach().numpy(), expected, atol=2e-9)


@pytest.mark.parametrize('options', [{'truncation':'wrong'}, {'max_refinement_sweeps':-1},
                                    {'max_refinement_sweeps':True}, {'region_selection':'wrong'}])
def test_options_fail_transactionally(options):
    rho = _density()
    before = rho.to_dense()
    with pytest.raises(ValueError):
        rho.compress_dmt(max_bond=12, inplace=True, **options)
    np.testing.assert_array_equal(rho.to_dense(), before)
