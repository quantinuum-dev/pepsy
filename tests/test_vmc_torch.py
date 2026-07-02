"""Tests for Pepsy's optional torch VMC kernels."""

import pytest
import quimb.tensor as qtn

torch = pytest.importorskip("torch")

from pepsy.vmc.torch import (  # noqa: E402
    FermionSiteEncoding,
    TorchPEPSAmplitude,
    TorchSquareLattice,
    count_spinful_particles,
    heisenberg_connections,
    local_energy_from_connections,
    make_torch_peps_amplitude_model,
    metropolis_exchange_sweep,
    propose_spin_exchange,
    propose_spinful_exchange_or_hopping,
    random_spin_configs,
    random_spinful_configs,
    spinful_fermi_hubbard_connections,
    transverse_ising_connections,
)


def test_fermion_site_encoding_supports_symmray_and_vmc_torch_orders():
    symm = FermionSiteEncoding.symmray()
    vmct = FermionSiteEncoding.vmc_torch()

    configs = torch.tensor([[symm.empty, symm.double, symm.up, symm.down]])
    n_up, n_down = symm.decode(configs)
    assert n_up.tolist() == [[0, 1, 1, 0]]
    assert n_down.tolist() == [[0, 1, 0, 1]]
    assert symm.encode(n_up, n_down).tolist() == configs.tolist()

    configs = torch.tensor([[vmct.empty, vmct.double, vmct.up, vmct.down]])
    n_up, n_down = vmct.decode(configs)
    assert n_up.tolist() == [[0, 1, 1, 0]]
    assert n_down.tolist() == [[0, 1, 0, 1]]
    assert vmct.encode(n_up, n_down).tolist() == configs.tolist()

    with pytest.raises(ValueError, match="Unknown fermion site code"):
        symm.decode(torch.tensor([[9]]))


def test_torch_square_lattice_edges_match_row_major_open_boundary():
    graph = TorchSquareLattice(2, 3)
    assert graph.row_edges == {
        0: ((0, 1), (1, 2)),
        1: ((3, 4), (4, 5)),
    }
    assert graph.col_edges == {
        0: ((0, 3),),
        1: ((1, 4),),
        2: ((2, 5),),
    }
    assert graph.edges == (
        (0, 1),
        (1, 2),
        (3, 4),
        (4, 5),
        (0, 3),
        (1, 4),
        (2, 5),
    )


def test_torch_peps_amplitude_matches_direct_quimb_contraction():
    peps = qtn.PEPS.rand(
        Lx=2,
        Ly=2,
        bond_dim=2,
        phys_dim=2,
        seed=10,
        dtype="float64",
    )
    model = TorchPEPSAmplitude(peps, contraction="exact", dtype=torch.float64)
    rows = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]])
    amps = model(rows)

    direct = []
    for row in rows:
        tnx = peps.isel({
            peps.site_ind(site): int(row[i])
            for i, site in enumerate(peps.sites)
        })
        direct.append(tnx.contract(all))

    assert torch.allclose(amps, torch.as_tensor(direct, dtype=torch.float64))
    phase, log_abs = model.forward_log(rows)
    assert torch.allclose(phase * torch.exp(log_abs), amps)
    assert model.n_sites == 4
    assert model.n_params == sum(p.numel() for p in model.parameters())


def test_torch_peps_amplitude_supports_torch_optimizer_step():
    peps = qtn.PEPS.rand(
        Lx=2,
        Ly=2,
        bond_dim=2,
        phys_dim=2,
        seed=11,
        dtype="float64",
    )
    model = make_torch_peps_amplitude_model(
        peps,
        contraction="exact",
        dtype=torch.float64,
    )
    rows = torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1]])
    target = torch.tensor([0.25, -0.1], dtype=torch.float64)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)

    opt.zero_grad()
    before = model(rows).detach().clone()
    loss = (model(rows) - target).square().sum()
    loss.backward()
    assert all(param.grad is not None for param in model.parameters())
    opt.step()
    after = model(rows).detach()

    assert not torch.allclose(before, after)
    peps_after = model.to_peps()
    assert tuple(peps_after.sites) == tuple(peps.sites)


def test_spinful_exchange_hopping_proposal_preserves_particle_counts():
    encoding = FermionSiteEncoding.symmray()
    configs = torch.tensor([
        [encoding.empty, encoding.up, encoding.down, encoding.double],
        [encoding.up, encoding.down, encoding.empty, encoding.double],
    ])
    before = count_spinful_particles(configs, encoding=encoding)
    proposed, changed = propose_spinful_exchange_or_hopping(
        0,
        1,
        configs,
        hopping_rate=1.0,
        encoding=encoding,
        generator=torch.Generator().manual_seed(1),
    )
    after = count_spinful_particles(proposed, encoding=encoding)
    assert changed.tolist() == [True, True]
    assert after[0].tolist() == before[0].tolist()
    assert after[1].tolist() == before[1].tolist()


def test_spin_exchange_proposal_swaps_only_different_binary_spins():
    configs = torch.tensor([[0, 1], [1, 1]])
    proposed, changed = propose_spin_exchange(0, 1, configs)
    assert proposed.tolist() == [[1, 0], [1, 1]]
    assert changed.tolist() == [True, False]


def test_spinful_fermi_hubbard_connections_include_fermionic_signs():
    encoding = FermionSiteEncoding.symmray()
    graph = [(0, 1)]
    configs = torch.tensor([
        [encoding.up, encoding.down],
        [encoding.double, encoding.empty],
    ])
    conn = spinful_fermi_hubbard_connections(
        configs,
        graph,
        t=1.0,
        U=8.0,
        encoding=encoding,
    )
    rows = [
        (tuple(row.tolist()), float(coeff), int(batch_id))
        for row, coeff, batch_id in zip(conn.configs, conn.coeffs, conn.batch_ids)
    ]

    assert ((encoding.empty, encoding.double), 1.0, 0) in rows
    assert ((encoding.double, encoding.empty), 1.0, 0) in rows
    assert ((encoding.up, encoding.down), 1.0, 1) in rows
    assert ((encoding.down, encoding.up), -1.0, 1) in rows
    assert ((encoding.double, encoding.empty), 8.0, 1) in rows


def test_heisenberg_and_transverse_ising_connections():
    graph = [(0, 1)]
    configs = torch.tensor([[0, 1]])

    heis = heisenberg_connections(configs, graph, J=2.0)
    heis_rows = [
        (tuple(row.tolist()), float(coeff), int(batch_id))
        for row, coeff, batch_id in zip(heis.configs, heis.coeffs, heis.batch_ids)
    ]
    assert ((1, 0), 1.0, 0) in heis_rows
    assert ((0, 1), -0.5, 0) in heis_rows

    ising = transverse_ising_connections(configs, graph, J=2.0, h=3.0)
    ising_rows = [
        (tuple(row.tolist()), float(coeff), int(batch_id))
        for row, coeff, batch_id in zip(ising.configs, ising.coeffs, ising.batch_ids)
    ]
    assert ((0, 1), -0.5, 0) in ising_rows
    assert ((1, 1), 1.5, 0) in ising_rows
    assert ((0, 0), 1.5, 0) in ising_rows


def test_local_energy_from_connections_matches_constant_amplitude_sum():
    configs = torch.tensor([[0, 1]])
    amps = torch.ones(1, dtype=torch.float64)
    conn = heisenberg_connections(configs, [(0, 1)], J=2.0)

    def amplitude_fn(rows):
        return torch.ones(rows.shape[0], dtype=torch.float64)

    energy = local_energy_from_connections(configs, amps, conn, amplitude_fn)
    assert energy.tolist() == [0.5]


def test_torch_peps_amplitude_feeds_local_energy_kernel():
    peps = qtn.PEPS.rand(
        Lx=2,
        Ly=2,
        bond_dim=2,
        phys_dim=2,
        seed=12,
        dtype="float64",
    )
    model = TorchPEPSAmplitude(peps, contraction="exact", dtype=torch.float64)
    graph = TorchSquareLattice(2, 2)
    configs = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]])
    amps = model(configs)
    conn = heisenberg_connections(configs, graph, J=1.0)
    energy = local_energy_from_connections(configs, amps, conn, model)

    assert energy.shape == (2,)
    assert torch.isfinite(energy).all()


def test_metropolis_exchange_sweep_accepts_constant_amplitude_proposals():
    graph = TorchSquareLattice(1, 2)
    configs = torch.tensor([[0, 1], [1, 0]])

    def amplitude_fn(rows):
        return torch.ones(rows.shape[0], dtype=torch.float64)

    result = metropolis_exchange_sweep(
        configs,
        amplitude_fn,
        graph,
        proposal="spin",
        generator=torch.Generator().manual_seed(2),
    )
    assert result.configs.tolist() == [[1, 0], [0, 1]]
    assert result.n_proposed == 2
    assert result.n_accepted == 2
    assert result.acceptance_rate == 1.0


def test_random_sector_initializers_fix_particle_numbers():
    spin = random_spin_configs(
        4,
        6,
        2,
        generator=torch.Generator().manual_seed(3),
    )
    assert spin.sum(dim=1).tolist() == [2, 2, 2, 2]

    encoding = FermionSiteEncoding.symmray()
    fermion = random_spinful_configs(
        4,
        6,
        2,
        3,
        encoding=encoding,
        generator=torch.Generator().manual_seed(4),
    )
    n_up, n_down = count_spinful_particles(fermion, encoding=encoding)
    assert n_up.tolist() == [2, 2, 2, 2]
    assert n_down.tolist() == [3, 3, 3, 3]
