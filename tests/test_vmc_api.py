"""Tests for the backend-neutral VMC API contracts."""

import numpy as np
import pytest

from pepsy.vmc import (
    BackendCapabilityWarning,
    ContractionConfig,
    LocalMatrixTerm,
    MCState,
    OperatorFactor,
    OperatorSum,
    ProductTerm,
    SamplingConfig,
    OptimizationConfig,
    VMCMeasurement,
    VMCBackendCapabilityError,
    VMCOptimizationResult,
    VMC,
    VMCProblem,
    VMCSamples,
    compile_operator_sum_netket,
    compile_operator_sum_torch,
)


def test_operator_sum_normalizes_symbolic_and_matrix_terms():
    hopping = ProductTerm(
        coefficient=-1.0,
        factors=(
            OperatorFactor(0, "fermion", spin="up", dagger=True),
            OperatorFactor(1, "fermion", spin="up", dagger=False),
        ),
    )
    onsite = LocalMatrixTerm(
        support=(0,),
        matrix=np.eye(2),
        coefficient=0.5,
    )
    terms = OperatorSum.from_terms(
        (term for term in (hopping, onsite)),
        constant=1.25,
        metadata={"statistics": "fermion"},
    )

    assert len(terms) == 2
    assert terms.sites == (0, 1)
    assert terms.metadata["statistics"] == "fermion"
    assert hopping.support == (0, 1)

    with pytest.raises(TypeError, match="ProductTerm or LocalMatrixTerm"):
        OperatorSum(terms=(object(),))

    with pytest.raises(ValueError, match="expected rank 4"):
        LocalMatrixTerm(support=(0, 1), matrix=np.eye(4))
    with pytest.raises(ValueError, match="dimensions must match"):
        LocalMatrixTerm(support=(0,), matrix=np.zeros((2, 3)))


def test_vmc_problem_freezes_observables_and_site_order():
    problem = VMCProblem(
        peps=object(),
        hamiltonian=OperatorSum(),
        observables={"density": OperatorSum()},
        symmetry="U1U1",
        site_order=((0, 0), (0, 1)),
    )

    assert isinstance(problem.observables["density"], OperatorSum)
    with pytest.raises(TypeError):
        problem.observables["other"] = OperatorSum()


def test_mcstate_uses_netket_total_sample_convention_and_bridges_to_problem():
    state = MCState(
        object(),
        n_samples=12,
        n_chains=3,
        n_discard_per_chain=4,
        symmetry="U1U1",
        site_order=(0, 1),
    )

    assert state.n_samples == 12
    assert state.n_chains == 3
    assert state.n_discard_per_chain == 4
    assert state.sampling.n_samples_per_chain == 4
    assert state.ansatz is state.peps
    assert state.to_problem(OperatorSum()).site_order == (0, 1)

    with pytest.raises(ValueError, match="divisible"):
        MCState(object(), n_samples=5, n_chains=2)
    with pytest.raises(ValueError, match="either sampling"):
        MCState(object(), sampling=SamplingConfig(), n_samples=8)


def test_netket_shaped_vmc_driver_builds_from_mcstate(monkeypatch):
    import pepsy.vmc.torch as torch_vmc

    seen = {}
    native = object()

    class FakeSetup:
        @property
        def native(self):
            return native

        def sample(self, sampling=None):
            return VMCSamples(
                configs=np.zeros((2, 2, 1), dtype=np.int64),
                n_samples_per_chain=2,
                n_chains=2,
            )

        def measure(self, observables=None, *, sampling=None, **kwargs):
            seen["measure_kwargs"] = kwargs
            values = {"energy": "native-energy"}
            if observables:
                values.update({name: value for name, value in observables.items()})
            return VMCMeasurement(energy_mean=-1.0, observables=values)

        def optimize(self, optimization=None, *, n_steps=None, **kwargs):
            seen["optimization"] = optimization
            seen["n_steps"] = n_steps
            return VMCOptimizationResult(
                steps=np.arange(1, (n_steps or optimization.n_steps) + 1),
                energies=np.full(n_steps or optimization.n_steps, -1.0),
                errors=np.zeros(n_steps or optimization.n_steps),
            )

    def fake_builder(problem, **kwargs):
        seen["problem"] = problem
        seen.update(kwargs)
        return FakeSetup()

    monkeypatch.setattr(torch_vmc, "build_torch_vmc", fake_builder)
    state = MCState(object(), n_samples=4, n_chains=2, symmetry="U1U1")
    hamiltonian = OperatorSum()
    driver = VMC(
        hamiltonian,
        state,
        backend="torch",
        fermion=object(),
        observables={"density": OperatorSum()},
    )

    assert driver.state is state
    assert driver.native is native
    assert seen["problem"].hamiltonian is hamiltonian
    assert seen["problem"].observables["density"] == OperatorSum()
    assert seen["sampling"] is state.sampling
    assert driver.sample().chain_shape == (2, 2)
    assert driver.expect().energy == -1.0
    assert driver.expect(OperatorSum()).observables["expectation"] == OperatorSum()
    supplied = VMCSamples(
        configs=np.zeros((2, 1), dtype=np.int64),
        proposal_log_probs=np.zeros(2),
    )
    assert driver.measure(samples=supplied).energy == -1.0
    assert seen["measure_kwargs"]["samples"] is supplied
    assert driver.run(3).final_energy == -1.0
    assert seen["n_steps"] == 3


def test_common_results_keep_chain_shape_and_shifted_energy():
    samples = VMCSamples(
        configs=np.zeros((4, 2, 3), dtype=np.int64),
        n_samples_per_chain=4,
        n_chains=2,
    )
    measurement = VMCMeasurement(
        energy_mean=-1.5,
        observables={"density": 0.25},
    )
    history = VMCOptimizationResult(
        steps=np.arange(2),
        energies=np.array([-1.0, -1.5]),
        errors=np.array([0.1, 0.05]),
        energy_shift=0.25,
        per_site=2,
    )

    assert samples.chain_shape == (4, 2)
    assert measurement.energy == -1.5
    assert measurement.observables["density"] == 0.25
    assert history.final_energy == -1.5
    assert history.final_error == 0.05
    assert np.allclose(history.shifted_energies, [-0.75, -1.25])
    assert np.allclose(history.displayed_energies, [-0.375, -0.625])


def test_common_samples_distinguish_fixed_weights_from_proposal_density():
    samples = VMCSamples(
        configs=np.zeros((3, 1), dtype=np.int64),
        weights=np.array([0.2, 0.3, 0.5]),
    )
    assert np.allclose(samples.weights, [0.2, 0.3, 0.5])

    with pytest.raises(ValueError, match="either weights or proposal_log_probs"):
        VMCSamples(
            configs=np.zeros((3, 1), dtype=np.int64),
            weights=np.ones(3),
            proposal_log_probs=np.zeros(3),
        )


def test_warning_types_are_backend_neutral():
    assert issubclass(BackendCapabilityWarning, UserWarning)
    assert issubclass(VMCBackendCapabilityError, NotImplementedError)


def test_netket_portable_adapter_rejects_external_weighted_batches():
    from pepsy.vmc.netket import NetKetVMCSetup

    setup = NetKetVMCSetup(setup=object(), problem=object())
    with pytest.raises(VMCBackendCapabilityError, match="externally supplied"):
        setup.measure(samples=np.zeros((2, 1), dtype=np.int64))
    with pytest.raises(VMCBackendCapabilityError, match="externally supplied"):
        setup.optimize(n_steps=1, weights=np.ones(2))


def test_shared_configuration_objects_normalize_aliases_and_defaults():
    contraction = ContractionConfig(method="boundary_mps", chi=4, cutoff=1e-8)
    sampling = SamplingConfig(
        n_samples_per_chain=8,
        n_chains=2,
        burn_in=3,
        thin=2,
    )
    optimization = OptimizationConfig(method="min-sr", n_steps=4)

    assert contraction.method == "boundary"
    assert contraction.chi == 4
    assert sampling.thin == 2
    assert sampling.n_samples == 16
    assert sampling.torch_kwargs()["n_samples"] == 16
    assert sampling.netket_kwargs()["n_samples"] == 16
    assert optimization.method == "minsr"

    with pytest.raises(ValueError, match="chi is required"):
        ContractionConfig(method="ctmrg")
    with pytest.raises(ValueError, match="n_samples_per_chain"):
        SamplingConfig(n_samples_per_chain=0)


def test_torch_driver_consumes_shared_sampling_and_optimization_configs():
    torch = pytest.importorskip("torch")
    from pepsy.vmc import TorchVMCDriver

    class ProductAmplitude(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weights = torch.nn.Parameter(
                torch.tensor([1.0, 2.0], dtype=torch.float64)
            )

        def forward(self, configs):
            return self.weights[configs].prod(dim=1)

    driver = TorchVMCDriver(
        ProductAmplitude(),
        [(0, 1)],
        torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        terms={0: torch.tensor([[0.0, 1.0], [1.0, 0.0]])},
        proposal="spin",
    )
    samples = driver.sample(
        sampling=SamplingConfig(
            n_samples_per_chain=2,
            n_chains=2,
            burn_in=0,
            thin=1,
            seed=12,
        )
    )
    assert samples.configs.shape == (2, 2, 2)
    assert samples.to_common().chain_shape == (2, 2)

    history = driver.optimize(
        optimization=OptimizationConfig(
            method="sgd",
            n_steps=1,
            learning_rate=1e-2,
            progress=False,
        ),
        sample_sweeps=1,
    )
    assert len(history) == 1


def test_torch_compiler_lowers_common_matrix_terms():
    torch = pytest.importorskip("torch")
    from pepsy.vmc import torch_hamiltonian_connections

    common = OperatorSum.from_terms(
        [
            LocalMatrixTerm(
                support=(0,),
                matrix=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
                coefficient=0.5,
            )
        ],
        constant=0.25,
    )
    compiled = compile_operator_sum_torch(common)
    configs = torch.tensor([[0], [1]], dtype=torch.long)
    connections = torch_hamiltonian_connections(
        configs,
        compiled.terms,
        constant=compiled.constant,
    )

    values = {}
    for config, coefficient, batch_id in zip(
        connections.configs.tolist(),
        connections.coeffs.tolist(),
        connections.batch_ids.tolist(),
    ):
        values[(int(batch_id), tuple(config))] = (
            values.get((int(batch_id), tuple(config)), 0.0) + coefficient
        )
    assert values[(0, (0,))] == pytest.approx(0.75)
    assert values[(0, (1,))] == pytest.approx(1.5)
    assert values[(1, (0,))] == pytest.approx(1.0)
    assert values[(1, (1,))] == pytest.approx(2.25)


def test_torch_driver_consumes_compiled_common_term_constants():
    torch = pytest.importorskip("torch")
    from pepsy.vmc import TorchVMCDriver

    class ConstantAmplitude(torch.nn.Module):
        def forward(self, configs):
            return torch.ones(configs.shape[0], dtype=torch.float64)

    driver = TorchVMCDriver(
        ConstantAmplitude(),
        [],
        torch.tensor([[0]], dtype=torch.long),
        terms={0: torch.eye(2, dtype=torch.float64)},
        proposal="spin",
    )
    compiled = compile_operator_sum_torch(
        OperatorSum.from_terms(
            [
                LocalMatrixTerm(
                    support=(0,),
                    matrix=np.eye(2),
                )
            ],
            constant=0.5,
        )
    )
    connections = driver.make_connections(
        torch.tensor([[0]], dtype=torch.long),
        terms=compiled,
    )
    assert connections.batch_ids.tolist() == [0, 0]
    assert connections.configs.tolist() == [[0], [0]]
    assert sum(connections.coeffs.tolist()) == pytest.approx(1.5)


def test_netket_facade_rejects_shared_sampling_settings_it_cannot_apply():
    from pepsy.vmc.netket import NetKetVMCSetup

    class FakeNativeSetup:
        def sample(self, sampling=None):
            return VMCSamples(
                configs=np.zeros((2, 2, 1), dtype=np.int64),
                n_samples_per_chain=2,
                n_chains=2,
            )

    facade = NetKetVMCSetup(
        setup=FakeNativeSetup(),
        problem=VMCProblem(peps=object(), hamiltonian=object()),
    )
    with pytest.raises(VMCBackendCapabilityError, match="thin"):
        facade.sample(SamplingConfig(n_samples_per_chain=2, n_chains=2, thin=2))
    with pytest.raises(VMCBackendCapabilityError, match="seed/sampler_seed"):
        facade.sample(
            SamplingConfig(n_samples_per_chain=2, n_chains=2, seed=3)
        )


def test_build_netket_vmc_passes_the_common_problem_to_native_builder(monkeypatch):
    import pepsy.vmc.netket as netket_vmc

    native = object()
    seen = {}

    def fake_builder(peps, **kwargs):
        seen["peps"] = peps
        seen.update(kwargs)
        return native

    monkeypatch.setattr(netket_vmc, "build_fermion_vmc", fake_builder)
    problem = VMCProblem(
        peps=object(),
        hamiltonian=OperatorSum(),
        observables={"density": OperatorSum()},
        symmetry="U1U1",
    )
    sampling = SamplingConfig(n_samples_per_chain=2, n_chains=2, burn_in=0)
    facade = netket_vmc.build_netket_vmc(
        problem,
        fermion=object(),
        sampling=sampling,
    )

    assert facade.problem is problem
    assert facade.native is native
    assert seen["peps"] is problem.peps
    assert seen["hamiltonian"] is problem.hamiltonian
    assert seen["observables"] == problem.observables
    assert seen["sampling"] is sampling


def test_torch_amplitude_accepts_common_contraction_config():
    qtn = pytest.importorskip("quimb.tensor")
    torch = pytest.importorskip("torch")
    from pepsy.vmc import TorchPEPSAmplitude

    peps = qtn.PEPS.rand(
        Lx=2,
        Ly=2,
        bond_dim=2,
        phys_dim=2,
        seed=193,
        dtype="float64",
    )
    model = TorchPEPSAmplitude(
        peps,
        contraction=ContractionConfig(method="boundary", chi=4, cutoff=0.0),
        dtype=torch.float64,
    )
    assert model.contraction == "boundary"
    assert model.chi == 4


def test_netket_setup_consumes_shared_sampling_config():
    nk = pytest.importorskip("netket")
    from pepsy.vmc.netket import NetKetPEPSVMC

    class Ansatz:
        n_sites = 4
        n_params = 4

    hilbert = nk.hilbert.Spin(s=1 / 2, N=4)
    graph = nk.graph.Hypercube(length=4, n_dim=1, pbc=False)
    hamiltonian = nk.operator.Ising(hilbert, graph=graph, h=1.0, J=1.0)
    sampler = nk.sampler.MetropolisLocal(hilbert, n_chains=2)
    vstate = nk.vqs.MCState(
        sampler,
        nk.models.RBM(alpha=1),
        n_samples=8,
        n_discard_per_chain=0,
        seed=4,
    )
    setup = NetKetPEPSVMC(
        hilbert,
        graph,
        hamiltonian,
        sampler,
        vstate,
        vstate.model,
        Ansatz(),
        None,
        None,
    )
    samples = setup.sample(
        SamplingConfig(n_samples_per_chain=3, n_chains=2, burn_in=0)
    )
    assert samples.chain_shape == (3, 2)
    assert samples.configs.shape == (3, 2, 4)


def test_netket_compiler_lowers_common_fermion_terms():
    nk = pytest.importorskip("netket")
    from pepsy.vmc import OperatorFactor

    hilbert = nk.hilbert.SpinOrbitalFermions(
        2,
        s=1 / 2,
        n_fermions_per_spin=(1, 1),
    )
    common = OperatorSum.from_terms(
        [
            ProductTerm(
                coefficient=-1.0,
                factors=(
                    OperatorFactor(0, "fermion", spin="up", dagger=True),
                    OperatorFactor(1, "fermion", spin="up", dagger=False),
                ),
            )
        ],
        constant=0.5,
    )

    compiled = compile_operator_sum_netket(hilbert, common)
    assert compiled.hilbert is hilbert


def test_common_spinful_fermion_operator_has_matching_exact_local_energies():
    """The Torch and NetKet compilers agree in a tiny exact Fock sector.

    This is deliberately an operator/local-energy oracle rather than a VMC
    optimization test. It fixes the public ``empty, down, up, double`` Torch
    codes, the NetKet spin-orbital columns, and the one fermionic coordinate
    phase needed to compare their two-site Fock bases.
    """
    torch = pytest.importorskip("torch")
    nk = pytest.importorskip("netket")
    import pepsy as py
    from pepsy.vmc.netket import (
        netket_spin_orbital_columns,
        occupation_to_phys_indices,
    )
    from pepsy.vmc.torch import torch_hamiltonian_connections

    terms = OperatorSum.from_terms(
        (
            ProductTerm(
                -1.0,
                (
                    OperatorFactor(0, "fermion", spin="up", dagger=True),
                    OperatorFactor(1, "fermion", spin="up", dagger=False),
                ),
            ),
            ProductTerm(
                -1.0,
                (
                    OperatorFactor(1, "fermion", spin="up", dagger=True),
                    OperatorFactor(0, "fermion", spin="up", dagger=False),
                ),
            ),
            ProductTerm(
                -0.4,
                (
                    OperatorFactor(0, "fermion", spin="down", dagger=True),
                    OperatorFactor(1, "fermion", spin="down", dagger=False),
                ),
            ),
            ProductTerm(
                -0.4,
                (
                    OperatorFactor(1, "fermion", spin="down", dagger=True),
                    OperatorFactor(0, "fermion", spin="down", dagger=False),
                ),
            ),
            ProductTerm(
                0.7,
                (
                    OperatorFactor(0, "number", spin="up"),
                    OperatorFactor(0, "number", spin="down"),
                ),
            ),
            ProductTerm(0.2, (OperatorFactor(1, "number", spin="up"),)),
        ),
        constant=0.13,
    )
    hilbert = nk.hilbert.SpinOrbitalFermions(
        2,
        s=1 / 2,
        n_fermions_per_spin=(1, 1),
    )
    netket_matrix = np.asarray(
        compile_operator_sum_netket(hilbert, terms).to_dense()
    )
    rows = np.asarray(hilbert.all_states(), dtype=int)
    columns = netket_spin_orbital_columns(hilbert)
    configs = occupation_to_phys_indices(
        rows,
        columns,
        phys_charges=((0, 0), (0, 1), (1, 0), (1, 1)),
    )

    compiled = compile_operator_sum_torch(
        terms,
        fermion=py.Fermion(symmetry="U1U1", spinful=True),
        site_order=(0, 1),
    )
    connections = torch_hamiltonian_connections(
        torch.as_tensor(configs, dtype=torch.long),
        compiled.terms,
        site_order=(0, 1),
        constant=compiled.constant,
    )
    config_index = {tuple(config): index for index, config in enumerate(configs)}
    torch_matrix = np.zeros((len(configs), len(configs)), dtype=complex)
    for eta, coefficient, source in zip(
        connections.configs.tolist(),
        connections.coeffs.tolist(),
        connections.batch_ids.tolist(),
    ):
        torch_matrix[config_index[tuple(eta)], source] += coefficient

    # NetKet orders modes as (down_0, down_1, up_0, up_1), while the Torch
    # connection table uses the site-local (down_0, up_0, down_1, up_1)
    # coordinate gauge. The fixed-number sector makes this basis phase exact.
    n_up = rows[:, columns.up]
    n_down = rows[:, columns.down]
    phase = 1 - 2 * n_up[:, 0] * n_down[:, 1]
    netket_matrix = phase[:, None] * netket_matrix * phase[None, :]

    assert np.allclose(torch_matrix, torch_matrix.conj().T, atol=1e-7)
    assert np.allclose(torch_matrix, netket_matrix, atol=1e-7)

    amplitudes = np.asarray([1.0, 0.6 - 0.2j, -0.4 + 0.5j, 0.3 + 0.7j])
    torch_local_energy = (torch_matrix @ amplitudes) / amplitudes
    netket_local_energy = (netket_matrix @ amplitudes) / amplitudes
    assert np.allclose(torch_local_energy, netket_local_energy, atol=1e-7)


def test_netket_compiler_flattens_common_multisite_matrix_axes():
    nk = pytest.importorskip("netket")
    hilbert = nk.hilbert.Spin(s=1 / 2, N=2)
    matrix = np.arange(16, dtype=float).reshape(2, 2, 2, 2)
    common = OperatorSum.from_terms(
        [LocalMatrixTerm(support=(0, 1), matrix=matrix)]
    )

    compiled = compile_operator_sum_netket(hilbert, common)
    reference = nk.operator.LocalOperator(
        hilbert,
        matrix.reshape(4, 4),
        acting_on=[0, 1],
    )
    assert np.allclose(compiled.to_dense(), reference.to_dense())
