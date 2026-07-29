"""Regression tests for NetKet's flat-Z2 PEPS and warmup reporting."""

from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.smoke
def test_prepare_fermionic_peps_flattens_odd_z2_bond_dimension():
    """Odd D creates unequal Z2 blocks that must be padded before JAX VMC."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    import pepsy as py
    from pepsy.vmc.netket import prepare_fermionic_peps_for_netket

    Lx = Ly = 2
    sites = tuple((x, y) for x in range(Lx) for y in range(Ly))
    occupations = {
        (0, 0): 1,
        (0, 1): 1,
        (1, 0): 1,
        (1, 1): 0,
    }
    fermion = py.Fermion(
        spinful=True,
        symmetry="Z2",
        to_backend=py.backend_jax(
            device=jax.devices()[0], dtype=jnp.complex64
        ),
    )
    peps = py.hrs_to_peps(
        (Lx, Ly),
        fermion=fermion,
        occupations=occupations,
        chi=3,
        seed=31,
        dtype="float32",
    )

    with pytest.warns(RuntimeWarning, match="Zero-padded unequal Z2"):
        prepared = prepare_fermionic_peps_for_netket(peps)

    assert tuple(prepared.sites) == sites
    assert all("Flat" in type(prepared[site].data).__name__ for site in sites)


def test_explicit_native_hubbard_terms_compile_to_matching_netket_terms():
    """NetKet uses the supplied native term mapping, not Fermion attributes."""
    import pepsy as py
    from pepsy.vmc.netket import (
        _native_fermi_hubbard_terms_to_netket,
        fermion_model_terms,
    )

    fermion = py.Fermion(spinful=True, symmetry="U1U1")
    t, U, mu = 1.25, 3.5, 0.2
    terms = {
        (0, 1): -t * fermion.hopping_operator(),
        0: fermion.onsite_term(0, U=U, mu=mu),
        1: fermion.onsite_term(1, U=U, mu=mu),
    }
    hamiltonian = fermion.hamiltonian(terms)
    actual, constant = _native_fermi_hubbard_terms_to_netket(
        fermion,
        hamiltonian,
        site_order=(0, 1),
    )
    expected = fermion_model_terms(
        fermion,
        ((0, 1),),
        t=t,
        U=U,
        mu=mu,
        n_sites=2,
    )

    def collect(entries):
        result = {}
        for coefficient, operators in entries:
            result[tuple(operators)] = result.get(tuple(operators), 0.0) + coefficient
        return result

    assert constant == pytest.approx(0.0)
    assert collect(actual) == pytest.approx(collect(expected))


@pytest.mark.smoke
def test_warmup_summary_reports_stage_times_and_amplitude_rows(capsys):
    """Warmup uses NetKet's JIT forward and gradient routes at chunk size."""
    from pepsy.vmc.netket import warmup_netket_vmc

    class State:
        sampler = SimpleNamespace(n_chains=2, sweep_size=4)
        sampler_state = None
        samples = np.zeros((2, 3, 4), dtype=np.int8)
        n_samples = 6
        n_discard_per_chain = 1
        chunk_size = 2

        def __init__(self):
            self.log_value_batches = []
            self.expect_and_grad_calls = 0

        def reset(self):
            pass

        def log_value(self, configs):
            self.log_value_batches.append(tuple(configs.shape))
            return configs.sum(axis=-1)

        def expect_and_grad(self, hamiltonian):
            self.expect_and_grad_calls += 1
            return SimpleNamespace(mean=-1.25), {"tensor": np.ones(1)}

    state = State()
    elapsed = warmup_netket_vmc(
        SimpleNamespace(vstate=state, hamiltonian=object()),
        progress=False,
        verbose=True,
    )
    output = capsys.readouterr().out

    assert elapsed >= 0.0
    assert "sampler" in output
    assert "JIT log amplitudes" in output
    assert "energy + gradient" in output
    assert "6 retained = 2 chains x 3/chain" in output
    assert "representative 2-row forward chunk" in output
    assert state.log_value_batches == [(2, 4)]
    assert state.expect_and_grad_calls == 1


@pytest.mark.smoke
def test_vmc_progress_starts_before_the_first_netket_update(monkeypatch):
    """The user sees a status while the first JIT-heavy update is running."""
    from pepsy.vmc import netket as netket_module

    class Bar:
        def __init__(self):
            self.description = None
            self.postfix = None
            self.updates = 0
            self.closed = False

        def set_description_str(self, value):
            self.description = value

        def set_postfix_str(self, value):
            self.postfix = value

        def update(self, value):
            self.updates += value

        def close(self):
            self.closed = True

    bar = Bar()
    monkeypatch.setattr(
        netket_module, "_make_progress_bar", lambda **kwargs: bar
    )
    callback = netket_module._VMCProgressCallback(3, enabled=True)

    callback.start("first update: sampling and compiling gradients")
    assert bar.description == "VMC 0/3: preparing"
    assert "compiling gradients" in bar.postfix

    class PendingParameter:
        def __init__(self):
            self.synchronized = False

        def block_until_ready(self):
            self.synchronized = True

    pending_parameter = PendingParameter()
    stats = SimpleNamespace(mean=-1.0, error_of_mean=0.1, variance=0.2)
    driver = SimpleNamespace(
        _loss_name="Energy",
        _loss_stats=stats,
        variational_state=SimpleNamespace(parameters={"t0": pending_parameter}),
    )
    callback(1, {"Energy": stats}, driver)

    assert pending_parameter.synchronized
    assert bar.description == "VMC energy"
    assert "first update" in bar.postfix
    assert bar.updates == 1
    callback.close()
    assert bar.closed


@pytest.mark.smoke
def test_vmc_progress_fails_fast_after_a_nonfinite_parameter_update():
    """A diverged PEPS update must not spend further VMC steps sampling NaNs."""
    from pepsy.vmc import netket as netket_module

    stats = SimpleNamespace(mean=-1.0, error_of_mean=0.1, variance=0.2)
    driver = SimpleNamespace(
        _loss_name="Energy",
        _loss_stats=stats,
        variational_state=SimpleNamespace(parameters={"t0": np.array([np.nan])}),
    )
    callback = netket_module._VMCProgressCallback(2, enabled=False)

    with pytest.raises(FloatingPointError, match="non-finite PEPS parameters"):
        callback(1, {"Energy": stats}, driver)


@pytest.mark.smoke
def test_sample_resource_monitor_reports_and_retains_metrics(monkeypatch, capsys):
    """Sampling forwards the opt-in GPU/RSS report into its diagnostics."""
    from pepsy.vmc import netket as netket_module

    report = netket_module.NetKetResourceUsage(
        elapsed_seconds=0.2,
        host_rss_before_mib=128.0,
        host_rss_after_mib=144.0,
        host_rss_peak_mib=160.0,
        gpu_after=(
            netket_module.NetKetGPUUsage(
                index=0,
                name="test GPU",
                memory_used_mib=1024,
                memory_total_mib=2048,
                utilization_percent=25,
                memory_utilization_percent=10,
                process_memory_mib=512,
            ),
        ),
        gpu_peak=(
            netket_module.NetKetGPUUsage(
                index=0,
                name="test GPU",
                memory_used_mib=1536,
                memory_total_mib=2048,
                utilization_percent=90,
                memory_utilization_percent=20,
                process_memory_mib=1024,
            ),
        ),
    )

    class Monitor:
        def __init__(self, *, interval):
            self.interval = interval
            self.started = False

        def start(self):
            self.started = True
            return self

        def stop(self):
            assert self.started
            return report

    class State:
        sampler_state = None
        samples = np.zeros((2, 3, 4), dtype=np.int8)
        n_samples = 6
        n_discard_per_chain = 1
        chunk_size = 2

    sampler = SimpleNamespace(n_chains=2, sweep_size=4)
    setup = SimpleNamespace(vstate=State(), sampler=sampler)
    monkeypatch.setattr(netket_module, "_NetKetResourceMonitor", Monitor)

    sampled = netket_module.NetKetPEPSVMC.sample(
        setup,
        resource_monitor=True,
        resource_interval=0.5,
    )
    output = capsys.readouterr().out

    assert sampled.diagnostics["resources"]["host_rss_peak_mib"] == 160.0
    assert sampled.diagnostics["resources"]["gpu_peak"][0]["utilization_percent"] == 90
    assert "NetKet sampling resources" in output


@pytest.mark.smoke
def test_sample_fresh_resets_the_retained_netket_cache():
    """``fresh=True`` guarantees a new batch without resetting chain state."""
    from pepsy.vmc.netket import NetKetPEPSVMC

    class State:
        sampler_state = None
        n_samples = 2
        n_discard_per_chain = 0
        chunk_size = 2

        def __init__(self):
            self._samples = np.zeros((1, 2, 3), dtype=np.int8)
            self.resets = 0

        def reset(self):
            self.resets += 1
            self._samples = None

        @property
        def samples(self):
            if self._samples is None:
                self._samples = np.full(
                    (1, 2, 3), self.resets, dtype=np.int8
                )
            return self._samples

    state = State()
    setup = SimpleNamespace(
        vstate=state,
        sampler=SimpleNamespace(n_chains=1, sweep_size=1),
    )
    sampled = NetKetPEPSVMC.sample(setup, fresh=True)

    assert state.resets == 1
    assert sampled.native is state._samples
    assert np.all(sampled.native == 1)


@pytest.mark.smoke
def test_warmup_stops_the_resource_monitor_when_jit_fails(monkeypatch):
    """Telemetry must not leave a polling thread running after a JIT error."""
    from pepsy.vmc import netket as netket_module

    monitor = SimpleNamespace(started=False, stopped=False)
    monitor.start = lambda: setattr(monitor, "started", True) or monitor
    monitor.stop = lambda: setattr(monitor, "stopped", True) or None
    monkeypatch.setattr(
        netket_module,
        "_NetKetResourceMonitor",
        lambda **kwargs: monitor,
    )

    class State:
        samples = np.zeros((1, 2, 3), dtype=np.int8)
        chunk_size = 2

        def reset(self):
            pass

        def log_value(self, configs):
            raise RuntimeError("synthetic JIT failure")

    with pytest.raises(RuntimeError, match="synthetic JIT failure"):
        netket_module.warmup_netket_vmc(
            SimpleNamespace(vstate=State(), hamiltonian=object()),
            progress=False,
            resource_monitor=True,
        )
    assert monitor.started
    assert monitor.stopped


@pytest.mark.smoke
def test_netket_mc_diagnostic_facade_forwards_and_version_gates():
    """Recent NetKet convergence helpers stay optional-dependency friendly."""
    from pepsy.vmc.netket import NetKetPEPSVMC

    class State:
        def check_mc_convergence(self, hamiltonian, **kwargs):
            return "check", hamiltonian, kwargs

        def thermalise(self, hamiltonian, **kwargs):
            return "thermalise", hamiltonian, kwargs

        def expect_to_precision(self, observable, **kwargs):
            return "precision", observable, kwargs

    setup = NetKetPEPSVMC(
        hilbert=None,
        graph=None,
        hamiltonian="energy",
        sampler=None,
        vstate=State(),
        model=None,
        ansatz=SimpleNamespace(n_sites=1, n_params=1),
        config_map=None,
        preconditioner=None,
    )
    assert setup.check_mc_convergence(min_chain_length=100) == (
        "check",
        "energy",
        {"min_chain_length": 100},
    )
    assert setup.thermalise(max_chain_length=200) == (
        "thermalise",
        "energy",
        {"max_chain_length": 200},
    )
    assert setup.expect_to_precision(rtol=0.01) == (
        "precision",
        "energy",
        {"rtol": 0.01},
    )

    missing = NetKetPEPSVMC(
        hilbert=None,
        graph=None,
        hamiltonian="energy",
        sampler=None,
        vstate=object(),
        model=None,
        ansatz=SimpleNamespace(n_sites=1, n_params=1),
        config_map=None,
        preconditioner=None,
    )
    with pytest.raises(RuntimeError, match="NetKet >= 3.22"):
        missing.check_mc_convergence()


@pytest.mark.smoke
def test_final_fermion_observables_use_conserving_netket_operators():
    """Fixed-sector final measurements avoid generic FermionOperator2nd."""
    nk = pytest.importorskip("netket")
    from pepsy.vmc.netket import (
        NetKetEtaPairObservable,
        _netket_eta_pair_operator,
        standard_fermion_observables,
    )

    hilbert = nk.hilbert.SpinOrbitalFermions(
        4,
        s=1 / 2,
        n_fermions_per_spin=(2, 2),
    )
    expected_name = "ParticleNumberAndSpinConservingFermioperator2nd"
    observables = standard_fermion_observables(hilbert)
    assert all(type(operator).__name__ == expected_name for operator in observables.values())

    ansatz = SimpleNamespace(
        orbital_sites=((0, 0), (0, 1), (1, 0), (1, 1))
    )
    eta_pair = _netket_eta_pair_operator(
        hilbert,
        ansatz,
        NetKetEtaPairObservable(1, 0),
    )
    assert type(eta_pair).__name__ == expected_name
