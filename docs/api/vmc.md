# `pepsy.vmc`

Optional VMC integrations live under `pepsy.vmc`. The NetKet bridge can now
wrap a packed PEPS as a Flax model for spin systems, while keeping the
Fermi-Hubbard helper as a specialization.

## Recommended public workflow

Use the model-specific builders for NetKet-backed VMC:

- `build_ising_vmc(...)`
- `build_heisenberg_vmc(...)`
- `build_fermi_hubbard_vmc(...)`

Each builder returns a `NetKetPEPSVMC` setup bundle with the NetKet Hilbert
space, Hamiltonian, sampler, variational state, packed PEPS ansatz, and optional
SR preconditioner. The bundle exposes `n_sites`, `n_params`,
`setup.expect_energy()`, and `setup.make_driver(...)` so notebooks can keep the
main path compact:

```python
import pepsy.vmc as pvmc

setup = pvmc.build_ising_vmc(
    peps,
    Lx=2,
    Ly=2,
    h=1.0,
    J=1.0,
    contraction="exact",
    n_samples=1024,
    n_chains=16,
    use_sr=False,
)

print(setup.n_sites, setup.n_params)
energy = setup.expect_energy()
driver = setup.make_driver(learning_rate=0.02)
```

Lower-level functions such as `pack_peps_ansatz(...)`,
`make_peps_batched_amplitude_function(...)`, and `make_netket_vmc_driver(...)`
remain public for custom flows and profiling.

## Torch sampling and local-energy kernels

`pepsy.vmc.torch` provides lightweight PyTorch kernels for custom VMC loops and
neuralized-PEPS experiments. They are intentionally lower-level than the
NetKet bridge: pass a batch amplitude function, choose a sampler proposal, and
accumulate local energies from connected configurations.

```python
import torch
import pepsy.vmc as pvmc

graph = pvmc.TorchSquareLattice(2, 2)
encoding = pvmc.FermionSiteEncoding.symmray()
configs = pvmc.random_spinful_configs(
    n_walkers=128,
    n_sites=graph.n_sites,
    n_up=2,
    n_down=2,
    encoding=encoding,
)

# For pure PEPS VMC, pack a quimb PEPS as torch trainable parameters.
model = pvmc.TorchPEPSAmplitude(
    peps,
    contraction="exact",  # or "boundary" / "ctmrg" / "hotrg" with chi=...
    dtype=torch.float64,
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

sample = pvmc.metropolis_exchange_sweep(
    configs,
    model,
    graph,
    proposal="spinful",
    hopping_rate=0.25,
    encoding=encoding,
)

# NetKet-like stateful sampling. ``n_samples`` is total across chains.
sampler = pvmc.TorchMetropolisSampler(
    model,
    graph,
    configs,
    proposal="spinful",
    n_chains=128,  # normally inferred from configs
    seed=2,
)
samples = sampler.sample(
    n_samples=1024,
    n_discard_per_chain=500,
    sweep_size=graph.n_sites,
)
print(samples.configs.shape)  # (chain_length, n_chains, n_sites)

# ``n_discard`` and ``n_thin`` are convenience aliases.
samples = pvmc.metropolis_local_sampler(
    configs,
    model,
    graph,
    n_samples=1024,
    n_discard=500,
    n_thin=graph.n_sites,
    seed=2,
)
conn = pvmc.spinful_fermi_hubbard_connections(
    sample.configs,
    graph,
    t=1.0,
    U=8.0,
    encoding=encoding,
)
eloc = pvmc.local_energy_from_connections(
    sample.configs,
    sample.amplitudes,
    conn,
    model,
)
```

If the amplitude model exposes `forward_log(configs)` and returns
`(phase, log_abs)`, Metropolis acceptance uses
`exp(min(0, 2 * (log_abs_new - log_abs_old)))` instead of dividing raw
amplitudes. Models without that method retain the raw-amplitude fallback.
After measuring an observable, `result.chain_diagnostics` reports `r_hat`, the
integrated autocorrelation time, and an effective sample size when there are
at least two chains and two retained samples per chain.

`TorchPEPSAmplitude.parameters()` returns the packed PEPS tensor leaves as
torch parameters, so plain torch optimizers and the lightweight SR/minSR
helpers can update the tensor network directly.

```python
log_derivatives = pvmc.torch_log_derivative_matrix(model, sample.configs)
sr = pvmc.solve_torch_sr(
    log_derivatives,
    eloc,
    method="auto",  # direct SR or sample-space minSR
    diag_shift=1e-3,
)
pvmc.apply_torch_sr_update(model, sr.direction, learning_rate=0.02)
```

For boundary-MPS contractions, `TorchPEPSBoundaryAmplitude` also reuses each
parent walker's row or column environments when evaluating local Hamiltonian
connections:

```python
model = pvmc.TorchPEPSBoundaryAmplitude(
    peps,
    chi=64,
    cutoff=1e-10,
    dtype=torch.float64,
)
```

For a compact no-JIT torch loop, `TorchVMCDriver` keeps walker configurations,
current amplitudes, Hamiltonian connection metadata, sampling, local-energy
evaluation, and optional SR updates together. Local-energy evaluation reuses
diagonal connected amplitudes, supports chunked off-diagonal amplitude calls,
and uses `connected_amplitudes(...)` automatically when the model provides it.
For native Pepsy Hamiltonians, pass the explicit term mapping and let the
driver build all connected configurations directly from those one- and
two-site operators:

```python
ham = fermion.hamiltonian(terms)
driver = pvmc.TorchVMCDriver(
    model,
    graph,
    configs,
    terms=ham.terms,
    site_order=tuple(peps.sites),
    proposal="spinful",
    hopping_rate=0.25,
    chunk_size=64,
)

result = driver.estimate_observable(
    burn_in=20,
    n_measurements=8,
    sweeps_between=2,
    progress=True,
)
print(
    result.energy_mean,
    result.energy_stderr,
    result.samples_per_second,
)
```

The explicit-term route is Hamiltonian-agnostic and is the preferred
measurement path when the caller already has `ham.terms`. The older
`connection_fn="fermi_hubbard"` / `connection_kwargs={...}` route remains
available for lightweight custom loops. `FermionSiteEncoding.vmc_torch()` is
the default torch spinful encoding: `0=empty, 1=down, 2=up, 3=double`.
The estimate result retains the legacy `energy_mean`, `energy_variance`, and
`energy_stderr` field names; they contain statistics for the configured
observable. `estimate_energy(...)` remains as a compatibility alias.

For a coordinate-labelled PEPS, use `TorchFermionVMC` to derive the lattice,
physical charge ordering, initial sector, and sampler rule in one place. Pass
`fermion` to generate the default Hamiltonian, or omit it when supplying
explicit `terms`:

```python
from pepsy import Fermion

fermion = Fermion(spinful=True, symmetry="U1U1", t=t, U=U)
vmc = pvmc.TorchFermionVMC(
    peps,
    fermion,
    n_walkers=128,
    contraction="exact",  # or "boundary" / "ctmrg" / "hotrg"
    chi=None,  # set this when using an approximate contraction
    dtype=torch.complex128,
    seed=1,
)
result = vmc.estimate_observable(
    burn_in=20,
    n_measurements=8,
    sweeps_between=2,
)
```

For the chain-preserving interface, call
`vmc.estimate_observable(n_samples=..., n_discard_per_chain=...,`
`sweep_size=...)`. The older `burn_in`/`n_measurements`/
`sweeps_between` form remains available. For spinful fermions, the selected
proposal is symmetry-aware rather than a literal single-site flip: `U1U1`
preserves both flavor counts, `U1` preserves total particle number, and `Z2`
preserves parity.

`U1U1` uses moves that preserve `(N_up, N_down)`. For spinful `U1`, the
default proposal also includes single-site spin flips, so only
`N_up + N_down` is fixed. Spinful `Z2` adds empty/double pair toggles, so it
preserves parity without freezing total particle number. Pass an explicit
`terms={site_or_edge: operator}` mapping to measure a custom observable; site
labels may be PEPS labels or positional integers. If a dense PEPS does not
expose its charge sector, pass `sector=...` or valid `configs=...` explicitly.
The adapter refuses to apply an unverified local basis permutation.

For an importance proposal from Pepsy's dense 2-norm BP sampler, use BP only
to propose configurations and let the torch PEPS model measure amplitudes and
local energies:

```python
from pepsy.sampling import PepsBpSampler

bp_sampler = PepsBpSampler(native_peps)
importance = driver.importance_energy_estimate(
    bp_sampler,
    n_samples=512,
    sample_kwargs={"method": "mps", "chi": 32, "cutoff": 1e-10},
    progress=True,
)
print(importance.energy_mean, importance.effective_sample_size)
```

The importance result uses self-normalized weights
`|psi(x)|**2 / q_BP(x)` and reports the effective sample size, so a small BP
proposal overlap is visible instead of being mistaken for a precise VMC
estimate.

For large nonlocal BP moves with an exact Metropolis correction, use the BP
sampler as an independence proposal. `TorchFermionVMC` infers the PEPS
encoding, symmetry, and sector automatically:

```python
bp_mcmc = vmc.make_bp_sampler(
    n_chains=64,
    bp_sampler_kwargs={"max_iterations": 100},
    sample_kwargs={"method": "mps", "chi": 32, "cutoff": 1e-10},
    seed=3,
)
samples = bp_mcmc.sample(
    n_samples=1024,
    n_discard=100,
    n_thin=4,
)
```

Each proposal uses

`min(1, |psi(y)|**2 q_BP(x) / (|psi(x)|**2 q_BP(y)))`.

The BP adapter supports four-state spinful fermion PEPS with `U1`, `U1U1`,
`Z2`, and `Z2Z2` physical symmetries. Since Quimb's current D2BP interface
samples binary output legs, a four-state physical leg is represented as two
occupation bits in a private dense BP copy. The original Symmray PEPS remains
block-sparse and is still used for Torch amplitudes and local energies. BP
proposals outside a fixed Fermion sector are rejected before they can enter a
chain.

The torch PEPS wrapper is validated for dense quimb PEPS and Symmray
block-sparse fermionic PEPS. Symmray tensors are packed through their own
`to_pytree` / `from_pytree` metadata via `quimb.tensor.pack`, then the numeric
block leaves are registered as torch parameters. This preserves fermionic
phases and charge sectors for sparse `Z2`, `U1`, `Z2Z2`, and `U1U1` PEPS. The
flat Symmray backend is currently a Symmray-side feature for `Z2`; Pepsy also
supports that path and it can be combined with `torch.vmap` in the style of
Symmray's batch-GPU fermionic-amplitude example. Do not assume the same flat
`vmap` batching behavior for `U1`, `Z2Z2`, or `U1U1`: those symmetries are
supported through sparse block contractions, so their batching/performance
profile should be benchmarked separately.

Torch VMC uses the physical-index order
`0=empty, 1=down, 2=up, 3=double`, matching the native four-sector fermionic
PEPS and the `sjdu10/vmc_torch` convention. Use
`FermionSiteEncoding.symmray()` only when interoperating with a caller that
explicitly uses its alternate legacy labels.

Available connected-config helpers include
`spinful_fermi_hubbard_connections`, `heisenberg_connections`, and
`transverse_ising_connections`. The Hubbard helper uses the site-major
fermionic mode order `down, up` by default, matching the Symmray-oriented
convention used by the reference implementation.

For spin models, NetKet spin-1/2 configurations use local values `+1` and
`-1`. The default `NetKetLocalConfigMap` sends `+1 -> physical index 0` and
`-1 -> physical index 1`; pass a custom map when a symmetric PEPS uses another
physical-sector order.

```python
import quimb.tensor as qtn
import pepsy.vmc as pvmc

pvmc.configure_jax_for_vmc()

peps = qtn.PEPS.rand(Lx=2, Ly=2, bond_dim=2, phys_dim=2, dtype="complex128")
setup = pvmc.build_ising_vmc(
    peps,
    Lx=2,
    Ly=2,
    h=1.0,
    J=1.0,
    contraction="exact",
    n_samples=1024,
    n_chains=16,
    chunk_size=256,
    seed=1,
    sampler_seed=2,
    use_sr=False,
)
driver = setup.make_driver(learning_rate=0.02)
```

Heisenberg VMC uses the same generic PEPS amplitude model with NetKet's
exchange sampler and, by default, the `total_sz=0` sector:

```python
setup = pvmc.build_heisenberg_vmc(
    peps,
    Lx=2,
    Ly=2,
    J=1.0,
    total_sz=0.0,
    contraction="exact",
    n_samples=1024,
    n_chains=16,
    chunk_size=256,
    use_sr=False,
)
```

For notebook profiling or custom NetKet models, use the generic batched JAX
path directly:

```python
ansatz = pvmc.pack_peps_ansatz(peps, lattice_shape=(2, 2))
batched_amp = pvmc.make_peps_batched_amplitude_function(
    ansatz,
    pvmc.NetKetLocalConfigMap.spin_half(),
    contraction="hotrg",
    chi=4,
    output="mantissa_exponent",
)
mantissa, exponent = batched_amp(spin_config_rows)
```

The Fermi-Hubbard helper builds the spin-orbital Hilbert space, fermionic
Hamiltonian, fermion-hop sampler, and Symmray spinful local-index map:

```python
import pepsy.vmc as pvmc

pvmc.configure_jax_for_vmc()
ansatz = pvmc.pack_fermionic_peps_ansatz(peps, lattice_shape=(2, 2))
settings = pvmc.recommend_netket_vmc_settings(
    n_params=ansatz.n_params,
    n_samples=4096,
    n_chains=32,
)
setup = pvmc.build_fermi_hubbard_vmc(
    peps,
    Lx=2,
    Ly=2,
    t=1.0,
    U=8.0,
    contraction="exact",
    n_samples=settings.n_samples,
    n_chains=settings.n_chains,
    chunk_size=settings.chunks.chunk_size,
    sampler_chunk_size=settings.chunks.sampler_chunk_size,
    seed=1,
    sampler_seed=2,
)
driver = setup.make_driver(
    driver=settings.driver,
    learning_rate=0.02,
    sr_mode=settings.sr_mode,
    use_ntk=settings.use_ntk,
    on_the_fly=settings.on_the_fly,
    chunk_size_bwd=settings.chunks.chunk_size_bwd,
)
```

The fermionic batch path mirrors Symmray's batch GPU amplitude example: pack
the PEPS once, keep `flat=True` Symmray data for JIT-friendly leaves, and
evaluate many spin-orbital occupation rows with one compiled function. In the
currently tested Symmray stack this flat path is the `Z2` route; sparse
fermionic `U1`, `Z2Z2`, and `U1U1` PEPS remain supported by the non-jitted
batch amplitude function and by `TorchPEPSAmplitude`, but should use those
sparse-block paths unless a matching flat backend is available and benchmarked.
For `U1U1`, `fermionic_peps_rand("U1U1", ...)` builds the block-sparse ansatz
and `make_fermionic_peps_batched_amplitude_function(..., jit=False)` is
validated with `contraction="exact"`, `"hotrg"`, `"ctmrg"`, and
`"boundary"`/`"mps"`. Full NetKet `MCState` VMC still requires a jitted Flax
model, so `build_fermi_hubbard_vmc(...)` raises clearly for block-sparse
`U1U1` PEPS until Symmray provides a flat U1U1 fermionic backend.
For an actual fixed-sector sparse-block VMC loop, use
`build_sparse_fermi_hubbard_vmc(...)`: it builds the NetKet
`SpinOrbitalFermions` Hilbert space and Fermi-Hubbard operator metadata, then
uses `TorchPEPSAmplitude`, `metropolis_exchange_sweep`, and
`spinful_fermi_hubbard_connections` for the non-jitted PEPS amplitudes and
local energies. The builder starts walkers on non-zero PEPS support; for tiny
sectors it can enumerate the NetKet Hilbert space, while larger runs can pass
`initial_configs=...` directly.

```python
peps = pvmc.fermionic_peps_rand(
    "U1U1",
    Lx=2,
    Ly=2,
    bond_dim=3,
    n_fermions_per_spin=(2, 2),
)
setup = pvmc.build_sparse_fermi_hubbard_vmc(
    peps,
    Lx=2,
    Ly=2,
    t=1.0,
    U=8.0,
    contraction="boundary",  # or "exact", "hotrg", "ctmrg"
    chi=4,
    n_samples=128,
    seed=1,
    sampler_seed=2,
)
eloc = setup.local_energy()
sample = setup.sample_sweep()
energy = setup.energy_estimate()
```

```python
batched_amp = pvmc.make_fermionic_peps_batched_amplitude_function(
    ansatz,
    setup.columns,
    contraction="hotrg",
    chi=4,
    output="mantissa_exponent",
)
mantissa, exponent = batched_amp(occupation_rows)
```

The approximate contraction choices are Quimb's finite 2D contractions:
`contraction="hotrg"` calls `contract_hotrg`, `contraction="ctmrg"` calls
`contract_ctmrg`, and `contraction="boundary"` or `"mps"` calls
`contract_boundary(mode="mps")`. Pass `contraction_opts={...}` for lower-level
Quimb options such as `sequence`, `max_separation`, `canonize`, or a non-default
boundary `mode`.

For larger lattices use an approximate contraction with `chi=...`, and treat
`chi`, `chunk_size`, `chunk_size_bwd`, `n_samples`, and the SR setting as
performance/accuracy knobs. The dense sign checks from the notebook should
remain capped to tiny systems. For large parameter counts prefer
`driver="vmc_sr"` with `sr_mode="real"` so NetKet can use its SR/minSR driver at
lower fermionic Jacobian cost, and use `make_netket_autochunk_callback(...)` on
the first GPU run to tune sampler, forward, and backward chunks after
compilation pressure is known.

```python
callback = pvmc.make_netket_autochunk_callback(
    sampler_chunk_size=settings.chunks.sampler_chunk_size,
    chunk_size=settings.chunks.chunk_size,
    chunk_size_bwd=settings.chunks.chunk_size_bwd,
)
driver.run(n_iter=100, out="vmc_run", callback=callback)
```

```{eval-rst}
.. automodule:: pepsy.vmc.netket
   :members:
   :undoc-members:

.. automodule:: pepsy.vmc.torch
   :members:
   :undoc-members:
```
