# `pepsy.vmc`

Optional VMC integrations live under `pepsy.vmc`. The NetKet bridge can now
wrap a packed PEPS as a Flax model for spin systems, while keeping the
Fermi-Hubbard helper as a specialization.

## Import boundaries

Use `pepsy.vmc` for the stable backend-neutral API and the common Torch/NetKet
builders. Use `pepsy.vmc.torch` for the lower-level native Torch workflow:

- `pepsy.vmc.api`: shared terms, sampling/optimization settings, portable
  state/measurement/result records, and the `VMC` façade.
- `pepsy.vmc.netket`: NetKet builders, packed ansätze, and JAX/NetKet-specific
  controls.
- `pepsy.vmc.torch`: public Torch models, samplers, connection builders,
  local-energy estimators, drivers, and SR helpers.
- `pepsy.vmc.torch.sr`: the implementation module for log derivatives and
  stochastic reconfiguration; import its public functions from
  `pepsy.vmc` or `pepsy.vmc.torch` in application code.
- `pepsy.vmc.torch._graded`: the private optional Symmray graded projector used
  by `TorchPEPSAmplitude(graded_torch=True)`; it is not a public construction
  API.

The old flat `pepsy.vmc.torch` implementation module has been replaced by the
responsibility-based package above. Compatibility imports remain in the
private `_core` module, but new code should use the public package exports or
the responsibility-specific leaf module when inspecting implementation code.

## Recommended public workflow

Use the model-specific builders for NetKet-backed VMC:

- `build_ising_vmc(...)`
- `build_heisenberg_vmc(...)`
- `build_fermi_hubbard_vmc(...)`
- `build_fermion_vmc(...)` — general spinful-fermion models (not only
  Fermi-Hubbard), built from a `pepsy.Fermion`, explicit symbolic terms, or a
  ready NetKet operator, with optional measurement observables.

Each builder returns a `NetKetPEPSVMC` setup bundle with the NetKet Hilbert
space, Hamiltonian, sampler, variational state, packed PEPS ansatz, and optional
SR preconditioner. The bundle exposes `n_sites`, `n_params`,
`setup.expect_energy()`, and `setup.make_driver(...)` so notebooks can keep the
main path compact. Pass `progress=True` to `build_fermion_vmc(...)` to show its
eight setup phases and inspect `setup.build_timing` for the measured phase costs.
It also provides a clean class-level workflow with progress bars:

- `setup.warmup(progress=True)` — compile the sampler/amplitude/energy kernels
  up front behind a small staged bar so the optimization ETA is meaningful.
- `setup.benchmark_amplitude(n_samples=1)` — time a synchronized amplitude
  batch; `.compile_seconds` and `.amplitude_seconds_per_sample` separate
  shape-specific JAX compilation from steady-state evaluation.
- `setup.to_peps()` — reconstruct the current optimized Flax parameter tree as
  the underlying quimb/Symmray PEPS network for Pepsy-side measurements or
  persistence.
- `setup.run(n_iter, ...)` — concise wrapper that warms up and optimizes, while
  returning separate warmup, optimization, total, and per-step timings.
- `setup.optimize(n_iter, *, learning_rate=..., driver="vmc"|"vmc_sr",
  energy_shift=..., per_site=..., warmup=True, progress=True)` — run VMC with a
  single live energy progress bar and return a `VMCOptimizeResult`
  (`steps/energies/errors/variances`, `.shifted_energies`, and `.plot(...)`).
- `setup.sample(sampling=None, progress=True)` — collect a retained batch and
  report chains, burn-in, sweep size, acceptance rate, and elapsed time in
  `VMCSamples.diagnostics`.
- `setup.measure_samples(samples, observables=None)` — evaluate one operator
  or a `{name: operator}` mapping on exactly that retained NetKet batch, with
  no additional sampling. A mapping value may be
  `NetKetEtaPairObservable(...)`; it is compiled from the PEPS lattice only
  after the shared sample cache is selected. `setup.measure(...,
  samples=samples)` is the equivalent convenience form.

For fermions, the supported JIT configuration intentionally combines a flat
`Z2` Symmray PEPS ansatz with NetKet's fixed `(N_up, N_down)` `U1U1`
`SpinOrbitalFermions` sampling sector. The `Z2` label describes the
JAX-friendly tensor storage; it does not relax the sampler's separate particle
number conservation. The fermion builders emit `SymmetryFallbackWarning` to
make this distinction visible. Block-sparse `U1`, `U1U1`, and `Z2Z2` PEPS still
cannot enter NetKet's jitted `MCState` until Symmray supplies matching flat
fermionic backends.

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

### Portable Torch / NetKet VMC façade

The native Torch and NetKet entry points deliberately retain their own
performance controls. The preferred portable workflow follows NetKet's shape:
an `MCState` owns the PEPS ansatz and sampling recipe, while `VMC` owns the
Hamiltonian and drives sampling, expectations, and optimization. Both
backends return `VMCSamples`, `VMCMeasurement`, and
`VMCOptimizationResult`; each result retains the engine-specific value in
`.native`.

```python
import pepsy.vmc as pvmc

from pepsy.vmc import (
    LocalMatrixTerm,
    MCState,
    OperatorFactor,
    OperatorSum,
    ProductTerm,
    VMC,
)

hamiltonian = OperatorSum.from_terms([
    ProductTerm(
        coefficient=-1.0,
        factors=(
            OperatorFactor(0, "fermion", spin="up", dagger=True),
            OperatorFactor(1, "fermion", spin="up", dagger=False),
        ),
    ),
    LocalMatrixTerm(support=(0,), matrix=onsite_matrix),
])
state = MCState(
    peps,
    n_samples=16_384,        # total over all chains, as in NetKet
    n_chains=32,
    n_discard_per_chain=64,
    symmetry="U1U1",
)

torch_vmc = VMC(
    hamiltonian,
    state,
    backend="torch",
    fermion=fermion,
    observables={"density": density_operator},
    contraction=pvmc.ContractionConfig(method="boundary", chi=32),
)
netket_vmc = VMC(
    hamiltonian,
    state,
    backend="netket",
    fermion=fermion,
    contraction=pvmc.ContractionConfig(method="exact"),
)

samples = torch_vmc.sample()
energy = torch_vmc.expect()
history = torch_vmc.run(
    optimization=pvmc.OptimizationConfig(n_steps=100, method="sr"),
)
```

`VMCProblem` and `build_torch_vmc(...)` / `build_netket_vmc(...)` remain as
compatible lower-level construction APIs. `MCState.to_problem(...)` bridges
to that representation when integrating existing code.

The shared contracts do not import Torch, JAX, Flax, or NetKet. Each adapter
compiles the same symbolic terms into its own connection/operator
representation and preserves its native sampling and contraction strategy.
`ContractionConfig`, `SamplingConfig`, and `OptimizationConfig` use the same
validated option names. The canonical sampling spelling follows NetKet:
`n_samples` (total), `n_chains`, `n_discard_per_chain`, and `sweep_size`.
The older `n_samples_per_chain`, `burn_in`, and `thin` spellings remain
compatibility aliases; `n_samples` must be divisible by `n_chains` so the
chain-preserving result has equal chain lengths.

`compile_operator_sum_torch(...)` lowers the common terms to Torch connection
tables (using a supplied `Fermion` object for graded symbolic factors), while
`compile_operator_sum_netket(...)` lowers them to a NetKet operator. The
existing `TorchFermionVMC(..., terms=operator_sum)` and
`build_fermion_vmc(..., hamiltonian=operator_sum)` entry points use these
compilers automatically.

The shared runtime settings are consumed directly by both façades:

```python
from pepsy.vmc import OptimizationConfig, SamplingConfig

sampling = SamplingConfig(
    n_samples=16_384,
    n_chains=32,
    n_discard_per_chain=64,
    sweep_size=2,
)
torch_samples = torch_vmc.sample(sampling)
netket_samples = netket_vmc.sample()  # its sampler was built from MCState

optimization = OptimizationConfig(
    n_steps=100,
    method="sr",             # "sgd" also works on both backends
    learning_rate=2.0e-2,
    diag_shift=1.0e-2,
)
torch_history = torch_vmc.optimize(optimization)
netket_history = netket_vmc.optimize(optimization)
```

### Three sampling layers

The portable Torch VMC interface keeps three deliberately separate workflows:

1. Metropolis-Hastings measurement: call `vmc.measure()` or
   `vmc.expect()` with no supplied batch.
2. Measurement of one or more observables from a supplied batch: pass
   `samples=...`. For a weighted empirical estimate also pass `weights=...`.
   For importance sampling, prefer `proposal_log_probs=log_q`, which causes
   the current `|psi(x)|**2 / q(x)` weights to be calculated stably.
3. Autodiff energy optimization: call `vmc.optimize(...)` normally for fresh
   Metropolis samples, or pass the same supplied batch and `proposal_log_probs`
   to reuse an importance-sampling proposal over several autodiff SR or minSR
   updates.

```python
samples = torch_vmc.sample()
measurement = torch_vmc.measure(samples=samples)

# `external_configs` was drawn from q(x), whose log-density is known.
proposal_samples = pvmc.VMCSamples(
    configs=external_configs,
    proposal_log_probs=external_log_q,
)
weighted_measurement = torch_vmc.measure(
    {"density": density_operator},
    samples=proposal_samples,
)
history = torch_vmc.optimize(
    pvmc.OptimizationConfig(n_steps=20, method="sr"),
    samples=proposal_samples,
)
```

Fixed `weights=` are useful for a single weighted empirical estimate or update,
but they do not change as the ansatz changes. Retain `proposal_log_probs` for a
reused optimization batch so the importance weights are refreshed at every
parameter value. NetKet's portable adapter currently supports the first layer
through its jitted `MCState`; it raises `VMCBackendCapabilityError` rather than
silently accepting external weighted batches.

For NetKet, `n_chains`, seeds, and sampler choice are fixed at construction,
so supply them on `MCState` (or `sampling=` on the compatible legacy
builders) and then call `sample()` without a new seed. Portable NetKet
currently supports the fixed spinful `U1U1` sector;
use the native builders or `build_torch_vmc` for the other symmetry families.
`thin`, a replacement proposal, and post-construction reseeding raise
`VMCBackendCapabilityError` rather than being silently ignored. `minsr` is
currently Torch-specific; use portable `method="sr"` for NetKet.

The original `TorchFermionVMC`, `TorchVMCDriver`, and `NetKetPEPSVMC` APIs
remain available when native result objects or backend-specific controls are
required.

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
For a small float64 PEPS where raw amplitudes are safely representable, pass
`log_amplitude_fn=False` to `TorchMetropolisSampler`, `TorchVMCDriver`, or
`TorchFermionVMC` to skip the additional stable-log contractions; leave the
default enabled for larger states that can underflow.

`TorchPEPSBoundaryAmplitude` and `TorchFermionVMC` also accept
`proposal_batching="auto"`, `"vmap"`, or `"cache"`. The automatic policy
uses full native `torch.vmap` proposal batches on CUDA once enough walkers
changed; it retains the cached boundary-environment path when the selected
Symmray representation or contraction cannot be vmapped. Use `"vmap"` and
`"cache"` to benchmark the two numerically equivalent routes explicitly.
Independent configuration batches have a separate
`amplitude_batching="auto"` / `"vmap"` / `"serial"` policy on
`TorchPEPSAmplitude`, `TorchPEPSBoundaryAmplitude`, and `TorchFermionVMC`.
`"auto"` probes the pure `torch.vmap` path and falls back permanently to
serial contraction when the backend is not vmappable; use `"vmap"` for flat
Z2 PEPS and `"serial"` for native U1/U1U1 block-sparse PEPS. The last selected
route is available as `model.last_amplitude_batching` for profiling.
`TorchFermionVMC` defaults to `contraction="boundary"` with `chi=4`; pass a
larger `chi` for production accuracy.
The driver also deduplicates identical connected targets across current
walkers during `local_energies`, `estimate_observable(s)`, and `step`; this is
especially useful for serial U1/U1U1 boundary contractions.
For native U1/U1U1 CPU measurements, pass `boundary_workers > 1` to evaluate
independent cached boundary-window closures concurrently. This is restricted
to no-grad inference and defaults to `1` to avoid accidental BLAS
oversubscription; keep it at `1` when using CUDA or a shared cotengra optimizer
that is not thread-safe. The connected profile reports `num_requests`,
`num_reused`, `num_parallel`, and `num_fallback` for the current measurement.
For no-grad calls on that serial boundary route, completed amplitudes are also
cached by configuration and the current torch-parameter version, so repeated
walkers can reuse the full boundary contraction. Inspect
`model.last_amplitude_cache_stats` (or the `profile["cache"]["amplitude"]`
entry); the cache is invalidated automatically after parameter updates and is
never used for gradient-enabled or custom-parameter calls.

### Native Torch throughput probes and rank-sharded sampling

Use `benchmark_torch_amplitudes` (or
`TorchVMCDriver.benchmark_amplitudes`) with an existing configuration batch to
compare the supported amplitude batching and chunk sizes without drawing more
Markov samples. It moves the batch to the amplitude model's device, uses
`torch.no_grad()`, synchronizes CUDA timing, and verifies each candidate
against the first result by default. Boundary-amplitude cache hits are disabled
by default so the result reflects contraction throughput; pass
`include_cache=True` to time the cache-aware serving path. The returned
`TorchAmplitudeBenchmark.executed_batching` is the route actually selected,
which makes an unavailable `"vmap"` request visible as `"serial"`.

```python
timing = vmc.benchmark_amplitudes(
    samples.configs,
    chunk_sizes=(None, 32, 64),
    amplitude_batchings=("serial", "auto", "vmap"),
    repeats=5,
)
print(timing.best)
```

After the application initializes an optional `torch.distributed` process
group, `TorchVMCDriver.sample(..., distributed=True)` and the corresponding
`TorchFermionVMC.sample`, `run_measurement`, and measurement-mode `run` accept
a rank-sharded native Markov calculation. `SamplingConfig.n_chains` is the
global count and must be at least the world size; it is split deterministically
among ranks. Set `seed` or `sampler_seed` in that configuration so each rank
receives a distinct reproducible stream. Each rank retains only its local PEPS
configurations and amplitudes, while unweighted observable moments, acceptance
counts, and rank-local effective sample sizes are reduced to global estimates.

```python
# Launch this script with torchrun after selecting each rank's CUDA device.
torch.distributed.init_process_group("nccl")
samples = vmc.sample(
    sampling=pvmc.SamplingConfig(
        n_samples=4096,
        n_chains=128,  # global count
        sampler_seed=7,
    ),
    distributed=True,
)
energy = vmc.measure(samples)  # detects the distributed sample metadata
print(energy.n_samples, energy.distributed.global_n_chains)
```

`TorchMCMCSamples.distributed` and `TorchVMCEnergyEstimate.distributed` record
the local and global counts; `TorchMCMCSamples.to_common()` preserves the same
metadata in `VMCSamples.diagnostics["distributed"]`. Distributed measurement
does not all-gather histories, so global R-hat is intentionally unavailable.
It currently supports only unweighted native Markov samples—not external
proposal/importance weights or multi-rank SR optimization.

After measuring an observable, `result.chain_diagnostics` reports `r_hat`, the
integrated autocorrelation time, and an effective sample size when there are
at least two chains and two retained samples per chain. In that case,
`result.energy_stderr` is the autocorrelation-corrected
`sqrt(variance / effective_sample_size)`, while
`result.energy_stderr_naive` retains the independent-sample value and
`result.effective_sample_size` is available directly on the result. For a
single retained chain or sample, both error fields use the ordinary sample
count.

`TorchPEPSAmplitude.parameters()` returns the packed PEPS tensor leaves as
torch parameters, so plain torch optimizers and the lightweight SR/minSR
helpers can update the tensor network directly.

```python
log_derivatives = pvmc.torch_log_derivative_matrix(
    model,
    sample.configs,
    derivative_backend="auto",  # batched PEPS Jacobian with scalar fallback
)
sr = pvmc.solve_torch_sr(
    log_derivatives,
    eloc,
    method="auto",  # direct SR or sample-space minSR
    diag_shift=1e-3,
)
pvmc.apply_torch_sr_update(model, sr.direction, learning_rate=0.02)
```

`diag_shift` can instead be a callable of the SR update number. The solver
uses Cholesky for the positive-definite SR metric and falls back to a Hermitian
pseudoinverse when a small batch leaves it rank-deficient. For noisy,
small-batch updates, `sr_momentum` enables a SPRING-style update: it retains
only the component of the previous SR direction outside the current sampled
tangent span, rather than adding conventional momentum directly.

```python
def sr_shift(step):
    return max(1.0e-3, 1.0e-1 * 0.8**step)

step = vmc.step(
    sample_sweeps=1,
    sr=True,
    learning_rate=2.0e-3,
    sr_diag_shift=sr_shift,
    sr_pinv_rtol=1.0e-10,
    sr_momentum=0.8,  # optional; disabled by default
)
print(step.sr.info["solver"], step.sr.info["relative_residual"])
```

For PEPS models, `derivative_backend="auto"` uses one batched Torch Jacobian
when the amplitude model exposes its tensor parameters, and falls back to the
scalar loop for generic models. Use `"batched"` to require the optimized path
or `"loop"`/`"scalar"` for compatibility diagnostics.

Local-energy evaluation also coalesces repeated `(walker, configuration)`
connections and batches each unique off-diagonal target amplitude once before
scattering the values back to the Hamiltonian terms.

For boundary-MPS contractions, `make_torch_peps_amplitude_model(...,
contraction="boundary")` now constructs `TorchPEPSBoundaryAmplitude`. It
reuses bounded row or column environment caches for local Hamiltonian
connections and for local Metropolis proposals, rather than rebuilding a full
boundary contraction for every proposal. During a local-energy measurement,
connected configurations are grouped by parent and preferred boundary strip.
A cached parent-selected strip is copied and only its changed physical
projectors are replaced, so all terms in that strip share two-sided boundary
environments and unchanged tensor data. The cache is keyed by the parent and
target configurations and automatically invalidated when torch PEPS leaves
change. The optional VMC profile exposes `num_groups`,
`num_strip_cache_hits`, and `num_strip_builds` alongside reuse/fallback counts:

```python
model = pvmc.TorchPEPSBoundaryAmplitude(
    peps,
    chi=64,
    cutoff=1e-10,
    dtype=torch.float64,
    boundary_workers=4,  # CPU inference only; benchmark this on your machine.
)
```

For approximate PEPS, pass a cotengra optimizer in
`final_contract_opts`. It controls every remaining *exact scalar closure*:
the ordinary boundary result, cached local-energy/proposal closures, and the
final closure after CTMRG. It does not change the boundary/CTMRG truncation
set by `chi` and `cutoff`.

```python
from pepsy import build_optimizer

final_contraction_opt = build_optimizer(
    progbar=False,
    directory="ctg_cash",
    parallel=True,
)
boundary_opts = {
    "mode": "mps",
    "final_contract": True,
    "final_contract_opts": {"optimize": final_contraction_opt},
    "progbar": False,
}
model = pvmc.TorchPEPSBoundaryAmplitude(
    peps,
    chi=64,
    cutoff=1e-10,
    contraction_opts=boundary_opts,
)
```

If a reusable cotengra search cannot produce a tree for a previously unseen
closure, VMC emits one warning and retries that closure with `"auto-hq"`.

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

For a repeated VMC workflow, use the driver's phase-specific methods rather
than writing an outer sampling loop. `warmup_proposal_mix(...)` is optional
adaptive local-move warm-up, `burn_in(...)` is fixed-kernel equilibration,
`optimize(...)` performs repeated `step(...)` updates, and
`estimate_observable(...)` performs the final measurement. All accept
`progress=True` where useful; the internal optimization bar reports energy per
site, acceptance, optional no-op fraction, and the SR solver.

```python
driver.warmup_proposal_mix(n_sweeps=16, progress=True)
driver.burn_in(50, progress=True, track_proposal_stats=True)
history = driver.optimize(
    10,
    progress=True,
    sample_sweeps=1,
    sr=True,
    learning_rate=2.0e-3,
    sr_method="minsr",
    track_proposal_stats=True,
)
final_step = history[-1]
```

`run(n_steps, ...)` remains a compatibility alias for `optimize(n_steps, ...)`.

The local exchange/hopping sampler can expose move-wise diagnostics without
using a BP proposal. Pass `track_proposal_stats=True` to a sweep, `sample`, or
`step` to obtain `selected`, `no_op`, `proposed`, and `accepted` counts for
`exchange`, `hopping`, `spin_flip`, and `pair_toggle` branches. This helps
separate rejected PEPS-amplitude proposals from invalid local moves:

```python
step = driver.step(track_proposal_stats=True)
print(step.proposal_stats)
```

For a short local-sampler warm-up, `warmup_proposal_mix(...)` uses those
counters to rebalance the enabled exchange/hop, spin-flip, and pair-toggle
branches. It holds all rates fixed during each whole graph sweep, adapts only
between completed warm-up sweeps, and then leaves the final rates fixed for
normal VMC sampling. The warm-up configurations are not production samples:

```python
tuning = driver.warmup_proposal_mix(
    n_sweeps=32,
    adaptation_rate=1.0,
    min_probability=0.05,
)
print(tuning["rates"])

# The following measurement uses the frozen tuned local proposal mix.
result = driver.estimate_observable(
    burn_in=20,
    n_measurements=64,
    sweeps_between=2,
)
```

An exactly disabled branch (`rate=0`) or an always-selected branch (`rate=1`)
is treated as an explicit user choice and is not re-enabled by the tuner.

When energy and another native-term observable (for example an eta-pair
correlator) are needed from the same Markov chain, use
`estimate_observables`. It merges matching `(walker, connected configuration)`
targets before amplitude evaluation, so boundary-MPS environments are shared
as well:

```python
measurements = driver.estimate_observables(
    {
        "energy": ham.terms,
        "eta": eta_terms,
    },
    burn_in=50,
    n_measurements=50,
    sweeps_between=1,
    profile=True,
)
energy = measurements["energy"]
eta = measurements["eta"]
print(energy.energy_mean, eta.energy_mean)
print(energy.profile)  # sampling / connection / estimator timings and cache stats
```

With the modern `n_samples=...` interface, `progress=True` first reports the
Metropolis sweeps and then shows a staged evaluation bar for shared connection
building, all requested observable contractions, and final statistics.

`local_observables({"energy": ham.terms, "eta": eta_terms})` provides the
same connected-amplitude sharing for the driver's current walkers without a
sampling pass. For one observable, pass `profile=True` to
`estimate_observable` or `step` to record phase timings and the available PEPS
boundary-cache counters.

To measure samples saved from an earlier run without starting another
Metropolis chain, pass the `TorchMCMCSamples` object directly to
`measure_samples`. A chain-shaped configuration tensor with shape
`(n_samples_per_chain, n_chains, n_sites)` is accepted as well; stored
amplitudes are reused when available, or recomputed with the active batching
backend when omitted. The returned estimate retains the chain axis and
reports autocorrelation-aware error bars. Repeated parent configurations and
connected targets are deduplicated before amplitude contraction by default;
the profile includes the number of unique parent samples:

```python
saved = driver.sample(n_samples=4096, n_chains=16, n_thin=2)
energy = driver.measure_samples(saved)
observables = driver.measure_samples(
    saved,
    observables={"energy": ham.terms, "eta": eta_terms},
)
print(energy.energy_mean, energy.chain_diagnostics)
```

For repeated runs with a fixed walker shape, set `compile_kernels=True` on
`TorchVMCDriver`, `TorchFermionVMC`, or `TorchMetropolisSampler`. It only
compiles cheap tensor work: proposal construction, connection-key packing, and
the local-energy `index_add` scatter. PEPS/Symmray configuration selection and
all contractions remain eager. If the active Torch installation cannot compile
those kernels, Pepsy automatically keeps the eager implementation.

The explicit-term route is Hamiltonian-agnostic and is the preferred
measurement path when the caller already has `ham.terms`. The older
`connection_fn="fermi_hubbard"` / `connection_kwargs={...}` route remains
available for lightweight custom loops. `FermionSiteEncoding.vmc_torch()` is
the default torch spinful encoding: `0=empty, 1=down, 2=up, 3=double`.
Native graded fermion terms are compiled into sparse configuration transitions
on their first measurement and reused across subsequent VMC sweeps. The
compiled form includes the endpoint crossing phase and the parity string for
non-adjacent sites, and applies the Jordan–Wigner prefix for odd one-site
operators. Its cache is separated by term placement and `coefficient_cutoff`.
Flat Z2 Symmray physical indices are supported as well; their duplicated
sector blocks are interpreted using the canonical fermion physical ordering.
The estimate result retains the legacy `energy_mean`, `energy_variance`, and
`energy_stderr` field names; they contain statistics for the configured
observable. `estimate_energy(...)` remains as a compatibility alias.

For compatibility with an existing lower-level sweep loop, coordinate-labelled
PEPS can still initialize `TorchFermionVMC` in the constructor. New code
should prefer the first-run recipe below instead. Supply an explicit native
Hamiltonian (or its terms) because `Fermion` intentionally stores no model
couplings:

```python
from pepsy import Fermion

fermion = Fermion(spinful=True, symmetry="U1U1")
edges = ((0, 1), (1, 2), (2, 3))
terms = {edge: -t * fermion.hopping_operator() for edge in edges}
terms |= {site: fermion.onsite_term(site, U=U) for site in range(4)}
vmc = pvmc.TorchFermionVMC(
    peps,
    fermion=fermion,
    terms=terms,
    n_walkers=128,
    contraction="boundary",  # or "exact" / "ctmrg" / "hotrg"
    chi=32,  # required for an approximate contraction
    dtype=torch.complex128,
    seed=1,
)
result = vmc.estimate_observable(
    burn_in=20,
    n_measurements=8,
    sweeps_between=2,
)
```

For a native fermionic PEPS measurement, construct `TorchFermionVMC` from the
state and native Fermion terms only. The first `sample` (or `warmup`) owns both
the chain recipe and PEPS contraction recipe:

1. `SamplingConfig` owns the total retained samples, number of chains,
   per-chain discard count, sweep spacing, and RNG seeds. The canonical
   keywords are `n_samples`, `n_chains`, `n_discard_per_chain`, and
   `sweep_size`. In the native Torch sampler, the Metropolis total is
   `(n_discard_per_chain + n_samples_per_chain) * sweep_size` batched
   sweeps; every batched sweep advances all chains once.
2. `contraction_opts` is a single mapping with `method`, `chi`, `cutoff`, and
   any backend options such as `mode`. It is consumed when the first operation builds
   the amplitude model, then remains fixed with the Markov state.
3. `observables` is a name-to-term mapping. `measure(samples, observables=...)`
   uses one retained batch for every entry. Include `"energy": terms`
   explicitly when the Hamiltonian should be visible in the measurement recipe.

```python
sampling = pvmc.SamplingConfig(
    n_samples=8192,
    n_chains=32,
    n_discard_per_chain=64,
    sweep_size=2,
    seed=7,
)
contraction_opts = {
    "method": "boundary",
    "chi": 32,
    "cutoff": 1e-10,
    "mode": "mps",
}

vmc = pvmc.TorchFermionVMC(
    peps,
    fermion=fermion,
    terms=terms,            # native Fermi-Hubbard terms; no JW conversion
)

# Optional, non-MCMC warm-up: inspect one valid PEPS amplitude.
warmup = vmc.warmup(
    sampling=sampling,
    contraction_opts=contraction_opts,
)

# Phase 1: exactly one Metropolis pass. `samples` retains psi(x) for every x.
samples = vmc.sample(sampling=sampling, progress=True)

# Phase 2: no new Metropolis work. Reuse this batch for energy, eta, density, ...
estimates = vmc.measure(
    samples,
    observables={"energy": terms, "eta": eta_terms},
    progress=True,
)
print(warmup.amplitude)
print(estimates["energy"].energy_mean, estimates["eta"].energy_stderr)
```

`estimate_observables({...}, sampling=..., contraction_opts=...)` provides the
same one-batch behavior without retaining a separately named sample batch.
`run(...)` is the one-command convenience form: warm up, sample once, then
measure. `measure_samples(...)` remains the lower-level spelling of
`measure(samples, ...)`. Native sample batches carry the PEPS parameter
versions and contraction signature used to draw them, so `measure` rejects a
batch after either changes; draw fresh samples after an optimization update.
`progress=True` reports optional burn-in sweeps, MCMC
sampling, then the shared connection-building, amplitude-contraction, and
statistics phases. The `Metropolis` bar reports walkers (chains), retained
samples per walker, burn-in/thinning, proposal, contraction method/`chi`,
acceptance, and live boundary-environment cache reuse/build activity. Its
`phase` is `equilibrate` while discarded intervals run, then `retain i/n` as
each retained configuration per walker is recorded. The
`Evaluation` bar reports the shared sample shape, observables, whether parent
amplitudes were stored, connection count, and the diagonal/environment/direct
target-amplitude split. The warm-up amplitude is a representative PEPS
amplitude, not an energy estimate. The legacy constructor-level `n_walkers`,
`contraction`, `chi`, and `cutoff` options remain supported for existing
scripts, but new measurement code should keep them in `SamplingConfig` and
`contraction_opts` as above.

External MPS/BP/tree proposal sampling uses the same explicit two-stage
shape, but has no Metropolis burn-in or thinning. Pass its independent count
as `n_samples`, rather than a `SamplingConfig`:

```python
importance_samples = vmc.sample(
    proposal=mps_sampler,
    n_samples=512,
    fermion=proposal_fermion,
    one_d_to_two_d=mps_site_to_peps_coordinate,
)
importance_estimates = vmc.measure(
    importance_samples,
    observables={"energy": terms, "eta": eta_terms},
)
```

`importance_samples` stores PEPS-code configurations, `log q(x)`, and target
parent amplitudes. The later `measure` call automatically forms the
self-normalized weights `|psi(x)|**2 / q(x)` once and shares the resulting
target-amplitude work across every observable. Unlike a target-Metropolis
batch, an external-proposal batch remains valid after a PEPS update: `measure`
refreshes its target amplitudes while retaining the fixed proposal density.
`measure_from_proposal(...)` remains the one-call compatibility shortcut.

By default, `pbc=None` reads the PEPS cyclic axes through Quimb's
`is_cyclic_x()` and `is_cyclic_y()` metadata; pass `pbc=` or `edges=` to
override that inference. When explicit two-site `terms` are supplied, their
supports are added to the proposal graph, including separated and
long-range terms, while the estimator retains the exact term placements and
fermionic parity strings.

For one VMC update, call ``vmc.step(sr=True, ...)``; for repeated updates and
internal progress reporting, call ``vmc.optimize(n_steps, sr=True,
progress=True, ...)``. Boundary-MPS, CTMRG, and HOTRG use the differentiable
truncated Torch contraction path when ``chi`` is supplied. These approximate
SR updates are more sensitive to cutoff and gauge choices than exact
contraction, so monitor finite amplitudes and the SR linear-solve residual
while tuning them.

For the chain-preserving interface, call
`vmc.estimate_observable(n_samples=..., n_discard_per_chain=...,`
`sweep_size=...)`. The older `burn_in`/`n_measurements`/
`sweeps_between` form remains available. For spinful fermions, the selected
proposal is symmetry-aware rather than a literal single-site flip: `U1U1`
preserves both flavor counts, `U1` preserves total particle number, and `Z2`
preserves parity.

With the older sweep-based interface, `target_effective_sample_size` makes
`n_measurements` a maximum rather than a fixed count. The driver checks both
ESS and (by default) `R-hat <= 1.05`; with several observables it waits until
all of them meet the target. `auto_thin=True` uses the current integrated
autocorrelation-time estimate as the spacing for later measurements:

```python
result = vmc.estimate_observable(
    burn_in=50,
    n_measurements=200,  # safety cap
    sweeps_between=1,
    target_effective_sample_size=128,
    min_measurements=8,
    rhat_threshold=1.05,
    auto_thin=True,
)
print(result.n_measurements, result.effective_sample_size, result.energy_stderr)
```

This adaptive stopping path intentionally does not change the explicit
chain-preserving `n_samples=...` sampler interface, whose sample count remains
fixed and reproducible.

### Measuring from an external sampler

`TorchVMCDriver.measure_from_proposal(...)` accepts a sampled MPS batch, a
`MpsSampler`, a PEPS-BP proposal, or a tree-sampler batch. Fermionic MPS
occupations are reordered by their coordinate map and encoded using the PEPS
metadata before the usual weighted `measure_samples(...)` path runs:

```python
from pepsy.sampling import MpsSampler

mps_sampler = MpsSampler(
    mps,
    one_d_to_two_d=one_d_to_two_d,
    fermion=fermion,
    backend="auto",
)
estimate = vmc.measure_from_proposal(
    mps_sampler,
    n_samples=512,
    seed=7,
)
print(estimate.energy_mean, estimate.effective_sample_size)
```

For a `TorchFermionVMC`, `measure_from_mps(...)` is the equivalent
fermion-specific convenience method. A bare MPS requires both
`one_d_to_two_d=` and `fermion=`; sampled batches carry their own mapping.
Tree/qubit batches require an explicit `occupation_map=` so binary qubit
columns are not silently interpreted as fermion physical codes. Nodes with
zero or non-finite proposal probability or PEPS amplitude are removed before
local-energy evaluation.

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

### General fermion models and observables

`build_fermion_vmc(...)` lifts the Fermi-Hubbard specialization to any spinful
fermion model. Pass a `pepsy.Fermion` together with an explicit native
Hamiltonian/term mapping, an explicit list of symbolic terms, or a ready
NetKet operator. `Fermion` owns only local symmetry and backend conventions;
the hopping `t`, on-site `U`, density `V`, and chemical potential `mu` live in
the explicit term arrays. Optional observables are stored on the setup and
evaluated with `setup.measure(...)`.

Like `TorchFermionVMC`, it can also infer the lattice geometry directly from a
native Pepsy Hamiltonian. Pass the `SymHamiltonian` from
`fermion.hamiltonian(...)` as `hamiltonian=`, or its coordinate-keyed
`.terms` mapping as `terms=`, together with `fermion=`; the builder reads the
integer edges and periodic axes (`pbc`) from those terms and compiles the
supplied native local operators to NetKet. Explicit `edges` / `graph` / `pbc`
still take precedence.

`SamplingConfig(n_samples=..., n_chains=..., n_discard_per_chain=...,
sweep_size=..., chunk_size=...)` controls retained samples, chains, per-chain
discard, sweep spacing, and batching. NetKet's Metropolis sampler also performs
`sweep_size` proposals
between retained samples; pass `sampler_sweep_size=...` to make that cost
explicit (the NetKet default is the Hilbert-space size).

The generic fermion builder defaults to `conserving=False`, which constructs
the exact ordinary NetKet fermion operator without a first-use conversion
compile. Set `conserving="auto"` when the specialized conserving operator is
worth the one-time conversion cost for a long production run.

For a compact construction call, put these numerical settings in
`NetKetVMCConfig(...)` and pass it as `config=`. Boundary/PEPS contraction
settings belong there because changing `chi` changes the compiled amplitude
model; retained sample counts and burn-in can instead be overridden at
`setup.sample(...)` time.

```python
import pepsy as py
import pepsy.vmc as pvmc

fermion = py.Fermion(spinful=True, symmetry="U1U1")
terms = {edge: -1.0 * fermion.hopping_operator() for edge in edges}
terms |= {site: fermion.onsite_term(site, U=8.0) for site in sites}
terms |= {edge: 0.5 * fermion.density_operator() for edge in edges}
ham = fermion.hamiltonian(terms)     # authoritative native Hamiltonian

# Clean API: infer edges + PBC from the native terms, build the NetKet model.
setup = pvmc.build_fermion_vmc(
    peps, fermion=fermion, terms=ham.terms, Lx=4, Ly=4,
    contraction="ctmrg", chi=16,
)
```

```python
fermion = py.Fermion(spinful=True, symmetry="U1U1")
terms = {edge: -1.0 * fermion.hopping_operator() for edge in edges}
terms |= {site: fermion.onsite_term(site, U=8.0) for site in sites}
ham = fermion.hamiltonian(terms)
setup = pvmc.build_fermion_vmc(
    peps,
    fermion=fermion,
    hamiltonian=ham,
    Lx=4,
    Ly=4,
    contraction="ctmrg",
    chi=16,
    observables={
        "double_occupancy": pvmc.standard_fermion_observables(
            # any SpinOrbitalFermions Hilbert space with the same orbitals
        )["double_occupancy"],
    },
)

result = setup.optimize(
    2000,
    driver="vmc_sr",
    learning_rate=0.01,
    energy_shift=-8.0 / 4,          # arXiv:2511.02125 E/N - U/4 convention
    per_site=setup.n_sites,
)
result.plot(per_site=setup.n_sites)

# Sample once, then measure each operator on those exact Markov chains.
samples = setup.sample()
stats = setup.measure_samples(
    samples,
    {
        "energy": setup.hamiltonian,
        # The coordinate lattice is inferred from setup.ansatz.orbital_sites.
        "eta_pair": pvmc.NetKetEtaPairObservable(
            1,
            0,
            periodic=True,
            staggered=True,
        ),
        **pvmc.standard_fermion_observables(setup.hilbert),
    },
)  # {name: nk.stats.Stats}
```

Two lower-level helpers back this path:

- `netket_fermion_operator(hilbert, terms, *, constant=0.0, conserving="auto")`
  builds any jittable NetKet fermion operator from symbolic
  `(coefficient, ops)` terms, where each op is `(site, sz, dagger)` with
  `sz` in `{+1, -1, None}`. It doubles as a Hamiltonian or an observable, and
  can optionally convert to NetKet's particle-number/spin-conserving operator
  for cheaper local energies.
- `fermion_model_terms(fermion, edges, *, t, U, V=0.0, mu=0.0, n_sites=None)`
  returns symbolic uniform Fermi-Hubbard terms from explicit couplings. For
  non-uniform models, pass the authoritative native term mapping directly to
  `build_fermion_vmc(...)`.
- `standard_fermion_observables(hilbert)` returns common observables
  (`n_up`, `n_down`, `n_total`, `double_occupancy`) as a ready
  `{name: operator}` mapping for `setup.measure(...)`.
- `NetKetEtaPairObservable(dx, dy, *, periodic=True, staggered=False)` is a
  declarative eta-pair correlator for `measure_samples(...)`. Its zero-offset
  form is the mean double occupancy; nonzero offsets measure
  `Delta_i^dag Delta_j + h.c.` and use either periodic or in-bounds pairs.
the PEPS once, keep `flat=True` Symmray data for JIT-friendly leaves, and
evaluate many spin-orbital occupation rows with one compiled function. In the
currently tested Symmray stack this flat path is the `Z2` route; sparse
fermionic `U1`, `Z2Z2`, and `U1U1` PEPS remain supported by the non-jitted
batch amplitude function and by `TorchPEPSAmplitude`, but should use those
sparse-block paths unless a matching flat backend is available and benchmarked.
For native sparse `U1U1` PEPS, `TorchPEPSAmplitude(...,
graded_torch=True, contraction="exact")` enables a separate fixed-shape Torch
projector. It densifies only the numerical leaves, compiles the U1U1 charge
transitions and Symmray Koszul phase masks once, and then contracts the
projected leaves under `torch.vmap`; it does not use the unsupported
`U1U1FermionicArrayFlat` conversion or a naive `to_dense()` contraction.
This opt-in path is exact and differentiable, but currently supports only
native sparse U1U1 and exact contraction; approximate boundary contractions
continue to use the established sparse path.
The same option is forwarded by `TorchFermionVMC` when its contraction is
exact.
For `U1`, `U1U1`, and `Z2Z2`, `fermionic_peps_rand(...)` builds the
block-sparse ansatz and `make_fermionic_peps_batched_amplitude_function(...,
jit=False)` is validated with `contraction="exact"`, `"hotrg"`, `"ctmrg"`,
and `"boundary"`/`"mps"`. Full NetKet `MCState` VMC still requires a jitted
Flax/JAX model, so `build_fermi_hubbard_vmc(...)` raises clearly for these
block-sparse PEPS until Symmray provides a flat backend for the requested
symmetry.
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
    "U1",  # or "U1U1" for separately fixed spin sectors
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
Quimb options. These options are method-specific: `sequence`,
`max_separation`, and `canonize` are boundary-MPS controls, while CTMRG's
stable default is `mode="projector"`.

For flat fermionic Symmray tensors under JAX, `max_separation=0` is accepted
but has a guarded compatibility fallback to `1` if Quimb reaches its upstream
empty-boundary-axis or block-matmul path. This applies to both boundary-MPS
and CTMRG. The requested `sequence`, `chi`, and `canonize` settings remain
active; dense/non-Symmray contractions are not modified.

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

### Torch MCMC convergence check

After a native Torch run, `TorchFermionVMC.check_mc_convergence(...)` runs a
separate, non-mutating diagnostic sampler from the current walker positions.
It retains one local-observable value after every raw Metropolis sweep, then
reports ordinary and split R-hat, average and maximum integrated
autocorrelation time, effective sample size, acceptance, and a suggested
production `sweep_size`:

```python
report = vmc.check_mc_convergence(
    observables={"energy": hamiltonian},
    min_chain_length=100,
    max_chain_length=500,
    target_effective_samples_per_chain=50,
    rhat_threshold=1.05,
    progress=True,
)

print(report.reliable, report.recommended_sweep_size)
print(report.energy.split_r_hat, report.energy.tau_max)
```

The live progress display is sampling acceptance only. The check uses a
cloned random stream, leaves the active walker configurations and RNG state
unchanged, and is intentionally separate from fixed-size production sampling.


> API details are maintained as handwritten Markdown in this page.
