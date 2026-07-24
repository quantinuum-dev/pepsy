---
name: pepsy-vmc
description: Design, implement, debug, or review Pepsy variational Monte Carlo workflows with PyTorch, Quimb PEPS/MPS amplitudes, Symmray fermionic tensors, Metropolis-Hastings sampling, local estimators, observables, gradients, and stochastic reconfiguration. Use for work in src/pepsy/vmc, Fermi-Hubbard VMC examples, PEPS amplitude batching, symmetry-constrained samplers, or integrations inspired by sjdu10/vmc_torch.
---

# Pepsy VMC

Use this skill for Pepsy VMC work. Keep the public workflow small and
automatic while making the internal state layout, conserved charges, fermionic
signs, amplitude contraction mode, and numerical scaling explicit.

## Read first

- Read `AGENTS.md` and nested instructions.
- Read the current VMC implementation in `src/pepsy/vmc/torch.py` and its
  focused tests in `tests/test_vmc_torch.py`.
- Read the native fermion guidance in
  `.github/skills/pepsy-fermion-operators/SKILL.md` and
  `.github/skills/pepsy-fermion-operators/references/design.md` when changing
  spinful operators, basis conventions, or Symmray signs.
- For Fermi-Hubbard workflows, inspect
  `/home/reza.haghshenas@quantinuum.com/pepsy_examples/fermi_hubbard/fh_peps.ipynb`
  and `fh_mps.ipynb`. Treat notebook source cells as authoritative over
  stale stored outputs.
- Read `references/vmc_torch.md` when using patterns from the upstream
  `sjdu10/vmc_torch` implementation.

## Core architecture

Keep these responsibilities separate:

1. **PEPS amplitude model**: owns packed Quimb/Symmray tensor parameters and
   evaluates `psi(configs)` or stable `(phase, log_abs)` values.
2. **Fermion/layout adapter**: derives PEPS site order, local basis, geometry,
   symmetry, and charge-sector metadata from the PEPS and `Fermion` object.
3. **Sampler**: proposes valid configurations, evaluates acceptance ratios,
   and retains current configurations and amplitudes in sync.
4. **Connection compiler**: turns local Fermion operators into connected
   configurations, coefficients, and source-sample indices.
5. **Estimator**: evaluates local observables and returns mean, variance,
   standard error, samples, and diagnostics.
6. **VMC driver**: orchestrates sampling, observable evaluation,
   log-derivatives, SR/minSR or optimizer updates, checkpoints, and optional
   distributed reductions.

Do not make the sampler construct Hamiltonians or make the amplitude model
decide fermionic operator signs. The public VMC entry point should normally
accept a PEPS and an explicit observable mapping; a `Fermion` is needed when
the default Hamiltonian must be constructed, but can be omitted when the PEPS
exposes symmetry metadata and the caller supplies `terms`. `Lx`, `Ly`,
`site_order`, local encoding, and sampler-rule details should be inferred
internally.

The current native Torch entry point is `TorchFermionVMC`. Treat it as the
integration seam: it should prepare validated metadata and delegate amplitude,
connection, sampling, and estimation work to the existing lower-level kernels.

The stateful native sampler is `TorchMetropolisSampler`. Its public sampling
contract follows NetKet-style chain semantics: `n_samples` is the requested
total across chains, `n_discard_per_chain` controls per-chain burn-in, and
`sweep_size` controls thinning. Accept `n_discard` and `n_thin` as convenience
aliases at user-facing boundaries. Preserve the chain axis in returned arrays
with shape `(n_samples_per_chain, n_chains, n_sites)`; flatten only when a
local-estimator kernel requires a batch matrix. For spinful fermions, the
proposal remains symmetry-aware and is not necessarily a literal one-site
flip.

When the amplitude model exposes `forward_log(configs) -> (phase, log_abs)`,
Metropolis acceptance must use the clipped log ratio
`exp(min(0, 2 * (log_abs_new - log_abs_old)))`; retain the raw amplitudes for
local-estimator compatibility and fall back to raw ratios only for models
without a log interface. For chain-shaped scalar observables, use
`torch_chain_diagnostics(...)` to report `r_hat`, integrated autocorrelation
time, and effective sample size. Do not report diagnostics from flattened
samples because the chain axis is required for between-chain variance.

For an approximate global proposal, use `PepsBpSampler` with
`TorchBPMetropolisSampler`. Quimb's public `sample_d2bp` currently samples
binary output indices, so four-state spinful PEPS legs must be represented as
two occupation-bit legs in a private dense BP copy. Preserve the original
Symmray PEPS for Torch amplitudes, gradients, local estimators, and fermionic
signs. Pass the resolved `FermionSiteEncoding` explicitly when constructing
the BP sampler, or let `TorchFermionVMC.make_bp_sampler(...)` do so.

Treat BP as a proposal `q_BP`, not as the VMC target. For independent BP
Metropolis proposals, retain `log q_BP` for every current chain and accept
`x -> y` using
`min(1, |psi(y)|**2 q_BP(x) / (|psi(x)|**2 q_BP(y)))`. Initialize chains from
BP when `q_BP` is not available for caller-supplied initial configurations.
For direct importance sampling, use self-normalized weights
`|psi(x)|**2 / q_BP(x)` and report effective sample size. Reject or discard
non-finite/zero proposal probabilities and configurations outside the target
sector; do not silently reinterpret them as valid fermion states.

The BP adapter and sector validator support spinful `U1`, `U1U1`, `Z2`, and
`Z2Z2`. `Z2Z2` moves must preserve each flavor parity independently; a move
that toggles only one empty/double site preserves total `Z2` parity but is not
valid for `Z2Z2`.

## Metadata inference and validation

Derive, in this order:

- PEPS `sites` and physical-index order;
- `Lx`, `Ly`, and lattice edges from the PEPS geometry/tags;
- physical dimension and Symmray sector/index metadata from PEPS arrays;
- spinful/spinless local space, symmetry, operator basis, and native terms from
  `Fermion`;
- the actual global charge sector from PEPS metadata or a validated initial
  configuration.

Validate PEPS and Fermion compatibility before sampling. At minimum check
site count, physical dimension, local sector map, spinful flag, and symmetry.
Never silently guess a charge sector or silently permute local basis values.
If the PEPS cannot expose its target sector, raise a concise error requesting
an explicit sector or valid initial configuration.

Keep the native Pepsy/Symmray metadata as the source of truth, but distinguish
two local orders: native `Fermion.dense_operator(...)` matrices are built from
`empty, up, down, double`, while a Symmray PEPS physical index is ordered by
its charge map and commonly exposes the physical codes
`empty, down, up, double`. Upstream `vmc_torch` examples use a different
semantic site-code order in some paths. Derive the PEPS encoding from the
physical charge map and convert exactly once; never infer it from integer
labels or dense-operator positions alone.

## Symmetry-constrained sampling

Choose the proposal rule from the resolved symmetry and sector:

| symmetry | conserved sector | valid default move |
| --- | --- | --- |
| spinless `U1` | total particle number | exchange or hop one particle |
| spinful `U1` | total `N_up + N_down` | any ergodic move preserving total number |
| spinful `U1U1` | `(N_up, N_down)` separately | same-spin hop or full-site exchange |
| spinful `Z2` | total parity | exchange/hop, spin flip, and empty/double pair toggle |
| `Z2Z2` | parity per flavor | flavor-parity-preserving custom rule |

For a spinful `U1` state, do not impose separate spin-number conservation
merely because the local space has two flavors. Separate conservation can be
an additional physical-sector choice, but `U1U1` must enforce it.

Initialize walkers inside the target sector. Every proposal must preserve the
sector, and the move set must be checked for ergodicity within that sector.
Track no-op, proposed, accepted, and effective configuration-changing moves.
For symmetric exchange/hop proposals, the acceptance ratio is simply based on
`abs(psi_new)**2 / abs(psi_old)**2`. If proposal probabilities differ in the
forward and reverse directions, include the Hastings correction.

Prefer fully vectorized Torch proposals that keep configurations, random
choices, and masks on the target device. Retain current amplitudes and update
only accepted walkers. For `torch.compile`, use fixed-shape full-batch forward
passes when variable changed-subset shapes would trigger recompilation.

## Amplitude evaluation

Use Quimb's public packing/unpacking path or the existing Pepsy wrappers:

- pack the PEPS tensor leaves and retain an immutable skeleton;
- register numeric leaves as Torch parameters without losing Symmray metadata;
- evaluate configurations in PEPS site order;
- use exact contraction for tiny reference systems and boundary/HOTRG/CTMRG
  style contraction for production;
- batch over configurations, not parameters;
- use `torch.vmap` only when the traced contraction path is pure, non-mutating,
  and compatible with the selected backend;
- keep gradients enabled only for gradient/optimization phases.

There is no PEPS analogue of a globally canonical MPS form. Do not assume
canonicalization makes PEPS amplitude evaluation cheap. The contraction mode,
bond cutoff, boundary bond, and approximation error are part of the VMC
configuration and should be reported.

Support a stable log-amplitude representation for wide dynamic ranges. A
convenient contract is `(phase, log_abs)` or `(mantissa, exponent)`. Compute
Metropolis ratios and connected-amplitude ratios from log magnitudes or
exponent differences; do not divide underflowed raw amplitudes. Preserve
complex phases for local estimators.

## Observable and Hamiltonian connections

Accept explicit term containers, including a mapping, a Fermion Hamiltonian,
or an object exposing `.terms`. Compile each supported local term into a
batched connection structure equivalent to:

- `configs`: connected configurations;
- `coeffs`: matrix elements for each connection;
- `batch_ids`: source walker for each connection.

Use output axes followed by input axes for dense one- and two-site operators,
and validate rank, local dimension, site support, and PEPS site order. Reuse
parent amplitudes for diagonal connections. Chunk connected amplitudes when
the number of walkers times nonzero operator transitions is large.

For sampled configurations `x`, evaluate

`O_loc(x) = sum_{x'} O[x, x'] * psi(x') / psi(x)`.

Average local estimators over the Markov-chain samples. The canonical sampled
driver method is `estimate_observable(...)`; `estimate_energy(...)` remains a
compatibility wrapper. Return complex means
when appropriate but report Hermitian observables using their real part only
after checking the imaginary residual. Keep energy, diagonal observables,
hopping, correlations, and arbitrary supported Fermion observables on the
same connection/estimator path.

Construct native Symmray fermion terms with Pepsy's `Fermion` operators. Do
not add Jordan-Wigner strings to the native graded path. A dense/JW reference
is acceptable only as a separately named small-system oracle. Never combine
native fermionic signs with an additional manually inserted parity sign.

## Optimization and gradients

Separate measurement from optimization:

- inference mode: Metropolis, amplitudes, connections, local estimators;
- gradient mode: differentiable amplitudes and log-derivative matrix;
- update mode: Adam/SGD or SR/minSR direction and parameter update.

For a parameterized amplitude, use log derivatives
`O_k(x) = d log(psi(x)) / d theta_k` and the standard covariance form for the
energy gradient. Center sample means consistently. For SR, expose diagonal
shift, solver choice, convergence, residual, and condition diagnostics. Use
double precision for reference and ill-conditioned SR calculations; mixed
precision requires explicit validation.

Do not detach connected amplitudes when a differentiable energy estimator is
requested. Conversely, do not retain autograd graphs during ordinary
Metropolis sampling. Reset gradients and temporary contraction/cache state at
the end of every VMC step.

## Diagnostics and validation

Always validate the smallest available system before performance work:

- direct Quimb amplitude versus Torch amplitude for every local basis state;
- native Fermion local basis and sector map;
- sampler invariants for `U1`, `U1U1`, and `Z2`;
- two-site hopping signs against the native Fermion operator and a dense/JW
  oracle kept as a reference path;
- connection local estimator versus direct dense expectation;
- zero, empty, full, single-particle, and doublon configurations;
- PEPS deterministic local expectation versus VMC mean within sampling error;
- acceptance rate, no-op rate, effective sample count, and chain mixing;
- finite, non-NaN amplitudes, local energies, gradients, and SR directions.

Report contraction approximation (`exact`, boundary, HOTRG, or other), bond
cutoff, sample count, burn-in, sweeps between measurements, acceptance, and
standard error. Keep energy and observable measurement tests separate from
optimizer tests.

## Pepsy-specific validation

Use Python 3.12 and run focused checks first:

```bash
source ~/envs/py312/bin/activate
pytest -q tests/test_vmc_torch.py
pytest -q tests/test_symmetric_tensors.py
```

For public API changes also run the public API and package-layout tests. For
notebook changes, do not modify generated outputs unless explicitly requested;
validate the relevant setup and measurement cells in a controlled run.

## Performance baseline and run planning

The native `TorchFermionVMC` path is functionally usable for the current
4x6, `D=4`, spinful `U1U1` workflow. A completed CPU run with target sector
`(N_up, N_down) = (12, 12)` produced finite amplitudes, valid-sector
diagnostics, stable SR linear solves, consistent boundary/CTMRG estimates,
and a final JSONL record with `status="ok"`. The deterministic PEPS energy
was `E/N = -0.425322`; the post-optimization boundary estimate was
`E/N = -0.424972 +/- 0.000691`, and the exact final diagnostic was
`E/N = -0.425612 +/- 0.000726`. These values are a functional baseline, not
an optimizer or ground-state convergence certificate.

The same run took about 10.5 hours on CPU. The dominant cost was repeated
local-estimator amplitude contraction, not DMRG or SR: each 2,000-sample
boundary or CTMRG measurement took roughly 2.1--2.3 hours, while the eight
SR steps took only minutes. The final exact 2,000-sample diagnostic took about
1.3 hours. Boundary and CTMRG were numerically indistinguishable at the
reported precision in this baseline, but both were still evaluated. The
connected-amplitude cache also recorded tens of thousands of fallback
contractions, which explains why sampler acceptance and chain mixing alone do
not predict runtime.

For development runs, use one measurement contraction, a smaller measurement
bond (for example `chi=32`), 256--512 diagnostic samples, and one or two VMC
steps. Omit redundant `measurement_grid` entries until the amplitude and
estimator paths are validated. Re-enable larger samples, boundary/CTMRG
cross-checks, and final diagnostics only for a production measurement. CPU
thread limits such as `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` should be
changed only in controlled benchmarks because extra threads can increase
memory use and oversubscription.

Preserve unrelated user changes in the worktree. Do not recreate Gaugy,
Tensy, or an upstream `vmc_torch` package inside Pepsy.
