# Upstream `sjdu10/vmc_torch` reference

Use this document to borrow implementation patterns, not to copy the package
or override Pepsy's native conventions.

Source repository: <https://github.com/sjdu10/vmc_torch>

## Useful separation of concerns

The upstream repository separates:

- `vmc_torch/VMC.py`: high-level optimization driver, statistics, MPI
  reductions, checkpointing, and optional preconditioning;
- `vmc_torch/sampler.py`: CPU Metropolis state, burn-in, chain sampling, and
  configuration conversion;
- `vmc_torch/GPU/sampler.py`: GPU-vectorized proposal and acceptance kernels;
- `vmc_torch/GPU/VMC.py`: GPU driver that orchestrates sampler, local energy,
  gradient, SR, optimizer, and distributed synchronization;
- `vmc_torch/hamiltonian_torch.py`: Hilbert-space and operator connection
  abstractions;
- `vmc_torch/GPU/vmc_utils.py`: batched energy and gradient accumulation;
- `vmc_torch/GPU/models/pureTNS.py`: packed Quimb tensor-network parameters,
  boundary contractions, basis remapping, and optional compiled/vmapped PEPS
  amplitude paths.

Preserve this division in Pepsy. A sampler should only propose and accept
walkers. An estimator should only build connections and evaluate local
estimators. A VMC driver should coordinate the phases.

## Configuration conventions

The upstream spinful Hilbert representation commonly uses a flattened mode
configuration:

- first all spin-up modes;
- then all spin-down modes.

Its Quimb site encoding is commonly `0=empty, 1=down, 2=up, 3=double`, with
conversion to/from the flattened mode representation. This is not the current
native Pepsy `Fermion` basis. Native `Fermion` dense local operators use
`empty, up, down, double`, while the Symmray PEPS physical index is ordered
by its charge map and commonly appears as `empty, down, up, double`.

Therefore:

1. Choose the ordered Symmray PEPS physical charge map as the public PEPS
   configuration contract; for the usual `U1U1` map this is
   `empty, down, up, double`.
2. Keep the dense native `Fermion` operator basis (`empty, up, down, double`)
   separate from the PEPS physical-index order.
3. Convert upstream-style or NetKet-style flattened configurations exactly
   once at an explicit boundary.
4. Test the conversion on all four local states and on hopping transitions.
5. Never infer a permutation from the integer values alone.

## Sampler patterns

The upstream GPU sampler has several useful patterns:

- keep current configurations and amplitudes synchronized;
- propose all walkers for one graph edge at once;
- evaluate only changed walkers in eager mode;
- evaluate a fixed full batch when export/compile requires static shapes;
- accept using the squared-amplitude ratio;
- provide log-amplitude mode using `2 * (log_abs_new - log_abs_old)`;
- count accepted and effective configuration-changing moves;
- cache graph edges and same-spin mode-neighbor tables on the device.

Its spinful exchange/hopping rule can preserve both spin counts by combining:

- full-site exchange on a lattice edge;
- a same-spin particle hop along a lattice edge;
- local four-state transitions `empty <-> up/down <-> double` that preserve
  each flavor count.

For Pepsy, choose the move set from the resolved symmetry. The upstream
spinful rule is suitable for `U1U1`; for total `U1`, add spin flips so the
chain can change spin-resolved counts. For spinful `Z2`, add an
`empty <-> double` pair toggle so the chain can change particle number by two
while preserving parity.

## Amplitude patterns

The upstream PEPS/TNS path follows the Quimb packing pattern:

1. copy the tensor network so the source network is not mutated;
2. call `qtn.pack` to obtain numeric leaves and a skeleton;
3. flatten leaves into Torch parameters;
4. reconstruct with `qtn.unpack` inside the amplitude function;
5. select physical indices in the network's site order;
6. use exact or boundary contraction;
7. optionally trace/vmap/compile over configurations.

Some upstream fPEPS models clear a Symmray physical-leg line map and store a
local basis permutation. That permutation is model-specific. In Pepsy, read
the live PEPS/Symmray index map and the `Fermion` registry instead of copying a
hard-coded permutation such as `[0, 2, 3, 1]`.

For large dynamic ranges, the upstream code supports sign/phase plus log
magnitude or mantissa/exponent outputs. Preserve this idea in Pepsy, but keep
complex phase and backend/autograd behavior correct for native Symmray data.

## Local-energy and gradient patterns

The upstream VMC estimator follows the standard connected-configuration
contract:

- each operator produces connected configurations and matrix elements;
- the model evaluates amplitudes on those configurations;
- local values are matrix-element-weighted amplitude ratios;
- sample means and variances are accumulated by chain/rank;
- gradients use the covariance between local energy and log-amplitude
  derivatives;
- SR/minSR preconditions the gradient before the optimizer update.

Use this as the conceptual model for Pepsy's `TorchConnections` and local
energy functions. Keep connection construction separate from the PEPS
contraction backend so exact, boundary, and future compiled paths can share
the estimator.

## Distributed and diagnostic patterns

The upstream driver supports MPI/`torch.distributed` reductions, rank-local
sample batches, parameter synchronization, checkpointing, and diagnostics for
local-energy outliers, log-amplitude spread, gradient norms, NaNs, and Infs.

Pepsy should first provide a correct single-process/device implementation.
Add distributed execution only after sector invariants, amplitude ratios,
local estimators, and SR results are validated. Reuse the same diagnostics:
acceptance/no-op rates, log-amplitude range, local-energy variance, effective
sample count, gradient norm, SR residual, and parameter norm.
