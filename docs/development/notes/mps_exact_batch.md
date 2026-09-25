# Bounded exact gate batching

Audit date: 2026-09-24. Scope: opt-in `MpsOptimizer(mode="exact-batch")`.
The existing default and the reference `_run_exact` gate kernel are unchanged.

## Implementation and invariants

`mps/_exact_batch.py` owns a sequential block compiler and backend-native
application. It assembles only small operators: dense blocks cover at most four
qubits (16 x 16), diagonal blocks at most twelve (4096 entries). Consecutive
single-qubit gates participate automatically. Switching between diagonal and
dense gates flushes a block; controls and stochastic events delimit segments
before compilation. No commuting-gate reordering is attempted.

Dense application uses Autoray `tensordot` and records its output ordering in
tensor indices. Only the small fused operator is reordered explicitly. Diagonal
application broadcasts a small factor in the current index order. Output arrays
remain out-of-place to preserve aliases and Torch autograd; already-contracted
networks receive only a shallow metadata copy on replay. Backend contraction
packing and logical-order readout can still allocate full-state buffers.

Gate classification tests structural zeros exactly, never with a tolerance.
Only tiny matrices are transferred to the host. Trainable Torch matrices bypass
diagonal specialization so zero off-diagonal entries retain their derivatives.
Torch lazy conjugate/negative bits are resolved only on the tiny gate. Plans are
rebuilt on every replay to avoid stale mutable payloads or autograd graphs.

Ordinary NumPy/Torch/CuPy qubit arrays opt in by capability, not dependency
version. Native symmetry, fermionic states, qudits, and other backends retain
the reference kernel; nothing is silently coerced to dense NumPy. Operator
scale, tags, gate-event accounting, and physical site order are preserved.
Exact-mode restrictions also apply to canonical normalization, layout pilots,
persistent layouts, and Gibbs replay. Controls retain the existing MPS
rebuild boundary, with backend and custom site-index preservation in the new
mode. The noise module excludes both exact modes from canonical-center-only
Kraus shortcuts.

Fusion changes floating-point association, not the intended operator or
truncation policy. It does not make statevector memory subexponential. No
thread count, device policy, SVD/QR registration, or running simulation is changed.

## Installed environment and API probes

Shared Python 3.12 environment:

| Dependency | Installed version |
| --- | --- |
| Pepsy | 0.4.1 |
| Quimb | 1.15.1.dev66+ge927f06e1 |
| Autoray | 0.11.1.dev3+g1b476b305 |
| Cotengra | 0.8.3.dev7+g1d7fd333f |
| Cotengrust | 0.2.1 |
| Symmray | 0.4.1.dev7+g83fb22865 |
| Torch | 2.6.0+cu124 |
| NumPy | 2.5.2 |

Inspected the installed callables, not just release versions:

- `Tensor.modify(self, **kwargs)`.
- `TensorNetwork.copy(self, virtual=False, deep=False)`: shallow array sharing
  with independent tensor metadata, used only with replacement output arrays.
- `TensorNetwork.contract(self, tags=Ellipsis, output_inds=None, optimize=None,
  get=None, max_bond=None, strip_exponent=False, preserve_tensor=False,
  backend=None, inplace=False, **kwargs)`.
- `MatrixProductState.from_dense(psi, dims=2, tags=None, site_ind_id='k{}',
  site_tag_id='I{}', **split_opts)`: accepts native Torch arrays for rebuilds.
- `tensor_network_gate_inds(self, G, inds, contract=False, dagger=False,
  transpose=False, tags=None, info=None, inplace=False, **compress_opts)`:
  retained as the unchanged reference route.
- `infer_backend_converter_from_sample(sample_data,
  *, cast_complex_to_real=False)`: creates the tiny fusion identity with the
  gate's backend/dtype/device.
- Autoray dispatch resolves `tensordot`, `diagonal`, `reshape`, and
  `transpose` to NumPy or CuPy functions on those backends. Torch uses
  `torch.functional.tensordot`, `torch.diagonal`, `torch.reshape`, and
  Autoray's `torch_transpose`. Torch `resolve_conj/resolve_neg` were exercised
  on CPU and CUDA.

### Upstream review and disposition

Reviewed the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/),
[Cotengra changelog](https://cotengra.readthedocs.io/en/latest/changelog.html),
and [Symmray repository](https://github.com/jcmgray/symmray).
The required [Symmray Abelian-array page](https://symmray.readthedocs.io/en/latest/abelian_arrays.html)
was attempted repeatedly but unavailable through the web tool; installed
Symmray behavior and repository documentation were inspected instead.

- **Adopt:** existing public Quimb tensor metadata/copy/contraction interfaces
  and Autoray's native dense operations, behind the new opt-in mode.
- **Defer:** upstream compression/cutoff changes, randomized SVD routes, and
  native Symmray fusion. These do not justify changing exact replay defaults.
- **Defer:** JAX fusion until a separate tracing/autodiff-aware capability path
  is designed. Current JAX execution uses the reference fallback.
- **Defer:** integration of compiled kernels, cross-replay plan caching,
  adaptive block sizes, and in-place state mutation until the focused
  prototypes below are evaluated. Caching mutable gates or retaining
  autograd graphs would require a stronger ownership contract.
- No compatibility shim or installed-package edits are needed for batching.

On broad regression failures, rechecked the installed Symmray randomized-SVD
route: `svd_rand_truncated(x, *args, **kwargs)` dispatches into a split that
rejects cumulative cutoffs with `max_bond_mode="eager"`. A representative
DMRG/native-SRC failure reproduces with the original HEAD optimizer loaded
in memory. Fixing that independent compatibility issue is deferred.
The inherited trajectory environment route also emits Quimb's deprecation
warning for `method="envs"` versus `route`; that API migration is deferred.

## Validation

The focused `tests/test_mps_exact_batch.py` checks mixed one-/two-qubit
streams, reversed endpoints, repeated scaled replay, input ownership, dense
reuse, diagonal block bounds, tiny off-diagonal elements, mutable gates,
Torch gradients, CPU/CUDA device and dtype, custom physical indices,
measurement/reset/cap/feed-forward boundaries, coalesced Kraus sampling,
mode transitions, cyclic input, and qudit/native-Symmray fallback. Additional
cases cover persistent-layout/Gibbs rejection and optional CuPy execution.

Integration run of exact-batch, dynamic controls, trajectory noise, sampler,
Gibbs, Quimb compatibility, contraction dependencies, public API, and package
layout: 339 passed, one JAX precision failure. That failure reproduces with
original HEAD optimizer/noise modules loaded through an in-memory import hook.
The adjacent MPS metadata, audit, and layout suites gave 59 passed and two
failures in mixed-mode quality repair and layered-FIT tag-cache expectations;
these do not execute the exact-batch route. The layered-FIT expectation failure
also reproduces on HEAD.

The broader MPS/native run was interrupted during the costly 3x4 Hubbard grid:
383 passed, 83 failures, with native randomized-SVD incompatibility dominating.
A completed MPS run excluding test names containing `native` and the known
bond-cache test gave 482 passed, 10 failed, 171 deselected. All ten remaining
failures are spinful Symmray DMRG variants with the same eager/cumulative-cutoff
exception. No attempt was made to modify those independent algorithms or
claim the whole repository suite is green.

The original exact-mode transition and cyclic-boundary tests passed (six
selected cases). `python -m ruff check src tests` and `git diff --check` passed.

## Performance spot check

Xeon W-3375 CPU, Torch complex64, four Torch/BLAS threads, 5x5 open lattice
with snake site mapping. Starting at the all-zero product state, replay RX(0.05)
on every site, RZZ(-0.1) on the twenty x-edges then the twenty y-edges, then
RX(0.05) on every site. Each step has 90 gates compiled into 20 blocks.
Timing covers `run()`, including compilation, but not readout/measurements.

| Successive depth | Reference exact (s) | exact-batch (s) | Relative state L2 difference |
| --- | --- | --- | --- |
| 1 | 11.538 | 1.711 | 8.37e-7 |
| 2 | 10.411 | 1.254 | 1.48e-6 |
| 3 | 11.007 | 1.319 | 2.27e-6 |

This is approximately 6.7–8.3x faster in this small spot check, not a guarantee
for the 32-thread 5x6 production sweep. Timings were collected on a shared
machine while that existing simulation continued unchanged. Different gate
orders change block counts; measuring full measurement/readout workloads,
peak memory, and 32-thread performance remains future benchmarking work.

## CUDA follow-up investigation (2026-09-24)

The available GPU is an NVIDIA RTX A5000 with 24 GB of memory. The shared
environment has CuPy, Torch CUDA, and Triton, but no cuQuantum installation.
All measurements below used temporary in-memory CuPy RawKernel prototypes,
complex64, one GPU, and 23 qubits (64 MiB state). These are exploratory
measurements, not a production 5x6 benchmark.

| 23-qubit operation | Current exact-batch path | Temporary CUDA prototype |
| --- | ---: | ---: |
| One RX at a selected bit | CuPy tensordot: 0.97–1.13 ms | In-place pair kernel: 0.19 ms |
| One RZZ | CuPy diagonal broadcast: 0.19 ms | In-place elementwise kernel: 0.19 ms |
| 23 RX then 22 adjacent RZZ | Eight fused blocks: 2.52 ms | 45 separate in-place kernels: 8.55 ms |
| 22 adjacent, equal-angle RZZ | Two diagonal blocks: 0.414 ms | One phase/count kernel: 0.200 ms |

The full-layer prototypes agreed with exact-batch within relative state-vector
errors of 3.5e-7 for separate kernels and 3.2e-7 for the one-pass RZZ kernel.
Replacing every fused block with separate small in-place kernels would regress
this workload. The one-pass RZZ kernel uses the identity that a commuting
equal-angle RZZ layer gives each basis amplitude one phase determined by the
number of disagreeing edge bits. The 23-site chain used a bitwise population
count. A grid implementation needs a correct edge map and a capability gate
for contiguous, eligible RZZ layers.

After warm-up, rebuilding the same eight-block plan took 2.79 ms median,
versus 2.52 ms to apply the blocks. Cold compilation included first-use GPU
setup and was slower. A reusable plan could help repeated static streams, but
cached matrix values would go stale for mutable gates or retain Torch autograd
graphs. Cache only an explicit immutable descriptor or structural plan, and
keep trainable and mutable payloads on the existing rebuild path.

## Pauli and parity structure on CPU and GPU

For each P in {X, Y, Z}, a two-qubit rotation has the form

    R_PP(theta) = cos(theta/2) I - i sin(theta/2) (P tensor P).

At generic angles its operator Schmidt rank is two, although the 4x4
unitary itself has full matrix rank. Splitting it into two operator terms is useful for planning, but
forming two full state vectors would violate the memory goal. X and Y
rotations instead pair each basis amplitude with the amplitude whose two
target bits are flipped; Z is a diagonal phase. An in-place pair update needs
constant scratch per worker and no state-sized intermediate. Products on
different edges can increase operator Schmidt rank, so there is no general
rank-two representation for a long mixed layer.

Consecutive RXX, RYY, and RZZ on the same pair all preserve even and odd
two-qubit parity. Their product is exactly two independent 2x2 matrices on
basis sectors (00, 11) and (01, 10), regardless of rotation angles. For
angles alpha, beta, gamma in the RXX/RYY/RZZ convention, the even sector is
exp(-i gamma/2) times an X rotation by alpha-beta; the odd sector is
exp(+i gamma/2) times an X rotation by alpha+beta. This was checked against
Pepsy's gate matrices to 1.1e-16. More generally, any exactly
parity-preserving 4x4 matrix can use the same two-block kernel without
inferring angles or requiring unitarity. Do not discard tiny cross-parity
entries; a raw matrix qualifies only when they are structurally zero.

A temporary Numba CPU prototype used four threads and complex64 on a
20-qubit vector. One RXX took 1.09 ms versus 2.16 ms for the current NumPy
dense contraction; one RYY took 1.03 ms versus 2.40 ms. Applying consecutive
RXX, RYY, and RZZ on one distant pair took 0.70 ms in one parity pass versus
3.05 ms in the current two exact-batch blocks, with relative state error
6.5e-8. JIT compilation and input cloning were excluded from both timed
paths. By contrast, a 19-gate RXX chain took 14.87 ms in seven current fused
blocks and 21.09 ms in 19 separate in-place pair passes, with relative error
2.2e-7. The planner must choose from measured state passes and arithmetic
cost, not apply the pair kernel indiscriminately. These are bounded examples,
not a CPU production benchmark.

For a contiguous same-axis layer on many edges, all PP rotations commute.
RXX can be conjugated to diagonal ZZ rotations by Hadamards, and RYY by the
corresponding Y-to-Z local basis change. Fuse those basis changes into bounded
local transforms, apply one graph phase pass, and undo them only when the
following stream requires another basis. A planner should select this route
only when its estimated state passes beat direct parity-pair or ordinary
four-qubit fusion. Mixed-axis gates on edges sharing one qubit generally do
not commute; preserve their stream order. The earlier RZZ chain prototype is
the zero-basis-change case.

Revised prototype order:

1. **Prototype:** one shared structural planner for exact rotations and
   parity-preserving same-pair blocks, with bounded CPU and CUDA kernels.
   Compare out-of-place and owned in-place execution, define copy-on-write
   for aliased states and shot branches, and keep arbitrary dense gates on
   the existing path.
2. **Prototype:** costed commuting-layer application for RZZ, RXX, and RYY.
   Start with equal-angle graph layers, then per-edge angles; test complete
   5x5 and 5x6 sweeps and peak memory.
3. **Prototype:** compare a capability-gated cuStateVec Pauli-rotation path and
   its queued Ex updater in a separate environment. cuQuantum is not installed
   here. Classic apply_matrix_batched batches state vectors, not consecutive
   gates on one vector.
4. **Prototype:** reuse batch structure for explicitly immutable streams and
   avoid host inspection when a symbolic descriptor proves the gate class.
   Mutable arrays and trainable Torch gates keep the existing rebuild path.
5. **Defer:** CUDA Graph replay and direct GPU measurement/collapse until
   fixed-buffer lifetimes and dynamic control handling are specified. An
   isolated owned buffer can be used for kernel prototypes without changing
   public ownership behavior.

At 30 qubits, a complex64 vector alone occupies 8 GiB. One ideal read and
write is 16 GiB of traffic, giving a roughly 22 ms bandwidth floor at the
A5000's advertised 768 GB/s; real gates can require more transfers. This
makes full-state passes and peak live buffers central metrics. It is a
theoretical floor, not a prediction of the 5x6 replay time.

Sources: [NVIDIA A5000 specifications](https://www.nvidia.com/content/dam/en-zz/Solutions/products/workstations/nvidia-rtx-a5000-datasheet.pdf),
[NVIDIA cuStateVec Python bindings](https://docs.nvidia.com/cuda/cuquantum/latest/python/bindings/custatevec.html),
[NVIDIA Pauli-rotation API](https://docs.nvidia.com/cuda/cuquantum/latest/python/bindings/generated/cuquantum.bindings.custatevecEx.apply_pauli_rotation.html),
[NVIDIA cuStateVec Ex updater](https://docs.nvidia.com/cuda/cuquantum/26.03.1/custatevec/overview/ex-svupdater.html),
[CuPy custom kernels](https://docs.cupy.dev/en/stable/reference/kernel.html),
and [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html).


## Structured exact replay follow-up (2026-09-24)

The opt-in batch planner now checks fixed 4x4 matrices for exact structural
zeros. Large consecutive ZZ-like diagonal runs with identical even/odd
values use one grouped-XOR population-count phase pass. Equal values are
required: the varied-angle prototype was slower than existing compact
diagonal blocks. Consecutive parity-preserving gates on one unordered pair
compose as two 2x2 sectors, including reversed endpoint order, and use one
pair pass. A tiny cross-parity entry prevents specialization; repeated
edges are kept out of one grouped mask so their multiplicity is retained.
These kernels are out-of-place and keep peak live state storage at the existing input plus
one output; masks, coefficients, and phase tables are small. NumPy uses
optional Numba with four local worker chunks at most; CuPy uses RawKernel.
Torch, missing Numba, incompatible dtypes, noncontiguous data, and
unsupported layouts retain the original fusion. No global CPU thread
setting or input-array mutation is introduced.

Installed versions remain those recorded above. API probes for this task
confirmed Quimb's gate, contract, copy, and tensor-modify signatures;
Autoray's NumPy/Torch/CuPy dispatch for the existing operations; optional
Numba 0.67.0; and CuPy 14.1.1 RawKernel. The Quimb, Autoray, Cotengra,
and Symmray upstream sources listed above were checked again. The Symmray
Abelian-array web page still returned an internal error. Classification:
adopt optional Numba/CuPy local kernels for supported dense qubits;
defer native Symmray and Torch structured kernels, varied-angle phase
fusion, cross-replay plan caches, and in-place mutation. No compatibility
shim is needed. The affected paths are the exact-batch block iterator and
the optimizer's exact-batch dispatch; the original exact route is unchanged.

Focused optimizer-level tests cover NumPy and CuPy, reversed endpoints,
nonunitary parity scale, equal-value RZZ replay, varied-value fallback,
and tiny cross-parity entries. The earlier exact-batch Torch gradient and
backend tests remain part of the same suite. Warm application spot checks
on a shared host (complex64, 20 qubits) showed an equal-angle 19-edge
RZZ chain at 1.75 ms with the grouped pass versus 4.10 ms with two
existing diagonal blocks; a three-gate same-pair RXX/RYY/RZZ block took
2.12 ms versus 14.66 ms in this measurement. On an RTX A5000 (23 qubits),
the grouped RZZ pass took 0.32 ms versus 0.41 ms, and same-pair parity
took 0.26 ms versus 1.02 ms. These timings exclude plan creation and
first-use JIT compilation; scheduling noise was visible across repeats.
Numba compilation adds first-use latency. Repeated complete 5x5
RX-RZZ-RX sweeps measured 1.54 s versus 1.14 s on CPU and 24.0 ms
versus 20.5 ms on GPU (old/new median replay, complex64). On a
complete 5x6 CPU sweep in separate fresh processes, old/new took
124.08/97.84 s with the same 32.56 GiB peak RSS. The first amplitude
agreed within complex64 roundoff; complete-state agreement was checked
on 5x5. The 5x6 RSS is larger than two 8-GiB state arrays because the
other dense contractions also allocate full-state temporaries. A 5x6
GPU benchmark was not run on the 24-GiB A5000 given that peak-memory
risk. These measurements are single shared-machine runs, not a
multi-seed or device-portable speed claim.


Final validation for this follow-up: the focused exact-batch suite passed
(22 tests), including NumPy/CuPy complex64 and exploratory complex128
kernel comparisons. The adjacent controls, Quimb compatibility, public
API, and package-layout run had 112 passing tests and one inherited
JAX complex64 Kraus-probability precision failure in direct mode,
previously reproduced against the original optimizer. Repository Ruff
and git diff whitespace checks passed.

## Alignment with Pepsy develop (2026-09-24)

Rebased the exact-batch commit onto origin/develop at `28d9b6c`. The newer
Pepsy MPS rebuild path preserves backend, device, physical index format, and
site tags for both exact modes. The batch mode now calls that shared
`_ensure_mps_state(self)` boundary; it does not carry a separate backend
conversion switch. Its dense replay and exact fallback still use the
installed Quimb and Autoray callables and dispatch listed above. The local
environment versions and inspected signatures were unchanged after the
rebase. Quimb, Autoray, Cotengra, and Symmray upstream sources were rechecked;
the Symmray Abelian-array documentation page remained unavailable.

**Adopt:** the upstream Pepsy shared backend-preserving reconstruction.
**Defer:** additional compression and native Symmray changes, which do not
change this opt-in dense gate route. Post-rebase exact-batch tests passed (22),
as did public API/package-layout tests (58). A wider MPS/control/API run had
128 passes and one JAX complex64 Kraus-probability precision failure in direct
mode; that exact failure reproduced on an isolated archive of remote
`28d9b6c`. The remote exact-reconstruction backend selection had nine passes
and three JAX complex64 precision failures; the direct case reproduced on the
same isolated baseline. The [session handoff](../../../history/2026-09-24-mps-exact-batch-rebase.md)
records the checks and remaining scope.
