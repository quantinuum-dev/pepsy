# PauliMPO DMT: implementation path and feasibility evidence

Use native Pauli cores, an explicit protected bond compressor, and a local
gate driver; generic FIT or ordinary Quimb compression does not supply the
preservation constraints.

**Status:** implemented on 2026-11-13 after the user's follow-up authorization.
The API, local kernel and persistent driver are in the working tree; no commit
or transport-convergence claim. The design refinements and completed checks are
recorded below and in the [handoff](../../../history/2026-11-13-pauli-mpo-dmt-assessment.md).

## Choose the physical problem explicitly

| Target | Compression policy | Reference |
| --- | --- | --- |
| Density-matrix evolution with nonzero trace | Preserve local partial traces and truncate connected correlations | [White et al. 2018, III](https://link.aps.org/accepted/10.1103/PhysRevB.97.035127) |
| Infinite-temperature correlations via traceless operator evolution | Preserve local partial traces and truncate the unprotected raw block, without dividing by the trace | [Yi-Thomas et al. 2024, Appendix D](https://arxiv.org/pdf/2310.06886) |

The second policy is already published; it does not require inventing a
zero-trace regularization. Do not silently replace a small trace by an epsilon
or select a different algorithm using a numerical trace threshold. For
infinite-temperature correlations, evolving the traceless observable directly
avoids carrying an identity background or a small perturbation parameter.

The [2026 lattice-DMT end matter](https://arxiv.org/pdf/2610.40776) also explains
the protected-subspace formulation. Geometry-dependent constraints are a
separate extension from this 1D task. The
[ITensor implementation](https://github.com/Jack-Kemp/dmt/blob/master/DMT.h)
provides a useful independent algorithm reference, not a new runtime dependency.

## Installed capabilities and upstream review

Checked this checkout with Quimb 1.15.0, Autoray 0.8.4, Cotengra
0.8.3.dev8+g8954240f2, Cotengrust 0.2.1, Symmray
0.4.1.dev15+g0374aaa3c, Torch 2.9.1 and JAX 0.8.2.

| Component | Finding | Decision |
| --- | --- | --- |
| [PauliMPO](../../../src/pepsy/operators/pauli_mpo.py) | Native `(left, right, 4)` cores, exact algebra/readouts, QR and ordinary SVD rounding already exist. Core-backed gates embed a gate MPO and multiply chains. | **Adopt** storage and helpers; add a separate protected policy and local gate path. |
| Quimb dimension-four MPS | Public constructor, canonicalization and `gate_split` work for Pauli coefficient vectors. A normal split has no physical preservation information. Installed 1D registry has no DMT. | **Adopt** public canonicalization/interop for references; retain explicit environment-aware compression in Pepsy. |
| Quimb `method="dm"`, FIT, SRC | These optimize ordinary overlap/norm objectives and do not enforce DMT's partial-trace constraints. | **Defer** as substitutes. Constrained FIT could be later research, with a separately specified objective. |
| Autoray | Existing dispatch reaches backend-native QR, SVD, matrix products and contractions. | **Adopt**; preserve dtype/device and existing `TorchLinalgConfig`. |
| Cotengra/Cotengrust | Useful for general contraction paths; this 1D hot path consists of fixed two-site contractions and trace-vector recurrences. | **Defer** new path optimization in the local kernel. |
| Symmray | Native PauliMPO does not yet provide the required symmetry representation. | **Defer**; do not densify symmetry data or imply support. |

Reviewed official [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[1D compression guide](https://quimb.readthedocs.io/en/latest/tensor/tensor-1d-compress.html),
[Autoray source](https://github.com/jcmgray/autoray),
[Cotengra docs](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html),
[Symmray array docs](https://symmray.readthedocs.io/en/latest/abelian_arrays.html)
and [source](https://github.com/jcmgray/symmray). Installed source and signatures
were inspected rather than assuming current online APIs match this environment.
Online Quimb 1.16 changes decomposition defaults and dependencies; no upgrade
is required for this design and none was performed. The first attempted
compression-guide URL was unavailable; the linked guide was reached through
the official changelog navigation.

Quimb's split-driver registry receives a matrix and split options, while the
DMT caller must also supply bond-specific physical constraints. Registering
a globally named SVD driver alone cannot discover that context. Installed
TEBD calls ordinary `gate_split_` during its sweeps; using it for DMT requires
an explicit integration point, not merely a compressor-name change.

## Protected matrix kernel

In canonical operator bases let `M` be the active bond matrix and let the
columns of `Q_L`, `Q_R` span the protected left/right directions. With
orthogonal projectors `P_L`, `P_R`, the raw-operator policy is

$$
{\color{blue}R}=(I-P_L)M(I-P_R),\qquad
{\color{purple}M'}=M-R+\operatorname{SVD}_r(R).
$$

It follows that `Q_L† (M'-M) = 0` and `(M'-M) Q_R = 0`. For the density
policy, form the connected matrix `C = M-B`, where

$$
{\color{blue}B}=\frac{(M b)(a^\dagger M)}{a^\dagger M b},\qquad
{\color{purple}R}=(I-P_L)C(I-P_R).
$$

Here `a,b` are the identity trace directions, included in their protected
spaces; the denominator must be nonzero. Use the same reconstruction
`M'=M-R+SVD_r(R)`. This preserves the protected components exactly in exact
arithmetic; it is not a penalty objective.

For one protected site on each side of a cut, at most four directions per
side are needed. Preserving the overlapping partial operators on sites
`1..j+1` and `j..L` preserves every three-contiguous-site marginal at that
split. The safe rank bound is `p + q + r`, with `p,q <= 4`, hence the usual
eight-direction reservation. Boundary and dependent directions can cost less.
A cap below the conservative reservation does not itself prove mathematical
infeasibility; preserve an already fitting operator or report the unsupported
budget instead of silently violating constraints.

Implementation improvements worth adopting:

- Use thin protected bases and products with `Q`, avoiding explicit complete
  basis rotations or dense projector matrices.
- Reconstruct the fixed part and truncated residual as thin factors. QR only
  the side away from the new canonical centre and absorb the triangular
  factor into the centre. No final SVD is needed. Use zero cutoff in support
  factorizations; do not apply the user's residual cutoff to protected factors.
- Keep identity contractions scaled, reuse them along sweeps, and recompute
  only environments invalidated by local writes or gauge moves.
- Reproject residual factors into the complementary subspaces to limit
  floating-point constraint leakage. Keep renormalization disabled.
- Use real Pauli coefficients for Hermitian operators where the representation
  permits it; unitary conjugation has a real Pauli transfer matrix.

The implementation uses a compact bond/support basis to match the paper's
Schmidt-space prescription, without preliminary support trimming.
A direct two-site
protected split can avoid its preparatory SVD, but can use additional zero
Schmidt directions and therefore need not reproduce the same approximation.
The probe below verifies preservation, not equivalence of these variants.

## Local gate integration

For a two-qubit gate `U`, build the 16-by-16 Pauli transfer matrix locally:

$$
{\color{blue}S}_{ab}=\tfrac14\operatorname{Tr}
\left(P_a U P_b U^\dagger\right).
$$

For Heisenberg evolution exchange `U` and `U†`. This acts on two physical
dimension-four legs, not on a physical qubit MPS. Merge only the acted-on
cores, apply `S`, then use protected splitting. Avoid the current embedded
gate-MPO product path and avoid ordinary truncation before DMT. A later
constrained step cannot recover information already discarded.

Keep one sweep's canonical centre and trace environments live. Repeated full
chain canonicalization per gate would turn a linear-length sweep into an
unnecessary quadratic-length setup. A fixed-dimension full-SVD design has
the usual cubic bond-dimension scaling, with additional constant work. The
larger-system review below records a limited optimization benchmark.

## Implemented interfaces and remaining extensions

1. [operators/_pauli_dmt.py](../../../src/pepsy/operators/_pauli_dmt.py) owns
   protected splitting, thin-factor reconstruction and explicit reports.
   `PauliMPO.compress_dmt` retains existing ordinary compression semantics.
   It supports dense full SVD, open chains, three-site protection and explicit
   `max_bond`.
2. `PauliMPO.apply_gate_dmt` shares that kernel and forms a local two-site
   target. [optimizers/pauli_dmt.py](../../../src/pepsy/optimizers/pauli_dmt.py)
   owns `PauliDMT`, its second-order evolution schedule, persistent canonical
   and trace environments, and bounded diagnostic history.
3. Arbitrary protected strings, randomized residual SVD, larger preservation
   radii, native symmetry and geometry-dependent constraints remain deferred.

Report protected ranks, residual rank, output rank, unprotected discarded
weight and constraint residual separately. The existing ordinary compression
report's norm proxy is not an observable-preservation guarantee. Preserving
trace and marginals does not guarantee global positivity, exact future local
dynamics, or energy conservation by a finite-step integrator.

## Pre-implementation feasibility measurements

These temporary-script measurements preceded the compact-support production
implementation. They establish the algebraic approach, not its final runtime.

A temporary deterministic probe at `/tmp/pepsy_dmt_probe.py` constructs a
positive six-spin density matrix (seed 20261113, minimum eigenvalue 0.0130054)
and compresses all internal bonds to at most 12. It uses repeated full-chain
QR preparation as a slow reference; it is not the proposed scalable driver.

| Compression | Trace change | Normalized energy change | Largest three-site marginal error |
| --- | ---: | ---: | ---: |
| Ordinary SVD | 3.78e-6 | 5.81e-4 | 2.17e-3 |
| Protected raw block | 6.66e-16 | 5.55e-17 | 9.78e-16 |
| Protected connected block | 9.99e-16 | 1.39e-16 | 9.91e-16 |

Both protected results have bonds `(4,12,12,12,4)`. Their global Frobenius
errors, about 0.00634, exceed ordinary SVD's 0.00506; these results establish
the protected quantities, not a universal accuracy improvement.

- Raw protection also preserved traceless-operator partial traces to 2.42e-16.
- Complex matrix-kernel constraints held to 1.01e-14 or better on NumPy,
  Torch and JAX CPU with double precision. Compact factor reconstruction
  agreed with the full formula to 6.62e-14 or better.
- Local untruncated Pauli gate application agreed with a dense Hilbert-space
  reference to 1.36e-16. A following protected split changed a three-site
  marginal by at most 2.87e-16.
- A public Quimb dimension-four MPS gate agreed with the native coefficient
  vector to 1.89e-17. Identity and rank-deficient raw cases were also checked.
- An initial dense readout of the existing global-product gate reference was
  interrupted after large intermediate allocation; the completed gate probe
  instead uses a direct small dense oracle. No timed performance comparison
  is claimed.

The final temporary script completed with preservation, rank, backend and
reference assertions. No production tests or full suite were needed for the
unchanged source. No long-time evolution, transport, gradients, JIT, GPU,
single-precision or native symmetry claims follow from these probes.

Acceptance checks for implementation: actual truncation with every three-site
marginal preserved at each split; zero/identity/product and dependent
constraint cases; density and traceless variants; trace/energy diagnostics;
dense evolution and independent timestep/bond convergence; NumPy/Torch/JAX
dtype/device preservation; unchanged ordinary compression; no Pauli-word
expansion or full Hilbert-space matrices in the production path.

## Implementation validation and numerical limits

- The initial implementation's focused Pauli/DMT/damping/public-API/package selection produced
  **226 passed, 1 skipped, 1 failed**. The failure is the existing installed
  Pepsy 0.5.0 metadata versus the checkout's 0.6.0 declaration. All **44**
  tests in [test_pauli_dmt.py](../../../tests/test_pauli_dmt.py) passed.
- Coverage includes independent complete-basis block references for both
  variants, actual truncation, all three-site marginals, trace and local
  energy, both gate pictures, second-order timestep convergence, finite-step
  truncated evolution, real/complex NumPy/Torch/JAX CPU data, native EIGH gate
  preparation, a nonzero Torch fixed-rank gradient versus finite differences,
  cache invalidation in both directions, explicit invalid-input failures,
  no word/global-matrix expansion, and bounded existing-bond SVD sizes.
- Before the schedule refinement below, the documented eight-site infinite-temperature example completed 20 steps
  at `dt=0.05`, with maximum reported constraint residual 1.39e-15 and final
  autocorrelation 0.6154199701. This is an execution check, not transport
  convergence evidence.
- Before the larger-system review below, an 18-site NumPy microbenchmark with input bonds 24 and output cap 12
  compared the optimized existing-bond path with the same workspace forced
  through a merged pair/identity transfer. Seven-run medians were 6.12 ms
  versus 8.59 ms, about 1.4x faster in this one case. A stable QR-centre norm
  measured relative output difference 5.10e-14. This is not a universal
  optimality or scaling claim; the current physical sweep check adds work,
  so these timings do not measure the revised implementation.
- Protected-basis rank revelation accounts for the Schmidt support condition
  number. Before accepting a numerically dependent direction, the kernel
  checks the original physical partial-trace constraints, retrying with the
  tighter rank tolerance if necessary. A separate regression keeps a small
  but physically resolved constraint. The larger-system review removed
  support cutoffs because small channels can carry resolved local data.
- Complex-gauge equivalence is checked on resolved generic density operators.
  A nearly singular sparse source can still select different global
  approximations under gauge/roundoff changes, even though its local partial
  traces remain preserved. The regression retains that source and checks the
  promised local data, without assuming a unique global approximation at an
  unresolved rank boundary. Real Pauli cores avoid introducing complex gauge
  freedom for Hermitian operators when the input representation permits them.
- Native Torch QR warns about rank-deficient derivatives in the scoped
  stabilized-gradient test. This is the existing explicit policy; fixed-rank
  finite-difference agreement does not establish derivatives at rank changes.
- Ruff over `src tests`, whitespace and local Markdown targets pass. Smoke
  has 111 passed and the same version-metadata failure. The full-suite attempt
  stops on the pre-existing BP backend-dispatch failure described in the
  handoff; no full-suite success is claimed.

GPU, end-to-end JIT, native Symmray, periodic chains and transport-scale
convergence have not been validated. No installed dependency was modified.

## Larger-system and optimization review

The follow-up review found a real long-chain failure: absolute coefficient
errors could be tiny while physical local density errors were large.
Normalized density coefficients decay exponentially with chain length.
Early support rounding lost physically resolved channels, and ordinary
norms underflowed at small global scales. The revised kernel retains support,
rescales centres and norms, and exposes relative panel diagnostics. A linear
standalone-sweep check uses log-scaled identity environments to compare every
three-site marginal before committing the result. Traceless, nearly zero
partial traces use a normwise local split guard; their raw relative residual
is not a meaningful accuracy certificate.

The new [large-chain tests](../../../tests/test_pauli_dmt_large.py) use PSD
mixtures of 24 product states with analytic three-site marginals, avoiding
global Hilbert-space matrices. They cover both DMT variants and NumPy
single/double precision through 128 sites, double precision through 512 sites, and rank
48 to cap 24 compression at 64 sites. Stronger mixtures at 128 sites and cap
12 are explicitly rejected when numerical preservation fails; cap 24 recovers
the source in standalone compression. Extreme global scales from `1e-180` to `1e180` (double) and
`1e-25` to `1e25` (single) are checked separately.
The revised 512-site result at cap 12 has maximum three-site Frobenius error
`1.61e-13` and trace error `1.84e-13` against the analytic mixture.

Evolution checks include a 128-site XX-chain infinite-temperature correlation
against the independent one-particle matrix exponential, timestep convergence,
a 256-site commuting chain against an exact three-spin oracle, and truncated
32-site noncommuting density evolution with trace and energy-convergence
checks. None establishes long-time transport convergence. Hamiltonian
Hermiticity validation now rescales its input to avoid overflow, and mixed
core dtypes fail clearly.

The [optimization tests](../../../tests/test_pauli_dmt_optimizations.py) cover:

- One directional final QR: only the side away from the centre must be
  isometric. Removing the second QR and final central SVD preserves the
  product without truncating protected factors.
- Local reuse of the unchanged initial Schmidt decomposition on a stricter
  protected-basis retry, and one pass for relative/normwise constraint checks.
- Persistent canonical-centre and identity-environment metadata, explicit
  isometry checks after jumps, and bounded centre travel. Parity layers now
  turn at their ends; subsequent timesteps start near the current centre.
  Finite-bond compression is nonlinear, so this schedule is explicit API
  behavior. Splitting evolution across calls preserves the schedule.
- Per-call EIGH and transfer reuse for shared Hamiltonian objects; preparation
  is repeated after parameter changes or on a fresh autograd call. A nonzero
  Torch Hamiltonian-parameter gradient matches finite differences. Native
  rank-deficient QR still warns under the existing explicit stabilized policy.

A warmed NumPy benchmark uses the same 64-site, rank-24, Bloch-strength-0.2
mixture at cap 16, one BLAS thread, and seven timing repetitions. Evolution
uses three steps at `dt=0.025`. This compares the immediately preceding
implementation, including its physical sweep check, with the optimized one:

| Operation | Earlier median | Revised median | Decomposition counts before → after |
| --- | ---: | ---: | --- |
| Compression | 62.0 ms | 54.7 ms | SVD 295 → 236; QR 366 → 307 |
| Evolution | 717 ms | 562 ms | SVD 2520 → 1779; QR 2500 → 1872 |

This is about 22% less evolution time in this case, not a claim of universal
optimality. Trace errors were below `1.5e-13` for both evolution runs. An
experiment forcing the strictest protected basis on every split failed the
existing cached/fresh and single-precision backend comparisons near singular
supports; it was **not adopted**. Approximate SVDs, Cholesky QR, global mutable
gate caches, and reusing canonical metadata after external tensor writes
remain deferred because they need separate accuracy or ownership contracts.

Final revised validation: **81 DMT tests passed** within a combined selection
of **263 passed, 1 skipped, 1 deselected**. The deselection is the separately
confirmed installed-version metadata mismatch (0.5.0 installed, 0.6.0 source).
The skip requires two logical JAX CPU devices. The full-suite retry still
stops at the baseline-reproduced BP NumPy/Torch reshape failure after
102 passes and seven skips. Ruff and whitespace checks pass; no clean full
suite, GPU or transport-convergence claim is made.

## Further review: per-gate precision gap and open-source comparison

The standalone sweep-level check does **not** run on `apply_gate_dmt` or on
`PauliDMT` gates. The per-split normwise check cannot certify physical local
preservation when partial traces are small compared with the operator norm.
For a 128-site, 24-product-state mixture (Bloch strength 0.65), 127 identity
gates at cap 12 changed trace 1 to 79.95 in float32, and to 1.00000017 in
float64, with `variant='operator'` and no exception. The float32 case is an
expected-failure regression in `test_pauli_dmt_large.py`. This corrects the
earlier broad preservation wording: these gate paths need a physical error
monitor or a better-conditioned formulation; their current check is not one.
Checking all windows after each gate would cost at least quadratic work in
chain length per sweep; a linear-cost, per-gate certificate needs further
design and validation. Monitor physical trace and target observables and test
bond/precision convergence in the meantime.

With the same float32 identity-gate sequence, caps 16 and 24 yielded traces
280.34 and 112.57 respectively. The nominal input rank is 24, so increasing
the gate cap to that rank did not recover the source. This contrasts with
standalone compression and requires further investigation; a larger cap is
not by itself a reliable workaround for this gate path.

The [Jack Kemp ITensor DMT library](https://github.com/Jack-Kemp/dmt) uses
cached partial traces, a Hermitian operator basis, configurable preservation
radius and finite support cutoffs. Those cutoffs differ from Pepsy's explicit
zero-support-cutoff policy, chosen after finding a long-chain loss of local
data. It also supports swap-gate long-range interactions and checkpointing;
these are not present in Pepsy's driver. The Julia
[MPSToolkit](https://github.com/jayren3996/MPSToolkit) reports DMT and a
constraint-checkpointed variant, and uses an MIT license. Its source was
subsequently fetched through the GitHub connector; see the deeper review
below. Neither external implementation was executed and no code was copied.

## Conditioning and reference-factorization investigation

In the same seeded 128-site identity-gate test, at bond 60 the trace
component was about `5.4e-5` of a norm-based bond scale and the single-split
relative trace change was `7.1e-4` in float32. At bond 100 the measured
component was about `1.1e-7` of that scale, comparable to float32 epsilon,
and the relative trace change was about `0.048`. The latter measurement is
already affected by earlier drift. A machine-small *normwise* error can thus
be large relative to a physically important local component. This identifies
conditioning as a contributor, not necessarily the only source of error.

Temporary NumPy/Quimb variants tested the current directional QR, a final
QR/QR/small-SVD refactor resembling the reference strategy, and a relative
`8 * eps` cutoff in the preparatory SVD. The cutoff experiment does not
account for discarded support and has no physical-preservation safeguard;
it is evidence, not a candidate to ship. All results use the same 128-site
identity sequence, raw/operator variant and cap 12. Errors are maximum
three-site Frobenius errors against the independent product-mixture oracle.

| Working dtype and factorization | Final trace | Maximum marginal error |
| --- | ---: | ---: |
| float32, current | 79.95 | 73.09 |
| float32, final SVD | 60.90 | 44.49 |
| float32, preparatory cutoff | 0.613 | 0.184 |
| float32, both | 0.707 | 0.167 |
| float64, current | 1.000000171 | `1.26e-7` |
| float64, both | 1.000000010 | `6.96e-9` |

The reference-like choices improve this example but do not reliably preserve
its local data. The float32 density/connected variant rejected all four
configurations for unresolved trace; in float64 all completed with marginal
errors between `8.47e-9` and `1.42e-7`. Scaling every input core by two, with
the common scale factored out of the readout, did not fix the float32
operator case: trace became `-94.66`. The issue is not solely underflow from
the overall `2^-L` normalization.

Explicitly converting the float32 input cores to a float64 **working state**
before creating `PauliDMT` reduced the cap-12 marginal error to `1.31e-7` and
the cap-24 error to `1.59e-7`, across all three-site windows. Casting the
final result back to float32 gave errors below `3.5e-7` in these two cases.
This is a useful workaround for this example, not a general stability
guarantee. No implicit dtype promotion was added.

Source inspection through the GitHub connector established that Julia
[MPSToolkit's low-rank kernel](https://github.com/jayren3996/MPSToolkit/blob/main/src/operator_space/dmt/lowrank.jl)
uses two QR factors and a small repair SVD, plus a numerical support cutoff;
its [bond kernel](https://github.com/jayren3996/MPSToolkit/blob/main/src/operator_space/dmt/bond.jl)
keeps the physics cutoff on the complement. Its
[constraint checkpoints](https://github.com/jayren3996/MPSToolkit/blob/main/src/operator_space/constrained.jl)
project back into a model's constrained sector, e.g. PXP; they do not certify
the local-density numerical error here. Its connector automatically switches
off below a trace threshold; Pepsy deliberately requires explicit variant
selection, since a poorly resolved nonzero density trace is not a traceless
observable. The source and the Kemp reference were inspected, not run: the
local Julia installation has no ITensor packages, and no dependency was
installed. **Defer** importing a cutoff, repair SVD, automatic connector
selection, or projection as a claimed fix pending a constraint-sensitive
algorithm and wider tests.

## Cutoff policy and double-precision follow-up

The user requested investigation of `float64`/`complex128`, `cutoff="auto"`
and Quimb/MPO-optimizer cutoff modes. Installed Quimb 1.15's low-level
`array_split` takes a numeric cutoff and defaults to `rsum2`; its
`tensor_split` defaults to `rel` and `1e-10`. Newer upstream documentation
describes `cutoff="auto"` as `1e-10` for exact SVD. Pepsy's existing
`MpoOptimizer` and shared `dtype_auto_cutoff` instead define `auto` as
`1e-12` for double/complex-double and `1e-6` for single/complex-single.
**Adopt** the Pepsy dtype policy, keep DMT's existing `cutoff=0` and
`cutoff_mode="rel"` defaults, and pass explicit cutoff options to Quimb.

All six Quimb modes now act exclusively on the unprotected residual SVD.
Absolute modes are defined in unnormalized physical Hilbert–Schmidt units,
matching `PauliMPO.compress_pauli`: the internal coefficient norm differs by
`2^(L/2)`, and the centre matrix is additionally rescaled for numerical
stability. The cutoff conversion is computed in logarithmic form. Relative
modes need no conversion. The report records the resolved cutoff and mode.
When an already-fitting operator cannot accommodate the conservative
protected-plus-residual rank construction, the existing exact pass-through
can leave a positive cutoff unused. The cap can also force more truncation
than the requested cutoff. Neither cutoff nor dtype makes the per-gate
normwise check a physical local-error certificate.

Before the boundary pass-through described below, for the difficult 128-site
mixture and 127 identity gates, cap 12, the measured maximum three-site
errors were:

| Dtype and variant | `cutoff=0` | `cutoff="auto"`, `"rel"` |
| --- | ---: | ---: |
| float64 density | `1.42e-7` | `6.08e-8` |
| float64 operator | `1.26e-7` | `7.72e-8` |
| complex128 density, complex virtual gauges | `1.33e-7` | `2.48e-7` |
| complex128 operator, complex virtual gauges | `1.24e-7` | `2.25e-7` |

The auto cutoff improved the real cases and worsened these complex-gauge
cases; these data do not establish universal stability or long-time accuracy.
Additional regressions check the complex-gauge cases, both variants, both
cutoffs and every three-site marginal. Independent cutoff regressions compare
the residual to public Quimb SVD for six modes at global scales `1e-120`, 1
and `1e120`, verify physical MPO unit conversion and protected panels, cover
NumPy/Torch/JAX CPU, and exercise public compression and gate calls.

## Auto-cutoff boundary and trace-guard follow-up

Compared `cutoff=0` and `cutoff="auto"` bond by bond on the same complex-gauge
128-site case. Their retained residual ranks differed only at the first and
last bonds. There the protected directions span the entire bond; the
positive-cutoff path nonetheless reconstructed a mathematically unchanged
bond. Its rounding altered later truncations. An exact QR pass-through now
avoids that reconstruction whenever a protected side spans the fitting bond.
At open-chain boundary gates, the scalar trace environment proves this
condition without the initial SVD. Arbitrary two-qubit boundary gates were
checked against dense four-spin targets with a high positive cutoff; no SVD
was called and the whole target operator was preserved.

With the same seeded 128-site cases and cap 12, auto and zero cutoff now
produce identical results in each tested case: maximum three-site errors are
`1.42e-7` and `1.26e-7` for float64 density and operator variants, and
`1.33e-7` and `1.24e-7` for the complex-gauge variants. Their earlier
differences were caused by the avoidable edge reconstruction, not necessarily
by deliberately discarding a residual direction. Cutoffs can still change
other inputs where they discard resolved residual singular values.

The underlying conditioning problem is not solved by this pass-through. On
the stronger 192-site complex128 operator test, auto cutoff and cap 12
previously yielded trace `0.9982` and local error `5.49e-3`; at 256 sites
trace became about `2058`. A new optional per-gate relative trace guard
detects a single-split defect above `32 * sqrt(eps)` before committing that
split. With an initially resolved nonzero trace, the 192-site float64 and
complex128 tests now stop before severe drift. For a nonzero trace known to
be physically meaningful but below the auto classifier's resolution,
`trace_guard=True` catches the 256-site complex128 case. An attempted retry
with the stricter protected basis did not reduce the failing trace defects
in the 160/192/256-site probes and was **not adopted**.

`trace_guard="auto"` uses the declared density policy or a log-scaled global
trace-coefficient/centre-norm ratio greater than `8 * eps` for operator
inputs. The log recurrence stays on the source backend until its final scalar
classification. This is an inference, not a proof of nonzero trace: at 256
sites the mixture's ratio was `1.2e-19`, below double precision, so automatic
classification cannot distinguish it reliably from a traceless observable.
Users can inspect `run.trace_guard_active`, force the guard, or disable it.
The guard checks the gate's actual target trace, does not undo earlier gates,
does not bound accumulated error, and does not certify local marginals.
The previously failing float32 case stops with `trace_guard=True`; its
default-auto variant remains an expected failure. The user-facing API states
these limits rather than claiming unconditional stability.

The [Kemp ITensor implementation](https://github.com/Jack-Kemp/dmt/blob/master/DMT.h)
checks whether its protected block spans the bond, uses full protected-basis
rotations and a final SVD; the [Julia implementation](https://github.com/jayren3996/MPSToolkit/blob/main/src/operator_space/dmt/bond.jl)
uses thin protected bases and a QR/SVD repair. The source was inspected, not
executed; no external code was copied. The pass-through follows directly
from the full protected-side constraint. The earlier numerical-cutoff and
repair-SVD probes did not justify adopting those choices as a general fix.

For 100 repeated complex128 boundary identity gates on a 64-site input, the
seven-run median decreased from 45.3 ms to 28.6 ms, with SVD calls 300 to 0;
both traces agreed. A separate 64-site, cap-16, three-timestep density run
measured 573 ms without and 583 ms with the relative trace guard (seven-run
medians, one BLAS thread), about 1.6% overhead in this workload. These are
limited CPU microbenchmarks, not universal performance claims.
For a 128-site traceless `Z` product operator, median workspace initialization
was 1.78 ms with the guard disabled and 3.44 ms with auto classification
(eight timed repetitions). This is a one-time linear scan per workspace;
persistent updates reuse the resulting decision.

Final combined Pauli/DMT/damping/public-API/package selection: **327 passed,
1 skipped, 1 deselected, 1 expected failure**. The skip requires two logical
JAX CPU devices; the deselection is the known installed-version metadata
mismatch. The expected failure is the unresolved float32 nonzero trace under
the default auto classifier. Ruff, whitespace and edited local-link checks
passed. The full-suite retry stopped at the previously baseline-reproduced BP
NumPy/Torch reshape failure after **102 passed and 7 skipped**.
