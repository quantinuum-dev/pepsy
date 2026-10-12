# Native strip refinement and per-gate fidelity

Implemented on `develop`, baseline `fe11e24`, in the working tree on
2026-10-08. This follows the user's approval of the
[strip ALS proposal](2026-10-08-full-update-cupy-and-strip-refinement.md)
and request for efficient per-gate fidelity accumulation. Public controls
are described in the [PEPS optimizer guide](../../api/optimizers/peps.md).

## Implementation and invariants

`full_update_kwargs['refine_sweeps']` defaults to zero. A positive value
enables fixed-rank one-site ALS over the row or column of a completed
contiguous gate block. A saved pre-block state receives the same gates
without truncation. Repeated bonds and single-site/strip boundaries close
the block, bounding exact qubit target bonds to at most four times the
retained rank. Exterior site arrays stay fixed. This first implementation
does not expand the variational region to neighboring strips.

The adapter reuses public Quimb `tensor_network_fit_als` with prebuilt local
norm/overlap networks, native eigensystem solves, and a Hermitian norm matrix.
Both layers use the shared boundary-MPS and prefix/suffix infrastructure;
Cotengra receives the original contraction policy. There is no NumPy PEPS
conversion. Native Torch and CuPy complex64/complex128 were exercised.
The pair update and refinement use the same extracted `boundary_strip`
helper. Adaptive settings recalibrate against the block target. Reported
passes alternate direction, have explicit stopping reasons, and retain
the best valid cost/fidelity candidate after a failed later pass.

Two Quimb details matter for correctness. First, selected subnetworks omit
the parent network exponent; the strip adapter explicitly restores it.
Second, public ALS selects hole networks internally, so the adapter supplies
zero-exponent local equations and applies the relative norm/overlap scale
to the RHS. Unit-norm objectives are converted back to the physical target
scale at the end. Candidate bra virtual labels are aligned between norm
and overlap layers even when the exact target has different bond labels.
Tests cover large network exponents, least-squares scale, and mutated labels.

Each two-qubit step now retains its chosen pair's `local_fidelity` and
`local_infidelity`, including a warm-start fidelity if outer acceptance
rejects the proposed ALS state. These need only tiny reduced contractions
in the existing positive norm environment. Optional accumulation (enabled
by default) uses `-expm1(sum(log(F)))`; zero fidelity stays zero in the
product. An unavailable estimate invalidates the cumulative diagnostic.
Resetting traces resets the product. No whole-lattice metric contraction
is added by this bookkeeping.

Pair estimates precede strip refinement. The refinement's before/after
scores refer to its fixed exact block target and stay separate. After an
accepted refinement, an unmeasured final-state infidelity is `None`, rather
than copying the stale pair score. Requested final metrics are recomputed.
The product of pair fidelities is neither exact circuit fidelity nor an
error bound; coherent error accumulation can differ substantially.

## Measured 4×4 transverse-field Ising evolution

The independent reference setup is unchanged: open lattice, initial
`|+>^16`, `H = -sum_nn ZZ - sum_i X`, exact state-vector Trotter reference
and independent sparse-Hamiltonian evolution. Zero boundary cutoff;
complex128; real time uses four steps of 0.1 and imaginary time three
steps of 0.05. Torch uses input strip order, CuPy column ordering; each
backend's ordinary/refined comparison uses its same schedule. One
refinement pass per completed strip was enabled.

| Backend / evolution | D / boundary chi | Ordinary FU final infidelity | Refined final infidelity |
| --- | --- | ---: | ---: |
| Torch CPU, real | 2 / 32 | 1.65300502e-3 | 1.65486901e-3 |
| CuPy GPU, real | 2 / 32 | 1.65302779e-3 | 1.65426642e-3 |
| CuPy GPU, imaginary | 2 / 32 | 4.31221072e-8 | 4.05430436e-8 |
| Torch CPU, real | 4 / 64 | 1.46945938e-8 | 8.44353432e-9 |

Infidelities are against exact Trotter states, not the continuous-time
reference. D=2 ordinary/refined runs took 3.63/6.12 seconds on Torch and
6.05/10.59 seconds on CuPy in these single measurements. These are not
controlled performance benchmarks. The imaginary and D=4 ordinary values
come from the preceding review's same-setup runs. All outputs remained
finite and normalized, with retained rank bounded by D.

Refinement improves each accepted local fit, but the final real-time D=2
circuit infidelity is slightly worse (about 0.1% relative). Thus the option
remains off by default. Imaginary D=2 and real D=4 improve in these runs;
this is not a general guarantee. The D=4 run rejected six invalid initial
fixed-boundary estimates and one invalid later candidate across 32 blocks;
those paths preserve a valid state instead of forcing an update. Dense
one-site norm matrices and exact block targets can be costly at larger D.
Larger lattices, multi-GPU use, and neighboring-strip expansion were not
tested.

Temporary reproducible drivers are `/tmp/pepsy_strip_4x4_torch.py` and
`/tmp/pepsy_strip_4x4_cupy.py`, using `--case` and `--refine 0|1`; JSON and
vector outputs are under `/tmp/pepsy_strip_4x4_{torch,cupy}_{0,1}/`.

## Validation

New tests verify native dtype/device, improvement against dense overlaps,
unchanged exterior arrays and metadata, both cache layers, cached/fresh
equivalence, exponent handling, rollback, exact block targets, boundaries
at repeated bonds/one-site gates, adaptive non-unitary refinement, final
remeasurement, and per-gate products/reset/rollback without extra metrics.

Focused refinement plus existing full-update tests: **60 passed**.
Combined PEPS/full-update/CuPy/reduced-update/cache/boundary/scheduling
regression: **448 passed**, 98 upstream warnings, 111.58 seconds.
Default smoke: **94 passed**, two warnings, 34.83 seconds. Ruff and whitespace
checks passed. No clean full-suite claim: the preceding
review's 24 reproduced baseline BP failures were not re-investigated here.

Final publication review additionally separated physical norm validity from
the ALS stopping tolerance and guarded nonfinite imaginary norms and
overflow-sized overlaps. Three new regression cases pass. An isolated
checkout containing only the selected commit changes passed **451 affected
tests** and **94 smoke tests**, plus Ruff and whitespace checks. The unrelated
backend/MPS working-tree changes were excluded from this verification.
