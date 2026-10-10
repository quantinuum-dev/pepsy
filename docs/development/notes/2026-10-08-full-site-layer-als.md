# Fixed-layer full-site PEPS ALS

Date: 2026-10-08. Branch `develop`, baseline `21f883b`. Implementation is
in the working tree, not committed. This continues the
[cap fix and Quimb/paper assessment](2026-10-08-full-update-caps-quimb-refinement.md).
The user approved incremental broader refinement with attention to caching
and large-D cost, then explicitly excluded environment gauge conditioning.

## Implemented

- `refine_scope="layer"` freezes an exact ordered gate-window target before
  sequential pair FU, then fits complete site tensors across every row and
  column against that same target. One sweep means one row-plus-column cycle.
  The exact automatic-batch builder limits target bonds to 2D for diagonal
  qubit gates or 4D when other gates are present. It closes the window instead
  of truncating the target. Intervening singles retain their order; trailing
  singles execute after the last pair and its refinement callback.
- Fresh whole-lattice norm/overlap checks accept only finite lower-cost,
  non-worsening-fidelity cycles. Failure restores the best previous state.
  Only the fixed target norm is retained between checks; fitting and checking
  caps are distinct, including separate norm/overlap caps. A layer report
  accompanies the closing pair record; pair fidelity remains a separate score.
- `refine_solver="pinv"` uses native Torch/CuPy Hermitian eigendecomposition,
  discarding negative and relatively small eigenvalues. It reports retained
  rank, condition on the retained support, negative spectral weight, and
  discarded RHS weight. No inverse gauge transformations are applied.
- Environment gauges now default to off in pair FU too. Its older explicit
  `gauge=True` option remains compatible. The attempted new full-site gauge
  option was removed in response to the user's instruction. Internal QR/LQ
  reduction and bond balancing are separate existing pair-ALS operations.
- The owned directional strip cursor builds one future cache per pass and
  extends only the updated past. For L sites, each objective requires
  `2*(L-1)` extensions per pass. The transverse boundary store still validates
  source mutations; the cursor trusts only its synchronous, privately owned
  network. Norm and overlap stores remain distinct.
- `refine_max_matrix_size=1024` checks the product of virtual dimensions
  before dense norm construction. Layer mode also avoids building a window
  target that cannot be refined. A skip records `matrix_size_limit` and
  preserves FU. Larger values or `None` are explicit overrides.
- Optional `refine_balance=True` rescales active tensor entries while
  compensating the network exponent, preserving the represented state.
  Defaults remain refinement disabled, strip scope, Quimb solver, and no
  additional scalar balancing. See the [API guide](../../api/optimizers/peps.md).

The objective is normalized least-squares compression of a fixed target,
with a separate normalized-fidelity check. It is not energy minimization or
a globally optimal PEPS fit. Boundary estimates can still be inaccurate at
insufficient caps; local positive projection cannot correct that error.

## Upstream decisions

The same-task upstream audit and installed source inspection are reused from
the linked preceding note; dependencies were not changed. Versions: Quimb
1.15.1.dev90+g6a3906cbe, Autoray 0.11.1.dev14+g014a3f69a, Cotengra
0.8.3.dev8+g8954240f2, Symmray 0.4.1.dev15+g0374aaa3c,
Torch 2.6.0+cu124, CuPy 14.1.1.

- **Adopt:** full-site variational freedom, Hermitian/positive norm handling,
  support-cutoff solves, shared exact targets, and directional caching.
- **Retain:** public Quimb generic ALS as the existing default refinement
  solver. Its positive eigensolver floors small eigenvalues; the new native
  pseudoinverse discards them. Small well-supported examples below produce
  indistinguishable fidelity, while null-space RHS regression tests verify
  the intended numerical difference.
- **Defer:** matrix-free solves, rank enrichment, overlapping 2x2 patches,
  and Quimb FullUpdate replacement. No compatibility shim was needed.
- **Excluded by user preference:** new environment gauge conditioning.

## Measurements without environment gauges

Torch CPU complex128, one BLAS/OpenMP/Numba thread, greedy contraction,
direct zero-cutoff boundary compression, fixed fit/evaluation/normalization
cap 32, eight pair ALS iterations. Random PEPS seed 24. Timings include FU
and requested refinement, excluding constructor and independent reference
metrics. They are small scoped measurements, not statistical performance
claims or GPU/long-time evolution benchmarks.

For 3x3, use two gates on row 1,
`diag(exp([-.3j,.3j,.3j,-.3j]))`, normalized initial state, no final
normalization. Strip uses two passes; layer uses one full cycle. Infidelity
uses the exact dense target vector. Seconds below are the second of two runs.

| D | FU seconds / infidelity | Strip seconds / infidelity | Layer pinv seconds / infidelity |
| --- | --- | --- | --- |
| 2 | 0.053 / 0.0775294 | 0.094 / 0.0651526 | 0.206 / 0.0474077 |
| 3 | 0.062 / 0.0927881 | 0.122 / 0.0450644 | 0.269 / 0.0276392 |
| 4 | 0.156 / 0.0691706 | 0.285 / roundoff | 1.146 / roundoff |

The tiny D4 case can represent this target exactly; it is not a general
high-D accuracy result. Quimb layer solves match these infidelities to
roundoff and take 0.208/0.334/1.097 seconds in the same second-run comparison.

For 4x4, use three row-1 gates with angle .21, requested final normalization,
and the same fit controls. Compare outputs against the same exact target
using freshly contracted cap-32 metrics, outside the timer. These are
boundary estimates, not dense exact fidelities, and differ somewhat from
the fixed-strip diagnostics. Single-run timings:

| D | FU seconds / infidelity | Strip seconds / infidelity | Layer pinv seconds / infidelity |
| --- | --- | --- | --- |
| 2 | 0.200 / 0.0561913 | 0.320 / 0.0530217 | 0.701 / 0.0472191 |
| 3 | 0.444 / 0.0729907 | 0.630 / 0.0637092 | 1.584 / 0.0575128 |
| 4 | 2.027 / 0.1017561 | 3.433 / 0.0929652 | 11.214 / 0.0882704 |

The D4 layer result improves this common estimated infidelity by about 5%
relative to strip refinement at 3.3 times its runtime. The separately reported
strip and layer scores (0.087412 and 0.087458) misleadingly suggest no gain
when compared directly: they use different approximate objective contractions
and precede final normalization. Always compare a common independently
contracted target metric, and converge its cap for close claims. Together with
the exact 3x3 D4 example, this supports keeping layer refinement optional,
not claiming a universal accuracy-per-time improvement.

The reference diagonal-gate target uses the known exact operator-Schmidt
bound `max_bond=2*D`. An initial reference script retained numerically zero
SVD directions with `cutoff=0` and no bond bound, greatly increasing pure-target
contraction cost; that run was interrupted and replaced. Production target
construction already uses the bounded exact gate-window builder. The table
above is from the corrected reference script; no target approximation was
introduced by discarding structurally unsupported directions.

For isolated interior-site solves, generate a complex random square root R
scaled by `1/sqrt(n)`, `N=R.H@R+0.1I`, and two exact RHS columns, seed 72.
Run two native pseudoinverse solves with `rcond=1e-12`. Second-run results:

| D | n = D^4 | Raw complex128 N, MiB | Solve seconds |
| --- | ---: | ---: | ---: |
| 2 | 16 | 0.0039 | 0.00038 |
| 3 | 81 | 0.100 | 0.00120 |
| 4 | 256 | 1.00 | 0.0121 |
| 5 | 625 | 5.96 | 0.108 |
| 6 | 1296 | 25.63 | 0.860 |

Relative solution errors are 2e-15 to 1.4e-14. These exclude environment
construction and eigensolver workspace. Dense cost grows as D^12, with
D^8 storage; D8 would require 256 MiB for N alone and was not measured.
The default guard excludes interior D6 and above. It does not bound the
cost of transverse boundaries, exact targets, or a complete cycle.

## Validation

- Broad PEPS FU/strip/layer, gate ordering/batching, optimizer, adaptive caps,
  cache reuse, shared reduced ALS, CuPy, public API and package layout:
  **488 passed**, 100 warnings, 75.03 seconds.
- Final focused FU/strip/layer run after adding ungauged-default assertions
  and extending cached/fresh exponent tests to pseudoinverse and balancing:
  **93 passed**, one warning, 24.53 seconds. These overlap the broad selection.
- Regressions cover complex64/128 Torch/CuPy exact dense fidelity improvement,
  backend/dtype/device/index preservation, repeated-bond window boundaries,
  nonunitary normalization, unsupported/negative norm support, whole-cycle
  rollback, unequal adaptive caps, skips before target/matrix construction,
  and linear cursor extension counts in both directions.
- Ruff (`src tests`) and `git diff --check` pass. No fresh full-repository
  suite or large-D complete-evolution/GPU timing benchmark.

Temporary scripts/logs are under `/tmp/pepsy_native_layer_ungauged_bench.*`,
`/tmp/pepsy_layer_4x4_common_bench.*`, `/tmp/pepsy_native_layer_validation.log`,
and `/tmp/pepsy_native_layer_final_focused.log`. Unrelated MPS edits are
preserved. Nothing was staged, committed, or published by this task.
