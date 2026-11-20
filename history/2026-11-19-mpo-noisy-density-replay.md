# 2026-11-19 — Noisy density MPO replay

The `direct` and DMRG modes can evolve a density MPO through a mixed stream
of unitary gates and deterministic one- or two-site Kraus events.

- Branch `develop`, baseline `e2857dc`. The changes are in the working tree.
- The [trace diagnostics](../src/pepsy/optimizers/mpo/optimizer.py) now measure
  the input and exact channel target before compression can overwrite either
  MPO. Previously the direct path read the target trace after mutating it in
  place, attributing trace loss to the channel instead of compression.
- The [lossy regression](../tests/test_optimize_mpo.py) verifies the reported
  traces in `direct` and `dmrg` at bond cap one. Mixed qubit evolution and
  qutrit leakage and seepage checks agreed with dense references within
  `5e-16` in `direct`, `dmrg`, `dmrg1`, `dmrg2`, and `dmrg3` at sufficient rank.
- `tests/test_optimize_mpo.py` and `tests/test_mpo_alignment.py` passed with
  152 passes and one skip. Ruff and `git diff --check` passed.

## Named streams and 4×4 verification

The MPO queue now lowers symbolic gates through the shared MPS resolver and
the named local noise through the trajectory channel definitions. Named gates
apply physical unitary conjugation. Dense two-site DMRG channels use two
factors of the exact local superoperator over the ket and bra legs. The
fitter splits the output tensors. The earlier materialized sum enlarged the
exterior virtual bonds and caused an interior-window dimension mismatch.

The [4×4 example](../examples/noisy_mpo_tfim.py) uses open-boundary
$H=-\sum_{\langle i,j\rangle} Z_i Z_j-1.05\sum_i X_i$, row-wise snake
storage, the all-$|0\rangle$ state, two Strang steps of 0.1, and a noise
layer after each step. Each layer applies 1% single-qubit depolarization
on all 16 sites and 1% two-qubit depolarization on the bond at sites 5 and 6.
The exact reference evaluates the same discrete circuit by statevector and
untruncated Pauli propagation. A 2×2 comparison of all 256 Pauli expectations
agreed with dense evolution to `6.7e-16`.

| Mode | χ cap | Max error at t=0.1 | Max error at t=0.2 | MPO trace at t=0.2 |
| --- | ---: | ---: | ---: | ---: |
| dmrg2 | 8 | 3.95e-5 | 1.244e-3 | 0.999134 |
| dmrg2 | 16 | 3.95e-5 | 1.244e-3 | 0.999271 |
| dmrg2 | 32 | 3.95e-5 | 1.244e-3 | 0.999272 |
| direct | 16 | 7.78e-6 | 1.926e-4 | 0.999755 |

Errors are the largest absolute difference among all single-site X and Z and
all nearest-neighbor ZZ expectations. MPO expectations were divided by the
reported trace for readout. The replay used NumPy complex128, two BLAS threads,
cutoff `1e-12`, and a two-sweep budget for DMRG2. Exact mean Z was
0.9656669205 and 0.9004605290 at the two recorded times.

The final selection across MPO, FIT schedules and the public package API had
217 passes, one skip, and two MPS schedule failures. The failures reproduce on
the baseline with Autoray 0.8.4 because its NumPy backend lacks `stop_gradient`.
The named-noise checks passed on NumPy and Torch CPU, and a three-qutrit
layered leakage probe matched the dense reference. Ruff and
`git diff --check` passed. The changes are in the working tree.

## Ten-step comparison with depolarization on all bonds

The longer benchmark used the same open 4×4 lattice and ten 0.1 Strang
steps. Each step ended with 16 one-qubit and 24 two-qubit depolarizing
events at probability 0.02 per event. The matched run used χ=16, NumPy
complex128, one BLAS thread per process, and cutoff `1e-12`. Both DMRG
modes used two sweeps and direct initialization. A χ=32 DMRG2 run probed
bond-cap sensitivity. The command accepts `--mode dmrg`, `dmrg2`, or
`direct`, and the sweep used `--steps 10 --p1 0.02 --p2 0.02
--all-noise-bonds --exact-steps 2`. [Benchmark script](../examples/noisy_mpo_tfim.py)

| Mode | χ | Mean Z at t=0.3 | Mean Z at t=1.0 | Real trace at t=1.0 | Replay seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| dmrg | 16 | 0.659168 | 0.181251 | 0.859930 | 16.61 |
| dmrg2 | 16 | 0.659168 | 0.181251 | 0.859930 | 16.26 |
| direct | 16 | 0.652929 | 0.185393 | 0.915598 | 340.49 |
| dmrg2 | 32 | 0.657154 | 0.179906 | 0.926218 | 47.45 |

The exact mean Z at t=0.1, 0.2, and 0.3 was 0.894226969, 0.771755415,
and 0.650008897. Lattice symmetry supplied the corner, edge, and interior
averaging weights at t=0.3. The exact corner Z at t=0.4 was 0.548736870.
At t=0.2, the maximum error across 56 observables was 0.0256712 for both
χ=16 DMRG schedules and 0.00207367 for `direct`. Mean Z is trace-normalized.

The χ=16 DMRG schedules agreed within `2.21e-11` across the recorded
observables. In the first step the DMRG2 trace targets preserved trace within
`1.8e-15`, and two-site channel fitting contributed 0.02856 of trace loss.
The `direct` trace had an imaginary component of `-9.86e-4` at t=1.0.

All runs completed with finite readouts and respected the requested bond
caps. A replay of the previous one-bond defaults matched the archived
first-step exact and MPO values. Ruff and `git diff --check` passed.

## Timing diagnosis and independent exact check

The original `direct` result includes materializing and adding 16 full-chain
Kraus branch MPOs per two-site depolarizing event. The resulting intermediate
bond reached 256 at χ=16. On the second step of an instrumented replay,
building these targets took 9.02 s, target Frobenius norm diagnostics took
10.34 s, and chain compression took 12.06 s. Two-site noise consumed
32.33 of 33.43 s. DMRG2 kept the channel layered over the input MPO and
spent 0.44 s on the same two-site events, including 0.20 s fitting.

A NumPy benchmark prototype applied the one-site transfer directly and
factored each two-site transfer across its physical endpoints. The
depolarizing transfer has rank two, so only bonds within the affected
interval doubled. The prototype retained the original direct compression
settings and matched all 56 real local readouts across ten steps within
`8.24e-12`. In sequential, single-BLAS-thread replays, it took 10.48 s
compared with 15.56 s for DMRG2 at the same cap. The prototype is scratch
code. No production replay changes were made during the timing diagnosis.

An independent check propagated reduced density matrices on the exact
observable causal cones, using at most eight qubits for the second step.
It agreed with the earlier Pauli reference on every observable at t=0.1
and t=0.2 within `2.4e-15`. At χ=16 the error before the first two-site
noise layer was `3.89e-5` for DMRG2 and grew to `0.01748` after the layer.
Essentially all first-step trace loss came from the two-site fits spanning
five or seven bonds in snake order. Increasing the DMRG2 sweep budget per fit
from two to eight reduced the maximum t=0.2 error from 0.02567
to 0.01454. The exact trace is one throughout, and no exact t=1
magnetization was calculated.

## SRC initialization for noisy FIT

The initial noisy replays explicitly selected `fit_init_strategy="direct"`.
The two-site channel path also bypassed the initializer, even if gate
replay selected SRC. Dense two-site channels now use the same initialization
policy as gates. The SRC guess is capped on the active tagged interval and
FIT uses the separate exact layered target. Resolving trace diagonals
locally also fixed the oversized trace contraction exposed after the SRC
gate warm starts. [Compatibility notes](../docs/development/notes/2026-11-19-mpo-channel-src.md)

With the same ten-step, 2% noise benchmark, NumPy complex128, χ=16,
`cutoff=1e-12`, and two sweeps per fit, the initialization ablation
gave these maximum errors across 56 trace-normalized observables.

| Mode | Gate guess | Channel guess | t=0.1 | t=0.2 | Real trace at t=1 |
| --- | --- | --- | ---: | ---: | ---: |
| dmrg2 | incoming MPO | incoming MPO | 0.0174780 | 0.0256712 | 0.859930 |
| dmrg2 | SRC | incoming MPO | 0.0167315 | 0.0136339 | — |
| dmrg2 | incoming MPO | SRC | 0.0000311 | 0.0026889 | — |
| dmrg | SRC | SRC | 0.0002093 | 0.0021842 | 0.954416 |
| dmrg2 | SRC | SRC | 0.0002116 | 0.0021847 | 0.954473 |
| direct | — | — | 0.0001698 | 0.0020737 | 0.915598 |

The channel-only ablation recovered the first-step accuracy, which isolates
the poor starting subspace on the nonlocal noise fits. At t=0.2 the two
SRC-initialized DMRG modes are within 6% of the direct result at the same cap.
The four-sweep runs for both modes and eight-sweep run for DMRG2 remained
near 0.00219 at χ=16. At χ=32 the t=0.2 errors were 0.001555 for `dmrg`,
0.001451 for `dmrg2`, and 0.001423 for the factorized direct prototype.
The [benchmark](../examples/noisy_mpo_tfim.py) now defaults to SRC for the
DMRG modes and records the selected settings.

The MPO, alignment, and public package checks finished with 215 passes and
one skip. The long-interval product-state regression checked both DMRG modes
against the full density matrix, including the MPO exponent and traces,
and verified SRC selection with the initializer argument omitted.
Three MPS FIT reference tests hit the same missing NumPy `stop_gradient`
dispatch at HEAD and in the working tree. Those three tests passed with a
process-local NumPy identity registration.
Ruff and `git diff --check` passed. The review scoped publication to the
noisy MPO changes.
