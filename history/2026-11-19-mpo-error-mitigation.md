# 2026-11-19 — Middle-out tensor-network error mitigation

Implemented `MpoErrorMit`, `PauliShots`, and `TemEstimate` in the MPO
optimizer package. The [API guide](../docs/api/optimizers/mpo.md#tensor-network-error-mitigation)
shows the flat MPS gate-stream constructor, independent replay/compression
modes, and external shot format. An explicit layer list sets compression
boundaries.

The unitary and inverse-noise PTMs share the trajectory stream's gate order.
Each middle-out update attaches the full layer to a tagged target and
compresses it once. The first unitary layer cancels around the identity map.
DMRG uses the existing MPO product FIT kernel and an SRC warm start.
Readouts use the existing independent, coalesced, and auto trajectory
runners and preserve logical qubit order. Exact replay uses the dense vector
sampler. The NumPy norm dispatch guards were taken from `origin/develop` so
trajectory replay also works from the older local baseline.

The tests compare the PTM against a dense inverse with reversed gate
supports and noncommuting channels, check biased-basis shot means and
standard errors, and count one FIT per layer across repeated estimates.
The two-layer six-qubit run used 4,096 coalesced shots and a χ=8 DMRG2 cap.
The map retained χ=5. Contracting it with the independently evolved dense
noisy state recovered both selected ideal observables within `6e-15`.
For the endpoint correlation, the shot estimate moved from 0.5466 to
0.7650 ± 0.0339, with ideal value 0.8400.

An independent seed with 65,536 shots gave 0.8505 ± 0.0085. For this
observable, the corrected Pauli row was a single term with coefficient
1.39946 and an off-term norm below `2e-16`. Sampling the existing
trajectory terminal bitstrings in Z gave 0.8384 ± 0.0175 at 4,096 shots.

Verified the installed Quimb MPO compressors and MPS direct, SVD, swap,
permutation, DMRG, mixed, exact, and exact-batch routes on a small circuit.
The mixed dense/native `compress_mpo_product` test selection reproduces two
Symmray contraction failures at HEAD. Both native cases pass in isolation.
The implementation is in the working tree.
