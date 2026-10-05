# 2026-10-05 — Noisy MPS ensemble and compression-bias reference tests

- Scope: user requested implementing the first two follow-up validations as
  pytest tests: exact density-matrix ensembles and separating compression bias
  from sampling uncertainty.
- Branch / baseline commit: develop / d7d509b.
- Commit status: working-tree tests/docs only for this follow-up; no commit or
  push. Existing stabilizer and trajectory corrections were preserved.
- Environment: local genpy Python 3.12 and unchanged session dependencies.

## New validation

Added tests/test_mps_trajectory_density_reference.py, marked integration.
The dense oracle uses explicit computational-basis embeddings and
rho -> sum(K rho K†), rather than Pepsy or Quimb evolution. Analytic checks
cover amplitude damping, Bell measurement/feed-forward and nonlocal CNOT bit
ordering. A separate pure-branch enumeration reproduces the density matrix and
computes the target/proposal path law and estimator variance.

Eight two-qubit ensemble checks combine amplitude damping, a phase-flip
mixture, measurement and conditional X, then another amplitude-damping event:
direct/dmrg2, independent/coalesced, ordinary/importance sampling. All sixteen
Pauli products form a complete observable basis for the final density matrix.
Fixed seeds use 512 independent shots or 32768 count-coalesced shots. Test
tolerances are six standard errors from the independent proposal oracle plus
2e-10 numerical tolerance, not empirical deviations or ESS. Weighted trace is
also checked; estimates are not self-normalized by the observed weight sum.
Six-standard-error checks are statistical regressions, not a general coverage
guarantee for arbitrary rare-event distributions.

A four-qubit SVD circuit creates Schmidt rank four across the middle cut.
The test varies chi=1/2/4, absolute cutoff=0/.4, and shots=8192/65536. Leaf
physical branch probabilities, rather than random counts, reconstruct the
compressed model's infinite-shot density matrix. This identifies its bias
separately from sampling error. For this circuit, full rank and zero cutoff
agree with the exact reference to <1e-10 Frobenius error; smaller ranks and
the large cutoff yield >.05 error. Monotone rank improvement is asserted only
for this chosen reference, not all variational optimizations.

At chi=2, eight times more shots reduces the observable standard error by
sqrt(8), while the reconstructed model bias is unchanged within 1e-12.
The observable X0 X2 has nonzero sampling variance, avoiding a vacuous
shot-count comparison of a deterministic observable. The first draft expected
four leaves even after severe truncation; actual chi=1 removes the excitation
responsible for one Kraus outcome. Corrected the test to require all positive
model branch probabilities sum to one, and four leaves only in the exact-rank
zero-cutoff case. No numerical implementation or tolerance was changed to
conceal that difference.

The noise API guide documents the command and limits of these references.
No production algorithm changes were made by this follow-up. Real hardware
profiling and multi-rank MPI execution remain outside this test addition.

## Validation

- Initial full new reference file: 10 passed, one existing Quimb SVD default
  warning, about nine seconds on this CPU.
- Final complete-Pauli-basis selection with existing trajectory and importance
  regressions: 156 passed, three skipped, one Quimb warning, 14.34 seconds.
  Skips: second JAX device not configured in this run, CUDA unavailable, and
  CuPy missing. The second-device regression was validated separately during
  the preceding trajectory fix; no fresh GPU claim is made here.
- python -m ruff check src tests and git diff --check passed. No full repository
  suite, package changes, staging, commit or push.
