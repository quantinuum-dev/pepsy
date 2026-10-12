# 2026-10-07 — CuPy sampling correction and exact reruns

- Scope: validate the reported fix, correct CuPy, recompute prior exact
  sweeps, and synchronize/commit/push both repositories as requested.
- Branch/baseline: develop, synchronized from 6e9cef6 to 095cc7e.
- Implementation: CuPy float64 CDF/uniform draws in VecSampler, focused
  GPU regressions, API/changelog documentation. Remote Torch fix retained.
- Evidence: [CuPy audit](../docs/development/notes/2026-10-07-cupy-vector-sampling-precision.md).
- Validation: actual GPU uniform states through 30 qubits pass the per-bit
  check; sampler/public API/layout after synchronization: 168 passed,
  one unrelated installed-version mismatch. Ruff unavailable; no full suite.
- Publication: user explicitly requested committing and pushing. Final
  synchronization validation and push outcome are reported in the session.
- Runtime: preserve old data, replace biased exact sweeps in distinct roots,
  keep GPU3 MPS running. Campaign pointer:
  `/tmp/pepsy_exact_corrected_campaign.txt`.
