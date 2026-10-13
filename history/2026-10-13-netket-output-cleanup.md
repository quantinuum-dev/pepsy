# 2026-10-13 — Remove generated PEPS/NetKet simulation outputs

- Scope: user requested deleting saved simulation outputs from Pepsy and `/tmp`.
- Branch / baseline: `develop` / `578cb6a`.
- Commit status: included in the simulation-output cleanup commit; no push.
- Removed the full scalar convergence JSON from the checkout, both temporary
  long/partial run directories and earlier standalone NetKet trajectory files.
- Updated notes to state that generated outputs were deleted. Code, tests,
  API documentation and concise measured summaries remain. Concurrent MPS
  work and installed/dependency environments were left intact.
- Validation: deleted paths and local documentation links checked; whitespace
  check passed. No numerical tests or simulations were run for this cleanup.
