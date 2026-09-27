# 2026-09-25 — qMERA final API review

- Scope: review the public qMERA API and TFIM example, then commit qMERA work.
- Branch / baseline commit: `develop` / `4f62cb0`.
- Commit status: included in the final qMERA commit on `develop`; the sibling
  `pepsy_examples/qmera/` example is committed separately in that repository.

## Review and validation

- Corrected the API page's scope statement. Live builder, optimizer, cost-report,
  and drawing signatures agree with the documented notebook calls.
- qMERA module: 103 passed, with one non-failing loky worker warning.
- Public API/layout: 57 passed; the version assertion fails because the shared
  installed distribution is `0.4.0` while this checkout declares `0.5.0`.
  The shared environment was not changed.
- Ruff (`src tests`) and staged whitespace checks passed.
- The CPU TFIM notebook executed into `/tmp/pepsy_examples_executed/`:
  final energy `-5.1873339644`, exact reference `-5.2262518595`; local-term
  sum and Torch full-graph AOT energy/gradients agreed. Generated notebook
  outputs were excluded from the example commit.
- Only qMERA files and qMERA changelog entries were staged. Existing Tree
  work and the sibling roughening/Dmrg edits were left outside these commits.

## Limits

- Inductor remains unverified on this machine because Python 3.12 development
  headers are absent. Dense preflight memory and FLOPs are path estimates,
  not measured process memory or runtime; native Symmray costs are unsupported.
