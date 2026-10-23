# 2026-10-22 — Optional adaptive PEPS gauge QR

- Scope: implement the approved optional dense full-update `qr:auto` path.
- Branch / baseline: `develop` / `5ff1fe6`.
- Commit status: included with the implementation, tests, and documentation
  in the user-requested commit. No dependency change. Earlier review notes
  are included; unrelated diagonal handoffs remain outside this commit.

Added `full_update_kwargs={"gauge": True, "gauge_qr": "qr:auto"}`, requesting
Quimb's triangular factor only. Defaults and global linalg policies remain
unchanged. New tests also exposed and fixed a pre-existing Torch CUDA gauge
SVD dispatch failure. See the
[evidence and limits](../docs/development/notes/2026-10-22-peps-gauge-qr.md).

Validation: 202 focused tests passed, including CPU/CUDA Torch and CuPy;
the new 29 cases passed again after final validation ordering. Ruff, local
Markdown links, and whitespace checks passed. No full-suite run.

The final synthetic-environment pair benchmarks showed no useful overall
speedup at D=4 or D=8. Keep this opt-in; larger workloads and other algorithms
need separate measurement. Autodiff and native symmetry are not added.
