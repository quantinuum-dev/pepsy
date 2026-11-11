# 2026-11-05 — Hermitian-only local-cluster environment

User selected Hermitian-only N_env for the six-site χ6 GPU experiment.
On `develop` baseline `ee22198`, these are uncommitted additions to the same
local-cluster task. No staging, commit or publication occurred.

`psd_project=False` now only Hermitianizes the open reduced environment:
the exact, SU-cluster and loop-cluster builders skip diagnostic eigenvalues
and return `raw_min_eigenvalue=None`, with zero clipped eigenvalues.
The public builders retain their existing default projection policy.
The downstream runner and profiler default explicitly to Hermitian-only,
with an explicit option to reproduce PSD runs. Local ALS conditioning is
unchanged; this selection concerns the contracted open environment.

Validation: **88** reduced-update, cluster-patch and local-gauge tests passed;
**6** downstream tests passed, including complete-operator dense comparisons
on actual CPU/CUDA in both projection modes. Independent weighted-cluster
checks reject any eigensolve during Hermitian-only environment construction.
Full Pepsy Ruff and changed downstream Ruff/whitespace checks passed.
Three additional native Symmray reduced/gauge cases also passed.
The maintained BP skill and unchanged-environment upstream audit were reused.
The complete repository test suite was not run.

The earlier PSD full GPU worker was stopped after verifying its command;
outputs and frozen sources remain preserved. The fresh frozen Hermitian-only
trial, driver comparison and timing scope are recorded in the
[downstream profile](../../pepsy_examples/experiments/mps_magnetization/benchmark/docs/2026-11-05-eli-loop-cluster-gpu-profile.md).
The Hermitian-only prefix completed 1,056 updates in 84.34 s; independent
inspection found 100 tensors/180 gauges on CUDA and no environment spectrum
in all pair records. The full worker launched successfully. A twelve-pair
driver study retained exact `gesvd` (68.66 ms/pair versus 166.50 ms/pair for
`gesvdj`, comparable local errors). No final T=6 convergence result is claimed.
