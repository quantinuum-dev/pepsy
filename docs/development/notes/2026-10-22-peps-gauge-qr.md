# Optional adaptive QR for PEPS gauges — 2026-10-22

Dense reduced full updates can opt into Quimb `qr:auto` for their two
environment gauges. This adopts an upstream factorization without changing
Pepsy's default or its autodiff policies. See the
[public controls](../../api/optimizers/peps.md#two-site-full-update).

## Scope and numerical contract

`full_update_kwargs={"gauge": True, "gauge_qr": "qr:auto"}` selects the
new path. `gauge_qr="qr"` preserves the existing Autoray QR call; gauges
remain disabled by default. Unsupported methods, disabled gauges with an
adaptive request, and missing upstream capability fail explicitly.

The reduced-pair constructor already requires non-differentiable dense
Torch/CuPy complex64/complex128 tensors. JAX, native Symmray, differentiable
states, and full-tensor fitting are outside this option's contract. Initial
state reduction, ALS regauging, boundary compression, and global linalg
registrations are unchanged.

For each unfolding, request only the triangular factor from Quimb:

$$
A={\color{blue}Q}{\color{purple}R},\qquad
A^\dagger A={\color{purple}R}^\dagger {\color{purple}R}.
$$

The existing singular-value threshold still determines whether either gauge
can be inverted. A report's `gauge_qr` records the requested method, not
whether Quimb selected Cholesky internally. Exact-SVD shortcuts skip gauges.

New GPU integration checks exposed a pre-existing failure with `gauge=True`:
Quimb's values-only SVD driver attempted NumPy conversion of Torch CUDA
gauges. Computing the small gauge spectra through Autoray's backend-native
SVD fixes both QR choices without moving tensor data to the host.

## Upstream audit

Reused the same-session [Quimb review](2026-10-22-quimb-opportunities.md),
then reconfirmed installed signatures and actual CPU/GPU dispatch:
Quimb `1.15.1.dev96+g5656a9bb2`, Autoray `0.11.1.dev14+g014a3f69a`,
Cotengra `0.8.3.dev8+g8954240f2`, Symmray `0.4.1.dev20+ge8c907157`,
Torch `2.14.1+cu126`, and CuPy `14.1.1`.

Public `qr_auto(x, absorb, stabilized, min_aspect=8, min_work=2**20)` accepts
`absorb='rfactor'`. It tries unshifted CholeskyQR2 with an orthogonality check
and falls back to ordinary QR. Direct probes exercised Cholesky on Torch
CPU/CUDA and CuPy. No dependency was upgraded or patched.

Also checked official [Autoray source](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/), and
[Symmray source](https://github.com/jcmgray/symmray). Symmray's array-doc URL
was unavailable; installed source and its official repository supplied context.
Classification: **adopt** for this opt-in dense gauge path; **defer** wider
FIT/tree/default/autodiff changes. No compatibility shim was introduced.

## Validation

- `test_peps_gauge_qr.py`, `test_peps_full_update.py`,
  `test_peps_full_update_cupy.py`, `test_peps_pair_lbfgs.py`,
  `test_peps_exact_warmstart.py`, `test_public_api.py`, and
  `test_package_layout.py`, with `python -m pytest -q -o addopts=''`:
  **202 passed**, three existing warnings, 34.88 seconds.
- The new 29 cases passed again after final option-validation ordering.
  Checks cover Gram preservation, input/dtype/device preservation, actual
  Cholesky execution, small-shape and singular fallback, missing capability,
  and full-update agreement with ordinary QR and exact dense fidelities.
  Both complex dtypes ran on Torch CPU, Torch CUDA, and CuPy CUDA.
- Whole-tree Ruff, Markdown local links, and `git diff --check` passed.
  No full repository suite or new autodiff validation was performed.

## Timing evidence

The final isolated benchmark timed `ReducedPair.optimize` on seeded 3-by-3
PEPS pairs with fixed synthetic positive environment matrices, complex128,
two weighted-QR ALS iterations, and one CPU/BLAS thread. Each method had
one warmup and three measured repetitions; CUDA timing synchronized an
RTX A5000. Boundary construction and initial pair reduction are excluded.

| Backend | D | Gauge unfolding | Ordinary QR | Adaptive QR |
| --- | --- | --- | --- | --- |
| Torch CPU | 4 | 512 by 8 | 21.88 ms | 22.55 ms |
| Torch CPU | 8 | 4096 by 16 | 382.83 ms | 381.11 ms |
| Torch CUDA | 4 | 512 by 8 | 87.62 ms | 90.02 ms |
| Torch CUDA | 8 | 4096 by 16 | 211.52 ms | 222.67 ms |

Infidelities agreed within `4e-16`. These measurements establish **no useful
overall pair-solve speedup** for the tested cases; adaptive QR can be slower.
The earlier standalone NumPy QR speedup is not a full-update claim. Keeping
the option explicit allows larger-shape investigation without changing defaults.
An earlier exploratory timing overlapped regression execution and is not used
as performance evidence.

Temporary reproduction: `/tmp/pepsy-gauge-qr-benchmark.py`, with final results
in `/tmp/pepsy-gauge-qr-benchmark-final.jsonl`; focused test logs use the
`/tmp/pepsy-gauge-qr-` prefix. This note accompanies the implementation commit.
