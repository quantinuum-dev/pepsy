# 2026-11-05 — Six-site PEPO update stage timings

The exact cluster environment becomes the principal GPU cost at larger bond
dimension; CPU threading benefits the contraction and can penalize ALS.

- Scope: user-requested CPU/GPU measurements at χ6, 8, 10 and 12.
- Branch / baseline: `develop`, `0bd0824224a8c66e725cea04521a0ac794175b53`;
  examples `main`, `b80779ab034d7aa5dc3d43e57d793bf214d75e9a`.
- Status: profiling only; this record is uncommitted. No package behavior,
  production jobs, saved results or downstream files were changed.

## Method and scope

Use the [prepared path](../docs/api/bp.md#prepared-ordered-pepo-cluster-replay)
and the saved corrected χ6 residual from the 23-spin homogeneous D20 run:
`loop-cluster-chi6-b18-homogeneous-d20-eli23-fixed-gauges-scaled-gpu-20261105T182312Z/full/case-000-residual.pkl`
under the examples benchmark's `store/eli_global_cost/` directory. This
benchmark embeds its 23 physical spins in a 5×5 PEPO with 25 site tensors.

The saved χ6 state is measured separately. For the scaling ladder, replace
the cores with seeded complex128 dense tensors and identity gauges, assigning
every virtual bond χ6, 8, 10 or 12 on the same topology. These are synthetic
scaling states, not evolved χ8–12 circuit results. The lower-leg diagnostic
gate is diagonal, with entries `exp(-0.05j * [1,-1,-1,1])`.

Measure two six-site patches: `(3,1)—(3,2)`, adjacent to the lower boundary,
and the interior `(2,2)—(3,2)`. Apply a fresh single pair update to the same
starting state on each repeat. Use ordinary eager Torch ALS, cap10,
tol1e-8, Hermitian-only environments, local-fidelity tracking and the existing
QR/LQ, exact SVD and gauge extraction. The CUDA SVD driver is `gesvd`.
No whole-network exact or boundary-MPS readout is performed in this profile.

Exclude plan construction, gate/state construction and input copies; include
the working copies/gauge insertion performed inside the update. Discard one
cold call and report medians of three warm calls. The contraction program has
one miss and three hits for every measured configuration. Synchronize CUDA
around each timed stage. `QR/prep` includes pair preparation, working-network
copies and gauge insertion; it is not a timer for the QR kernels alone.
The environment column times the numeric Cotengra program, including its
boundary checks and operand preparation. ALS includes its initialization,
conditioning guards, stopping and fidelity measurement. Gauge extraction
includes its additional local QR/LQ and SVD. Other setup/validation remains
in the total. Independent column medians need not sum exactly.

Hardware: Xeon W-3375, 38 visible CPU cores, RTX A5000 24 GiB. Both CPU and
CUDA were accessible during the measurements; an earlier separate CUDA
availability check had failed. The workstation is shared. Software is Torch
2.14.1+cu126 with the same installed numerical dependencies as the
[prepared-replay record](2026-11-05-prepared-cluster-replay.md).

## Interior patch: one numerical host thread

All times are **milliseconds per pair update**. Synthetic states converge
in two ALS sweeps, with four Cholesky solves and no spectral fallback.

| χ | Device | QR/prep | Environment | ALS | Gauge extraction | Total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 6 | CPU1 | 3.9 | 113.6 | 43.7 | 4.7 | 167.8 |
| 6 | CUDA | 5.5 | 13.9 | 39.8 | 8.4 | 68.2 |
| 8 | CPU1 | 6.2 | 788.4 | 124.3 | 6.4 | 965.7 |
| 8 | CUDA | 5.2 | 68.0 | 52.7 | 7.5 | 136.1 |
| 10 | CPU1 | 10.8 | 3209.5 | 507.3 | 11.7 | 3850.3 |
| 10 | CUDA | 7.4 | 404.8 | 93.3 | 8.4 | 516.8 |
| 12 | CPU1 | 18.2 | 10894.6 | 1029.5 | 21.3 | 12194.0 |
| 12 | CUDA | 6.4 | 1007.0 | 95.9 | 7.7 | 1119.3 |

For the **saved evolved χ6** interior pair, four ALS sweeps are taken:
CPU1 QR/prep **4.1 ms**, environment **105.2 ms**, ALS **78.5 ms**, gauge
extraction **5.0 ms**, total **199.2 ms**. CUDA QR/prep **4.9 ms**,
environment **10.8 ms**, ALS **37.0 ms**, gauge extraction **6.8 ms**,
total **61.5 ms**. The second patch has the same qualitative ordering.

## Interior patch: 16 numerical host threads

Repeat on CPU and CUDA, again with three warm samples. Independent GPU
repeats show timing variability, especially at χ6–10; conclusions below
do not rely on those differences being a host-thread speedup.

| χ | Device | QR/prep | Environment | ALS | Gauge extraction | Total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 6 | CPU16 | 15.6 | 100.2 | 446.6 | 26.8 | 613.1 |
| 6 | CUDA | 4.4 | 11.2 | 25.2 | 5.6 | 48.2 |
| 8 | CPU16 | 29.7 | 220.9 | 648.6 | 39.6 | 918.1 |
| 8 | CUDA | 5.3 | 47.0 | 33.8 | 6.0 | 94.2 |
| 10 | CPU16 | 79.3 | 580.8 | 1035.8 | 208.8 | 2065.2 |
| 10 | CUDA | 5.7 | 283.6 | 64.7 | 6.3 | 362.9 |
| 12 | CPU16 | 87.4 | 1630.5 | 1017.5 | 340.2 | 3117.2 |
| 12 | CUDA | 6.8 | 1007.5 | 96.4 | 8.3 | 1122.0 |

For saved χ6, CPU16 totals **1050.2 ms**, with **105.7 ms** environment and
**866.3 ms** ALS. CUDA repeats at **63.4 ms** total. Increasing every CPU
operation to 16 threads is substantially worse for this small-bond case.

## Ten-sweep ALS control and memory

To distinguish contraction cost from early stopping, reuse the already
constructed interior problem and time ALS with cap10/tol0. Discard one call
and take three warm medians. All probes execute exactly ten sweeps.

| χ | CPU16 ALS, ms | CUDA ALS, ms |
| --- | ---: | ---: |
| 6 | 2079.1 | 73.9 |
| 8 | 3189.0 | 114.2 |
| 10 | 2785.2 | 239.4 |
| 12 | 3501.3 | 377.5 |

At χ12, CUDA contraction is still about **1.01 s**, versus **0.378 s** for
ten-sweep ALS. CPU16 ten-sweep ALS instead exceeds its **1.63 s**
contraction. The dominant stage therefore depends on the hardware,
thread count, conditioning and actual convergence iterations.

Measured peak Torch allocated GPU memory for interior patches is **107 MiB,
542 MiB, 2015 MiB and 5951 MiB** at χ6, 8, 10 and 12. This is process-local
Torch allocation, including the update, not the device-wide usage or only
the environment tensor. The χ12 environment alone is **81 MiB**; its
contraction intermediates cause the much larger peak.

## Findings and validation

- On saved χ6 CUDA, ALS is the largest stage; QR/LQ preparation is small.
- For the seeded χ12 CUDA update with two sweeps, environment contraction
  is about 90% of total time. Ten-sweep ALS does not reverse that ordering.
- CPU1 spends about 89% in the χ12 contraction. CPU16 reduces contraction
  from 10.89 s to 1.63 s, but does not similarly improve ALS or local gauge
  extraction. Separate CPU thread policies by stage are a candidate for a
  subsequent measurement, not an implemented optimization.
- Improving large-χ contraction paths/layouts is the main CUDA target.
  Compiling the small QR/LQ stage alone cannot remove that bottleneck.
- All 30 configuration measurements completed without exceptions. In all
  matched CPU/CUDA comparisons, iteration counts agree and normalized local
  squared errors agree within absolute 1e-10. Every measured local fidelity
  is defined; ALS reports use Cholesky without spectral fallback.
- This measures two diagnostic gates, not full-trajectory throughput or
  accuracy. No package code changed, so numerical test suites were not rerun.

Reproduction script: `/tmp/pepsy-cluster-stage-profile-20261105.py`.
The first command uses one thread for OMP/OpenBLAS/MKL/Numba:

```bash
python /tmp/pepsy-cluster-stage-profile-20261105.py
```

The second uses 16 threads for those libraries:

```bash
python /tmp/pepsy-cluster-stage-profile-20261105.py \
  --threads 16 --devices cpu cuda --pair interior --fixed-als-probe \
  --output /tmp/pepsy-cluster-stage-profile-20261105-cpu16
```

Activate the session's Python environment before either command. Full raw
samples and metadata are in `results.json` and `metadata.json` under the two
output directories. Timers instrument the existing module functions only
inside the profiling process; production code is unchanged.
