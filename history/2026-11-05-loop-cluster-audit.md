# 2026-11-05 — Local-cluster implementation audit

User requested a bug audit after χ6 homogeneous depth20 gave cost 0.864885,
worse than χ2. Branch `develop`, baseline `ee22198`, uncommitted changes.
No staging, commit, publication or installed-library edit occurred. Reused
the maintained BP skill and unchanged-environment upstream audit.

Two defects were reproduced before fixing:

- Active gauge extraction used a bare reduced-factor SVD, although the
  physical QR/LQ work copy includes only sqrt(lambda) on exterior legs.
  It preserved the physical state but supplied incorrect subsequent boundary
  weights. An untruncated three-site chain embedded in PEPS gave normalized
  spectrum [0.91138987, 0.41154405], versus the independent Schmidt spectrum
  [0.92985918, 0.36791563]. The PEPO variant also failed. The zero-cluster
  gate stream differed from native SU by 17.6% after its first truncation.
- Tree-cache keys sorted operands. Reusing a Cotengra tree after tensor
  permutation returned a wrong scalar on a closed three-tensor network.
  Keys now retain operand order. A frozen production-prefix audit recorded
  252 calls, 139 hits and zero reordered hits, so that sample does not
  attribute the production failure to the cache bug.

The corrected local extraction restores endpoint factors, includes the
remaining exterior sqrt(lambda), performs endpoint QR/LQ and a bond-sized
SVD, then removes full exterior weights on their support. Only the two
active cores and active gauge change. State amplitude, fixed spectator
arrays, exterior gauges, backend/device and native charge spaces are retained.
Hermitian-only N_env and the exact CUDA SVD policy are unchanged.

Independent truncated SU-metric ALS checks match the full weighted-SVD
oracle: relative optimum 0.03067871687502619, versus Quimb/QR results
0.03067871687502624/0.03067871687502621. Both costs decrease monotonically.
The corrected zero-cluster stream agrees with native SU within 6.5e-7 after
18 random gates, versus 17–24% discrepancies before the fix.

Validation: **142** focused compression, reduced-update, local-gauge,
cluster-patch, selected native Symmray and SymPEPS cases passed; **6**
downstream complete-operator/prefix/preflight cases passed. CPU/CUDA
Schmidt tests execute. Full-suite testing was not run; known unrelated
baseline fermionic readout failures remain outside scope.

Earlier production trial artifacts remain frozen and unchanged. A fresh
same-control χ6 GPU rerun is required before reporting the corrected
method's long-time behavior. The corrected frozen trial launched at
18:19:22 UTC, PID 2250878, with the original controls. See the
[downstream failure-test record](../../pepsy_examples/experiments/mps_magnetization/benchmark/docs/2026-11-05-eli-loop-cluster-chi6-su-failure.md).
Full Pepsy Ruff and whitespace checks passed. Earlier benchmark records
are explicitly annotated with the gauge defect and remain preserved.

The weighted-but-unscaled attempt then exposed an integration omission:
unlike the existing SU driver, the example cluster driver did not move each
active gauge's RMS into the network exponent. Local norms grew past 1e228
and overflowed before source slice one completed. The runner and profiler
now call public `renorm_gauge` after every pair, preserving the full operator
amplitude. Six downstream CPU/CUDA complete-operator/prefix/preflight cases
pass with explicit unit-RMS gauge checks. The failed attempt is preserved.
A fresh weighted-and-scaled trial launched at 18:23:19 UTC, PID 2262537.

That same-control corrected χ6 trial completed: exact scalar cost
**0.05162553834410777**, versus **0.864884908921097** before the audit,
and matched-target HRS **0.0476403551 ± 1.27e-5**. Boundary18/36 costs
0.05162555247249179 / 0.05162555249445078 differ from exact by about 1.4e-8.
Construction took 588.02 s. All 25 site tensors/40 gauges remain CUDA;
RMS scales are unity to roundoff and the extracted amplitude remains in
network exponent 8.511193258727571. No PSD projection was introduced.

484/5,040 fits stopped at the ten-sweep cap. Three fixed CPU environments
from source-block10 improved their normalized errors by at most 5.5e-8
from 10 to 30 sweeps, and at most 1.9e-12 from 30 to 60. This sample does
not certify global convergence; the full matched-control run kept cap10.
Independent six-site PEPS/PEPO trial-state error checks also pass. Earlier
results are preserved and annotated rather than overwritten. Ruff and
both repositories' whitespace checks pass; no full repository suite ran.
