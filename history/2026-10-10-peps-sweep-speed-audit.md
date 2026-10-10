# 2026-10-10 — Measured opportunities for faster PEPS refinement

- Scope: review opportunities to make PEPS optimization faster and more useful.
- Branch / baseline: `develop`, `4adb5f9` (includes sweep commit `56dfe09`).
- Status: diagnostic probes and this uncommitted journal only. No solver,
  default, notebook, or running campaign was changed; no new commit or push.

## Reproduction and limitations

Read saved explicit-CG D=2 checkpoints at steps 8, 20, and 35 from campaign
`sweep_4x4_20261010_192437UTC`. Apply the next timestep's X/2 gates and first
ZZ gate on `(0,0)-(1,0)`, then prepare the same fresh reduced-pair FU initializer.
Fit the untruncated target with Torch CPU complex128, four boundary fitting
iterations, both axes, and the default explicit Hermitian CG solver. Initializer
and target are fixed within each probe; exact normalized local fidelity is
measured independently after refinement. Timing excludes pair preparation and
the final exact measurement. Background jobs remained active, so these are
shared-machine timings, not isolated throughput measurements.

The temporary reproduction script is `/tmp/pepsy_sweep_speed_audit.py`.
Thread results: `/tmp/pepsy_sweep_speed_audit.json`; accuracy probes:
`/tmp/pepsy_sweep_speed_audit_accuracy_{8,20,35}.json`. The latter use
`AUDIT_MODE=accuracy AUDIT_CHECKPOINT=<step>`. cProfile output is
`/tmp/pepsy_sweep_speed_audit_profile.txt`. Temporary artifacts may disappear;
the essential results are retained below.

## Measurements

With chi=8 and two inner round trips, the step-20 probe's median times over
three trials in varied order were:

| Torch/BLAS threads | Seconds per refinement |
| --- | --- |
| 1 | 1.305 |
| 2 | 1.332 |
| 4 | 1.325 |
| 8 | 1.386 |

One thread was about 6% faster than eight, with identical fidelity to roundoff.
Thread overhead is not the main bottleneck here. The general reason to measure
thread counts rather than infer speed from CPU utilization is also discussed
in [PyTorch's threading study](https://pytorch.org/blog/optimizing-libtorch/).
These results do not establish the best thread count at D=3,4,5.

The completed explicit-CG D=2 trajectory contains 1,200 gates: 48 exact pair
fits skipped refinement, 519 refinements were accepted, and 633 were rejected
by the final exact-fidelity guard. Thus 55% of attempted refinements supplied
no accepted improvement. Among 116,400 site solves, no CG failures were recorded;
28,842 returned at zero iterations. There were 1,883 invalid strips out of 9,184.
Those counts establish wasted work, not that every rejection shares one cause.

At step 20, pair fidelity was 0.999533973089. The same refinement gives:

| Boundary chi | Inner round trips | Local fidelity | Seconds |
| --- | --- | --- | --- |
| 8 | 1 | 0.998878378411 | 1.185 |
| 8 | 2 | 0.998573188055 | 1.323 |
| 16 | 1 | 0.999655900909 | 1.141 |
| 16 | 2 | 0.999659151466 | 1.400 |
| 32 | 2 | 0.999659151465 | 1.401 |

The chi=8 candidates would be rejected. Here a more accurate boundary makes
refinement useful, whereas extra inner iterations at chi=8 make it worse.
Chi=16 and chi=32 agree to about 1e-12 across all three checkpoint probes.
This is small-lattice evidence, not a universal rule that doubling chi is cheap.

At chi=16, one versus two inner trips retained 99.6%, 97.4%, and 97.9% of the
fidelity gain over the pair initializer at steps 8, 20, and 35. Their respective
times were 1.123/1.327, 1.141/1.400, and 1.129/1.375 seconds. However, at chi=8
and step 35, one trip worsened fidelity while two improved it. Therefore do not
replace the fixed two-trip policy blindly; evaluate adaptive stopping.

The step-20 profile recorded about 2.58 million Python calls. In the instrumented
2.28-second refinement, all CG solves together took 0.17 seconds; site optimization
took 0.95 seconds. Boundary FIT took 0.78 seconds and initial boundary construction
0.45 seconds. Contraction-expression construction alone accounted for 0.33 seconds.
These are cumulative profile categories and must not all be added together.
Profiler overhead means the proportions are indicative, not exact speedup bounds.

## Recommended order of work (proposed, not implemented)

1. Add an optional boundary-accuracy check and retry with larger chi when the
   estimated improvement is unreliable. Keep chi=2D² fixed in the existing
   comparison; evaluate adaptive chi as a separately labeled method. At larger
   sizes, use independent higher-chi checks rather than exact state vectors.
2. Add adaptive inner-trip budgets with a meaningful improvement threshold.
   Avoid oversolving below boundary accuracy; retain the existing rollback and
   true-residual checks. Benchmark full trajectories before changing defaults.
3. Reuse contraction expressions for fixed topology/shapes, passing current
   arrays each time. Reduce duplicate endpoint solves and unnecessary boundary
   initialization. Never reuse stale tensor values after state changes.
4. Compare local strip cleanup plus occasional global sweeps against a full
   lattice refinement after every gate. This changes the approximation schedule
   and needs an independent fidelity/runtime comparison.
5. Tune threads separately by D. Keep CG for the tested explicit quadratic path;
   replacing it with another optimizer does not address the dominant measured
   work. Larger-D solver comparisons remain incomplete.

No production behavior changed, so the earlier numerical regression suite was
not repeated. The probes completed successfully; this journal was checked with
`git diff --check`.
