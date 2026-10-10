# 2026-10-10 — Fixed-chi sweep efficiency controls

The user requested the improvements from the
[performance audit](../../../history/2026-10-10-peps-sweep-speed-audit.md),
excluding adaptive boundary accuracy. Boundary chi remains fixed. The
[public controls](../../api/optimizers/sweep.md#fixed-chi-performance-controls)
separate numerical work budgets from contraction reuse.

## Implemented

- A bounded per-strip expression cache supplies fresh arrays and exponents
  on every contraction. Keys include ordered topology, shapes, output order,
  backend, and contraction policy. No tensor values are retained as constants.
  This is enabled by default and can be disabled for reference comparisons.
- Optional immediate endpoint skipping requires the previous solve to be
  both accepted and converged. An unfinished L-BFGS solve is never skipped.
- Absolute improvement tolerance, minimum inner trips, and patience extend
  the existing relative stopping rule. They do not weaken CG residual checks.
- Optional lazy boundary guesses reuse the existing BdyMPS lazy interface;
  requested boundaries receive the same initialization and fitting policy.
- `SweepOptimizer.optimize_strip` advances environments to a selected strip
  while freezing all other sites. The example runner can perform strip or
  pair-only cleanup between periodic global refinements. Gate counting spans
  timesteps and is restart-stable; each gate retains exact local acceptance.

The existing two-trip schedule, explicit CG, and non-lazy DMRG initialization
remain defaults. Approximation-changing stopping/cadence controls are opt-in.
The existing campaign runs from its frozen source and was not changed.

## Measurements

Torch CPU complex128, D=2, chi=8, one thread, same saved gate target at step 20:
median refinement times over three trials were 1.334 s uncached, 1.275 s with
expression reuse, 1.248 s with endpoint skipping too, and 1.139 s with lazy
boundaries as well. All four gave the same local fidelity to displayed precision
(0.998573188051), with site solves reduced from 128 to 104. This particular
candidate is worse than its pair initializer and would be rejected by the
trajectory runner; these timings establish equivalent computation, not an
improvement in physical fidelity. A strip-only candidate took about 0.197 s
and has different fidelity, so it is not an equivalent-work comparison.

A separate full timestep started all three variants from the same mature
step-8 checkpoint and evolved all 24 gates to t=0.9. True fidelity is measured
against the same saved uncompressed circuit reference:

| Schedule | Seconds | True fidelity | Site solves |
| --- | --- | --- | --- |
| Original full sweep after every gate | 36.701 | 0.994331450856 | 3072 |
| Reuse + endpoint skipping + lazy guesses + atol=1e-7 | 29.361 | 0.994331183649 | 2065 |
| Same controls; strip cleanup, global every 4 gates | 15.533 | 0.993850119082 | 696 |

All 24 refinements in each trajectory passed the exact local acceptance guard.
The second schedule is about 20% faster in this probe with a fidelity change of
2.67e-7. The third is about 58% faster but loses 4.81e-4 circuit fidelity versus
the original, demonstrating why it remains optional. These are one-timestep,
shared-machine measurements, not long-time or larger-D speed guarantees.
Temporary reproductions: `/tmp/pepsy_sweep_speed_after.py` and
`/tmp/pepsy_sweep_cadence_probe.py`, with corresponding JSON/log files.

## Upstream check

Installed versions: Quimb `1.15.1.dev90+g6a3906cbe`, Cotengra
`0.8.3.dev8+g8954240f2`, Autoray `0.11.1.dev14+g014a3f69a`, Symmray
`0.4.1.dev15+g0374aaa3c`, Torch `2.6.0+cu124`.

Reviewed the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html),
[Autoray source](https://github.com/jcmgray/autoray), and
[Symmray source](https://github.com/jcmgray/symmray). The Symmray array docs URL
returned an internal error; installed APIs and official source remain the
reference. Symmray ALS support is unchanged (explicitly rejected).

Classification: **adopt** public `cotengra.array_contract_expression` with
`strip_exponent=True`, and existing `BdyMPS(lazy=True)` behavior. Installed
signatures and exponent handling were inspected, then checked against uncached
Quimb contractions, changed values/shapes/output orders, and native backend
regressions. No dependency changes, private upstream patches, or compatibility
shims were introduced.
