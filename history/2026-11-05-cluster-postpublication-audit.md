# 2026-11-05 — Independent cluster-update audit

Scope: user-requested Pepsy-level review of cluster methods, gauges and
logical consistency. Branch/baseline: `develop`, `4d50e53`; initially clean.
This is an audit report, not an implementation fix or publication. No
production simulation, restart, commit or push was performed.

## Confirmed defects and robustness gap

1. **Full-system selection can lose exactness under tree reduction.**
   `ReducedLoopClusterCache.regions_for` prunes the system-covering region
   when `tree_reduction=True`, despite `include_full_system=True`. On a
   four-site PEPS chain with arbitrary SU weights, four retained sites become
   two. The environment relative error is 0.94455 versus the directly
   contracted finite-system oracle; disabling reduction gives 2.1e-16.
   With `run_bp=False` and no gauges/messages, the same full-system request
   instead raises a missing-boundary-gauge error. Protect the complete region
   from reduction; retain the approximation only for intermediate regions.
   Source: [region reduction](../src/pepsy/bp/reduced_update.py#L644).

2. **Native U1 environment contraction drops required empty sectors.**
   A random fermionic U1 2×3 PEPS, bond two, alternating site charges and
   seeds 17/18/19 prepares reduced spaces of dimension four, but its contracted
   environment reports dimension three. The public
   `compress_reduced_loop_cluster` six-site path raises
   `open reduced environment has shape (3,3,3,3), expected (4,4,4,4)`.
   This occurs with PSD projection enabled or disabled. Local SU extraction
   itself reconstructs the state to 2.5e-16; the defect is subsequent
   environment charge-space alignment, not that extraction. Restore the
   complete reduced charge spaces before validation/Hermitianization, keeping
   native blocks and fermionic metadata. Source:
   [environment validation](../src/pepsy/bp/reduced_update.py#L206).

3. **Positive relative PSD floors depend on arbitrary message scale.**
   `_project_boundary_message` uses `max(1, spectral_scale)` before message
   normalization. With `message_psd_floor=.001`, scaling every directed PSD
   message by 1e-9 changes the reduced environment by relative error 1.1324.
   At floor zero, the same rescaling agrees to 7.9e-16. Small spectra are
   therefore treated with an absolute floor, contrary to the relative-floor
   contract. The open-metric projection uses the same lower bound. Normalize
   before flooring or use a genuinely relative spectral scale, with an
   explicit zero-spectrum policy. Sources:
   [message floor](../src/pepsy/bp/reduced_update.py#L164),
   [metric floor](../src/pepsy/bp/reduced_update.py#L2109).

4. **The package stream needs caller-managed scalar gauge stabilization.**
   `gate_loop_cluster` applies successive pairs without RMS-to-exponent
   extraction. A seeded 180-gate unitary 2×3 PEPO test at bond two fails
   after 78 completed updates, with gauge magnitudes reaching 8.7e245.
   Calling public `renorm_gauge(..., smudge=0)` after each pair completes all
   180, with maximum recorded gauge magnitude 1.35. A single batched stream
   also fails. This caller responsibility is documented, and the corrected
   downstream runner already handles it; a package-level per-pair
   `strip_exponent` option would make the multigate API robust directly.
   Source: [stream loop](../src/pepsy/operators/gates.py#L2437).

## Passing checks and limits

- Existing compression/reduced-update/patch/local-gauge suites: **137 passed**,
  including actual CPU/CUDA cases, in 15.03 s. No tests were weakened.
- Independent dense PEPS/PEPO reconstruction, exterior-gauge preservation,
  mixed physical-input streams and RMS representation changes agree at
  roughly 1e-14. Rectangular PEPOs with reversed active sites and complex
  nonunitary upper/lower gates agree with independently embedded finite
  operators to 9.2e-14 or better.
- Current six-site dense runs use zero extra loop corrections, zero PSD
  floor and explicit per-pair scale handling. The three edge-case defects
  above do not demonstrate corruption of the corrected recorded χ6 result.
- Raw reduced costs/norms exclude the network exponent: exponent 1.25 leaves
  reported target norm 4538.64 while physical norm squared is 1435244.90.
  Relative objectives and reconstructed amplitudes remain consistent; this
  mantissa convention should be stated when presenting absolute diagnostics.
- Native Z2 reconstruction passed the sampled seeds. Native Symmray solving
  remains a graded SVD adapter, rather than dense weighted ALS. A sampled Z2
  PSD-projection request was explicitly rejected by its existing graded
  spectral-clipping guard. These are separate coverage/capability limits.
- Installed dependency versions match the earlier upstream audit. The full
  repository suite was not rerun; the five previously reproduced baseline
  fermionic readout failures are separate from the new environment defect.

Reproducers and numerical JSON are under
`/tmp/pepsy-cluster-audit-20261105/`; the focused-suite log is
`/tmp/pepsy-cluster-audit-20261105-tests.log`. No numerical implementation or
running frozen-source campaign was modified during this review.
