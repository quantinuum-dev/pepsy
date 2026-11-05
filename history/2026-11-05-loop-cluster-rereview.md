# 2026-11-05 — Loop-cluster gate update re-review

The current dense six-site prepared route passes the scoped checks; three
previously recorded optional-path defects remain reproducible.

- Scope: user-requested review of loop compression and cluster gate updates.
- Branch / baseline: `develop`, `859fda0`.
- Commit status: this review note is uncommitted. No numerical code, running
  job, checkpoint or downstream repository changed.

## Confirmed findings

These reproduce the [earlier independent audit](2026-11-05-cluster-postpublication-audit.md)
on the current checkout; they are not new regressions from prepared replay.

1. **Full-system exactness can be lost with tree reduction.**
   [Region reduction](../src/pepsy/bp/reduced_update.py#L654) prunes the
   complete region as well as intermediate regions. A four-site PEPS chain
   with arbitrary SU closures retains only the active two sites despite
   `include_full_system=True`. Relative environment error against the exact
   contraction is 0.9445501; `tree_reduction=False` gives 2.08e-16. Protect the
   system-covering region from reduction. The current workaround is to
   disable tree reduction for a finite-system exactness check.

2. **Native U1 reduced environments lose empty charge sectors.**
   Three seeded fermionic U1 2x3 inputs with bond two reach
   [shape validation](../src/pepsy/bp/reduced_update.py#L228) with contracted
   shape `(3,3,3,3)` but reduced-pair shape `(4,4,4,4)`. Public six-site
   `compress_reduced_loop_cluster` fails with PSD projection enabled or
   disabled. Local SU reconstruction agrees to 2.53e-16, isolating this
   from the gauge-extraction step. Align the contracted native environment
   with all reduced charge spaces before Hermitianization and validation.

3. **Nonzero relative PSD floors depend on arbitrary scalar normalization.**
   The [boundary-message floor](../src/pepsy/bp/reduced_update.py#L172)
   uses `max(1, spectral_scale)` before message normalization. With
   `message_psd_floor=.001`, multiplying the same directed PSD messages by
   1e-9 changes the environment by relative error 1.13242. Floor zero gives
   7.94e-16. The open-metric floor uses the same lower bound. Use a genuinely
   relative floor with an explicit zero-spectrum policy; floor zero avoids
   the demonstrated message-rescaling discrepancy.

The dense prepared route uses a direct base patch, no tree-reduced added
regions, no native U1 arrays and zero PSD floor. These findings do not
demonstrate a failure of that route or of the frozen D=5 full-update jobs.

## Fresh validation

Using `~/envs/py312`, one OMP/MKL/OpenBLAS/Numba thread:

```bash
python -m pytest -q -o addopts='' \
  tests/test_bp_prepared_update.py tests/test_bp_reduced_update.py \
  tests/test_bp_torch_als.py tests/test_bp_local_gauges.py \
  tests/test_bp_cluster_patches.py tests/test_bp_compression.py
```

**212 passed, 20 warnings in 39.73 s**, including CPU and CUDA cases.
This is a focused selection; the full repository suite was not rerun.

Additional independent prepared-replay checks use nonsymmetric complex
nonunitary upper/lower gates, reversed site ordering, nonuniform gauges and
a nonzero network exponent. Untruncated physical operators match direct
Quimb physical-index gating to relative error 3.56e-15 on CPU and 9.93e-15
on CUDA. A boundary-gauge derivative through two truncated updates matches
finite differences to relative error 2.01e-9 with both native and stabilized
Torch linalg. The scalar stabilization and physical-input reconstruction
probes also reproduce their earlier passing results.

Installed versions remain Quimb `1.15.1.dev101+g1b64353d9`, Autoray
`0.11.1.dev14+g014a3f69a`, Cotengra `0.8.3`, Symmray
`0.4.1.dev25+gdc377f033` and Torch `2.14.1+cu126`. No dependency changed.
Ruff was not rerun because this review changes no Python implementation or
tests. Local link targets and whitespace were checked.

Fresh reproducer copies/results are under `/tmp/pepsy-loop-review-20261105/`;
the suite log is `/tmp/pepsy-loop-review-20261105-tests.log`. Existing audit
evidence was preserved. No fixes were implemented or published in this review.
