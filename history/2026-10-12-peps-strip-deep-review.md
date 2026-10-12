# 2026-10-12 — Deeper strip and boundary review

- Scope: review the pending strip/global changes in more depth and fix remaining issues.
- Branch / baseline: `develop` / `192561a`.
- Commit status: working-tree edits only; nothing staged, committed, or published.

## Changes and findings

- Reproduced Quimb MPS crashes for split-gate targets containing several
  tensors at a site. A shared exact site/layer preparation now covers strip
  environments, metrics, and global norm/loss evaluation. Temporary tensor
  size can increase; lazy target storage and gradients are preserved.
- Reproduced Quimb environment construction indexing a nonexistent second
  plane on singleton axes. Such axes now expose no boundary cuts.
- Reproduced global objective evaluation renaming caller target indices when
  candidate and target share labels. Temporary metadata copies prevent this.
- Constructor-only boundary settings in run options now raise actionable
  errors. Sweep counts reject negative, fractional, and boolean values;
  boundary iteration counts must be positive. Zero outer cycles/round trips
  remain supported.
- Three boundary-engine numerical tests explicitly select their documented
  NLopt recipe instead of accidentally exercising the new ALS strip defaults.
  Their numerical acceptance assertions are unchanged.

## Validation

- New strip review selection: **25 passed**, including NumPy/Torch dense
  per-visit fidelity references, both axes and directions, singleton lattices,
  target preservation, metric gradients, and invalid run controls.
- Initial wider boundary selection: **209 passed, 3 failed**. The three
  failures were the legacy NLopt recipe assumptions corrected above.
- Combined strip/driver regression: **746 passed**, including the corrected
  three boundary-engine tests, all sweep/strip suites, batching, convergence,
  full update, and public API/package layout checks.
- Default global MPS layered-target tests: **2 passed**, checking dense loss
  and target norm, Torch/JAX gradients, and candidate/target data and metadata
  preservation. Broader global regression: **44 passed**, covering global
  batches, JAX defaults, failed-optimizer recovery, dtype/backend ownership,
  and native U1/U1U1 output.
- Full-suite fail-fast audit: **550 passed, 2 skipped**, then manually
  interrupted after 304 seconds in unrelated cluster factorization coverage.
  No failures were reported. This is not a full-suite pass; the collection
  contains over 8,700 tests, and this process started before the final global
  objective fixes. Final global behavior is checked separately above.
- Ruff and `git diff --check` passed after the implementation changes.

## Evidence and limitations

The unchanged isolated validation environment and earlier upstream audit
were reused; installed Quimb boundary/environment implementations confirmed
both compatibility constraints. See the
[implementation note](../docs/development/notes/2026-10-12-peps-strip-global-alignment.md#deeper-compatibility-review)
and the [earlier fixes](2026-10-12-peps-strip-fixes.md).
No long-run speedup or peak-memory improvement is established by this review.
