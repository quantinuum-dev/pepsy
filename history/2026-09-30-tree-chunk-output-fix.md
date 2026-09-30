# 2026-09-30 — Fix TreeSampler chunk output retention

- Scope: user authorized fixing chunk output memory and preserving the selected
  tree backend/device after the careful sampler review.
- Branch / baseline: `develop`, `dacc200`.
- Publication: the user authorized committing and pushing the TreeSampler
  fix and its review evidence. This entry accompanies that commit; its exact
  revision is recorded in Git history. Independent PEPS edits remain outside
  this publication.

Resolved the review's P2 finding by filling one final output allocation,
releasing each completed chunk promptly, and reusing outputs when one batch
suffices. Broke the recursive visitor's reference cycle. CuPy sampling and
scoring now explicitly use the captured tree device. Added regression tests,
updated the API guide and the TreeSampler changelog entry.

[Implementation, measurements, and dependency audit](../docs/development/notes/2026-09-30-tree-chunk-output-memory.md).

Validation: **120 passed, 1 skipped, 1 failed** across sampler/API/layout tests,
including real Torch CUDA and CuPy checks. The skip requires two GPUs; the
failure is the known installed-version mismatch. Two final explicit host-output
checks passed. New lifetime/output-reuse tests fail against the old code.
Ruff and diff whitespace checks pass. Small NumPy/Torch CUDA/CuPy allocation
probes produced identical outputs with roughly 32% lower peak allocation.
Two-GPU and full-package validation were not run. No production jobs changed.

The patch helper failed with a filesystem sandbox launcher error; scoped exact
text replacements were used after attempting the required apply_patch workflow.

## Follow-up review

A second requested review found no new implementation defect. Added dated
verification evidence to the linked note: 96 forced-tile sampling calls across
all four CPU/GPU execution paths matched seeded reference samples and an
independent dense statevector. Warmed throughput differed by less than 0.4%
from dacc200 in a small D=8 probe. Eight successive discarded-result calls
on each GPU backend showed zero retained live allocation growth with cyclic
GC disabled. No further code change, commit, or publication was made.
