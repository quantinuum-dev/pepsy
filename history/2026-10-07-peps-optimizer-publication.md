# 2026-10-07 — Publish pending PEPS optimizer controls

User explicitly requested committing and pushing the remaining Pepsy changes,
in addition to the pepsy_examples publication already in progress. Baseline:
develop `dfdd9c6920d22699eaf000322d2755ca8493ec90`, which already contains the
solver-storage/rollback correction.

This commit includes the previously requested bounded approximate fidelity
diagnostic policy (default allowance 1e-3, raw values retained, objectives
unclipped, exact/nonfinite/gross-error safeguards preserved) and Torch global
boundary cutoff default 1e-10. Explicit metric options win; JAX keeps its
existing zero-cutoff JIT loss policy. The corresponding tests, API docs,
changelog and dated investigation/automatic-batch validation records are
included. Earlier journals describing these changes as uncommitted record
their status before this publication.

Fresh prepublication checks: global optimizer safeguards, JAX global PEPS,
and PEPS timing suites: **23 passed**, 10 warnings, 52.27 s. The unchanged
optimizer implementation also passed the earlier same-session **350-test**
solver/JAX/PEPS selection, and automatic D=4 to D=8 CUDA batching completed
t=0.3 with 10 fits and seven exact rollbacks. See the linked dated journals
for those measured results; they are not a full t=6 trajectory check.

`git diff --check` passed. Ruff is unavailable in cloudspace. No full Pepsy
suite was run for this publication. The 12-angle D=4 production sweep on
CUDA:0 remains active; no worker was stopped or restarted for Git operations.
