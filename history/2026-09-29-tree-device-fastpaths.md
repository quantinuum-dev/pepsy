# 2026-09-29 — Tree DMRG device fast paths

- Scope: user authorized fixing the performance issues in the preceding
  [Tree DMRG review](2026-09-29-tree-dmrg-step-review.md), then committing and
  pushing. Iteration budgets, stopping semantics, physical evolution, and
  thread defaults remain unchanged.
- Branch / baseline: `develop` / `f1b9924` initially. Independent square-PEPO
  work was committed as `4e7d08e` during implementation and is outside this
  change. Examples remain at `1021497` with unrelated notebook edits preserved.
- Publication: this handoff accompanies the authorized implementation commit.
  No production job was restarted or stopped and no production data changed.

## Changes

- Dense Torch FIT exteriors can use the exact identity shortcut when target
  and fitted arrays are the same object, their leg positions match, and the
  current canonical region proves the exterior is isometric. No device
  equality comparison is performed. Distinct arrays, changed gauges,
  differentiable tensors, and native Symmray states retain full contractions.
  Identity messages inherit dtype and device. NumPy's existing equality proof
  remains available.
- TreeOptimizer reuses the existing pre-update norm ledger scalar when its
  exponent is zero. Otherwise it obtains a stripped backend norm. TreeFIT
  retains a backend target mantissa until the first required convergence
  read, transferring it together with that iteration's centre-norm square.
  This removes a separate pre-fit scalar transfer and, on the common ledger
  path, the duplicate centre contraction. Host target norms and stripped
  exponent pairs, including NumPy pair arrays, retain their behavior.
- The convergence rule still reads one centre norm per iteration; no stopping
  decision is skipped or delayed. Public norm traces remain host numbers.
  New backend target-norm inputs are validated at their first readout.
- Single-site Torch unitarity certification uses a bounded 64-entry cache
  keyed by source identity, storage/view, dtype/device, tolerance, and mutation
  version. Strong source references prevent id reuse; ordinary in-place edits
  invalidate certification. Stream/state/layout replacement clears it.
  NumPy, native inputs and unversioned inference tensors are checked directly.
  Raw external-storage or `.data` writes must explicitly increment the Torch
  version or reinstall the gate stream, as documented and tested.
- Updated API docs, changelog, Tree skill references/catalog/manifest, and
  corrected remaining stale performance-reference wording about default threads.

## Validation

Existing Python 3.12 environment, local sources, CUDA hidden and CPU library
limits confined to validation subprocesses. The GPU was occupied by an existing
job (100% utilization when checked); no GPU performance experiment was run.

- Initial closest FIT/messages/stability/sub-MPO/zipup suites: **149 passed**.
- Broad Tree, TreePeps, stabilizer, trajectory and public API/layout selection,
  excluding slow/benchmark markers: **1463 passed, 39 skipped**, 151.34 s.
  Log: `/tmp/tree-device-fastpaths-broad-20260929.log`.
- Final expanded new regressions: **9 passed, 6 CUDA skips**, 2.48 s.
  These include unchanged-exterior/reference agreement, gradient preservation,
  in-place/external-version/reinstalled-stream gate edits, inference fallback,
  and host/backend norm input compatibility. The readout regression observes
  one two-scalar transfer followed by one scalar per remaining iteration.
  Eight repeated gate applications need one certification; an edit triggers
  another. Log: `/tmp/tree-device-fastpaths-new-final-20260929.log`.
- Complete downstream benchmark suite: **475 passed, 6 skipped**, 74.35 s.
  Log: `/tmp/tree-device-fastpaths-examples-20260929.log`.
- Real frontend comparison, all 12 angles, 3x2, chi64, two steps, NumPy/Torch
  CPU: **72 states**, minimum exact-reference fidelity 0.9999999999987095,
  max norm error 1.34e-15. All 336 multi-node fits retain `(2,2,1,1,1)` and
  five-iteration stopping. Torch identity shortcuts increased from **0 to 384**,
  matching NumPy's 384. Summary:
  `/tmp/tree_dmrg_review_20260929_vuu34sec/summary.json`.
- The 16 finite-chi stress cases still pass **1528 local-update checks** with
  maximum squared-residual increase 1.11e-16 and cached/reference discrepancy
  1.16e-15. Twelve fused-target cases remain within 1.32e-11. Log:
  `/tmp/tree-device-fastpaths-numerics-20260929.log`.
- An additional even-parity native U1U1 Torch DMRG probe matches NumPy to
  1e-10 with stabilized norm 1.0; native identity shortcuts stay disabled.
- Package Ruff, skill quick validation, 12-skill catalog validation, and
  diff checks pass. The installed Pepsy metadata is now 0.5.0, and the public
  version check passed; this task did not reinstall the environment.

The broad selection preceded the final host-pair compatibility and cache-clear
adjustments; the final expanded tests cover those changes. No full-package or
production GPU timing result is claimed.

## Upstream audit and limits

Reused the same-task Quimb/Autoray/Cotengra/Symmray audit and unchanged installed
numerical versions. Inspected Autoray's like-array creation dispatch and Torch
view/storage behavior. Official Torch
[autograd guidance](https://docs.pytorch.org/docs/2.14/notes/autograd.html) and
[increment_version documentation](https://docs.pytorch.org/docs/2.14/generated/torch.autograd.graph.increment_version.html)
describe tracked in-place versions, untracked external writes, and inference
tensors without counters. Runtime remains Torch 2.6.0+cu124.

**Adopt:** existing like-array creation and backend scalar arithmetic.
**Narrow guarded optimization:** Torch view/version inspection with a direct
certification fallback when a version counter is unavailable. **Defer:**
dependency upgrades, broader backend identity shortcuts, and altered convergence
or iteration policies. The new counts demonstrate eliminated work, not a
measured GPU speedup. Finite-chi global optimality remains unproven.
