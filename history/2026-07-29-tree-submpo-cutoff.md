# 2026-07-29 — cap-aware Tree MPO cutoffs

- Milestone: Tree/MPO replay accuracy and shared TreeStab routing
- Branch / commit: `develop`

## What changed

- Tree subtree and MPO path replays now use lossless QR on bonds already at or
  below the active bond cap, while configured Quimb cutoff modes apply when a
  route expands a bond past that cap.
- Added a deterministic regression for tiny pre-existing state components and
  updated the Tree optimizer documentation.

## Why

- Distance-5, two-cycle SurfaceCode replay with `chi=64`,
  `cutoff=1e-12`, `cutoff_mode="rsum2"` changed from roughly 21.9% Tree
  logical errors to 0.098% while keeping the native sub-MPO route and the same
  layout.

## How it was validated

- Focused Tree/MPO/sub-MPO tests: passed.
- `tests/test_optimize_tree_stabilizer.py`: 57 passed.
- Notebook-derived distance-5, two-cycle native replay: 0.0009756 error rate,
  max bond 64.
- `py_compile`: passed.
- Ruff was unavailable in `/Users/rezah/envs/genpy`.

## Open questions / blockers

- Two broader Tree tests hit sandbox semaphore-permission errors while cotengra
  attempted to create loky workers; they were unrelated to this change.
