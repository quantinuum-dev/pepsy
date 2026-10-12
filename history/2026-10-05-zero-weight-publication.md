# 2026-10-05 — Integrate and publish the zero-weight SU correction

The scoped correction `3f316df` followed `d7d509b`. Its first push was rejected
because `origin/develop` had advanced to `6ab177b` with stabilizer/trajectory
work. Integrated both histories in `/tmp/pepsy-zero-weight-integration`, a
detached worktree, to preserve the original checkout's unrelated edits.
The only conflict was adjacent changelog entries; retained both unchanged.

Validation on the integrated tree: 240 gate/cutoff/scale/API/layout cases
passed and one CuPy test failed while allocating its initial CUDA array,
before exercising gate code. A separate CUDA probe likewise reported memory
allocation failure while the Gaugy workloads ran. After Gaugy integration
finished, the same CuPy test passed in 2.34 seconds without code changes.
Thus all 241 selected cases passed, with that resource-contention retry.
Full Ruff and whitespace checks passed; no full Pepsy suite was run.

This record accompanies the merge prepared for publication to
`origin/develop`. The original checkout remains at `3f316df` with its user's
uncommitted changes intact; the shared gate implementation is identical to
the integrated version. Git records the merge hash and remote publication.
