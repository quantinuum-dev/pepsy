# 2026-09-30 — Optional chunks for TreeSampler

- Scope: user requested optional 1,000-shot Tree sampling chunks on cuda:1,
  preserving chi=256, complex128, and 8,192 total samples per time point.
- Baselines: Pepsy `develop` at `fb31c15`; examples `main` at `3bd0ada`, with
  the existing local completion-reporting fix preserved.
- Commit status: the user subsequently authorized committing and pushing
  these changes in Pepsy and the sibling examples repository. This entry
  accompanies that change; exact revisions are recorded in Git history.

Added optional constructor `chunk_size` and bounded first-child density
environment tiles in [TreeSampler](../src/pepsy/sampling/tree.py), with
[API documentation](../docs/api/sampling/tree.md), changelog, and focused
regressions. The sibling examples runner exposes
`--tree-sample-chunk-size 1000` and records the option only when active.
This addresses both batched-shot memory and the separate 64 GiB environment
plus copy that caused the actual CuPy OOM.

See [implementation, dependency audit, and validation](../docs/development/notes/2026-09-30-tree-sampling-chunks.md).
The three existing MPS GPU sweeps were left running. The failed Tree results
were retained; the retry uses a distinct `_chunk1000` output directory.

The retry passed its previously failing depth-2 sampling point and saved the
checkpoint with `tree_sample_chunk_size=1000`, `samples=8192`, native CuPy
complex128, and alternating-XY layout. Observed memory during sampling was
about 11 GiB on cuda:1. This validates startup and that failing point, not
completion of the full 12-angle run. Exact commands, PIDs, and provenance
are in the runtime handoff linked from the note.
