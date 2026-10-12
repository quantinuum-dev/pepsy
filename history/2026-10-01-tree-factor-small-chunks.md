# 2026-10-01 — Factor sampling with small chunks

- Scope: verify whether factor sampling is faster and uses less memory at
  small chunks; correct the earlier broad conversational recommendation.
- Branch / baseline commit: `develop` at local commit `608b664`.
- Commit status: the initial measurement draft was uncommitted. The user
  subsequently authorized finalization, commit and push (see below). Existing
  unrelated solver/test edits and their handoffs were preserved.

## Findings

The [measurement record](../docs/development/notes/2026-10-01-tree-factor-small-chunks.md)
contains reproducible construction, environment, timings, allocator method and
limitations. On synthetic 30-site native CuPy complex128 trees, factor used
61–63% less extra live GPU memory at chi 256 / chunks 16 and 128. Chunk 16
timings were nearly tied; chunk 128 favored factor by about 1.4 times.
At chi 32 / chunk 16 factor was slower; at chunk 128 its extra memory was
5.9 times standard despite an empty cache. Neither speed nor lower memory
is universal. Keep the existing standard default; factor is opt-in.

The factor chi-256 / 8,192-sample / chunk-1,000 run completed at median
4.20 seconds with 2.09 GiB extra peak. Standard chunk 1,000 / 1,024 samples
hit an OOM while another process held about 18 GiB; no comparable time or full
peak was obtained. Shared GPU activity limits timing conclusions. No
production checkpoint or Torch memory validation was performed.

## Validation

- Two synchronized timing repeats and one allocator-profile call per completed
  case. Same-chunk strategies had identical configurations and probabilities
  agreeing at `rtol=2e-10, atol=1e-22`; every call separately scored its first
  16 configurations bottom-up.
- Relative local links and `git diff --check`: passed.
- No numerical suite rerun: tracked changes are measurement documentation
  only. The earlier 512-pass focused correctness selection belongs to the
  [implementation session](2026-10-01-tree-sampler-correctness-resume.md).

Production defaults, workspace policy and cache policy remain unchanged.
Future default promotion needs evidence on the actual target trees and
backends; this handoff does not authorize that work.

## Large-bond finalization and publication follow-up

The user confirmed that large bonds are the intended workload and requested
finalizing, committing and pushing this TreeSampler work. Finalized the
[API example and selection guidance](../docs/api/sampling/tree.md) around
`strategy="factor", chunk_size=1000, backend="native"`, retaining the existing
128 MiB cache and 512 MiB workspace targets. The guide links the measured
small-chunk benefits and counterexample; factor remains experimental and
opt-in. No further numerical change was needed: the factor implementation
and its correctness fixes are already committed, most recently in `608b664`.

This guide, measurement record and handoff are included in the follow-up
commit titled `Document large-bond TreeSampler factor usage and benchmarks`.
The authorized push targets `origin/develop` and also publishes the preceding
local correctness commit. Publication is verified against the remote ref at
handoff; unrelated working-tree changes are excluded from staging.

New validation: relative documentation links and whitespace checks pass.
No numerical tests were rerun for this documentation-only follow-up; the
earlier 512-pass focused implementation checks and the measurements above
retain their recorded scopes. Actual-checkpoint memory and throughput remain
unverified.
