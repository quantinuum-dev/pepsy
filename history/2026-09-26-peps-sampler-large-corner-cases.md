# 2026-09-26 — Large PEPS sampler memory/CPU corner cases

- Scope: user requested actual 9×9 or 10×10 D=4 resource testing.
- Branch / baseline: `develop`, `a13031b`.
- Commit status: uncommitted; no implementation edits, staging, publication,
  environment changes, or production-job changes.

Ran four limited NumPy complex128 probes on evolved D=4 states. Factored 9×9
sampling passed with 651 MiB peak RSS; factored 10×10 sampling passed with
781 MiB after selecting a better exact-amplitude plan. Greedy 9×9 reference
sampling reached the 150-second CPU limit in its second two-shot proposal
chunk. Greedy 10×10 exact amplitude was declined by the harness because its
largest planned intermediate was 4 GiB; 32 serial randomized-greedy trials
reduced that to 64 MiB and the full public sample call then passed.

These runs used chi=16, chi_prime=8, repair="absolute", one CPU thread, and
bounded chunks. They did not test the sampler's default auto-hq planner or GPU
throughput. All successful public outputs were finite; shared-seed 9×9
reference/factored log proposals agreed within 2.85e-14. No large dense oracle,
full test suite, or long-run memory-leak claim. No OOM occurred.

See the [durable report](../docs/development/notes/peps_sampler_large_corner_cases.md)
for exact settings, measurements, limits, reproduction artifacts, and remaining
work. All probe processes ended. Earlier production CPU/GPU jobs were observed
running during testing and were not modified.

Documentation checks: local links in the new report/handoff and the note-index
entries resolve; `git diff --check` passed. No numerical implementation changed,
so existing numerical suites were not repeated.
