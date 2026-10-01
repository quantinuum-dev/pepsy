# 2026-09-30 — TreeSampler further independent review

- Scope: user requested another review for correctness and missed speedups.
- Branch / baseline: `develop`, `414798e`, with the preceding uncommitted
  shared-density implementation and unrelated PEPS edits preserved.
- Status: new audit records and isolated `/tmp` prototypes only. No further
  production edits, staging, commit, or publication.

Found two more exact reuse opportunities: subtree-only keys for collapsed
messages, and whole-prefix grouping for later-sibling mixed densities.
The fullest prototype measured 2.894 s for the three-child-root and
3.493–3.499 s for binary-root synthetic 30-site chi=256 complex128 CuPy states,
8,192 samples/chunk_size=2048. All compared configurations matched and
probabilities passed reference checks. Production checkpoint unavailable.

Ran 648 additional small-state sampling checks across NumPy/CuPy, real/complex
dtypes, several arities, heterogeneous physical dimensions, physical roots,
chunk modes, forced sharing, and small memory budgets. All passed.

Confirmed wasted contractions on prototype cache exhaustion. Also measured
existing `_amplitudes` snapshot retention: chi=128 scoring/refresh retained
another 32 MiB per subsequent iteration with cyclic GC disabled. Isolated
visitor cleanup removed retention. Neither fix is integrated. The old OOM's
cause remains unproven, and full backend/gradient/device/memory coverage is
still required before promoting the prototypes.

See [detailed measurements, algebra, limitations, and reproducibility](../docs/development/notes/2026-09-30-tree-sampling-rereview.md).
No repository numerical suite rerun during this review-only turn. New local
documentation links and whitespace checked; production sampler verified
unchanged against the preceding snapshot. Upstream audit reused from the
same active task and unchanged environment.
