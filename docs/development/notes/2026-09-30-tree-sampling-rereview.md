# 2026-09-30 — TreeSampler independent re-review

Scope: the user requested another review of correctness and missed performance
opportunities. This follows the [factor-sampling review](2026-09-30-tree-factor-sampling-review.md).
All new algorithm changes and controls below remain isolated prototypes in
`/tmp`. Production sampler code was not changed in this review and still
matches `/tmp/pepsy_tree_sampler_shared_only.py` byte for byte. Reused the
unchanged environment and upstream audit from the preceding work.

## Two additional exact reuse opportunities

**Prototype: use subtree keys for collapsed child messages.** A returned
child message is the contraction of that child's tensors with its selected
physical values. It depends only on that subtree's configuration, even when
the probabilities used to select those values depended on outside measurements.
The previous prototype grouped the large first-child collapse using the
entire measurement prefix. That unnecessarily distinguished identical subtree
messages in different outside histories.

The refinement groups this collapse with exact mixed-radix keys restricted to
the completed child subtree. For a separate 2,048-shot diagnostic of the
binary-root case, one expensive collapse had 2,005 distinct whole prefixes
but only **107 distinct child-subtree configurations**. Another had 114 under
either key. The three-child root's large collapse had 474 under either key.

This narrower key is appropriate for the collapsed message, **not generally
for a conditional density**: the density also depends on outside measurements.
Do not apply subtree-only keys indiscriminately to density caches.

**Prototype: group later-sibling mixed densities too.** The earlier prototype
shared first-child density transfers, but its later-sibling mixed branch still
evaluated `X = rho @ K` and the child Gram contraction for every shot. The
refinement selects representative rows by the complete already-measured
prefix, computes these two contractions once per unique prefix, then gathers
the child density into shot order. A singleton incoming density broadcasts
over representatives. This preserves each shot's own uniform draws and does
not change chi, precision, physical order, or rank.

## Timings and larger-state checks

Same synthetic 30-site builder, actual chi=256, complex128, native CuPy,
8,192 samples, chunk_size=2048, state seed 19, sample seed 2, RTX A5000.
Synchronized GPU timing encloses the full sampling call, excluding state
construction/capture and scoring. Timing runs were serial. Cyclic collection
and release of unused pool blocks occurred between variants. These are
synthetic tree layouts, not the user's unavailable production 5x6 checkpoint.

| Root layout | Prior factor/cache prototype, remeasured | Add subtree keys | Also group later-sibling densities |
| --- | ---: | ---: | ---: |
| Three children | 3.355 s | 3.252 / 3.255 s | **2.894 / 2.894 s** |
| Two children | 9.967 s | 8.722 / 8.782 s | **3.499 / 3.493 s** |

The separate later-sibling comparison measured its subtree-key control at
3.334 s and 8.785 s respectively. The small three-child change from subtree
keys alone did not reduce distinct work at the measured large collapse and
should not be interpreted as an established algorithmic speedup there.
Later-sibling sharing improved both layouts, especially the binary-root case.

The three-child result is approximately **10 times faster** than the 28.90 s
current production implementation measured in the preceding integration audit.
That 28.90 s baseline was not rerun in this review. No completed current-code
binary-root baseline exists, so its table compares prototypes only.

All 8,192 configurations matched exactly between variants within each layout;
all probabilities passed rtol=1e-10, atol=1e-22. Independent bottom-up scoring
of 16 configurations had maximum relative errors 1.29e-14 and 2.13e-14 for
the three-child and binary-root cases respectively. All runs here completed;
this does not establish the cause of the preceding combined-run OOM.

## Expanded small-state checks

Ran **216 sampling calls per variant**, for the previous factor/cache
prototype, subtree-key refinement, and later-sibling refinement: **648 total**.
Checks cover NumPy and CuPy; float64, complex128, complex64; balanced maximum
arities 2/3/4; absent/present physical root; heterogeneous physical dimensions
`[2, 3, 2, 3, 2, 3, 2]`; sliced source arrays; chunk sizes None/1/13; explicit
and persistent RNG; 41 shots per call. Physical roots use supported top_arity=2.

The probes force the large-collapse sharing threshold to zero and the density
sharing threshold to parent dimension one, use a 128-byte factor tile target,
and a 1 KiB retained-cache budget. This exercises paths that small states did
not reach under the previous performance heuristics. They check exact seeded
configuration equality against current production sampling, returned
probabilities against an independently contracted dense statevector, and
cache release after each call. All passed. Maximum absolute probability
errors across dtypes were 3.25e-8, 2.78e-8, and 2.72e-8 respectively.

These do not establish Torch/gradient/native Symmray support for the full
prototype, multi-GPU behavior, or a global memory bound. Earlier checks of
simpler prototypes are not substitutes for those checks.

## Confirmed issues to address before integration

1. **Prototype cache exhaustion repeats contractions.** A targeted cache
   probe first stores two keys at its budget, then asks for one hit and one
   missing key. Observed transfer batch sizes are `[2, 1, 2]`: after computing
   the missing row, the failed admission path recomputes both requested rows.
   Results are correct but work is wasted. Assemble a temporary result from
   cached hits and already-computed misses without admitting new persistent
   entries. The retained-data budget still does not bound temporary copies.
2. **Existing production scoring retains GPU snapshots until cyclic GC.**
   `_amplitudes` leaves its recursive visitor self-reference intact. A native
   CuPy chi=128 probe disabled cyclic GC and performed four scoring/refresh
   iterations. Live allocation growth relative to setup was 0, 32, 64, and
   96 MiB. The discarded sampler stayed alive through a weak reference until
   collection; collection released 134,218,752 bytes. The equivalent sampling
   loop showed no growth or retained sampler. An isolated scoring control
   adding `try/finally: visit = None` also showed no growth or retention.
   These are live pool allocations, not merely unused cached pool blocks.
   This confirms actual GPU retention; it does not prove the old OOM's cause.
   Normal automatic GC was deliberately disabled to expose the lifetime.
3. **The prototype still needs production capability/memory handling.**
   Keep temporary allocation inside the captured device context; provide an
   exact fallback for mixed-radix overflow; preserve optional/native backend
   behavior; avoid mutable instance scratch flags for reentrant calls; and
   account for gathered batch tensors and layout copies as well as factor
   tiles. At B=2048, two remaining chi=256 child axes already require a
   2 GiB complex128 collapsed batch; grouping its arithmetic alone does not
   eliminate that gathered allocation.

Classification: **prototype** for the new reuse algorithms; **adopt next**
for the verified visitor cleanup and nonduplicating cache-full fallback,
subject to focused implementation checks. Neither fix was integrated here.
Keep the already-working canonical-region bookkeeping. Do not move the center
per shot or introduce numerical rank truncation to obtain these speedups.

## Reproducibility and validation scope

Temporary probes/results: `/tmp/pepsy_tree_rereview.py`,
`/tmp/pepsy_tree_rereview_memory.py`, `/tmp/pepsy_tree_subtree_keys.py`,
`/tmp/pepsy_tree_subtree_checks.py`, `/tmp/pepsy_tree_subtree_review.py`,
`/tmp/pepsy_tree_sibling_keys.py`, `/tmp/pepsy_tree_sibling_checks.py`, and
`/tmp/pepsy_tree_sibling_review.py`, with corresponding JSON/logs where
applicable. They use the prior prototype and state builder linked above.

No repository numerical suite was rerun and no production implementation was
changed in this review. Checked new local documentation links and
`git diff --check`; prior integration validation remains separate dated
evidence. Nothing was staged, committed, or published.
