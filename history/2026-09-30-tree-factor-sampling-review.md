# 2026-09-30 — Further exact TreeSampler speedup review

- Scope: user requested deeper review for a substantially faster algorithm.
- Branch / Git baseline: `develop`, `414798e`; current starting implementation
  includes the uncommitted shared-density optimization from the prior turn.
- Commit status: audit records only in this turn. No further production-code
  changes, staging, commit, or publication. Existing PEPS edits were preserved.

Isolated prototypes found a substantially better exact execution strategy:
carry structurally pure vectors, retain existing mixed factors, group exact
prefixes with compact integer keys, cache small child densities within one
sampling call, and reuse the large root contraction for repeated prefixes.
The representative 30-site chi=256 complex128 CuPy case fell from 28.90 s to
3.291/3.293 s for 8,192 samples in 2,048-shot chunks. Configurations matched
exactly, and probabilities passed reference/independent scoring checks.

See [evidence, algorithms, validation, and integration limits](../docs/development/notes/2026-09-30-tree-factor-sampling-review.md).
The pure/grouping prototype passed 192 small CPU/GPU calls; bounded-cache
and factor variants each passed 32 additional NumPy checks. The fullest
prototype is not yet portable/validated across all production backends.
Binary-root runs passed in a fresh process after one combined-run OOM;
that OOM's cause remains unproven.

Found an existing `_amplitudes` recursive-closure cycle retaining discarded
samplers until cyclic GC. It is not established as the OOM cause and was not
fixed during this review. Address lifetime and bounded-memory behavior along
with backend/device/key-overflow fallbacks before integrating the prototype.

No repository numerical suite was rerun because no production implementation
changed in this turn. Local documentation links and whitespace checks passed.
