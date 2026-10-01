# 2026-10-01 — Batched conditional-tree sampling experiment

- Scope: evaluate the user's proposal to carry a batch through the tree,
  similarly to MPS sampling, after they rejected the prefix-grouping direction.
- Branch / baseline commit: `develop` / `5eadd9f`.
- Commit status: no commit, staging or publication; earlier solver and tree
  sampler working-tree changes preserved. This turn added evidence documents
  and isolated `/tmp` prototypes, with no production numerical edits.

## Findings and validation

An exact prototype carries independent conditional trees as a protected
leading batch axis, samples physical centers, prunes measured branches and
uses batched QR for center moves. All 144 NumPy/Torch CPU/Torch CUDA/CuPy
sampling checks passed against the current sampler and dense probabilities;
four normalized Torch source-gradient checks passed.

On a synthetic 30-site chi=32 complex128 tree, 256 shots/chunk 128, the
literal implementation was 1.41x faster on NumPy but 26.6x slower on CuPy.
A chi=256/chunk 2048 probe hit the allocation guard at a 2 GiB conditional
tensor. Full timings, test coverage, caveats and the exact construction are
in the [experiment note](../docs/development/notes/2026-10-01-tree-batched-center-prototype.md).
These measurements concern the prototype, not a production checkpoint or
the earlier pure-vector sampler change. No full suite was rerun.

## Proposed follow-up

Keep the batch attached to compact conditional factors while sharing original
tree tensors. This preserves the user's traversal idea, but still requires
exact handling of correlations with unmeasured siblings and bounded
intermediates. Native Symmray/fermionic support and adverse rank/weight cases
remain unverified for the literal dense prototype. No integration was made.
