# 2026-09-28 — Review of differentiable construction without SVD

- Scope: user requested thinking through fast MPO/PEPO construction with
  differentiable parameters and an explicit reusable computation graph.
- Branch / baseline: `develop`, `4e398e4`; existing working-tree edits preserved.
- Status: review and isolated prototype, not production implementation.
  Nothing staged, committed or published.

## Findings

Fixed-channel Pauli PEPO paths already avoid parameter-dependent SVD for usual
orders through four and uncapped localized histories. Generic homogeneous
higher orders and Torch MPO local factorizations still use SVD. Structural
compile/cache APIs do not currently guarantee machine-compiled tensor graphs.
A zero-residual local MPO probe exposed loss of a nonzero parameter derivative
through numerical zero pruning; it needs a regression and correction in the
proposed fixed-shape construction work.

## Evidence

Six CPU float64 local reconstruction/gradient checks passed for an isolated
fixed-index 2/3/4-site MPO prototype, including theta=0. Local forward
factorization was about 8–11 times faster in this small CPU microbenchmark;
end-to-end speed, backward time, GPU and graph capture remain unmeasured.
See the [design and measurements](../docs/development/notes/2026-09-28-svd-free-cluster-design.md)
for current-code distinctions, the zero-pruning probe, proposed architecture,
and acceptance criteria. No production code or API changed; no numerical
regression suite was rerun for this design-only review.
