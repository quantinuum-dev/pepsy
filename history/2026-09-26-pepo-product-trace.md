# 2026-09-26 — Trace-first joint exponential-product PEPOs

- Scope: reuse Pepsy's joint exp(A) exp(B) exp(C) cluster machinery for the
  requested Gaugy local-cost companion.
- Branch / baseline: develop / 34393cbe6f5f1890b11430c722dbd9d0661b742c.
- Commit status: this entry accompanies the local implementation commit;
  no remote publication. Device-local instructions are excluded.

## Changes and rationale

Added `ActivePEPOBlocks.to_trace_network()` and `trace_nbytes`; fixed Torch
dtype inspection and integer-overflow-safe estimates. Batched located local
products without changing their algebraic factor order or cluster subtraction.
Updated API, implementation notes, changelog, and five focused regressions.

## Validation and limitations

125 focused cluster/ordered-product/API/layout tests passed; full Pepsy Ruff
and diff checks passed. Gaugy's 503 tests and reference notebook passed.
[Dated evidence](../docs/development/notes/2026-09-26-pepo-product-trace.md)
records installed versions, upstream classification, Torch exp accuracy, and
unresolved downstream MPS/CTMRG gradient failures. Full Pepsy suite, native
graded traces, and whole-objective JIT are outside this validation claim.
