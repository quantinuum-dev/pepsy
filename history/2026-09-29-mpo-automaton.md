# 2026-09-29 — Cluster MPO automaton mode

- Scope: implement operator-aware automaton construction and show it in the
  single/joint 1D notebooks.
- Baselines: Pepsy develop c1ff0f8; Gaugy Examples main dbabafa.
- Status: uncommitted working-tree edits; nothing staged, committed or
  published. Other ongoing PEPO work and earlier notebook changes preserved.

## Changes

Added explicit `preparation="automaton"`, combining frontier transitions,
Pauli closure certificates and rational linear state reduction. Reuses the
existing shared Pauli/rational helpers and fused assembler. Full channels
need no SVD/QR; unsupported native/operator families reject explicitly.
Updated API docs, changelog, module map, status ledger and examples README.
Both notebooks show a commented automaton call and five-method comparison,
retaining NN + NNN, one boundary per run and the joint XX/YY/ZZ matrix order.

## Evidence and limits

[Implementation, measurements and test details](../docs/development/notes/2026-09-29-mpo-automaton.md).
The commuting Pauli regression reduces largest bond 25 to 9. In the requested
six-site models, automaton and frontier dimensions coincide: OBC p=5 is 209.
This is not global minimization or a universal performance gain. Fixed maps
remain valid across the declared scalar parameter family. Raw supplied
residuals outside the certified Pauli span are projected.

## Validation

- Affected Pepsy suites: 217 passed, 10 CUDA cases skipped.
- Downstream Gaugy channel/binding/materialization suites: 31 passed.
- Both complete notebooks executed separately under OBC and PBC, including
  independent cluster matrices, cut-rank bounds, single-boundary controls,
  one-MPO examples and Pauli checks. Worst automaton reconstruction error:
  6.17e-14. Fresh default OBC outputs saved in 25 single / 26 joint code cells;
  PBC copies remain under /tmp/gaugy_examples_executed.
- Ruff across source/tests/notebooks, notebook schemas, strict KaTeX
  (62 single / 56 joint expressions), edited-document local links and
  whitespace passed. Visually inspected both five-method bond plots.
- No full repository suite. Test compiler cache isolation fixed an initial
  combined-run cache-limit failure; no production numerical policy changed.
