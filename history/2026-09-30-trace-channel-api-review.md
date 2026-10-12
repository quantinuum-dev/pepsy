# Trace options and compact-channel preparation review

Date: 2026-09-30

## Findings and fixes

Constructed active-block traces previously accepted `contract_opts` while
silently discarding `state_budget`. Compressed traces had the same gap.
All uniform, graph, square and routed-square entry points now share one
option validator. Sparse contraction accepts `state_budget`; materialized
contraction accepts mapping-valued `contract_opts` and rejects a custom
sparse budget. Compact channel plans expose the same controls. MPO channel
plans reject materialized contraction explicitly because that route returns
an MPO rather than a PEPO.

Full symbolic and algebraic channel preparation previously evaluated the base,
sample and tangent residuals even when no QR projection could occur. The
preparation path now validates those specifications, builds the exact
structural layout, and evaluates numerical references only if `max_bond`
actually cuts at least one sector. Exact reference, frontier and automaton
preparation still performs one base evaluation to discover its layout, but now
skips optional samples and tangent pairs when the cap cannot bind.

Reports previously labeled every full graph/square plan
`frozen-channel-qr`. They now name the exact reference, frontier, automaton,
symbolic or algebraic method, expose `projection_applied` and
`rank_selection`, and count `reference_evaluations`.

## Validation

The focused operator/channel suite passed 104 tests across PEPO trace,
compact channels, channel PEPO replay and algebraic preparation. Regression
coverage includes every PEPO layout, compressed traces, invalid option
combinations, non-mapping contraction options, exact nonbinding caps,
reference validation, report taxonomy and projected-plan counts.

## Publication state

These changes sit in the existing dirty Pepsy development tree, which contains
the broader unpublished compact-channel implementation and many user-owned
edits. They have not been staged or committed independently. Publish the
coherent Pepsy body only after deciding its intended commit boundary.
