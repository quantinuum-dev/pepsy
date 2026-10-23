# 2026-10-22 — Quimb feature and compatibility review

- Scope: review recent Quimb developments for Pepsy opportunities.
- Branch / baseline: `develop` / `5ff1fe6`.
- Status at review: notes only; no implementation or installation performed.
  These notes are included in the subsequent user-requested adaptive-gauge
  commit; see the [implementation handoff](2026-10-22-peps-gauge-qr.md).

The [audit](../docs/development/notes/2026-10-22-quimb-opportunities.md)
records exact upstream revisions, adoption/prototype/defer classifications,
and isolated CPU probes. Upstream fixes correct lost exponents, including a
100x error reproduced through Pepsy's unnormalized boundary measurement.
New exact-zero handling also needs a compatible Cotengra build. Adaptive QR,
cluster-boundary measurements, batched RDM reuse, and edge-induced loops are
prototype candidates; several other changelog features are already integrated.

Validation: small source-isolated numerical probes, local Markdown links,
and `git diff --check`. No full suite, GPU, or gradient validation. The shared
development environment and Pepsy implementation are unchanged.
