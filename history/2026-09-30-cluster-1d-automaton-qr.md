# 2026-09-30 — Add automaton plus delinearisation comparison

- Scope: user-requested automaton + QR delinearisation in the active
  [cluster_1d notebook](../../gaugy_examples/pauli_gaugy/cluster_1d.ipynb).
- Baselines: Gaugy Examples `main` at `dbabafa`; Pepsy `develop` at `c1ff0f8`.
- Status: uncommitted working-tree edits; nothing staged or published.

Added the existing API composition to the short example, cutoff sweep,
full-matrix and cut-rank checks, and bond/accuracy plots. Kept frontier + QR
for comparison and the preceding plot styling. Teal open triangles identify
the new curve. Updated the examples guide, owning API guide, and status
ledger. Package algorithms and the joint notebook were unchanged.

Both full single-exponential notebook runs (OBC/PBC) passed. The 27 existing
delinearisation tests, Ruff, notebook schema/preservation, local links, and
whitespace checks passed. Visually inspected both updated plots. Refreshed
only five affected OBC code-cell outputs; retained all other saved outputs.
No full package suite or joint notebook validation in this session.

[Measurements, upstream audit, and limits](../docs/development/notes/2026-09-30-automaton-delinearisation-example.md).
At p=5 OBC the largest bond falls from 209 to 50, matching frontier + QR;
there is no universal claim of an advantage over frontier + QR.
