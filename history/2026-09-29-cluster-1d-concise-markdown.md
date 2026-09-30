# 2026-09-29 — Concise, colored notebook Markdown

- Scope: shorten the 1D notebook explanations and use colored LaTeX formulas.
- Baselines: Gaugy Examples `main` at `dbabafa`; Pepsy `develop` at `c1ff0f8`.
- Status: uncommitted working-tree edits; nothing staged or published.

Updated only Markdown in
[cluster_1d.ipynb](../../gaugy_examples/pauli_gaugy/cluster_1d.ipynb).
Removed repeated formulas and narrative, shortened gallery captions, and made
NN, NNN, field, residual, and MPO quantities visually distinct. Markdown is
approximately 60% shorter by whitespace-delimited token count. Kept the model,
partition/embedding definitions, direct Pepsy construction, fixed versus compact
distinction, bond limits, and p=1 API restriction clear. Detailed API explanations
remain in the linked examples guide.

All code cells, saved outputs, figures, and notebook metadata were preserved
exactly against the pre-edit snapshot. All 74 math expressions passed strict
KaTeX rendering; notebook schema, local links, and whitespace checks passed.
No numerical rerun was needed for these presentation-only edits. The prior
[Pepsy facade handoff](2026-09-29-cluster-mpo-pepsy-facade.md) records the latest
full notebook run and numerical tests; those results were not rerun here.
