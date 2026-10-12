# 2026-09-30 — Keep only the MPS magnetization package example

- Scope: user explicitly requested removing everything in Pepsy's own
  `examples/` except `MpsMagnetization/`.
- Branch / baseline: `develop` at `1b4ca92`.
- Status: working-tree edits only; nothing committed or pushed.

Removed the six cluster/benchmark Python scripts and the top-level examples
bytecode cache. Preserved the entire MpsMagnetization directory, including
notebook outputs and its existing support/cache files. Updated the current
examples index and removed the obsolete runnable benchmark command from the
MPO API guide. Sibling examples repositories and unrelated pending tests
were untouched.

Validation: file hashes confirm all retained files are unchanged; the only
remaining top-level examples entry is MpsMagnetization. Edited-document links,
Ruff (`src tests`) and whitespace checks passed. No numerical tests needed.

Dated notes and prior handoffs retain their historical script references.
The removed sources remain recoverable from Git, for example
`git show 1b4ca92:examples/cluster_mpo_bond_compression.py`.
