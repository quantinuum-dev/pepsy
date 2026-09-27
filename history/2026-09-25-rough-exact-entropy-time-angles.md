# 2026-09-25 — Additional entropy-time angles stacked vertically

- Scope: add two entropy-versus-time plots below the current plot, targeting
  Δθ=π/8 and 2π/8, in the same `rough_exact.ipynb` cell.
- Branch / baseline: Pepsy `develop`, `80f451a`; examples `main`, `301159a`.
- Commit status: notebook working-tree edits only; nothing committed or published.
- The saved 12-angle grid contains neither target exactly. Asked the user
  whether nearest saved angles were acceptable; no answer arrived during the
  work. Proceeded with the explicitly stated nearest-angle assumption:
  theta_05=3π/22 and theta_08=21π/88. Notebook text and plot titles identify
  the saved angles; no interpolation or new simulation was performed.
- The cell now renders three sequential figures, vertically, for THETA_INDEX,
  theta_05, and theta_08. `ENTROPY_TIME_FIGURES` and per-index `FIGURES` keys
  retain all three figures; the existing primary export key remains intact.
- Executed setup and the edited cell in `/tmp`, verified three image outputs,
  9/8/7 available curves, notebook schema, and unchanged unrelated cells.
  Missing 5×5 traces are reported and will appear when their jobs reach them.
- Saved all three rendered outputs into the source cell. `git diff --check`
  passed. Numerical suites were not rerun for this notebook-only change.
- Exact π/8 and 2π/8 traces remain unavailable unless matching data are run.
