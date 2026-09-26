# 2026-09-25 — Entropy versus system size

- Scope: add entropy versus size in `rough_exact.ipynb` and clarify whether
  dividing by a linear lattice dimension bounds entropy by one.
- Branch / baseline: Pepsy `develop`, `80f451a`; examples `main`, `301159a`.
- Commit status: working-tree edits only; nothing staged, committed, or published.
- Added `ExactRougheningNotebook.plot_entropy_system_size` in the examples
  [plot helper](../../pepsy_examples/experiments/mps_magnetization/benchmark/plots/helper.py)
  and two cells in
  [rough_exact.ipynb](../../pepsy_examples/experiments/mps_magnetization/benchmark/plots/rough_exact.ipynb).
  Two vertically stacked axes show entropy in bits and S/S_max versus total
  qubits N=Lx*Ly at t=5, for THETA_INDEX, theta_05, and theta_08.
- Confirmed the runner uses log2 and cut N//2. The Hilbert-space maxima for
  4x4, 4x5, 5x5 are 8, 10, 12 bits. Area-law scaling with boundary length is
  not a universal maximum-entropy bound. The notebook includes the formula,
  a primary review link, and existing colored LaTeX conventions.
- Colors identify angle in the new size plot; timestep markers match the
  preceding entropy figures. Exact saved t=5 values only: dt=0.4 omitted,
  unavailable 5x5 dt=0.05 theta_08 left as NaN at execution time. No interpolation.
- Validation: focused notebook execution in
  `/tmp/pepsy_examples_executed/rough_exact_entropy_size.ipynb`; independently
  compared six series to checkpoint values and the 8/10/12 denominators,
  checked vertical layout, markers, and off-grid omission; visually inspected
  rendered PNG. Embedded only the new output; all 46 original cells preserved.
- `tests/test_plot_helper.py`: 39 passed, one existing empty-legend warning.
  Ruff on the helper and `git diff --check` passed. No full numerical suite.
- No changes to simulation jobs. No 5x6 entropy data are available.
