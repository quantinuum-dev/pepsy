# 2026-10-22 — Diagonal-wall alignment recheck

The current circuit and roughening implementations agree at matched settings,
but the rough notebooks do not all select the new geometry or angle grid.

- Scope: recheck `eiHt-compress/diagonal_wall` against the rough notebooks in
  `pepsy_examples/experiments/mps_magnetization/benchmark/plots`.
- Baselines: Pepsy `develop` at `5ff1fe6`; examples `main` at `c3a0279`;
  eiHt-compress `main` at `88497d2`. Both sibling repositories already had
  extensive working-tree changes, including the alignment implementation.
- Changes this session: this handoff only. No implementation, notebook,
  output or dataset edits; nothing committed or published.

## Findings

The active-site mask, removed corners, signed Hamiltonian, mean-field angle,
X-conjugate preparation, Strang step and fixed-domain imbalance agree.
Qiskit row ordering and Pepsy snake ordering require physical-coordinate
permutation before comparing state vectors.

Two distinctions qualify the earlier
[alignment record](../../pepsy_examples/experiments/mps_magnetization/benchmark/docs/2026-10-22-diagonal-wall-alignment.md):

- The diagonal-wall notebook configuration still offers 12 equally spaced
  offsets from zero through `3*pi/8`; the roughening sweep defaults to seven
  offsets `k*(0.533/6)`. The selected notebook offset is currently zero, so
  its single initial-state choice agrees with the runner default.
- `rough_exact.ipynb` and `rough_exact_mps.ipynb` still select the corrected
  October 7 campaign via `corrected_exact_run_roots()`. Inspected 4×4, 5×5
  and 5×6 metadata specify `wall_diagonal="x+y"`, with no `remove_corners`
  or `diagonal_convention` fields: historical full rectangles. These data
  cannot validate the new corner-removed system. The plotting guide's new
  geometry selection applies to the extrapolation notebooks; it does not
  migrate historical exact archives. `rough_9x10` also retains historical
  sweep paths; `rough_10x10_demo` explicitly illustrates `x+y` walls.

The circuit notebook currently builds 5×5 minus two corners (23 qubits),
`dt=0.1`, two steps. Its `N<16` guard intentionally skips local state-vector
readout. Saved rough exact comparisons select t=6. Matching comparisons need
the same lattice, geometry, angle and observation time.

## New validation

- eiHt-compress `diagonal_wall/test_helper.py`: **34 passed**.
- Examples `tests/test_diagonal_wall_alignment.py`: **23 passed**.
- Re-executed `/tmp/compare_diagonal_notebook.py`: eight direct Quimb-circuit
  versus untruncated Pepsy MPS cases, 2×3 and 3×3, both corner settings,
  offsets zero and 0.533, one `dt=0.1` step. Maximum amplitude difference
  **3.0043e-15** after coordinate permutation; MPS cutoff explicitly zero.
- Environment: shared Python 3.12; Quimb `1.15.1.dev96+g5656a9bb2`, Qiskit
  `2.4.2`, Autoray `0.11.1.dev14+g014a3f69a`, Cotengra
  `0.8.3.dev8+g8954240f2`, Symmray `0.4.1.dev20+ge8c907157`.
- Only dependency deprecation warnings in these focused tests. No full suite,
  production sweep, large-state convergence test or notebook execution.

Future alignment should explicitly choose the angle grid and matching new
dataset roots; changing labels cannot convert the historical dynamics.
