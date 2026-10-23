# 2026-10-22 — Seven-angle diagonal-wall contract

The runner and sweep follow the diagonal-wall circuit; historical rough
plotting notebooks remain unchanged in this session, as explicitly requested.

- Scope: `eiHt-compress/diagonal_wall`, the examples roughening frontend and
  its sweep, with seven offsets `k*(0.533/6)` for `k=0,...,6`.
- Baselines: Pepsy `develop` at `5ff1fe6`; examples `main` at `c3a0279`;
  eiHt-compress `main` at `88497d2`. Existing sibling working-tree changes
  already implement the matching active-site physics and default sweep.
- Commit status: working-tree edits only; nothing staged, committed or published.

## Changes and physics

Updated the diagonal-wall notebook's configuration from twelve offsets through
`3*pi/8` to the requested seven through `0.533`. Source outputs are preserved.
Clarified the sweep CLI help and both README files. Expanded the independent
dense/tree alignment checks to all seven offsets and added an optional
cross-repository CLI regression against the actual Qiskit circuit helper.

The existing roughening defaults use `J=-1`, `h=1`, `hz=0`, open boundaries,
and the mask `(Lx-1)y <= (Ly-1)x`. Opposite corners `(Lx-1,0)` and `(0,Ly-1)`
are removed, including their fields and bonds. Preparation is uniform
`RY(theta)` followed by `X` inside, with `theta=-asin(1/4)+delta_theta`.
Imbalance divides by active N and has initial value `cos(theta)`.
The Strang step is `RX(h*dt)`, `RZZ(2*J*dt)`, `RX(h*dt)`.
Lattice size and evolution duration remain explicit run controls.

The new CLI regression launches the real sweep entrypoint and all seven child
entrypoints for both dense exact and DMRG MPS on 2×3 minus two corners, with
`dt=0.1`, depth two, complex128 and zero cutoff. It checks saved configuration,
physical preparation angle, initial energy, and final direct energy/imbalance.
Direct contraction observables are final-time measurements; earlier archive
entries deliberately remain NaN. Sampling traces are a separate observable.

Expanding the tree comparison exposed its implicit default cutoff: errors
up to approximately `4.2e-7` near the pure-Z angle were truncation, not a
physics mismatch. The exact-reference test now explicitly uses cutoff zero;
its numerical tolerance and production truncation defaults are unchanged.

## Validation and limits

- Final combined alignment and sweep regression run: **93 passed**, one
  existing compatibility-alias deprecation warning. `git diff --check` passes.
- Independent active-geometry/dense/tree checks: **33 passed** after setting
  the exact-reference tree cutoff to zero.
- Both seven-child CLI sweeps completed. Maximum final imbalance differences
  from Qiskit: dense exact `2.11e-15`, DMRG `2.22e-15`; energy-density
  differences below `1.0e-15`. Only small validation jobs under `/tmp` ran.
- Circuit helper suite: **34 passed**, dependency deprecation warnings only.
- Notebook schema, every code cell's compilation, and exact seven-angle grid
  checked without executing or modifying saved outputs.
- Changed Python files pass Ruff. Benchmark-wide Ruff still reports six
  existing errors in effective-model code/tests, outside this task.
- Full numerical suite, GPU production runs, long-time convergence and thermal
  calibration of the supplied figure were not performed.

See the earlier [recheck](2026-10-22-diagonal-wall-recheck.md) for the historical
plot-data distinction. This clarification supersedes any suggestion to
migrate those plotting notebooks as part of this task.
