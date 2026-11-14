# 2026-11-13 — Geometry-aware Pauli-MPO DMT

- Scope: extend existing DMT to preserve configured physical lattice regions,
  including square-lattice L-shaped triples and plaquettes.
- Branch / baseline: `develop`, `d3612a15`. Existing unrelated sampling and
  changelog edits were preserved. This work was prepared for a DMT-only commit.

## Implementation

- `regions` on `PauliMPO.compress_dmt`, `apply_gate_dmt` and `PauliDMT`
  selects complete reduced operators by MPO-site indices. `None` retains the
  existing 1D contiguous-three-site implementation and its checks.
- `square_lattice_dmt_regions` uses the supplied `OneDMap` to generate open
  plaquettes, L-shaped triples, or all connected triples. Explicit lists also
  support other geometries and periodic neighborhoods.
- For each cut, protect identity directions and distinct Pauli factors on one
  side of each crossing region. Protecting the full corresponding matrix rows
  or columns is sufficient for every factor on the other side. Independent
  subspaces and the existing density/connected or raw DMT split provide the
  constrained factorization. This is conservative and can overprotect.
- Check every selected region through each split using log-scaled full Pauli
  panels. A failed standalone inplace sweep leaves its input unchanged;
  persistent gate evolution retains previously completed updates. Region
  verification costs `O(4**k)` output per `k`-site region. Geometry-specific
  Pauli environments are recomputed per split; only the static pattern plan
  is cached. No nonlocal geometry gate scheduler was added.

## Validation and limits

- `tests/test_pauli_dmt.py`, `test_pauli_dmt_cutoff.py`,
  `test_pauli_dmt_lattice.py`, and `test_pauli_dmt_optimizations.py`:
  116 passed, six pre-existing Torch rank-deficient QR warnings. A subsequent
  larger-lattice test parametrization was run separately: two passed.
- `tests/test_pauli_dmt_large.py -m 'not slow'`: 22 passed, 19 deselected.
- Final `tests/test_pauli_dmt_lattice.py`: 13 passed, including actual
  complex128 truncation after nontrivial virtual-gauge changes.
- The dense nine-site reference checks every selected region before and after
  each split for L shapes and plaquettes, including noncontiguous MPO sites,
  and establishes nonzero discarded residual. Tests cover insufficient bond
  failure, NumPy/Torch/JAX complex128, auto cutoff, traceless operators,
  adjacent gate evolution, and an exact nonlocal gate then DMT sweep.
- Independent product-mixture checks cover 12- and 16-site lattices and Ising
  energy. A 4x4 cap-48 run had nine plaquettes, maximum Frobenius error
  `3.22e-15`, trace error `2.22e-15`, and maximum reported discarded residual
  norm `3.48e-6`; cap 40 was explicitly rejected on protected budget.
- Public API/package layout: 54 passed, one failed due to the previously
  recorded installed distribution version `0.5.0` vs project version `0.6.0`.
  The shared environment was not modified. Ruff and `git diff --check` passed.
- Full-suite retry with `-x` stopped after 102 passed and seven skipped at the
  previously baseline-reproduced BP cluster test
  `test_loopunion_selects_both_plaquettes_and_clips_at_edge[where0-expected0]`;
  Quimb/Autoray dispatched a NumPy reshape through the Torch namespace.
- This is not a full-suite, long-time accuracy, GPU, arbitrary large-region,
  positivity, or performance guarantee. The Pauli-MPO compressed target uses
  explicit constraints but the selected conservative row/column subspace may
  need more bond dimension than a different constrained factorization.

## Upstream audit

Adopt existing public Quimb `array_split` and Autoray dispatch. Installed:
Quimb 1.15.0, Autoray 0.8.4, Cotengra 0.8.3.dev8+g8954240f2,
Cotengrust 0.2.1, Symmray 0.4.1.dev15+g0374aaa3c. Reviewed the linked Quimb,
Autoray, Cotengra and Symmray project sources/changelogs; Symmray's linked
array-doc page was unavailable. Native Symmray remains unsupported by DMT.

## Follow-up: OneDMap and MPS layout review

- `OneDMap` uses `snake` for a column-oriented snake,
  `snake-row-major` for a row-oriented snake, and `row-major` for a straight
  row traversal. The lattice-region generator uses `mapper.build()` in the
  same coordinate-to-chain direction as `MPOBasis.from_square_lattice`.
- Added optional `site_order`, matching `MpsOptimizer.logical_order`'s
  physical-position-to-logical-site convention. This maps regions to the
  storage positions of an already permuted density MPO. It is a snapshot,
  not a live link to the optimizer, and it does not build a density MPO.
  PauliMPO has no lattice metadata with which to catch mismatched mappings.
- Added explicit 3x2 position-grid and physical-neighborhood checks for all
  three map modes, a comparison with `MPOBasis`, and a real permuted
  `MpsOptimizer` case. The same nine-site physical density mixture is
  constructed independently in each of the three orders and in an explicit
  permutation; each run truncates and preserves all plaquettes against an
  independent physical-coordinate oracle.
- Review checks: the DMT, cutoff, lattice and optimization files passed 127
  tests with six existing Torch QR warnings; two focused square-lattice MPO
  mapping tests passed; public API and layout passed 54 with the known
  installed-version mismatch deselected. Ruff and `git diff --check` passed.

## Follow-up: larger-lattice review

- Added independent product-mixture checks for every plaquette and Ising
  energy on 5x5 float64 and 6x6 complex128 lattices, including automatic
  cutoff and actual residual truncation. A separate 5x5 test checks all 64
  L-shaped triples. The complete slow lattice selection, including the 8x8
  plaquette/energy regression below, passed four tests.
- An 8x8 complex128, rank-128, cap-112 run with nontrivial virtual gauges
  completed with trace error `5.85e-15` and nonzero reported residual. It
  took about 162 seconds on this CPU; this was a stability probe, not an
  independent verification of every plaquette. Cap 56 and 80 were rejected
  on protected-budget checks in separate lower-rank 8x8 probes.
- A separate 8x8 float64, rank-96, cap-92 run compared all 49 plaquettes
  and the Ising energy to an independent product-mixture oracle. Maximum
  plaquette Frobenius error was `6.37e-15`, trace error `8.44e-15`, and
  energy error `1.20e-14`, with a nonzero discarded residual. This one probe
  took 20.87 seconds on the same CPU; neither timing is a general benchmark.
- Focused DMT regressions excluding slow tests: 149 passed, 21 deselected,
  six existing Torch QR warnings. Public API/layout: 54 passed, one known
  installed-version mismatch deselected. Ruff and `git diff --check` passed.
- A further full-suite retry again stopped at the previously documented BP
  NumPy/Torch dispatch failure: 102 passed, seven skipped, one failed.
