# 2026-10-13 — JAX boundary compression and FIT differentiation

- Scope: wrap up Torch compilation, mature NetKet/JAX contraction through
  Quimb methods and Pepsy FIT, review stable gradients and fixed-point
  literature, then commit and push the reviewed changes.
- Branch / baseline: `develop` / `9b2bef2`.
- Commit status at validation: scoped changes staged for the requested commit
  and push. Existing TDVP edits and earlier draft journals remain outside this
  commit. The remote was fetched and matched the baseline before publication.

## What changed

- Boundary amplitude builders honor explicit Quimb `method` and legacy `mode`.
  `dmrg`/`dmrg2` select one-/two-site Quimb fitting with two sweeps by default,
  deterministic zip-up guesses and JIT checks for static iteration controls.
  Plain FIT variants retain their upstream iteration budgets. Seeded SRC
  preserves a fixed sketch; unavailable native Symmray RNGs raise clearly.
- JAX's custom SVD VJP retains discarded vectors and uses bounded relative
  spectral reciprocals, preserving native forward factors and complex/rectangular
  conventions. Accuracy references use resolved spectra; gapped subspace
  projector checks separately cover retained degeneracy.
- Pepsy `FIT.run_eff` has one-/two-site unrolled and checkpointed JAX
  gradient checks. Native flat fermionic Quimb compression has independent
  eager/JIT and directional-derivative checks. A fixed-sweep NetKet SR and
  real-coordinate TDVP regression checks integration.
- Torch boundary zero-sector logs return zero phase and `-inf` magnitude in
  eager and exported/compiled inference.
- Public usage and the changelog were updated. The
  [review](../docs/development/notes/2026-10-13-jax-compression-autodiff.md)
  records literature, exact-versus-surrogate limits and an implicit-FIT design.

## Review decisions

A generic symmetric correction at retained degeneracy repaired paired
reconstruction but incorrectly changed SVD projectors. It was removed.
Finite factor gradients alone do not establish correctness for every
gauge-invariant contraction. Cutting a degenerate multiplet stays nonsmooth.
Implicit fixed-point gradients remain deferred: they require convergence,
gauge treatment and checked matrix-free adjoint solves. Few-sweep gradients
continue to differentiate the finite algorithm, including initialization.

## Fresh validation

Activated the existing Python 3.12 environment, using CPU JAX and explicit
single-thread limits for these bounded checks; dependencies were unchanged.

- New SVD stability, boundary compression and Pepsy FIT selection:
  **67 passed**, 171.81 s. Includes native fermionic methods and bounded
  NetKet SR/TDVP integration.
- Backends, JAX QR and Torch compilation/zero-log selection:
  **87 passed**, 29.51 s. Rank-policy warnings were expected.
- Existing NetKet TDVP, flat-Z2 and global-JAX selection:
  **65 passed**, 315.21 s, with upstream sample-count/deprecation warnings.
  This working-tree selection also exercises the preserved earlier TDVP edits.
- Additional resolved-SVD and invalid-option selection:
  **24 passed**, 16.93 s. Native SVD is no longer used as an accuracy oracle
  on the old repeated-spectrum cotangent test matrix.
- Public API/package-layout checks passed in the earlier combined selection
  (**101 passed**, 232.87 s). The final staged-tree integration/API check is
  recorded below after it completes, to separate committed behavior from
  existing unstaged TDVP work.
- Whole-source/test Ruff, whitespace checks and changed documentation links
  passed. Full collection was inspected (8,986 tests); the full numerical
  suite was not run. No new GPU benchmark or physical trajectory certification.

## Final staged-tree check

An isolated snapshot produced with `git checkout-index` was imported through
its own `src` path, excluding unstaged TDVP edits. Its fixed-sweep boundary
NetKet SR/TDVP regression, Torch zero-log regression and public API/layout
selection passed: **57 passed**, 38.50 s, with two compatibility warnings.
The staged diff excludes earlier TDVP work and journals. No full numerical
suite or independently repeated GPU selection is claimed.
