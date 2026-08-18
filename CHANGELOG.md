# Changelog

All notable Pepsy changes are documented here.

Pepsy follows [Semantic Versioning](https://semver.org/):

- **MAJOR** versions may contain incompatible public API changes.
- **MINOR** versions add backwards-compatible public functionality.
- **PATCH** versions contain backwards-compatible fixes and documentation updates.

## [Unreleased]

Changes for the next release should be added here before the version is bumped.

### Added

- Core package facades now resolve implementation modules lazily, and the test
  suite exposes explicit `core`, `optional`, and responsibility-based domain
  markers with a scheduled full-suite workflow.
- The top-level `pepsy` namespace is documented and guarded as a frozen
  compatibility facade; new advanced APIs should live in their owning domain
  or under `pepsy.experimental`.
- Accelerated contraction search is now optional through the `contraction`
  extra. Without it, reusable contraction optimizers fall back to Cotengra's
  built-in `sbplx` search and native Python pathfinders.
- MPS FIT convergence controls now use mode-neutral `fit_min_iter`,
  `fit_rtol`, and `fit_patience` names, with deprecated `mix_fit_*` aliases,
  and `stabilize_unitary` now covers DMRG, mixed MPO warm-up/fallback, and the
  standalone MPO/swap/permutation/SVD compression modes.
- PEPS boundary contractions expose typed per-fit convergence diagnostics,
  opt-in detailed timing, `return_info=True` on scalar norm helpers, and an
  information-preserving `peps_fidelity(..., return_info=True)` path.
- Dense PEPS DMRG boundaries now support cached two-site FIT sweeps with
  native SVD rank growth, independent norm/overlap bond caps, configurable
  sweep and truncation policy, and optional adaptive stopping across the
  boundary metrics, `SweepOptimizer`, and `PepsOptimizer` APIs.
- `SimulatorPlanner` and `recommend_simulator` provide non-executing,
  chi-aware rankings across MPS, tree, MPS-stabilizer, and tree-stabilizer
  circuit strategies using physical and dressed-frame support geometry.
- `TreeOptimizer` and `TreeTensorNetwork.compress_edge_` now accept
  `cutoff_mode`, allowing Tree truncations to use the same Quimb
  singular-value cutoff conventions as MPS truncations.

### Deprecated

- Backend helpers imported from `pepsy.tensors` now warn and direct callers to
  their canonical `pepsy.backends` namespace. `pepsy.experimental.mera` now
  directs callers to `pepsy.experimental.qmera`.

### Fixed

- Stabilizer planner diagnostics now explicitly describe when a cap changes
  the logical MPS width, preserving the warning contract for unavailable
  static-frame candidates.
- Native Symmray MPS compression now measures non-unitary target norms from a
  sector-preserving canonical active-span overlap instead of constructing a
  routed target copy, and native bosonic FIT reuses audited reversed-sweep
  environments. Infidelity samples identify the target-norm source route.
- MPS/FIT diagnostics now rebase unitary norm tracking after state replacement,
  manual normalization, and layout changes; profile DMRG target-norm work; keep
  unclipped norm-ratio diagnostics with an overshoot guard; and compute one
  terminal canonical-center norm per FIT sweep. Reused FIT objects reset
  per-run traces and split metadata, and full-chain entry points reject invalid
  sweep counts consistently. Canonical modes reject cyclic MPS inputs, local
  normalization reuses FIT's singleton center without an extra QR sweep, and
  mixed in-place commits preserve Quimb isometry metadata.
- Two-site PEPS boundary warm starts retain their requested future bond caps
  instead of treating the current rank as the cap; target replacement,
  per-call FIT overrides, and lowering `chi` now preserve explicit policy.
- `TreeOptimizer` non-unitary scale control now preserves removed normalization
  in the TTN exponent, and fast centre-based norm reads include that exponent,
  so `normalize_every=True` no longer changes the represented state.

## [0.4.0] - 2026-07-27

This release removes obsolete package-layout compatibility layers and keeps
advanced-domain discovery under the single lazy `pepsy.experimental` namespace.

### Removed

- Old flat modules such as `pepsy.core`, `pepsy.gates`, and `pepsy.optimize_mps`.
- The duplicate `pepsy.extensions` namespace and unused re-export leaf modules.
- The in-package benchmark directory and its orphaned benchmark test.

### Changed

- Repository agent guidance is concise and delegates domain invariants to the
  relevant skills.
- Active documentation now points to public simulation and sampling APIs rather
  than deleted benchmark scripts.

## [0.3.0] - 2026-07-24

This release consolidates the tensor-network API refresh and the new native
TreeOptimizer and symmetric-tensor workflows.

### Added

- Native fermionic TreeTensorNetwork evolution, observables, measurements, and
  state-versioned norm caching with explicit mutation invalidation.
- TreeOptimizer support for direct and MPO execution paths, including native
  subtree and multi-site operator routing.
- Backend-aware symmetric-tensor and Symmray sweep support with regression
  coverage for Torch-backed block arrays.
- Public trajectory, stabilizer tensor-network, fermionic, and VMC workflow
  APIs with corresponding documentation and examples.

### Changed

- TreeOptimizer execution modes are now limited to `auto`, `direct`, and
  `mpo`; unsupported legacy mode names fail clearly.
- Dense and native TreeOptimizer measurements use consistent gauge and norm
  diagnostics semantics.
- Progress reporting uses a common norm-infidelity proxy, and Symmray
  truncation diagnostics use the actual retained block spectra.
- Public imports and package documentation are organized around the current
  `pepsy.backends`, `pepsy.boundary`, `pepsy.operators`, `pepsy.optimizers`,
  `pepsy.sampling`, `pepsy.solvers`, and `pepsy.tensors` namespaces.

### Fixed

- Fermionic local expectations and norm calculations now agree with complete
  graded-network reference contractions, including nonzero hopping terms.
- Norm-cache invalidation covers public optimizer mutation and normalization
  paths, including constructor normalization.
- Symmray backend conversion, soft MPO bond caps, and blockwise discarded
  weight reporting are now handled without dense global-spectrum assumptions.

### Removed

- Stale benchmark and example artifacts that no longer represent the current
  public API.

## [0.2.0] - Baseline

`0.2.0` is the package metadata baseline that preceded this changelog. Earlier
changes were not recorded in a versioned changelog, so historical entries are
intentionally not reconstructed here.
