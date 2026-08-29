# Changelog

All notable PePsY changes are documented here.

PePsY follows [Semantic Versioning](https://semver.org/):

- **MAJOR** versions may contain incompatible public API changes.
- **MINOR** versions add backwards-compatible public functionality.
- **PATCH** versions contain backwards-compatible fixes and documentation updates.

## [Unreleased]

Changes for the next release should be added here before the version is bumped.

### Added

- Added the initial `TreePepsPlan` and `TreePeps` state API for PEPS-like
  tensor networks with 2D/3D coordinate tags, stable 1D logical tags, and
  validated spanning-tree virtual bonds, including PEPS-style `show`, tree
  canonical-center movement, `info_c` synchronization, and compression hooks.
- Added a hard three-virtual-bond TreePeps rank invariant, explicit local and
  maximum tensor-rank diagnostics, a workload-aware `TreePepsLayoutFinder`,
  and Quimb-style 2D Unicode state schematics that show retained tree bonds.
- Added `left_inds`-backed isometry metadata to `TreePeps`, with canonical
  region recovery, path-only center movement, QR-free canonical edge moves,
  and a center-oriented compression sweep that avoids a redundant full QR.
- Added tree-native `TreePepo` and `TreeSubPepo` operators with separate
  input/output legs, support/span metadata, exact dense-factorized gates,
  term sums, tree-bond fusion on application, expectation values, and
  optional canonical compression.
- Added `TreePepsOptimizer` with direct tree-geodesic gate replay and
  `sub_treepepo` span replay, lossless routing, `left_inds`-aware canonical
  preparation, localized compression, per-update bond reports, persistent
  `set_gates`/`add_gates` streams, common one-/two-/multi-site aliases, and
  validated state replacement.
- Added TreePeps parity helpers for rooted topology traversal, bond-growth
  estimates and preflight, batched local expectations, dense state-vector
  conversion, normalization, optimizer state aliases, and truncation reports.
- Completed TreePeps optimizer parity for state canonicalization aliases,
  span-local explicit compression, normalization controls, intermediate-bond
  preflight, profile and transient-bond diagnostics, layout convenience, and
  chi convergence sweeps with optional dense-reference fidelity.
- Added `compression_mode="dm"` to the tree, TreePeps, and TreeStab optimizer
  families. It applies Quimb's density-matrix-equivalent local `svd:eig`
  decomposition after the complete state/operator network is fused; the
  existing direct SVD mode remains the default and native fermionic trees keep
  their graded direct-compression path.
- Added explicit `TreeOptimizer` `tree_mpo_direct` and `tree_mpo_dm` modes.
  These build and route a true TreeMPO over the active Steiner subtree rather
  than lowering gates to a chain sub-MPO; `auto` now promotes gates wider than
  four qubits to the TreeMPO route, while bounded dense direct gates fail with
  an actionable mode recommendation.
- Added `OneDMap` center-out/inside-out traversal aliases and independent
  `TreePepsPlan.tree_order` seeds. Row-major, Hilbert, diagonal, and
  center-out orders can now guide legal degree-bounded virtual trees, and
  `TreePepsLayoutFinder` compares those deterministic seeds with weighted
  growth and reports the selected seed.

## [0.4.1] - 2026-08-27

This patch release consolidates the recent sampling, optimizer, operator,
backend, and documentation improvements developed on `develop`.

### Added

- `MpsStabSampler` now supports shared-prefix branch sampling, direct
  tableau/coefficient-MPS construction, basis-absorbing measurements, explicit
  physical/MPS qubit orders, probability queries, and branch diagnostics while
  preserving the coefficient-MPS backend for batched outputs.
- `PepsSampler` now provides exact, Quimb-MPS, and DMRG/FIT boundary proposal
  engines with conditioned ket boundaries, future marginal environments,
  prefix-grouped batches, and compact-row transfer caching.
- MPI shot-ensemble execution now includes checkpoint-aware orchestration,
  progress reporting, robust reductions, and native MPS/tree entry points.
- The documentation build now includes a generated API reference site, and
  the Guppy gate-stream adapter is available through the public interoperability
  API.
- MPO cluster expansions now expose a reusable compiled topology for ordered
  `exp(A) @ exp(B) @ exp(C)` products, explicit local `max_bond` control, and
  stabilized Torch/JAX autodiff factorization paths.
- Graph-aware MPO cluster expansions now reuse `ClusterLattice` connectivity,
  support long-range two-site clusters on 2D coordinate graphs, preserve
  ordered local exponential factors, expose trace and rank diagnostics, and
  map noncontiguous graph residuals into controlled MPO paths.
- PEPO cluster products now support ordered `exp(A) @ exp(B) @ ...` factors,
  direct physical traces, optional intermediate PEPO compression, and
  Torch/JAX-safe factor and step autodiff.
- `MPOBasis.from_square_lattice(...)` compiles coordinate-based Pauli terms
  through a reusable `OneDMap`, aligns reversed location/Pauli descriptions,
  and preserves backend autodiff coefficients while sharing MPO channels.
- `exp_mpo(...)` provides a term-centric operator/location/coefficient entry
  point that infers 1D/2D/3D layouts, accepts custom `OneDMap` orderings,
  accepts Pepsy-style Pauli-keyed mappings such as `{"XX": ((2, 3), J)}`,
  shares common MPO paths, and returns a compiled Quimb MPO by default.
- Higher-order MPO symmetry metadata now accepts case-insensitive compact
  symmetry names and charge-to-multiplicity mappings for degenerate physical
  sectors, while retaining the per-basis-state charge sequence form.
- Core package facades now resolve implementation modules lazily, and the test
  suite exposes explicit `core`, `optional`, and responsibility-based domain
  markers with a scheduled full-suite workflow.
- The top-level `pepsy` namespace is documented and guarded as a frozen
  compatibility facade; new advanced APIs should live in their owning domain
  or under `pepsy.experimental`.
- Accelerated contraction search is now optional through the `contraction`
  extra. Without it, reusable contraction optimizers fall back to Cotengra's
  built-in `sbplx` search and native Python pathfinders.
- General `MPOLocalOperatorTerm` inputs compile arbitrary dense multi-site
  operators through an exact operator-Schmidt MPO decomposition while keeping
  coefficient slots differentiable.
- `MPOPhysicalSpace` and `MPOBraiding` make local dimensions, Abelian sectors,
  grading, and odd-factor exchange signs explicit MPO construction metadata.
- `history_storage="reduced"` streams reachable products directly into the
  Algorithms 1--2 reduced history space without materializing raw virtual
  tensors, including the Algorithm 3 and 4 policies.
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

### Changed

- MPS and tree optimizer truncation defaults now use dtype-aware automatic
  cutoffs and explicit automatic cutoff-mode resolution. MPS DMRG/FIT defaults
  now use adaptive schedules, tighter automatic fit tolerances, and a
  two-site pair policy that can be overridden explicitly.
- MPS quality checks are opt-in, and optimizer seed handling no longer leaks
  MPS sampling seeds into contraction options.
- Torch linear-algebra defaults now select the exact SVD path, with the
  configured backend policy preserved across optimizer workflows.
- MPS, tree, and stabilizer optimizer streams now enforce backend, device, and
  applicable dtype compatibility at their stream boundaries; callers must use
  an explicit backend converter for intentional cross-backend payloads.

### Deprecated

- Backend helpers imported from `pepsy.tensors` now warn and direct callers to
  their canonical `pepsy.backends` namespace. `pepsy.experimental.mera` now
  directs callers to `pepsy.experimental.qmera`; the equivalent
  `pepsy.optimizers.mera` compatibility namespace directs callers to
  `pepsy.optimizers.qmera`. The legacy tensor constructor spellings
  `build_contraction`, `SpinfulFermionHubbard`, and `hrps_to_*`, the generic
  boundary spellings `normalize` and `infidelity`, the qMERA optimizer alias
  `QMeraParametricEnergyOptimizer`, and the stabilizer alias
  `MpsStabOptimizer` now also warn and identify their canonical names.

### Fixed

- Graph MPO cluster assembly now carries singleton backgrounds through skipped
  chain sites and retains products of disjoint long-range clusters with
  crossing or nested MPO spans; ordered MPO-basis products also reject
  mismatched chain geometry. Ordered PEPO products now use the same joint
  local-residual construction instead of multiplying independent factor
  PEPOs. MPO
  fixed-rank SVD dispatch now remains compatible with custom JAX registrations
  and switches stabilized Torch mode to match real or complex inputs.
- Term-centric MPO parsing now accepts integer coefficients without confusing
  them with lattice sites, rejects fractional shapes and coordinates instead
  of truncating them, and reports when semantic history cannot survive Quimb
  compression.
- MPO and Pauli supports now preserve site/operator pairing while sorting,
  multiply repeated-site factors in supplied local order, retain the Pauli
  phase, and reject Boolean or fractional site labels instead of coercing
  them to integers.
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
- Stabilizer optimizer and sampler branch state handling now keeps temporary
  conditional branches isolated from the live optimizer and separates Born
  probabilities from compression-fidelity diagnostics.
- CuPy-backed tree gate application and recent stabilizer optimizer routes now
  preserve their backend and consistency contracts.

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
