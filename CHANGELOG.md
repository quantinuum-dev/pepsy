# Changelog

Release highlights. See the [migration guide](docs/development/api-migration.md)
before upgrading: 0.x minor versions may change APIs and defaults;
patch versions remain compatible.

[Full notes through 0.6.0](https://github.com/quantinuum-dev/pepsy/blob/1717c1b97b51919fb786ef2eb8116680d99002d8/CHANGELOG.md)

## [Unreleased]

- **Lattice Pauli DMT:** accept configurable physical-region site sets and
  generate square-lattice plaquettes or connected triples; protect their
  reduced operators during DMT compression and adjacent gate updates. Map
  permuted MPS storage positions with `site_order`.

- **Pauli DMT:** add explicit density/connected and raw-operator protected
  compression through `PauliMPO.compress_dmt`, adjacent local
  `apply_gate_dmt`, and `pepsy.optimizers.PauliDMT` second-order evolution.
  Reuse Quimb decompositions and Autoray backends; preserve three-site partial
  traces through truncation, with separate support/residual diagnostics,
  compact reconstruction and persistent trace/canonical environments.
  Rescale small centres/norms, retain protected Schmidt support, and verify
  standalone sweep marginals with log-scaled environments; reject insufficient
  precision rather than returning an unchecked local error.
  Use directional final QR, reuse unchanged SVD/prepared gates locally, and
  carry the canonical centre through serpentine parity sweeps.
  Support dtype-aware `cutoff="auto"` and Quimb cutoff modes on the
  unprotected spectrum, with physical-norm units for absolute policies.
  Avoid refactoring a fully protected bond for positive cutoffs and detect
  excessive per-gate trace defects for resolved nonzero-trace inputs.

- **Pauli MPO / DAOE:** add exact Pauli-weight damping through
  `PauliMPO.apply_daoe`, preserving native cores and separating filtering from
  SVD compression. Expose reusable `PauliDampingMPO` objects with explicit
  dimension-four Quimb MPO export, independent diagonal-core copies, scalar
  factor queries and `PauliMPO.apply_damping`. Fix even-chain native
  subtraction/commutators, real-core complex promotion, dtype/device-aware
  constants and gradient-preserving
  dense readout. Keep in-place channels native, invalidate stale gate/channel
  reports and reject noninteger gate sites. Partial-SVD diagnostics now include
  the uncomputed spectrum via reconstruction residuals. A scalable DAOE
  evolution driver remains future work.

- **Native BP open-rho sweeps:** treat empty Symmray correction blocks as
  zero trace instead of attempting an invalid dense conversion. Coverage
  includes U1, U1U1, Z2, and Z2Z2.
- **Native fermionic gate exponentials:** retain identity on zero-generator
  charge sectors before fusion, fixing generic spin/interaction gates and
  Etienne replay without changing Hubbard Hamiltonian coefficients.
- **Native one-site FIT:** correct each virtual environment leg's parity
  independently after reversed gates. Native MPS initialization also avoids
  dense identity scans that lack charge-index metadata.

- **Full-update boundary default:** two-site full update now uses direct
  boundary compression. Select `fit_mode="dmrg"` for the previous default;
  finite-cap environments and numerical results can differ.

- **Full-update gate ordering:** `PepsOptimizer` now defaults to smart ordering
  for two-site full updates. Pass `gate_order="input"` to retain the previous
  default, and submit separate runs for source/depth barriers.

- **CUDA full-update synchronization:** collect SVD-rank and norm-metric
  diagnostics in small grouped transfers, reuse an evaluated initial loss,
  and classify fixed CUDA gates in bounded batches for smart/row/column
  ordering. Keep exact gate-dependency checks and ALS stopping criteria.
- **Full-update SU warm starts:** use Quimb's public single-index diagonal
  multiplication so retained weights work on Quimb builds without the
  plural-index helper.

- **Full-update iteration default:** `PepsOptimizer` now defaults to a
  maximum of 20 pair-solver iterations instead of 50. Explicit
  `full_update_kwargs['max_iterations']` retains precedence.

- **Optional joint cluster updates:** ordinary and prepared PEPO cluster
  streams accept explicit `both` entries or opt-in `fuse_pepo_legs=True` to
  fit adjacent opposite-leg gates on one bond together. No reordering or
  pairing across call/source barriers; separate updates remain the default.
  Prepared replay records physical actions and joint fits separately.

- **Gate-dependent PEPO reduction:** one-sided cluster and full updates keep
  only the acted physical leg in reduced QR/LQ factors and trace the untouched
  leg on Q into the environment. Joint upper/lower fits retain both legs.
  `prepare_reduced_bond_pair(which=...)` exposes this grouping; its omitted
  selector preserves the full-physical compression layout. Finite-bond fit
  results can change; source gate order, scale and gauge extraction are retained.

- **Prepared cluster selection:** `PreparedLoopClusterUpdate` accepts
  `cluster_mode`, `base_radius` and `fillin`. Graph distance one with fill-in
  gives the rotated twelve-site 4×3 bulk patch, including correct edge clipping
  and batch read footprints. The package default remains the six-site patch.

- **Loop-cluster replay overhead:** `materialize_physical=False` skips unused
  physical-network snapshots in reduced gates, gate streams and prepared
  replay; existing defaults retain them. Dense ALS results expose the scalar
  `target_norm` already computed during solving for diagnostic reuse, without
  caching mutable problem tensors or changing ALS/stopping/gauge policy.

- **Independent cluster batches (experimental):** opt into
  `PreparedLoopClusterUpdate(..., batch_size="auto")` to discover compatible
  disjoint ready clusters from gate/lattice dependencies and current workspace;
  integer caps retain consecutive grouping. Batch patch contractions and Torch
  ALS while preserving per-pair conditioning,
  rank, stopping, gauges, exponent bookkeeping and source barriers; sequential
  replay remains the default. Preparation and final gauge extraction remain
  sequential, and AOT is unsupported with this option.

- **Prepared cluster replay:** `pepsy.bp.PreparedLoopClusterUpdate` binds
  dense Torch PEPO gate layouts and six-site Cotengra contraction programs,
  with fresh tensor/gauge values, bounded caches and rank-growth rebuilds.
  Preserve source barriers, gate order, pair truncations and ALS policy.
- **Reduced-update gradients:** clone Torch targets and SU gauges without
  detaching; Autoray's generic Torch copy detached singular-value derivatives
  from reconstructed operators, despite preserving forward values.

- **Dense Torch cluster ALS:** opt into `solver="torch"` for batched physical
  RHS contractions and certified Cholesky solves with the reference spectral
  fallback. Optional `torch_backend="aot_eager"` captures array contractions
  and backward, with separate charts for autocast state/dtype; eager remains
  the default. Preserve QR/LQ and stopping policy.
- **External reduced environments:** preserve gradients when a raw Torch
  environment is supplied; unwrap labeled tensors without using array
  `.data`. Raw NumPy/backend environments also retain their array types,
  with labeled Quimb views constructed when needed by its ALS solver.
- **Local cluster diagnostics:** optional normalized complex-overlap fidelity
  on reduced ALS results and `pepsy.bp.LoopClusterUpdateTracker` for scalar
  records across streams. Products of local fidelities are heuristic, not
  global fidelity. `gate_loop_cluster(strip_exponent=True)` performs
  amplitude-preserving per-pair RMS bookkeeping; unused full result snapshots
  are no longer retained across the stream.

- **Local SU gauge correction:** include full exterior SU weights in the
  endpoint QR/LQ and small-core SVD. Bare reduced-factor spectra preserved
  states but produced incorrect boundary gauges for later updates. Add
  independent PEPS/PEPO Schmidt-spectrum and zero-cluster SU regressions.
- **Contraction-plan cache:** retain tensor operand order in cache keys;
  a tree reused after operand permutation could contract the wrong network.

- **Hermitian-only reduced environments:** `psd_project=False` skips spectral
  diagnostics as well as clipping; `raw_min_eigenvalue` is then `None`.
  Hermitianization remains enabled for exact, SU-cluster and loop-cluster
  problem builders.

- **Loop-cluster GPU replay:** preserve dense Torch tensors, gates and SU
  gauges in `gate_loop_cluster`. Torch exposes `to_dense()` even for dense
  tensors; it must not trigger NumPy/CPU conversion. Add actual-device checks
  for PEPS/PEPO one-/two-site gate streams.

- **BP cluster evolution:** expose Quimb graph-distance/loop-union base-patch
  selection and fill-in through reduced gate/compression APIs. Loop-union
  cutoff four selects the centered six-site patch, with cached Cotengra
  paths and fresh contractions. Add dense PEPO upper/lower gate streams,
  preserve SU weights across physical-input streams, retain native charge
  spaces, and keep Torch/CuPy local ALS solves in their array backend.

- **BP reduced updates:** replace whole-network SU regauging with local
  QR/LQ and small-core SVD extraction of the active bond gauge. Preserve
  exterior gauges, spectator tensors and fitted amplitude. Loop-cluster gate
  streams initialize missing gauges as identities. Legacy `regauge_opts`
  sweep controls are deprecated.

- **Reduced full-update work:** reuse the initial SVD factors and balance
  ungauged ALS results with QR/LQ and a bond-sized SVD. Keep native Torch/CuPy
  local ALS solves dense when Quimb's automatic threshold would choose a
  SciPy LinearOperator. Source-barrier replay can retain checked environments
  through the existing `set_gates()` and `run()` interface.

- **FU boundary initialization:** default to `reuse-direct` and four boundary
  DMRG iterations. Direct compression seeds cold cuts; existing fitted MPS
  initialize later refits. Explicit initialization and iteration controls
  retain precedence; seeded `guess-src`/`reuse-src` remain selectable.

- **Joint PEPO leg updates:** opt into `run(fuse_pepo_legs=True)` to apply
  upper/lower gates on the same bond to one exact target and fit it once.
  Align reversed sites and retain both original gate positions. Reuse SVD/SU
  starts, reduced/full pair solvers, cached boundaries and strip/layer refinement.

- **PEPO scheduling and SU reuse:** reorder explicit upper/lower qubit gates
  with row/column/smart traversal and preserve same-leg dependencies and
  fusion order. Retain SU weights across full-update gates/runs, refreshing
  only the active bond. Report boundary-MPS absorption and cache reuse steps;
  direct boundaries retain separate BRA/KET compression.

- **Native PEPO full update:** support upper/lower-leg gate streams with
  shared QR/LQ reductions, ALS and pair L-BFGS, retaining both operator legs.
  Preserve PEPO amplitudes by default and measure target Frobenius norms;
  PEPS normalization defaults remain unchanged. Validate direct and DMRG
  boundaries against dense operator references. Add opt-in SU pair warm
  starts for PEPS/PEPO and pass their factors through the shared dense ALS
  solver. Independent norm/fidelity diagnostics remain off by default in FU.

- **Full Haar vectors:** allow an explicit `L_max>20` with a positive
  `max_dense_bytes` preallocation budget. The historical default size guard
  and seeded complex Haar distribution remain unchanged.

- **Torch SVD backward:** allow `TorchLinalgConfig.svd_gap_rtol` to select
  the relative stabilized gap threshold while preserving the `1e-6` default.
  The direct JAX `svd_jax` call likewise accepts `gap_rtol`. Narrower
  thresholds require independent gradient checks near degeneracies.

- **Prepared dense SU:** add `operators.PreparedSimpleUpdate` for reusable
  Torch PEPO gate schedules, disjoint-site/network batching, and optional
  AOT-eager forward/backward capture. Check per-member truncation decisions
  and replay adaptively on rank changes; preserve SU weights and scale.
  `apply_many` batches compatible ready updates across independent schedules
  without padding them with identity gates or changing per-plan cutoffs.

- **Torch QR backward:** reuse saved forward factors on nonsingular stabilized
  charts and avoid batch-mask gathers for single-matrix adaptive pullbacks.
  Preserve rank-deficient fallbacks, FX capture, dtype, and truncation policy.

- **Bounded dense CPU runs:** add opt-in Torch complex64 exact-batch gate
  replay with in-place one-qubit/structured kernels, and chunked final Z
  sampling. These avoid state-sized gate and CDF temporaries for supported
  streams while preserving physical site order. Earlier references to the
  optimizer's dense state reflect later in-place replay.

- **Local energy readout:** add canonical and shared-environment routes to
  `MpsEnergyOptimizer`, plus joint energy/observable `measure()` calls for
  MPS, tree, and Gibbs classes. Dense readout reuses support density matrices;
  optional Torch einsum/vmap batches compatible final contractions with
  bounded chunks. Native arrays retain graded fallbacks. Gibbs keeps canonical
  readout by default; existing MPS energy objectives keep their exact default.

- **Gibbs MPS:** add normalized local `energy()` readout without constructing
  a density MPO, using a reusable Pepsy contraction optimizer; reject
  nonfinite inverse temperature and step size early. Default to second-order
  evolution with thermal increments `beta_step=0.05`, retaining explicit `dt`
  in purification-time units. Add `evolve()` for one trajectory with
  intermediate measurements, every N steps or at explicit beta targets;
  default to `measure_every=5`, reuse equal gate batches and fuse between
  measurement boundaries. Energy readout reuses the known canonical center
  on a copy and caches its contraction optimizer and bound operators.
  Explicit `n_steps` still takes precedence over the default step size.
- **Non-unitary MPS diagnostics:** automatically measure local compression
  overlap in direct/DMRG replay and expose it as Gibbs `local_infidelity`.
  Dense paths cancel the canonical exterior and reuse FIT's terminal norm;
  native arrays retain a graded full-network fallback. Keep unitary
  norm-survival behavior unchanged.

## [0.6.0] - 2026-10-23

- **PEPS:** full updates, strip ALS, full-PEPO fitting, adaptive boundary caps,
  and optional gauge QR.
- **Autodiff:** functional JAX/Torch FIT, dense implicit gradients, and
  improved QR/SVD derivatives. Native Symmray implicit FIT remains unsupported.
- **Dynamics:** Torch and NetKet PEPS TDVP, with solver diagnostics and
  restartable TFIM examples.
- **Operators:** reusable cluster plans, compact MPO/PEPO channels, and
  structural MPO compression.
- **Changed defaults:** one-site MPS/tree DMRG, PEPS sweep/boundary policies,
  dense tree factor sampling, PEPS proposal sampling, and dtype-aware gate
  cutoffs. MPS `dmrg1` was removed; use `dmrg`.
- **Fixes:** gradient stopping and state restoration, boundary scaling,
  backend/device preservation, measurement probabilities, and parallel
  trajectory initialization.

See [0.6.0 migration](docs/development/api-migration.md#upgrading-from-050-to-060)
for reproducibility controls and API guides.

## [0.5.0] - 2026-09-25

- **Environment:** Python 3.12+ and higher dependency minimums.
- **Added:** Gibbs MPS, native tree FIT/DMRG, more compression methods,
  state-aware layouts, and differentiable boundary/operator workflows.
- **Changed:** direct MPS replay by default, revised FIT schedules,
  term-compressed Hamiltonian builders, and lazy public imports.
- **Removed:** MPS `mode="su"`, `routing="perm"`, and composed permutation
  modes. See [0.5.0 migration](docs/development/api-migration.md#upgrading-from-041-to-050).

## [0.4.1] - 2026-08-27

- Added stabilizer and PEPS sampling engines, MPI shot ensembles, Guppy
  interoperability, and graph-aware MPO/PEPO expansions.
- Added simulator planning, boundary FIT diagnostics, and optional
  accelerated contraction search.
- Improved truncation controls, backend preservation, normalization, and
  fermionic ordering; deprecated legacy API aliases.

## [0.4.0] - 2026-07-27

- Removed obsolete flat modules and the duplicate `pepsy.extensions`
  namespace. Advanced APIs use `pepsy.experimental`.
- Updated documentation to use maintained public APIs and examples.

## [0.3.0] - 2026-07-24

- Added native fermionic tree evolution, TreeOptimizer, symmetric-tensor
  workflows, and public trajectory/stabilizer/VMC APIs.
- Organized public namespaces and corrected norm, gauge, and truncation
  bookkeeping.

## [0.2.0] - Baseline

Initial package-version baseline; earlier changes were not recorded.
