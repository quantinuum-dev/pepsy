# Changelog

All notable PePsY changes are documented here.

PePsY follows [Semantic Versioning](https://semver.org/). During the 0.x
series, minor releases may include documented incompatible changes; patch
releases remain backwards-compatible. From 1.0 onward:

- **MAJOR** versions may contain incompatible public API changes.
- **MINOR** versions add backwards-compatible public functionality.
- **PATCH** versions contain backwards-compatible fixes and documentation updates.

## [Unreleased]

### Added

- Shared PEPO `from_plan` factories now select the existing square 2D builder
  for compatible nearest-neighbor Pauli plans, with explicit `layout="square"`
  and `"graph"` controls. The square adapter preserves term slots, ordered
  factors, parameter identities, OBC/PBC metadata and repeated periodic terms,
  reuses the shared inventory, and supports fixed-channel Torch/JAX
  materialization. Active square blocks expose explicit `to_dense()`.

- TreeOptimizer adds opt-in `stabilize_unitary` at construction and replay,
  restoring incoming unitary norms while retaining compression loss and
  preserving explicitly non-unitary scale changes. TreeOptimizer and
  TreeSampler now default to `threads=None`. Disabling tree replay history
  also disables accumulated norm/FIT records; compact norm queries are
  available through `norm_diagnostics(include_history=False)`. Automatic
  cutoff and cutoff-mode defaults remain unchanged.

- Add `pepsy.operators.ClusterPlan` for inferred interaction graphs, lazy cached
  cluster inventories, exact partition/collection counts and verified local
  symmetry reuse. MPO and graph PEPO share `from_plan` factories; general graph
  products support nonuniform and higher-body terms. Graph materialization
  preserves asymmetric complex operators and mixed-block dtypes; graph traces
  and residuals preserve backend gradients. See the interaction-cluster guide.

- `MpsOptimizer.run(fit_single_pair_n_iter=None)` inherits `n_iter` by
  default; a positive integer caps adjacent two-site DMRG windows separately.
  Larger and batched
  windows use their full span to select the budget; convergence stopping,
  measurement/shot replay, and timing records preserve it. Named DMRG2
  honors the same budget policy; explicit `fit_single_pair_fast_path=True`
  still selects one update.

- Common cluster API conventions across MPO, PEPO and Gaugy: PEPO plans and
  builders accept `cluster_size` as an alias for spatial `order`; dense PEPO
  plans expose `compile_exp().exp(step)` with `build(-step)` semantics.
  MPO products expose `from_bases`, positional runtime `parameters`, and
  per-factor term `coefficients` in compiled and one-shot evaluations, plus
  `materialize=True` on reusable evaluators for direct Quimb MPO output. Their
  cached vector-binding plans preserve independent coefficient gradients.

- Compiled connected-cluster MPO and Pauli PEPO products now expose
  `trace_exp(..., normalized=False, state_budget=100000)`. It evaluates
  complete chosen-order ordered-product traces from scalar connected
  residuals and compatible-cluster subset recursion, without MPO/PEPO
  assembly or SVD. Single `PauliPEPOBasis` instances expose the same method.
  Tests cover noncommuting products, periodic parallel bonds, Torch gradients
  and JAX JIT gradients.

### Changed

- Tree DMRG reuses unchanged non-differentiable Torch exterior branches as
  identity environments and combines the target norm with its first required
  convergence readout. Single-site Torch unitary checks use a bounded cache
  guarded by storage/view and mutation versions, with direct-check fallbacks
  for unversioned arrays. Sweep budgets, stopping rules, and cutoffs are unchanged.

- Graph MPO runtime coefficient vectors retain the original graph-support
  contract, including interactions connected through intermediate sites.
  Graph active-block materialization preserves Torch storage metadata and
  gradients, including isolated sites. Shared-plan factories reject invalid
  term indices with an explicit validation error.

- Shared cluster bindings now accept `None` factor vectors on MPO products,
  matching PEPO/Gaugy defaults, and reject conflicting parameter/coefficient
  containers before evaluating PEPO callbacks. The dense PEPO cutoff alias
  preserves `dataclasses.replace(plan, order=...)` by storing only `order`.

- Joint located PEPO products now avoid building unused homogeneous
  Hamiltonian component maps. Independent noncommuting square-lattice
  set-partition checks cover joint MPO/PEPO orders two through four with
  symmetry reuse on/off; two-site fixed joint JAX JIT gradients are checked
  against dense ordered exponentials.

- Finite NumPy MPO compression matrices that trigger SVD nonconvergence now
  retry with SciPy `gesvd` when SciPy is installed. Other backends,
  nonfinite inputs and requested bond ranks keep their existing policies.

- Located Pauli PEPO construction now reuses lower-order support
  contractions under the same verified Hamiltonian and geometry symmetry
  plan used for exact targets. Residuals and sparse block insertion remain
  per placement; no numerical values persist between evaluations.

- Fixed cluster JAX evaluation is regression-tested through complete
  three-site MPO/PEPO construction and a 2×2 square at order four under
  `jit(value_and_grad)`, including zero parameters/time and complex-time
  two-site cases. Torch full-graph
  capture is verified for the local exponential/fixed-split numeric path.
  Native Torch identity and scalar promotion avoid two Autoray tracing
  failures, and the PEPO tree uses Python shape products during capture;
  complete Torch builder capture remains unsupported.

- Explicit post-compression of fixed-channel cluster MPOs now has a runnable
  chi/error benchmark. Dense numerical MPO error estimates QR-canonicalize the
  difference before its Frobenius contraction, avoiding cancellation when the
  compression error is small. This is an opt-in diagnostic and leaves
  SVD-free fixed construction unchanged.

- Fixed cluster MPOs align evaluated targets before residual subtraction,
  supporting host-valued time steps with positional parameters and callable
  coefficients without extra callback evaluations. Cluster MPO parameters
  now honor declared defaults when no parameter container is supplied.

- Cluster MPO and Pauli PEPO APIs accept `factorization="fixed"` for exact,
  SVD-free construction with parameter-independent channel shapes. Trainable
  zero MPO residuals retain their derivatives; generic homogeneous PEPO trees
  use fixed index splits and cached topology. Internal compression is rejected
  in this mode; the existing numerical policy remains the default.
- `spatial_symmetries` accepts verified finite-site permutations alongside
  automatic term-aware cluster reuse. Invalid geometry/term/parameter claims
  raise; independent coefficient overrides retain separate reuse plans.
- Mixed Python/Torch Pauli PEPO coefficients retain the trainable slots'
  precision instead of rounding constants through the default Torch dtype.


- Graph cluster MPOs support `assembly="recursive"`: shared remaining-site
  MPOs retain every disjoint residual collection, adding and compressing
  branches without enumerating collections. `assembly_state_budget` guards
  structural work and raises without an approximation fallback. Reports expose
  subproblem counts, live cache peaks and assembly rank reductions.
- Streaming graph MPOs on direct plans now retain products of separated
  residuals. Intermediate assembly compression prepares an orthonormal
  environment before truncation, avoiding gauge-dependent loss of large terms.


- Cluster MPOs and `PauliPEPOBasis` default to conservative
  `spatial_reuse=True`: proven translations/rotations and graph relabelings
  share local ordered targets, with structural plans cached at compilation.
  Parameter identity, directed/parallel bonds, factor order and gradients are
  preserved. Cache reports expose reuse; `spatial_reuse=False` enables
  unreduced comparisons. Residual and collection/truncation policies remain
  unchanged.

- Compiled cluster MPOs reuse their callable, expose `last_report`, and accept
  `return_report=True` to return the resolved construction report alongside
  the semantic MPO. Compiled PEPO calls expose cache and shape inventories;
  ordered PEPO diagnostics include each factor's preparation state.

- `PauliPEPOBasis.compile_exp()` prepares located operator maps and tree
  topology, and homogeneous higher-order source maps, before evaluation.
  Translated clusters share static local maps while preserving independent
  coefficients and directed/parallel bonds. Mixed ordered products compile
  all factors for their actual evaluation route.

- Cluster PEPO plans and Pauli bases expose `cluster_inventory` with per-size
  oriented and C4 shape counts, split into trees and loops. Bounded geometry
  caches reuse smaller shape levels across cutoffs and finite translated
  embeddings across builds; numerical residuals and gradients remain dynamic.

- Removed duplicate backend and linalg exports from `pepsy.tensors` and
  `pepsy.tensors.core`; import them from `pepsy.backends`. Also removed unused
  compatibility aliases for `experimental.mera`, `QMeraParametricEnergyOptimizer`,
  `MpsStabSampler`, `StabilizerMps`, `SpinfulFermionHubbard`, and
  `hrps_to_ttn`. See the [API migration guide](docs/development/api-migration.md).

### Added

- Torch boundary VMC exposes bounded stage/error records for recovered
  connected-amplitude and proposal retries, plus the latest sparse-cutoff
  retry reason. Numerical retry policies remain unchanged.

- Boundary PepsSampler now defaults to FIT-style factored numerical row
  environments with a 64 MiB estimated cache budget and independent auto-hq
  local contractions. Right suffixes are reused and left prefixes advance
  after each draw. Explicit zero-budget reference and dense modes remain.

- PEPS full-network contractions now default to Pepsy's reusable Cotengra
  `build_optimizer(parallel=False)`; cached-row steps use `auto-hq`. Explicit
  full/row optimizer overrides remain supported. Optional exact-amplitude
  intermediate-size/cost limits reject oversized plans before execution,
  with estimates exposed through `amplitude_plan_info`.

- Direct PEPS sampling supports `iter_samples` and optional `sample_batch`
  chunking to bound live prefix states, stable normalized weights/ESS and
  compact rho diagnostics, reusable scaled exact-amplitude plans, and
  factored row suffix caches. A reproducible evolved-state CPU/GPU benchmark
  reports stage costs and memory. Factored caches are the boundary default,
  retain an explicit reference fallback, and do not truncate amplitudes.

- `PepsSampler(..., rho_positivity="clip" | "absolute")` optionally repairs
  the Hermitian local proposal spectrum. Default sampling explicitly uses
  the Hermitian part with its existing diagonal policy. Qubit repairs use a
  batched 2×2 formula; larger physical dimensions use native `eigh`. All paths
  record repaired proposal probabilities and report the relative correction,
  while preserving original PEPS amplitudes and rejecting NaN/Inf input.

- Opt-in `contract_flat(..., method="mps", mps_factorization="projector")`
  for dense 2D direct boundary compression. A composed isometric-factor Torch
  VJP handles redundant virtual directions without singular QR inverses or
  regularized singular-vector derivatives. It supports first-order,
  gauge-invariant losses on locally fixed-rank, spectrally gapped subspaces;
  unsupported singular charts raise instead of returning a surrogate.

- `ActivePEPOBlocks.to_trace_network()` closes physical blocks before dense
  materialization, returning an unnormalized `TensorNetwork2D` with preserved
  NumPy/Torch/JAX values. `trace_nbytes` reports its dense site storage.

- `PepsSampler` exposes `chi` (future double-layer cap) and `chi_prime`
  (conditioned ket cap); `marginal_chi` and `sample_chi` remain compatible
  aliases with conflict checks. Automatic mode selection keeps uncapped calls
  exact and selects boundary sampling when caps are supplied. Explicit
  `ket_compression=None` permits an uncapped ket boundary. `PEPSSampleResult`
  adds natural-log `log_probabilities`, `log_abs_amplitudes`, and `log_weights`
  NumPy views of its scaled values.

- `PepsSampler(..., to_backend=callable)` converts a private PEPS copy;
  omitting the converter infers the source array backend, dtype, and device.
  Local density matrices, identity caps, probabilities, and categorical draws
  now stay on NumPy, Torch, or JAX. This fixes mixed NumPy/Torch identity caps
  when `marginal_chi=0`, respects float32 roundoff in conditional validation,
  and reuses the final amplitude for identical batch configurations.
  JAX array creation and RNG use a scoped source-device context.

- `QMeraBuilder.estimate_contraction_cost(...)` reports pre-run complex FLOPs
  and peak forward contraction bytes for dense qMERA local cones, with log10
  and log2 summaries. Its cache reuses unsliced Cotengra paths at compilation.

- `QMeraBuilder.compiled_parametric_loss_fn(torch_fullgraph=True)` now returns
  a Torch-only dense-spin energy callable with frozen Cotengra paths. It
  reuses each scheduled gate across local terms and supports
  `torch.compile(..., fullgraph=True)` with AOT eager. The
  `QMeraEnergyOptimizer` compiled loss and Torch solver can use the same mode.

- Opt-in `QMeraBuilder(hierarchy="retained")` builds a 2D spin qMERA
  hierarchy with retained qubit registers, x/y boundary disentanglers,
  inspectable coarse-grid blocks, and coarse-to-fine preparation order.
  Rectangular and square covering blocks absorb odd one-cell edge tails.

- `TreePlan.from_alternating_lattice((Lx, Ly))` and the
  `TreeLayoutFinder(order="alternating-xy")` / `map_mode` preset build an
  actual x-then-y recursive pairing hierarchy, preserving physical labels
  and odd edge blocks with a strictly binary root. Existing layout defaults
  and `coarse-*` traversal semantics are unchanged.

- Added opt-in `MpsOptimizer(mode="exact-batch")` (`batch-exact` is an alias).
  It fuses bounded one- and two-qubit gate runs, compacts repeated Z/ZZ
  supports, and applies one- or two-value diagonal phases in grouped passes.
  Supported NumPy/CuPy states also fuse consecutive same-pair parity-preserving
  gates. Dense replay preserves backend, device, and operator scale; unsupported
  states use the reference path. Numba acceleration is optional.

### Changed

- `PepsSampler` defaults to `cutoff="auto", cutoff_mode="auto"`, sharing
  the MPS dtype policy (complex64: `1e-6`, complex128: `1e-12`) and `rsum2`
  truncation convention. Resolution follows conversion and refresh; explicit
  numeric cutoffs and Quimb modes remain available. Both policies are passed
  to future-boundary preparation and conditioned ket compression.

- `PepsSampler` retains the simple conditioned-boundary reference sweep
  (`row_cache_max_bytes=0`) and explicit dense row transfers, alongside the
  default factored environment cache and its memory fallback. Future preparation builds
  only the required side; DMRG boundaries are initialized lazily. The new
  `log_probability(config)` returns a natural-log likelihood, with scaled
  contractions for exact/reference-boundary evaluation. Configuration values,
  finite cutoffs, missing future caches, and unsupported periodic boundary
  sweeps are validated explicitly. Prefix batches validate probabilities and draw all active groups
  together once per site, reducing host readbacks. Local-rho contractions copy
  only the tensors that need reindexing; row bonds, identity caps, and traced
  column contractions are reused. Scaled Hermiticity diagnostics avoid
  complex64 norm overflow. Grouped seeded sequences can differ from previous
  releases; same-backend/device/method reproducibility is preserved.

- The downstream roughening PEPS runner now defaults to initial SU gauge
  equilibration only (`--peps-gauge-every 0`). Periodic equilibration remains
  available with `--peps-gauge-every N`; gate updates still update bond gauges.

- The 1D qMERA clean schematic now draws a single left-to-right circuit
  across RG scales. It separates stage headings from gate markers, outlines
  isometry blocks, curves long pair links around intervening wires, and keeps
  explicit-mode wire labels distinct.

- The qMERA `draw_schematic(style="clean")` view now follows actual gate
  direction: retained circuits show coarse-to-fine W then D, while site
  schedules show fine-to-coarse D then W. It marks 1D periodic seams,
  isometry blocks, retained/product wires, and 2D parent registers. The 2D
  panels now mark each scheduled pair gate within its covering block;
  `style="register"` remains available.

- Existing 2D site-retention qMERA schedules now absorb a trailing one-cell
  axis segment into the previous covering block when a lattice dimension is
  odd; the default 2D hierarchy remains site retention.

- The main spin 1D `QMeraBuilder` uses retained-register blocks, ternary odd
  tails, boundary disentanglers, and coarse-to-fine gate execution. It accepts
  bond width and retention policy, with `system_size=N` as the 1D site count
  (`shape=N` remains accepted). `structure="ladder"` closes each isometry or
  disentangler block and repeats the complete pair sequence at each
  `circuit_depth`. The 2D and explicit-mode fermion schedules retain their
  existing behavior.
- Spin 1D qMERA exposes `pair_ansatz` with five global-X Z₂-preserving
  templates and an unrestricted Pauli template. `spin_symmetry` declares the
  symmetry contract separately from `initial_state`; descriptive ansatz names
  such as `z2_zz_yy_rx` and `z2_rx_zz_yy_xx_rx` identify gate order, while
  old names remain accepted. `QMeraPairSpec.rotation_sequence` reports the
  ordered rotations and pair wires.

### Fixed

- Ordinary `hrs_to_mps`, `hrs_to_peps`, and `hrs_to_ttn` now reject real
  storage, which previously discarded Haar phases and could reduce the norm.
  Use a complex dtype, or `ps_to_*` for real product states. Native fermion
  random constructors retain their existing dtype support.
- Real stabilizer single-site Y rotations avoid zero-imaginary casts;
  local coefficient operators requiring complex storage now raise clearly.
- D2 edge-loop suppression validates norm weights before using Quimb's real
  solver, accepting roundoff-sized imaginary residues and rejecting larger
  ones instead of losing them through implicit casts.

- Torch conversion copies read-only NumPy storage before creating a tensor,
  preventing writes through the tensor from changing an immutable source.
- Cold PEPS environment sweeps and MPS trajectory norm fallbacks select the
  installed Quimb `method`/`route` keywords without deprecated calls.
- BP reduced-update metric factorization falls back after numerical failures
  while allowing unrelated type errors to propagate.

- Torch VMC convergence estimates now resolve the `tau` property correctly.
  BP Metropolis sweeps report acceptance/proposal totals across every requested
  sweep. Unsupported local-move statistics raise a descriptive `ValueError`
  before drawing BP proposals.

- Tree-PEPS two-layer compression now initializes transient bond diagnostics
  consistently. Enabling `track_bond_diagnostics=True` no longer raises
  `UnboundLocalError` after applying the operator.
- `PepsOptimizer` now builds untruncated two-site targets before deciding that
  they fit within `chi`, forwards the selected cutoff mode to all warm-start
  compression paths, and rejects non-finite or substantially negative
  infidelity estimates. Explicit truncating target gate overrides now raise.
- `PepsSampler.sample_batch` gives each returned shot its own configuration
  list even when prefix grouping reuses its proposal and exact amplitude.

- PEPS amplitude cache initialization can be retried after preparation fails.
  Sample-result log and weight accessors reject mismatched field lengths rather
  than silently broadcasting; weight diagnostics avoid duplicate host transfers.
  Sampler API comments clarify cache ownership and diagnostic lifetimes.

- `PepsSampler` rescales within-row transfers and running prefixes/
  suffixes, fixing rare-configuration log probabilities that could underflow
  inside the cache. It reuses the initial-row cache until `refresh()`, counts
  that retained memory in the budget estimate, releases unused center networks,
  and skips terminal prefix contractions. Later rows remain conditioned on
  each incoming sampled prefix; the default factored-cache budget is 64 MiB.

- `PepsSampler` rescales private boundaries before relative-cutoff compression
  and equalizes Quimb future sweeps, preventing complex64 squared-singular-value
  overflow from spuriously collapsing retained ranks. This covers conditioned
  Quimb compression and FIT guesses while preserving absolute-cutoff semantics,
  source tensors, physical amplitudes, and cached-future reuse.

- Reduce complete traces of located exact Pauli-history PEPOs to certified
  identity-subtree sectors before dense allocation. Preserve the full PEPO,
  coefficient gradients, and parameter-independent trace topology; rank-capped
  and uncertified builders retain all sectors.
- Inspect active PEPO dense-storage size with differentiable Torch blocks
  without converting them to NumPy. Use Python integer products for dense
  size estimates so large virtual dimensions cannot overflow a machine integer.
- Batch located equal-size cluster products, preserving factor order and
  coefficient gradients while avoiding repeated scalar exponential dispatch.
  This also avoids measured small-matrix Torch exponential inaccuracies in
  the covered batches; local targets and cluster orders are unchanged.

- Preserve resolved Torch SVD derivatives under `stabilized=True`: replace
  global Lorentzian damping with compact relative stabilization, retaining
  the same gap threshold and finite singular-case extension. Use the
  dtype/shape numerical-rank threshold for inverse singular values, separately
  from the singular-gap threshold.
  Applies to both real/complex Autoray and native Symmray split paths.
- Renormalize native Symmray SU gauge vectors over their backend blocks,
  preserving Torch gradients and stable exponent accounting without densifying
  the vectors or calling an unsupported Symmray mean.
- Forward `cutoff_mode` through PEPO gate construction and its fallback
  compression, which also now respects the requested `cutoff`. Expose optional
  cutoff-mode selection for compressed hyper-contraction.
- Preserve simple-update gauge values and first derivatives when extracting
  their RMS scale: divide and restore the same detached positive scale, avoid
  overflow/underflow in norm evaluation, and keep zero gauges finite.
- Capture Torch QR rank policy during forward so backward remains consistent
  after a scoped configuration exits. Add opt-in `qr_rank_policy="adaptive"`
  to preserve finite native VJPs and warn when a singular/nonfinite block
  needs the regularized fallback.
- Build dense MPO/PEPO gate streams in Quimb's operator convention: upper
  `k...` indices are outputs, lower `b...` indices are inputs, and later gates
  multiply on the left. Remove the legacy per-gate transpose. `tns_align`
  now applies this convention by default, with `transpose=True` available
  for explicitly transposed operators. Rebuild old general gate streams;
  transposing their final operator alone may not fix their order.
- Apply lazy sub-MPOs when `gate_with_submpo(..., inplace_mpo=False)`;
  this flag controls the applied operator's defensive copy, not application.
- Correct the bra/ket orientation in native MPS sampling and probability
  evaluation with complex right environments. NumPy, Torch, and CuPy use
  the corrected Born-weight contraction; complex-state sample distributions
  can change compared with earlier versions.

### Development

- Remove obsolete standalone operator, PEPS-sampling, MPI-launcher, and MPS
  compression benchmark scripts. Current guides now point to maintained APIs
  and regression tests, and nightly MPI validation uses its integration suite.
- Separate sampler engines and records, MPS controls and norm bookkeeping,
  stabilizer advice and layout planning, and BP loop geometry into their
  owning modules. Preserve public APIs, historical serialization paths, and
  numerical implementations.
- Move symmetric MPO assembly and local-term factorization into a private
  operator module, preserving Hamiltonian APIs and historical helper imports.
- Extract MPS layout execution into a private module while retaining optimizer
  method signatures, documentation, subclass hooks, and class hierarchy.
- Separate native symmetry diagnostics and fermion model helpers from the
  shared Hamiltonian/conversion layer. Move MPS stream preparation and
  stateless compression adapters into focused modules. Preserve public imports,
  historical serialization paths, numerical implementations, and replay defaults.
- Separate symmetric MPS/PEPS state construction, evolution, and measurement
  into their owning module. Preserve existing namespace imports and serialized
  state-class paths; model and operator imports keep state loading lazy.
- Import contraction and fidelity helpers directly from their owning modules
  throughout fitting, boundary sweeps, optimizers, and sampling. Legacy
  `tensors.core` entry points retain their compatibility hooks.
- Correct tensor ownership documentation, fill empty API guides with tested
  examples, and point optional-dependency messages to local checkout extras.
- Keep release automation limited to GitHub build artifacts; remove unused
  PyPI/TestPyPI publishing jobs and their token permission.
- Move MPS layout report formatting and FIT timing summaries into the private
  diagnostics module, preserving report output, lazy imports, and replay behavior.
- Split the tree API guide into layout, state, operator, replay, fitting, and
  readout pages while preserving the original section links.

## [0.5.0] - 2026-09-25

This release changes supported environments and several optimizer defaults.
Read the [migration guide](docs/development/api-migration.md) before upgrading.

### Changed

- Require Python 3.12 or newer. Core dependency floors are NumPy 1.26,
  Quimb 1.15, Cotengra 0.8.0, Autoray 0.9, and tqdm 4.65. Optional dependency
  floors also increase; NetKet VMC requires JAX `>=0.7,<0.11.1` and NetKet
  3.22 or newer. See [installation](docs/installation.md) and the
  [dependency audit](docs/development/notes/dependency_minimums_2026_09.md).
- `MpsOptimizer` now defaults to `mode="direct"`. Request `mode="dmrg"`
  or a named DMRG schedule explicitly for variational replay. `MpoOptimizer`
  and `GibbsMps` also use the canonical `direct` spelling; `mpo` remains a
  compatibility alias.
- MPS `mix` replay now fits a disposable direct-compressed guess against the
  exact target during both bond growth and fixed-rank updates. It fixes
  `fit_block_size=1` and `fit_init_strategy="guess-direct"`; use ordinary
  DMRG for other block sizes or initializers.
- FIT schedules now include larger-block warm-up and one-site refinement,
  with a two-site transition in `dmrg3`. Tree optimizers select path or branch
  traversal through `fit_traversal="auto"`. Tree compression uses live-rank
  scheduling; explicit traversal and compression controls remain available.
  Finite-rank and seeded results can change with the updated algorithms.
- Hamiltonian `to_mpo`, `to_pepo`, `to_tree_mpo`, and `to_tree_pepo` default
  to `compress="term"`. Use `compress="auto"` for workload-selected assembly,
  `compress="automaton"` for shared state diagrams, or `compress=False` to
  disable numerical compression. Native tree builders can infer a layout
  from interaction supports.
- MPS runtime finite checks and MPI diagnostic collection are opt-in.
  Tree and stabilizer MPI replay likewise default to `collect_diagnostics=False`.
  Backend-native norm and fidelity bookkeeping defers host scalar conversion
  until readout where possible. Explicit diagnostic controls remain available.
- Stabilizer front ends use `StabilizerMpsSimulator`,
  `StabilizerTreeSimulator`, and `StabilizerMpsSampler`. Previous names remain
  deprecated aliases. Hamiltonian `build_mpo` / `build_pepo` wrappers remain
  deprecated aliases for `to_mpo` / `to_pepo`.
- Public namespaces load exports lazily and expose them through `dir()`.
  Internal helpers use their owning modules, optimizer event parsing has a
  shared owner, and VMC/test extras compose existing dependency profiles.

### Added

- `GibbsMps` finite-temperature purification, `bell_to_mps`, thermal MPO
  readout, exponent-aware partition functions, and graph-aware first-, second-,
  and fourth-order Trotter scheduling. See the
  [Gibbs-MPS API](docs/api/optimizers/gibbs_mps.md).
- Tree-native FIT/DMRG, compact `SubTreeMPO` / `TreeSubPEPO` application,
  zipup and mixed replay, and successive/randomized compression families.
  `TreePeps` supports coordinate-aware spanning trees, layout selection,
  canonical-region metadata, and up to four virtual bonds per site. Native
  symmetry/backend restrictions are explicit in the
  [tree](docs/api/optimizers/tree.md) and
  [TreePEPS](docs/api/optimizers/tree_peps.md) APIs.
- Opt-in SDC, SRC, SDCR, and oversampled compression variants across supported
  MPS, MPO, tree, and PEPS boundary paths. Intermediate and final compression
  controls remain independent, with execution-time capability checks and
  authoritative final bond limits. See
  [compression options](docs/api/boundary/compression.md).
- State-aware MPS layout and replay scheduling, including interaction-derived
  coordinates, site lifetimes, role-aware candidates, and an opt-in
  `measure-early` schedule that respects control dependencies.
- `mps_to_ttn` and `mps_to_treepeps` conversions with explicit bond caps;
  MPS transfer spectra and correlation lengths; MPS and tree entanglement
  diagnostics; and dense-backend/native-Symmray tree sampling. See
  [observables](docs/api/tensors/observables.md) and
  [tree sampling](docs/api/sampling/tree.md).
- Directional and middle-out flat contractions, layered boundary contraction,
  additional finite-CTMRG modes, and `preserve_backend=True` scalar returns
  for differentiable flat contractions. See
  [boundary metrics](docs/api/boundary/metrics.md).
- Term-centric and backend-aware Hamiltonian/MPO construction, higher-order
  exponential policies, structural MPO block inspection, charge-flow
  validation, bounded/streaming graph-cluster assembly, and ordered products.
  Inhomogeneous PEPO products support explicit locations and periodic square
  lattices through order nine. See [operators](docs/api/index.md) and the
  [exponential API](docs/api/operators/exponentials.md).
- Fourth-order symmetric Hamiltonian/fermion gate streams, additional BP
  option forwarding, and capability-gated upstream contraction integrations.
- Opt-in Torch PEPS export, batching, and compilation for exact amplitudes
  and reusable rectangular boundary environments, with eager fallbacks for
  unsupported paths. See [VMC](docs/api/vmc.md).

### Removed

- `MpsOptimizer(mode="su")` and its optimizer-owned gauge state. Use
  `pepsy.operators.gate_simple` or the dedicated PEPS simple-update APIs.
- MPS `routing="perm"` and composed `perm-*` / `*-perm` solver modes.
  `mode="perm"` is now a single lazy swap-and-split SVD route; it retains
  physical order and exposes the logical mapping. Use an explicit solver
  with a static layout for compression-aware ordering.

### Fixed

- Released Quimb compatibility for seeded compression, method/route options,
  optional compressors, and dtype-specific decomposition failures. Narrow
  fallbacks retain backend, dtype, and requested truncation semantics.
- Rare MPS/tree measurement branches, conditional and trajectory controls,
  dynamic caps, represented exponents, and projection-versus-compression norm
  accounting. Invalid replay settings preserve state and configuration.
- Tree entropy now uses physical Schmidt spectra from a canonical copy.
  Tree state invalidation clears stale isometry metadata, and compact operator
  application preserves exterior scale and identity semantics.
- Native Symmray MPO dense readout restores computational-basis order;
  fermionic FIT, BP overlaps, and cluster bra phases preserve graded
  conventions. Unsupported odd-parity TreeFIT projections fail explicitly.
- Torch/JAX/CuPy backend, dtype, device, and gradient preservation in replay,
  fitting, operator construction, and diagnostics. Complex PEPO coefficients
  and real-term/complex-prefactor MPO products retain correct promotion.
- Rectangular and single-slice PEPS boundaries, FIT target/guess ownership and
  cache reuse, JAX SVD options and derivatives, and Torch export/compile
  log-amplitude evaluation.

### Development

- Documentation builds now fail on warnings in CI and Read the Docs. Fixed
  API parameter formatting, heading levels, navigation coverage, and links
  to repository history so the strict HTML build completes cleanly.
- CI tests the five direct core dependency minimums and a combined extended
  profile with a 60% coverage gate, plus MPI, packaging, docs, type checks,
  and agent guidance. Backend imports and test failure annotations improve
  diagnostics. MPS/tree tests are split by responsibility.
- See the [release-readiness review](docs/development/notes/release_readiness_2026_09.md)
  for validation evidence, merge scope, and decisions required before release.

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
  and `stabilize_unitary` now covers DMRG, mixed direct/fallback, and the
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
