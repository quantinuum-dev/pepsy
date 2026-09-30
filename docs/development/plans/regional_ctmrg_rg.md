# BP-guided regional CTMRG/RG plan

Status: deferred

## Objective

Develop a systematic finite-PEPS renormalization mode that coarse-grains
chosen local regions, uses converged BP messages as their external
environments, preserves explicit bra/ket and additional operator layers, and
executes repeated contractions through reusable Cotengra contractors.

The current experimental `ctmrg_projector_region=(2, 3)` implementation is
only a precursor. It enlarges Quimb's ordinary local projector calculation by
one neighboring boundary site. It is not an arbitrary regional RG schedule,
does not construct and combine separate bra/ket isometries, does not transport
BP messages, and does not cache compiled regional contractors.

## Algorithm contract

1. **Systematic chosen regions.** Represent an ordered RG schedule explicitly.
   Start with 2x2 and 2x3 rectangular regions, then allow coordinate-selected
   regions and user-supplied schedules. Define overlap, boundary fallback,
   sweep-direction, retagging, and coarse-lattice rules rather than inferring
   them from temporary Quimb tags.
2. **BP messages as environments.** Run D2BP on the current layered network,
   check and report convergence, and attach its directed matrix messages to
   every cut leg of the selected region. Reuse or transport compatible messages
   after each coarse-graining step; do not silently substitute diagonal
   simple-update vectors.
3. **Explicit layer-preserving projectors.** Keep `KET`, `BRA`, and optional
   `PEPO` or other layers separate while constructing projectors. For a norm,
   derive a message-weighted ket isometry and its compatible bra adjoint so the
   combined double-layer map preserves Hermitian/positive structure. Define the
   corresponding paired oblique-map policy for overlaps before implementing
   it. Contract projectors into a coarse tensor only after the layer relation is
   established.
4. **Compiled Cotengra execution.** Build contraction trees or array contraction
   expressions for BP updates, reduced environments, projector formation, and
   coarse-tensor assembly. Cache contractors by region topology, index sizes,
   backend, dtype, output ordering, and contraction options; invalidate the
   cache whenever a coarse-graining step changes one of those keys.

This is regional BP-guided coarse graining, not generalized/Kikuchi BP. A true
GBP region-message solver remains a separate possible environment provider.

## Implementation stages

### 1. Region model and reference contraction

- Add a private immutable region/schedule representation before choosing a
  public API.
- Implement one exact, untruncated 2x2/2x3 regional contraction oracle.
- Return region coordinates, input/output tags, layer membership, and shape
  metadata in diagnostics.

### 2. BP environment preparation

- Run one D2BP solve per current coarse network and expose convergence,
  iterations, residual, and message snapshots.
- Extract directed messages for every region boundary leg and validate their
  index orientation and dimensions.
- Design message transport and warm-start fallback when tensors or bond
  dimensions change.

### 3. Bra/ket projector construction

- Construct ket and bra reduced environments separately without flattening the
  layers prematurely.
- For norms, enforce the adjoint relation between layer maps and test positivity
  and reality of the resulting contraction.
- For overlaps and PEPO insertions, define paired left/right oblique maps and
  their layer ordering explicitly.
- Insert, contract, and retag the combined projector/coarse tensor atomically.

### 4. Cotengra contractor cache

- Compile reusable expressions rather than invoking a new path search for each
  same-shaped region.
- Separate path-search time, expression-build time, and execution time in
  diagnostics.
- Support a supplied `PathOptimizer` or `ContractionTree` and deterministic
  cache keys. Keep uncached execution as a correctness reference.

### 5. Adaptive policy and hardening

- Compare 2x2 and 2x3 candidates using discarded weight or a reduced-density
  residual; use the larger region only when justified.
- Benchmark lattice size, PEPS bond dimension, boundary dimension, region
  shape, BP tolerance, contraction sequence, backend, runtime, and peak memory.
- Add explicit policies for BP non-convergence and contraction-budget failure.

## Acceptance criteria

- Untruncated regional RG matches exact contraction on deterministic small
  PEPS norms and overlaps.
- Norm results remain real and nonnegative within tolerance and are invariant
  under allowed gauge transformations and reflected sweep order.
- D2BP messages are demonstrably attached to all regional boundary legs;
  convergence and warm-start behavior are observable.
- BRA/KET tensors remain separately identifiable until the combined projector
  is formed.
- Repeated same-topology regions hit a compiled-contractor cache, with measured
  path-search savings and no numerical change from the uncached reference.
- A multi-seed/model benchmark establishes when 2x3 improves accuracy per unit
  time over native 2x2 CTMRG.
- Defaults remain unchanged; the new mode stays opt-in until these criteria are
  met.

## Initially deferred

- Periodic regional schedules.
- Native Symmray/fermionic projectors.
- Generalized/Kikuchi BP region messages.
- MERA-style disentangler optimization.
