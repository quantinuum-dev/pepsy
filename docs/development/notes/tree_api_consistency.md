# TreeOptimizer API consistency audit

Audit date: 2026-09-09. Scope: the optimizer facade around the current tree
path implementation, including constructor/run options, direct operator
calls, copies, state replacement, controls, and local/MPI shot dispatch.

## Confirmed fixes

| Problem | Corrected contract |
| --- | --- |
| `two_site_mode="dm"`, `"src"`, or `"dmrg1"` bypassed the main resolver | Constructor, legacy selector, run overrides, and copies agree on the effective route, compressor, and DMRG schedule |
| Rejected `run` calls could clear the DMRG alias or replace options before raising | Validate prospective configuration and stream labels before installing options, queue, or RNG |
| Shot overrides changed parent mode, seed, or tracking | Child-only overrides; explicit `run_kwargs` takes precedence; template-copy RNG restoration uses `finally` |
| DMRG `apply_subtreempo(..., max_bond=1)` inherited parent cap 8 and produced bond 2 | Forward effective cap and cutoff through both initial-guess construction and TreeFIT; retain parent defaults |
| Invalid resource limits were truncated with `int()`, and invalid FIT strategies/seeds failed late | Validate integer types, ranges, initialization names, and finite cutoffs at the API boundary |
| MPI-only options could silently fall through to ordinary single replay | Nondefault MPI controls trigger shot dispatch and reject use without MPI |
| The latest FIT getter remained stale after a non-FIT update or state replacement | Getter describes the latest completed update; state replacement clears latest and historical FIT diagnostics |
| Capping the default three-child root retained `top_arity=3`, so subsequent copies/shots rejected their own two-child plan | Synchronize stored root arity after caps and state installation |
| Direct `measure(q)` resolved compact positions twice after stable-label caps | Use compact positions for tensor access and original logical labels for public updates |
| Direct computational measurement counted projection probability as compression loss and rejected positive rare outcomes | Mark the projector non-unitary for norm accounting, use the local projected weights without a probability floor, and normalize positive branches with `eps=0` |

The new API regressions compare states with constructor-configured references,
independent dense vectors at sufficient rank, and queued controls. Finite-rank
SRC/FIT is not assumed to reach the optimal Schmidt approximation after a
fixed number of sweeps. Tests inspect the actual guess/final caps, effective
cutoff diagnostics, canonicality, parent state/configuration/RNG, and child
settings. A weak two-site rotation with a per-call cutoff drops the expected
Schmidt component even though the parent cutoff is zero.

## Option scope and intentional differences

- Ordinary run overrides of mode, compressor, compression seed, and norm
  tracking persist. Shot overrides affect children only. The full option table
  is in the [tree API](../../api/optimizers/tree.md).
- `fit_*` controls remain constructor-only, inherited by copies and shots.
  Both `run(fit_n_iter=...)` and child `run_kwargs` reject these unsupported
  keywords rather than silently ignoring them.
- The `direct` mode is a route selector compatible with an independently
  configured compressor. To reset both on an existing optimizer, specify
  `mode="direct", compression_mode="direct"`. Algorithm-owning shorthands
  reset the compressor to their named method.
- Explicit structured `apply_submpo` retains its chain-operator routing and
  final subtree compression in DMRG configurations. Use ordinary gates,
  `apply_subtree_operator`, or `apply_subtreempo` for FIT execution.
- A per-operator `None` means inherit. This audit does not introduce a sentinel
  for requesting an uncapped operator update on a capped optimizer.
- `measure` returns a computational bit; `measure_pauli` returns an eigenvalue
  and probability. Standalone `TreeFIT` retains its own lower-level interface
  and traversal default. Native Symmray's degeneracy-preserving cap policy is
  unchanged.
- Validation rejection preserves replay configuration and the queued stream;
  this is not transactional rollback of numerical failures after replay starts.

## Upstream compatibility audit

Reviewed the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray repository](https://github.com/jcmgray/symmray). The
[Abelian-array documentation](https://symmray.readthedocs.io/en/latest/abelian_arrays.html)
repeatedly failed to fetch; installed signatures and the repository were used
for that part of the audit.

The active genpy environment contains Quimb `1.15.1.dev51+g2e99c793e`, Autoray
`0.11.1.dev3+g1b476b305`, Cotengra `0.8.3.dev7+g1d7fd333f`, and Symmray
`0.3.2.dev8+g6c6dd34b5`. Inspected installed `tensor_split`, `tensor_contract`,
`ContractionTree.get_contractor`, `AbelianArray.svd_truncated`, split dispatch,
and cutoff-mode dispatch. Quimb's cutoff codes are `abs=1`, `rel=2`, `sum2=3`,
`rsum2=4`, `sum1=5`, `rsum1=6`; the facade accepts those public conventions.
The preceding path audit also exercised NumPy, Torch, JAX, and Symmray
QR/SVD/tensordot paths with these same installed versions.

Classification: **adopt** the installed public option/dispatch contracts and
validate them consistently; **defer** unrelated upstream compressor,
HilbertSpace, and fermionic ordering changes. No compatibility shim, installed
package patch, global numerical change, or new upstream algorithm selection
was required. Native QR retains the shared structural-zero safeguard.

## Validation and limits

The combined affected suite passed **980 tests, 4 skipped**, in 53.07 seconds:
the new 62 API cases, core tree optimizer/path/FIT/compression tests,
tree/MPS parity, tree sampling, TreePeps/TreePEPO, TreeMPO, noisy trajectories,
MPI unit tests with fake communicators, public imports, and package layout.
The strengthened MPI parent/child override test also passed separately after
the combined run. Ruff (`src tests`), the skill catalog and tree-skill
validators, and `git diff --check` pass.

No new performance benchmark, GPU benchmark, or live multi-rank MPI run was
performed for this API cleanup. The previous full-repository attempt remains
blocked by the independently reproduced BP/Symmray bond-vector mismatch
documented in the [path review](tree_path_execution.md). The affected suite
does not establish a passing full repository.

The first pass left a numerical discrepancy: the separate Pauli-event path
derived probabilities from an expectation and rejected probabilities at or
below `1e-12`. A computational state with amplitude `1e-9` in the selected
branch had dense probability `1e-18`, but `measure_pauli("Z", 0, outcome=-1)`
reported zero and rejected it. The follow-up below resolves that issue.

## Follow-up: primary operator API and control consistency

The second pass on 2026-09-09 makes `apply_sub_mpotree` the primary public
TreeMPO implementation and routes ordinary gates through it in every tree
compression/FIT mode. `apply_subtreempo`, `apply_sub_tree_mpo`,
`apply_sub_treempo`, and `apply_subttno` are aliases of the same function;
there is no wrapper or second algorithm. The matching `sub_mpotree_event`,
event-parts, and predicate helpers are available. The event builder retains
the established `subtreempo` wire marker, and TreeOptimizer also accepts the
new spelling in explicit tuples/mappings.

User traversal choices remain `fit_traversal="auto"`, `"depth"`, and
`"depth-first"`. They govern FIT on the actual operator support, whether the
input is a dense gate or a multi-site TreeMPO. Auto uses endpoint sweeps on
paths, including two-body geodesics, and depth-first on branches. Explicit
policies preserve their requested ordering. They are separate from the
direct/SRC/SDC/zipup compression algorithms. New tests execute branched
three-site operators with both explicit policies and auto in `dmrg1/2/3`,
check resolved diagnostics and copies, and compare with dense statevectors.

Pauli measurements now expose both probabilities through the private shared
runner protocol `_measurement_probabilities`. One-site queries apply the two
2x2 projectors to a normalized canonical tensor. Multi-site queries rotate
only the measured physical legs, attach their Z bits, and merge XOR parity
through dimension-two indices. Each active tree edge sends one lossless QR
message; only its `R` factor is requested (`absorb="rfactor"`), and the final
tensor carries both orthonormal-subtree amplitude norms.
This preserves coherent parity sectors and avoids cancellation in `1 +/-
expectation`. It needs neither a dense multi-site projector nor a full state
or optimizer copy, and never truncates a probability query. Live state changes
are lossless gauge moves only. Numerical messages are private to the query.

Direct Pauli measurements, queued controls, and coalesced shots share those
paired weights. The shared runner preserves MPS's existing logical-to-physical
mapping and uses the paired protocol for TreeOptimizer. Positive selected
branches normalize with `eps=0`, and probability calculation uses working
tensors independently of the represented exponent. The existing TreeMPO
projection/compression mode remains responsible for applying the collapse;
the probability query does not choose another compression algorithm.

Further API fixes reject fractional gate/operator supports, cap labels, and
compact-position lookups instead of truncating them to another qubit. The
`canonize_around_qubits` facade resolves logical labels after stable-label
caps. `normalize(eps=...)` rejects nonfinite thresholds consistently with
replay, and `cap(stable_labels=...)` requires a boolean value.

Upstream sources and installed versions were rechecked for this follow-up;
they are unchanged from the first pass above. Additional installed probes
covered `Tensor.gate` (`transpose` is the current spelling) and NumPy/Torch/JAX
QR, SVD, and tensordot dispatch. Classification remains **adopt** the public
tensor contraction/gate/QR contracts and **defer** unrelated upstream changes.
The new probability route explicitly rejects native fermionic qubit
measurements, preserving that existing boundary and native QR policy.

Follow-up validation:

- Broad affected suite: **1,018 passed, 5 skipped** in 56.24 seconds.
- Final API/probability cases with JAX x64 enabled in the test process:
  **103 passed**. These include independent dense-reference rare X/Y/Z and
  multi-site weights near `1e-18`, entangled complex64 states with physical
  root sites on NumPy/Torch, JAX complex128, parity coherence, represented
  exponent 400, stable labels, and a 24-site query forbidden from using dense
  operators, dense statevectors, or optimizer copies.
- After requesting only QR's `R` factor, reran probability, core tree, and
  trajectory suites: **511 passed, 4 skipped** in 34.51 seconds, with JAX x64
  enabled. This separately validates the final NumPy/Torch/JAX QR dispatch.
- Ruff, skill catalog validation, tree-skill validation, and diff whitespace
  checks pass. No GPU benchmark or live multi-rank MPI run was performed.
- A fresh headless full-repository fail-fast run again stopped at the same
  unrelated BP/Symmray bond-vector error: **182 passed, 1 failed** in
  11.14 seconds. No full-repository success is claimed.

## Follow-up: mode implementation and readout review

The third pass on 2026-09-09 traced the primary operator dispatch and its
lower-level compatibility methods. Comments now distinguish these algorithms:

| Route | Operator preparation and reduction |
| --- | --- |
| Direct / DM | Complete lossless QR routing, then canonical edge compression using SVD / `svd:eig` |
| SRC | Layered target, product-noise complementary environments, nested QR projections; cutoff ignored with warning |
| SDC | Layered target, deterministic truncated-SVD complementary environments, nested QR projections |
| Zipup | Absorb ready child messages and truncate outgoing messages immediately; unvisited boundaries are not canonical |
| DMRG1/2/3 | Exact layered target, independent initial guess, cached local FIT sweeps and configured block schedule |

Explicit direct/DM settings select FIT's local split too; SRC/SDC settings map
to direct local SVD. The guess algorithm is controlled independently by
`fit_init_strategy`. Diagnostics now expose the actual local `split_method`.
Auto traversal selects endpoint windows on paths and depth-first on branches;
explicit `depth` / `depth-first` remain available for all FIT supports.
The unused alternate private Pauli-projector implementation was removed;
live projection continues through the shared TreeMPO route in every mode.

Three additional bugs were reproduced and fixed:

1. `expectation_mpo` consumed the live sampling RNG when making its private
   copy, even for deterministic modes. Restore RNG state in `finally`; the
   private optimizer still receives its derived seed and configured method.
2. Replacing a state with the same number of qubits but another layout left
   cached TreeMPO gate factors on the old TreePlan. Replaying the same gate
   object then raised a plan-mismatch error. State replacement and caps clear
   both path and operator-factor caches; ordinary unchanged-layout replay
   retains caching.
3. Approximate expectation diagnostics depended on replay history. Identity
   readout of a normalized entangled state with private cap one returned about
   `0.681`, versus exact `1.0`, while history-disabled readout reported no cuts.
   FIT also reported no cuts despite its approximation. Requested measurement
   records now work independently of replay history without enabling spectral
   probes. Multi-node FIT warns and exposes `fit_diagnostics` plus
   `approximation_possible`. Single-node exact FIT is exempt. The existing
   `truncated` field describes recorded rank reductions, not certified error;
   exact MPO readout remains a separate contraction.

Rechecked the upstream sources listed above and the installed versions, which
are unchanged. Installed probes covered `tensor_split`, `tensor_contract`,
`Tensor.gate`, `ContractionTree.get_contractor`, Symmray truncated SVD/QR, split
driver dispatch, and NumPy/Torch/JAX QR/SVD/tensordot. The Abelian documentation
fetch again failed; installed signatures and the Symmray repository supplied
that evidence. Classification remains **adopt** the public contracts and
**defer** unrelated upstream algorithms and ordering changes. No installed
package, global numerical default, or native QR policy changed.

The affected integration suite passed **1,047 tests, 4 skipped**, in 58.13
seconds, with JAX x64 and a headless plotting backend. Coverage includes tree
paths and branches, SRC/SDC comparisons with Quimb, native Symmray routes,
NumPy/Torch/JAX cases, noise, MPI unit tests, public imports and package layout.
The new regressions cover all eight named compressor/FIT modes for cached
operators after layout replacement and private readout state/RNG preservation.
After adding the effective FIT split field and chain-input coverage, the final
API, measurement, and FIT suites passed **200 tests** in 17.85 seconds. Tests
observe actual Quimb split-driver calls for direct, DM, SRC, and SDC FIT
configurations and distinguish the independent guess method. Ruff (`src tests`),
skill catalog/tree-skill validation, and `git diff --check` pass.

This correctness review adds no performance claim. Readout still copies
optimizer histories; unusually long histories can make those copies expensive.
The tree one-site FIT neighbor-carry optimization remains a separate candidate
requiring rank-change and backend validation. No GPU benchmark or live MPI
run was performed. The previously reproduced full-repository BP/Symmray failure
above remains outside this tree change.

## Follow-up: represented operators and exterior identities

The fourth pass on 2026-09-09 found discrepancies that unscaled, freshly
built gate tests did not exercise:

- A local TreeMPO with `exponent=2` had dense transformed norm `100`, while
  direct, DM, SRC, SDC, and zipup returned norm `1`. DMRG's layered target
  already retained the exponent. Tensor-local routes now add the operator
  exponent once after successful reduction, without forming a power of ten.
  Tests also use operator exponent `400` plus state exponent `3`, comparing
  working tensors and the final exponent `403` without overflow.
- Canonicalizing the same gate operator changed its applied norm from `1`
  to about `2.828` in the five compression routes. Scaling every operator
  tensor by `1.1` returned norm `1.4641` instead of `2.14358881`. Both arose
  because operator factors outside the nominal support were omitted despite
  no longer being unit identities.
- The public operator exponent was separate from the exponent on its stored
  Quimb network. Copies, conjugation, exact expectation, and matrix elements
  could therefore disagree with dense operator readout. `_scaled_tree_networks`
  supplies private views with the public offset while preserving relative
  sector exponents. Addition aligns sector scales before the direct sum;
  composition accumulates the two represented exponents.
- Native statevector readout could return an object array containing a
  Symmray wrapper. Simply unpacking it was insufficient: native contraction
  and gauge moves can trim empty physical sectors, changing vector dimensions.
  Readout now keeps separate physical legs and restores declared sector maps
  on a private result before unpacking the numeric array. This is confined to
  explicit dense readout; application and measurement stay block sparse.

The support shortcut now requires builder-proven exterior identities whose
array references and index metadata are unchanged. No numerical contraction,
dense projector, hashing of tensor entries, or host transfer is added to this
check. Ordinary cached gates keep their minimal Steiner/path routes. Copies,
scalar scaling on the active support, conjugation, and internal backend
conversion preserve the proof. Arbitrary external operators and modified
exteriors use the complete tree in every mode, including FIT guesses. This
fallback is conservative and can cost more; it does not claim to recover an
optimal local route after arbitrary operator gauge changes. Unmanaged array
entry edits require `invalidate_canonical_form()`, which now also clears the
identity proof. A raw `operator_support` hint alone cannot certify elision.

The mode-specific algorithms, SRC numerical-environment lifetime, native QR
policy, FIT guess/split separation, and explicit `depth` / `depth-first`
traversals are unchanged. Comments at the shared application boundary now
explain both the scale handling and the conditions for a minimal route.

Upstream sources were checked again; the installed versions remain those
listed above. Probed the actual tensor split/contract/gate, network scalar
multiplication, Cotengra contractor, and Symmray truncated-SVD signatures,
plus NumPy/Torch/JAX dispatch. The native readout follow-up also inspected
Symmray `to_dense`, `copy_with`, `BlockIndex.copy_with`, and `unfuse_all`.
`to_dense(index_maps=...)` only reorders an existing basis and cannot restore
trimmed sectors; the readout therefore restores the original physical index
metadata before conversion. Reviewed Quimb's direct-sum implementation, which
combines tensor entries without aligning represented scales. Classification:
**adopt** those actual public contracts at Pepsy's boundary; **defer** unrelated
upstream algorithm and ordering changes. No package patch or global dispatch
change was needed. The Abelian documentation fetch still failed, so installed
source and the Symmray repository supplied that part of the evidence.

The final representation suite passed **36 tests** in 6.73 seconds: all eight
named modes, canonicalized/distributed operator factors, exponent `403`,
copies and arithmetic, relative sector scales, stale/unproven support hints,
rejection before mutation, native direct/zipup/even-parity FIT, and Torch/JAX
conversion. Native tests verify numeric vector shape and physical-site output
order, including nonzero occupied sites whose positions change on reversal.
The final affected integration suite passed **1,095 tests, 4 skipped**, in
60.34 seconds. Ruff (`src tests`), the skill catalog/tree-skill validators,
and `git diff --check` pass. GPU performance and live multi-rank MPI remain
untested; the previously reproduced unrelated full-suite BP/Symmray failure
remains outside this change.

## Consolidation follow-up — 2026-09-09

The approved [consolidation plan](../plans/tree_optimizer_consolidation.md)
is implemented. Pure helpers now own policy resolution, operator geometry and
local target preparation, projected-amplitude probability kernels, and
diagnostic record construction. TreeOptimizer keeps live state and numerical
dispatch, one nested-update finalizer, and one configuration/state-construction
path for copies, FIT guesses, and layout pilots. TreeLayoutFinder owns pilot
search orchestration. SRC/SDC and TreeFIT numerical environments stay with their
existing engines. Public signatures, defaults, mode precedence, seed draws,
and traversal behavior are preserved.

Approximate MPO readout now creates private work without copying retained
histories or queued gates. Public `copy()` retains both as before. Installing
a selected layout also clears cached gate factors tied to the old plan.
Regression tests cover private-readout history independence, RNG restoration,
new profiling events, and layout-cache invalidation. The implementation record
contains the exact characterization, integration results, retained boundaries,
and matched NumPy/Torch CPU timings.

The upstream audit rechecked the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray repository](https://github.com/jcmgray/symmray). The
[Abelian-array documentation](https://symmray.readthedocs.io/en/latest/abelian_arrays.html)
fetch failed again; installed source supplied that part of the API evidence.
Installed versions were Quimb `1.15.1.dev51+g2e99c793e`, Autoray
`0.11.1.dev3+g1b476b305`, Cotengra `0.8.3.dev7+g1d7fd333f`, and Symmray
`0.3.2.dev8+g6c6dd34b5`.

Probes inspected actual `Tensor.copy`, `TensorNetwork.copy`, `Tensor.gate`,
`tensor_contract`, `tensor_split`, and `ContractionTree.get_contractor`
signatures; NumPy/Torch/JAX/Symmray QR, SVD, and tensor-contraction dispatch;
and Symmray's dispatched `linalg.svd_truncated`. Classification: **adopt** the
existing public copy/gate/contraction and decomposition contracts while
preserving explicit transpose, represented exponents, and graded native
safeguards; **defer** unrelated upstream compression variants, ordering/default
changes, and contractor options. No compatibility shim, dependency edit, or
global dispatch change was needed. The affected paths are the new private
tree helpers and optimizer/layout orchestration; numerical engines remain
unchanged by this refactor.

Final verification: 14 bitwise-identical statevectors and 1,004 identical
baseline records; 1,181 affected checks passed with four skips across the
broad run and the corrected TreeStab instrumentation test. The full-suite
BP/Symmray failure was independently reproduced against the saved source
from before this refactor. Ruff passes. GPU and live multi-rank MPI remain
untested.
