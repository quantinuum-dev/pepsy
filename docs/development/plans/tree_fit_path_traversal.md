# Geometry-aware tree FIT traversal

Status: implemented on 2026-09-09. The user expanded the scope to ordinary
direct/path operator execution as well as FIT. See the
[implementation and validation record](../notes/tree_path_execution.md).

## Objective and scope

Exploit the actual active region of a TreeOptimizer update. A two-site gate
has a path-shaped Steiner region even when the complete tree branches. Some
few-body gates also have path-shaped regions. Use endpoint sweeps on paths,
retain depth-first traversal on branched regions, and retain the exact
one-node fast path. Apply this to generic DMRG and DMRG1/2/3 and coordinate
the SRC initial guess with the selected sweep.

The implementation also prepares direct/DM operators exactly along paths before
one directional canonical compression pass, streams layered path contractions,
and applies endpoint routing to SRC/SDC, zipup, subtree operators and sub-MPOs.
It retains their existing numerical algorithms and the optimized low-level
two-qubit threading/sibling kernels.

Profiling refined the direct preparation: exact TreeMPO QR can peel from both
ends before moving to the compression entry endpoint. This limits intermediate
QR ranks while preserving one endpoint-to-endpoint truncation pass.

Keep the TTN representation and the original layered target. Internal path
nodes can have exterior virtual legs and physical legs; they are not ordinary
qubit-only MPS sites. Do not convert the network to an MPS or fuse the full
active target into one tensor. Large exterior dimensions can still make
two-/three-node factorization expensive.

Relevant implementation: `src/pepsy/fitting/tree.py` owns block traversal,
local center selection, effective tensors, and environment invalidation.
`src/pepsy/optimizers/tree/optimizer.py` owns active-region/guess preparation
and the public FIT options. `src/pepsy/optimizers/tree/compression.py` already
accepts directed peel orders and a requested hub for SRC/SDC.

## Public policy and compatibility

Introduce `fit_traversal="auto"` in TreeOptimizer and `traversal="auto"`
in TreeFIT. Its resolution is:

| Active region | Resolved execution |
| --- | --- |
| One node | Existing exact local solve when enabled |
| Connected region with induced degree at most two | Endpoint path sweep |
| Connected region with a branch | Existing depth-first sweep |

Classify induced degrees inside the active region, not degrees in the complete
tree or the number of operator sites. Validate connectivity first. A node
with two active neighbors and several exterior neighbors still belongs to a
path-shaped region.

Preserve explicit `"depth"` and `"depth-first"` behavior as reproducible
baselines. Do not silently redefine the recently adopted depth-first option.
Develop and validate auto explicitly, then make it TreeOptimizer's default
for all four DMRG modes. Standalone TreeFIT keeps its existing default so
other callers do not implicitly change. Copies and trajectory workers must
preserve the requested option.

Do not alter the default SRC guess for dense trees, the native fermionic
direct guess, rank-growth policy, block-size transitions, iteration budgets,
cutoff semantics, stopping tolerance, or explicit user overrides. Finite-rank
results can change with the new update/factorization order; document this.

## Path ordering and direction

Build the ordered node path in linear time from its two endpoints. Before
constructing the guess, choose the reference entry endpoint nearest the
incoming state's tracked canonical center, breaking ties by node id. If the
center is unknown, use a deterministic endpoint-id tie-break rather than an
extra canonicalization. Freeze this orientation for the entire FIT target.

For auto paths, define the existing pass labels explicitly:

- `inward-outward` and its existing aliases: reference entry to opposite end,
  then reverse.
- `outward-inward` and its aliases: the reverse pass first, then return.

These labels select pass order on a path; they no longer describe radial
movement toward a medial node. The explicit legacy traversals retain their
current radial semantics. Reversing the requested sequence must really
reverse the first pass, rather than being canceled by endpoint selection.
The actual first endpoint also determines the desired initial-guess center.

For `P = (D, B, A, C, G)`, a forward pass has:

| Block size | Blocks | Desired center after each block |
| --- | --- | --- |
| 1 | D; B; A; C; G | Updated node |
| 2 | DB; BA; AC; CG | B; A; C; G |
| 3 | DBA; BAC; ACG | A; C; G |

Reverse both block order and directional centers on the return pass. Use
consecutive windows, covering every current path block once per pass. Keep
the existing short-window fallback (3 to 2 to 1). Directional centers must be
supplied through the existing `fit_block(center=...)` path; merely reordering
blocks leaves the current medial-center factorization policy in place.

Adjacent multi-node windows then contain the preceding center. Existing
canonical preparation can skip interior center moves, while factorization
establishes the next center. The three-node endpoint factorization has a
focused regression already; extend it across directions and phase changes.
Do not initially remove repeated turnaround updates or change sweep budgets.

## Canonical boundaries and environment lifetime

Prepare only the exterior gauge needed for the active region/block. A
tracked center already inside the next block needs no preliminary QR. For
an outside center, invalidate the changed path before moving it, and stop
at the first active-block node as the current implementation does.

Off-path branches are isometric boundaries, but mixed target/guess overlap
messages are not automatically identities: the two branches can have
different gauges. Keep exact cached overlap messages unless identity is
established by a specific proof. Do not introduce identity substitution as
part of this traversal change.

Retain the existing dependency-based message invalidation. A fitted update
invalidates messages containing the changed block; an unchanged opposite
branch retains its messages. Resolve fitted indices from live tensors after
QR/rank changes; target indices remain private and fixed. Verify actual
isometries as well as center metadata.

Cache the ordered path and block/center schedules once per run, keyed by
block size and pass direction. Reuse them across 3-to-2-to-1 transitions.
Keep numerical overlap environments local to that target and preserve their
dependency lifetimes. Existing bounded SRC caches may retain immutable
geometry only. No numerical reuse across gates, array edits, or failed fits.

## Align the SRC warm start

First validate endpoint FIT sweeps with the existing guess unchanged, so
traversal gains and numerical changes can be assessed independently.

Then allow the disposable SRC guess to use the same path plan. Peel from
the opposite endpoint toward the actual first FIT endpoint, so its final
canonical center is already in the first fitting block. The corresponding
SRC fixed environments run in the opposite direction and require only
`len(path)-1` directed messages. Retain the original local operator/state
layers and the existing Q-only projector implementation.

Pass a private validated order/hub through the shared guess/compression
boundary instead of duplicating SRC or adding a public hub-selection API.
Share the run's plan rather than recomputing geometry independently in the
optimizer, guess optimizer, and FIT. An explicitly selected SDC guess can
use the same geometry through its existing deterministic kernel. Direct and
native guesses may use existing preparation plus one necessary entrance
move; do not claim SRC support for native arrays.

Preserve the parent's child-seed draw and existing `fit_init_seed` policy.
Changed sketch visitation order can change seeded approximations, so require
reproducibility for the new policy and document that old bitwise results are
not promised. Keep parity tests against unmodified Quimb SRC on actual paths.
The original plan scoped this to FIT. The implementation request expanded
endpoint path routing to ordinary direct/DM, SRC/SDC and zipup gates too;
their compression methods and truncation semantics remain distinct.

## Validation and acceptance

Add focused tests that demonstrate the intended algorithm rather than only
checking an option string:

1. Geometry: one/two/three-node regions, long paths crossing the structural
   root, a physical qubit on an internal/root node, non-monotonic node ids,
   few-body collinear supports, true branching, and rejected disconnected
   regions. Operator argument order must not be confused with path order.
2. Schedules: both pass sequences and aliases; DMRG1/2/3; short windows;
   3-to-2-to-1 transitions; explicit legacy behavior; exactly the intended
   block coverage; first/last centers; copies and trajectory forwarding.
3. Mathematics: compare a few local updates against complete projected-target
   contractions, then lossless replays against independent statevector gates.
   Check actual isometries, norms/scales, and untouched exterior branches.
   Include zero targets, weak entanglement, rank changes, and finite caps.
4. Cache integrity: compare cached messages with full branch contractions,
   exercise changed fitted indices and direction reversals, and check failure
   recovery. Assert that unchanged exterior branches do not rebuild on every
   sweep and changed branches cannot reuse stale values.
5. Backends: dense NumPy/Torch and existing supported JAX paths; independently
   validate supported even-parity native Symmray/Torch FIT with direct guesses
   and graded QR. Keep unsupported cases explicit. Do not enable an untested
   native shortcut solely because dense tests passed.
6. Performance: compare explicit depth, depth-first, and auto at identical
   budgets/seeds. Separate initial guess, canonical preparation, effective
   tensors, factorization, cache misses, actual center travel, and total replay
   time. Measure without instrumentation after warmup; report fidelity too.

Repeat the existing small exact-reference case, 24-qubit product and entangled
replays, cap-200 single-gate compression, and a branched three-qubit workload.
Auto's branched fallback must reproduce explicit DFS given identical inputs.
Path measurements must demonstrate endpoint-consecutive execution and lower
unnecessary center travel; runtime improvement is measured, not assumed.
Report CPU/GPU scope accurately and do not transfer MPS complexity claims to
TTN nodes with large exterior legs.

Record requested/resolved traversal, oriented endpoints, and actual final
center in compact FIT diagnostics. Keep detailed work counters opt-in rather
than growing a per-block history by default. Run the closest FIT/tree suites,
native regressions, shared-caller checks, lint, and the required upstream API
audit before promoting auto to the TreeOptimizer default.

## Implementation sequence

1. Add immutable run planning and opt-in auto traversal; keep guesses fixed.
2. Implement and test consecutive blocks with directional centers and
   incremental message reuse for DMRG1/2/3.
3. Align SRC/SDC guess orientation with the first fitting block; validate
   seeded behavior, environment counts, and failure isolation separately.
4. Complete correctness and matched-budget benchmarks, then adopt auto as
   TreeOptimizer's default and update API docs, changelog, and tree skills.

Skipping the last QR-factor absorption into a tensor that will immediately
be overwritten is a separate possible optimization. It needs its own proof,
rank-change/backend checks, and failure handling; it is not a prerequisite
for this traversal work and must not be mixed into the first comparison.
