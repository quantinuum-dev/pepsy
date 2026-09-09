# Tree SRC, SDC, and zipup algorithm audit

## Compression review follow-up (2026-09-09)

Reviewed ordinary TreeMPO direct/DM routing, canonical path and branching
compression, SRC/SDC environment construction and release, and the focused
algorithm tests. No compression implementation was changed in this review.

The active `/Users/rezah/envs/genpy` environment now contains Quimb
`1.15.1.dev51+g2e99c793e`, Autoray `0.11.1.dev3+g1b476b305`, Cotengra
`0.8.3.dev7+g1d7fd333f`, and Symmray `0.3.2.dev8+g6c6dd34b5`.
Rechecked the Quimb changelog, Autoray repository, Cotengra documentation and
changelog, and Symmray repository; the Symmray Abelian HTML page still failed
retrieval. Probed installed `Tensor.split`, `tensor_contract`,
`tensor_compress_bond`, `TensorNetwork.compress_between`,
`autoray.get_namespace`, and Quimb's SRC/SDC callable signatures, plus NumPy
dispatch for `svd_truncated`, `qr_stabilized`, and `svd_via_eig_truncated`.
The historical standalone DM failure described below no longer reproduces:
3-by-3 identity splits with `svd:eig`, cap 2, complex64/complex128, and
cutoffs 0 and 1e-7 all pass. No compatibility shim is needed for those probes.

Findings and classifications:

- **Defer / performance opportunity:** direct/DM use layered local inputs
  on paths, but eagerly fuse local operator/state tensors on branching
  regions. At internal nodes this forms an outer product before incoming
  messages can reduce the contraction. Its element count is the product of
  the two tensor sizes. Extending delayed fusion to branching regions needs
  dense/native correctness and peak-memory measurements before adoption.
- **Defer / approximation policy:** SRC sketches use exactly `chi` samples;
  there is no separate oversampling rank. SDC similarly caps complementary
  environments at the output rank. Neither guarantees optimal retained
  subspaces of the complete target. Canonical direct/DM cuts are locally
  optimal in exact arithmetic, but a sequence of cuts is not a global
  fixed-rank TTN optimization. DM retains Gram-matrix conditioning concerns.
- **Defer / intermittent validation failure:** the combined selection of
  successive, path, hook, TreeMPO, and optimizer tests produced 488 passes,
  4 skips, and one failure in the immediate weak-reference-release assertion
  of `test_src_environment_objects_are_reused_then_released`. It passed
  alone, in 40 direct repetitions (20 with prior garbage collection), and
  with its preceding cache-plan test under the same five-file collection
  (2 passed, 491 deselected). Cause remains unresolved; this is not proof
  of a persistent leak, nor a clean full-suite result.

A small serial NumPy complex128 check used a balanced eight-site tree,
initial bond 4 (state seed 19), QR-generated complex gates (RNG seed 12),
output cap 2, zero cutoff, and SRC seed 31. Three warm timings followed one
warmup per case; operator construction and dense readout were excluded.
For support `(0, 2, 5, 7)`, direct/DM/SRC/SDC respectively took median
2.28/2.18/1.77/1.99 ms and yielded normalized squared fidelities to the exact
target of 0.442375/0.442375/0.149243/0.082012. All returned canonical states
and respected the active-region cap. The two-site path case also returned
canonical states; untouched exterior bonds correctly remained at 4.
These small examples establish neither general speed rankings nor error
bounds. In particular, deterministic SDC need not beat SRC or direct at the
same output cap. GPU performance and large-tree peak memory were not tested.

Follow-up source tracing of construction and canonical preparation found two
additional bookkeeping opportunities. `TreeMPO.from_gate` factorizes only
the active span but installs bond-one identities on the full plan; the
optimizer's 64-entry identity-keyed gate cache amortizes that construction
only for repeated immutable payload objects with matching support.
`plan_operator_application` still validates the complete operator and checks
exterior identity references on each application. Thus a local numerical
update can retain O(N) operator bookkeeping. Any validation cache would need
reliable mutation tracking; bypassing checks for arbitrary edited operators
would be unsafe.

Steiner selection correctly unions paths from one support node; TreePlan
caches the individual directed paths, while the union is rebuilt at its
call sites. Ordinary direct/DM/SRC/SDC preparation skips gauge work when the
tracked canonical region is contained in the active span, and otherwise
moves a known center only to its first entry. Actual edge moves consult live
`left_inds` (plus native charge alignment) and preserve the TTN-owned
`_canonical_region`; there is no independent optimizer `info_c` dictionary.
However, `_recover_center_from_region` rescans every remaining node for each
peeled leaf. Even when all QR operations are skipped using isometry proofs,
this traversal has quadratic bookkeeping for bounded-degree regions.
A leaf queue is a deferred optimization. `TreePlan._path_cache` is also
unbounded, unlike the bounded optimizer and successive-plan caches; workloads
with many distinct queried node pairs can retain substantial path storage.

### Implemented follow-up: region leaf queue and environment validation

Replaced the repeated scans in `_recover_center_from_region` with cached
region adjacency, degree counters, and a minimum heap of non-target leaves.
The smallest eligible node is still processed first, preserving both absorb
orientations, tensor operations, native charge checks, and `left_inds` skips.
Bookkeeping is O(E + R log R) rather than quadratic for a bounded-degree
region of R nodes; E includes incident adjacency entries read during setup.
The regression compares the complete edge sequence with the former scan
order and bounds neighbor queries on a 64-site tree for both orientations.

SRC/SDC compression mathematics and environment scheduling are unchanged.
Extended the existing weak-reference test to track deterministic SDC factors
as well as SRC sketches, verifying each directed environment is built once,
branch consumers reuse the same object, and numerical arrays are released
after return. The previous intermittent SRC failure did not reproduce in
100 additional direct repetitions with cyclic GC disabled; its historical
cause remains unresolved, and no speculative memory-management fix was made.

Validation: 462 tests passed and 4 skipped across optimizer, path execution,
and successive compression after the queue change. After extending the SDC
test, all 28 successive-compression tests passed. Repository-wide Ruff and
whitespace checks passed. These include installed Quimb path parity, dense
NumPy/Torch/JAX successive algorithms, native canonical movement, stale-cache
protection, and failed-call recovery. The full repository test suite was not
run. The upstream versions and API/dispatch probes recorded above remain
applicable; no upstream API or dependency change was introduced.

Date: 2026-09-08.

## Implemented algorithms

`optimizers/tree/compression.py` owns the shared dense successive compressor.
Inputs are groups of original target tensors at each tree node, an inward
peel order, and a final hub. TreeMPO application passes the separate operator
and state tensors directly. State-only and two-node edge compression use
the same engine. All exterior state branches are first made isometric toward
the active region, so their boundary bonds are valid output legs.

For a directed cut `u -> v`, SRC contracts the original target component on
the `u` side with independent local Gaussian output sketches. A common sample
index is retained as a hyperindex throughout the contraction. This produces
an environment with dimensions `(samples, original cut bonds...)`; it does
not apply randomized SVD to a local canonical core. Complex sketches use the
live array dtype/device and the supplied compression seed. Scalar environment
rescaling leaves sampled ranges unchanged and controls multiplicative drift.
The batch size is `chi`; projection can use fewer columns when the original
cut dimension is smaller. `chi=None` uses a full-cut rank bound. Nonzero SRC
cutoff is explicitly warned about and ignored, matching its fixed-rank nature.

SDC instead contracts each node with low-rank factors received from the
other neighboring branches, then retains a deterministic truncated-SVD factor
on the outgoing cut. Its environments are bounded by `chi` and the requested
cutoff convention. Direct SVD replaces Gram-matrix factorization here without
changing the low-rank-environment algorithm; it avoids squared conditioning
and the installed NumPy complex64 `svd:eig` compilation problem below.

Two iterative traversals construct directed environments without recursion.
The final inward pass contracts each node's original target layers, projected
child messages, and complementary environment. QR extracts the retained
isometry Q. Contracting Q-dagger with the original local target and child
messages produces the next exact projected message. The hub is the final
projected target, preserving its norm and scale; every other retained node
has an actual inward isometry and matching `left_inds` metadata.

On a path this construction reduces to Quimb's successive environment
algorithms. On a branching tree it is their hierarchical extension: nested
child subspaces are selected using directed sketches of the original
complementary components. We do not claim a published chain error bound
automatically applies to this extension. There is no Cholesky compressor here.

Zipup was already a genuine streamed tree algorithm: it contracts one node's
operator/state layers with incoming child messages and immediately truncates
the outgoing message. It has no complementary-environment pass. The new
regressions explicitly forbid the full-target routing and final canonical
compression paths while checking its state and isometries. It remains
different from both direct compression and SRC/SDC.

Edge histories record dimensions for all three algorithms; local singular
spectra of approximate environments are not reported as global discarded
weights. Norm-survival diagnostics retain their existing separate semantics.
TreeFIT's SRC/SDC guesses use the actual environment algorithms; its later
local variational refinements use ordinary direct SVD.

SRC/SDC currently reject native symmetry arrays rather than replacing the
algorithm or densifying the state. This also removes the former native SDC
alias to direct compression. Native direct and zipup retain their existing
graded routes. CuPy dispatch is supported by the dense operations but GPU
hardware has not been validated.

## Upstream compatibility audit

Reviewed the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray repository](https://github.com/jcmgray/symmray).
The [Abelian-array documentation](https://symmray.readthedocs.io/en/latest/abelian_arrays.html)
again returned a retrieval error.

Installed versions: Quimb `1.15.1.dev39+g369d09b9d`, Autoray
`0.11.1.dev1+gc56f64427`, Cotengra `0.8.3.dev6+g08fe1a3a1`, and Symmray
`0.3.2.dev6+ga17699db6`.

Inspected installed `tensor_network_1d_compress_src`,
`tensor_network_1d_compress_sdc`, and their low-rank-environment projection
sweep, including signatures and handling of `max_bond`, `cutoff`, `seed`,
`project_opts`, and `compress_opts`. The new implementation composes public
`Tensor.split`, `Tensor.isel`, `Tensor.H`, and `tensor_contract`, with explicit
output indices on sample hyperedges. Random arrays use Pepsy's existing
Autoray-backed random helper with explicit dtype, preserving its compatibility
fallback. No upstream code is vendored and no installed package is edited.

Classification: **adopt** successive complementary-environment compression
for the tree geometry; **adopt** stable direct SVD for deterministic factors;
**defer** native graded sketches, GPU benchmarking, and Cholesky compression.

The broader suite independently exposed an existing Quimb/Numba complex64
DM failure: `svd_via_eig_truncated_numpy` cannot compile its full-result branch
because two returns mix complex64/complex128 factors. An isolated public
`Tensor(np.eye(3, dtype='complex64')).split(method='svd:eig', ...)` reproduces
the failure at both zero and positive cutoff, without invoking Pepsy's tree
compressor. The existing tree DM cutoff test reproduces it alone. This change
does not alter DM dispatch or patch that unrelated upstream driver.

## Validation

- Direct algorithm comparison against Quimb on a five-site path, including
  identical SRC sketches and deterministic SDC output.
- Layered, branched MPS-like trees: NumPy, Torch, JAX; complex64/complex128;
  finite caps and lossless caps; a physical root; canonical/isometry metadata;
  exact target projection identities; and seeded reproducibility.
- Partial gate spans, ternary roots, weak branches, zero targets, state-only
  compression, and public two-node compression.
- Full-target routing and local randomized SVD are forbidden in dedicated
  tests, so passing a mode-name-only implementation is insufficient.

No universal speedup or GPU performance claim is made. Directed environments
remain bounded in sample/factor rank, but runtime and memory also depend on
original operator/state cut dimensions and tree arity.

Final checks (counts overlap): the dedicated algorithm suite passed 19 tests;
the broader tree optimizer, TreeFIT, trajectory, sampler, API, and package
layout selection passed 650 tests and skipped 4, with the one independently
reproduced upstream DM complex64 failure described above. The default smoke
suite passed 128 tests. Ruff, skill/catalog validation, and whitespace checks
passed. The full repository suite was not run: validation covered the changed
tree subsystem and its callers, and the known upstream DM failure remains
unresolved.

## Follow-up: paper and Quimb parity

Rechecked [the paper, v2, Sections 2.2 and 3](https://arxiv.org/html/2504.06475v2#S3)
and the installed `quimb.tensor.tn1d.compress.tensor_network_1d_compress_src`.
The core operations agree: reusable Khatri–Rao/product-state sketches, QR
range extraction, and projection of the original target. The paper treats
chains; our directed-environment, nested-QB construction for branching trees
is an extension, not a published tree algorithm or a claim of its error bounds.
Its dense reference test independently builds the complementary Khatri–Rao
columns and leaf QB projections of a three-branch tensor.

The follow-up removes avoidable work:

- Build only the dependency closure of complementary messages consumed by
  the projection sweep. A length-L path uses L-1 fixed environments instead
  of 2(L-1). Branching trees generally require both directions on more edges.
- Use one backend-native random generator and draw in first-use environment
  order. Seeded layered path results match the unmodified Quimb implementation
  in both directions; NumPy, Torch, and JAX are covered. This intentionally
  changes the earlier per-node seed-offset sequence.
- Use `Tensor.split(method='qr', absorb='lorthog')` so QR returns only Q.
- Release environments and projected messages after their final consumers.
- Drop intermediate tensor tags, as Quimb does, and restore only each output
  node's local tags. This avoids growing complementary-component tag unions.
- Prepare only exterior isometries. No center-moving QRs are needed inside
  the active subtree that the environment algorithm will replace.

Probed `autoray.get_namespace(like=None, device=None, dtype=None,
submodule=None)`, the namespace's `random.default_rng`, existing
`random.array` dispatch, and Quimb's Q-only split result `(Q, None)` in the
active environment. Dependency versions and the independently reproduced DM
complex64 failure are unchanged. Rechecked the upstream sources listed above;
Symmray's Abelian HTML retrieval still failed. Classification: adopt the
existing public RNG and Q-only QR interfaces; no compatibility shim or
dependency modification.

Serial Torch CPU kernel comparisons against commit `fe1f8ef`, using one Torch
thread and eight alternating post-warmup samples:

| Target | Previous median | Updated median | Quimb SRC |
| --- | ---: | ---: | ---: |
| 24-site layered path, MPS bond 16, MPO bond 4, output cap 16, complex64 | 7.393 ms | 5.273 ms | 5.378 ms |
| 12-site branching tree, 22 nodes/21 edges, state bond 8, six ZZ terms, output cap 8, complex64 | 5.528 ms | 5.364 ms | Not applicable |

These are compressor-kernel timings with target construction excluded. The
path result improves about 29%; the branching improvement is about 3% in this
small case. The Quimb comparison disables output array permutation; its call
includes segmentation/output assembly that the prepared tree kernel excludes,
so the close timings establish comparable scale rather than a speed advantage
over Quimb. Exterior-only preparation is outside these timed kernels.
RNG sequencing changed, so these compare equal shapes
and policies rather than claiming identical old/new random approximations.
No GPU or full replay speedup is established.

Final follow-up validation: 24 dedicated algorithm tests passed. The broader
tree/FIT/trajectory/sampler run passed 611 tests, skipped 4, and deselected the
one known DM complex64 case after reproducing its upstream failure in the
preceding runs. Default smoke passed 128 tests. Ruff, skill/catalog validation,
and whitespace checks passed. The final tag-only change was rechecked with
all 24 focused tests and Ruff. The full repository suite was not rerun; the
known upstream DM failure remains separate and unresolved.

## Follow-up: safe environment caching (2026-09-08)

SRC and SDC now share a bounded 128-entry LRU of immutable traversal plans,
keyed by the ordered directed edges and hub. Plans contain only neighbor
tuples, the required environment schedule, and read-only consumer counts.
Each invocation copies those counts and recomputes live tensor indices,
dimensions, sketches, and numerical messages. Numerical environments are
built once per required direction, reused by all dependent branch contractions,
and removed after their last consumer. No input tensor or backend array is
retained by the persistent plan cache. Failed calls cannot mutate shared plans.

Cross-call numerical caching is deferred: tensor identity is not a reliable
version because callers can edit arrays in place. Reusing such messages would
require versioned target tensors and sketch/rank policies, with invalidation
of every dependent directed environment. The structural cache needs none of
those numerical invalidation rules and safely survives changes in indices,
dimensions, operators, seeds, ranks, and backends.

Rechecked the required Quimb, Autoray, Cotengra, and Symmray sources in the
same environment; Symmray's Abelian HTML retrieval again failed. Installed
versions are unchanged from the audit above. Reprobed the public
`tensor_contract` and `Tensor.split` signatures, including `drop_tags` and
`absorb`. Classification: adopt immutable structural caching; defer persistent
numerical caching. No upstream dispatch, dependency, or shim changes.

Regression checks exercise immutable plans, cache bounds, changed sweep
direction, malformed orders, actual branch environment reuse, array release,
in-place input edits, changed dimensions/indices/ranks/seeds, and recovery
after an injected QR failure. Warm-plan results match fresh-plan results.

Validation: 27 focused algorithm tests passed; the broader tree/FIT/sampler/
trajectory selection passed 614 tests, skipped 4, and deselected the known
independently reproduced upstream DM complex64 failure described above.
Ruff, skill/catalog validation, and whitespace checks passed. The full
repository suite was not run because this follow-up is confined to tree
compression and validation covered its callers.
