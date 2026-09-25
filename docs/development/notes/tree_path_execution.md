# Tree path execution — 2026-09-09

Implements the [geometry-aware traversal plan](../plans/tree_fit_path_traversal.md)
and its user-requested extension to direct gate execution. The complete TTN
may branch, but the minimal subtree of a two-site operator is its unique
geodesic. Several physical sites, including a physical root, can also lie
on one such path. Inspect induced degrees; exterior virtual legs do not
make the active region branch.

## Execution

`TreeOptimizer(fit_traversal="auto")` is now the default. A geometry-only
helper chooses the endpoint nearest the incoming tracked center, with node-id
ties and deterministic node-id order if no center is known. TreeFIT freezes
this reference order per target before any guess construction or updates.
Inward follows it; outward reverses it. Explicit depth/depth-first block
orders and standalone TreeFIT's depth default remain available.

Path windows are consecutive structural nodes. Factor every two-/three-node
window toward the advancing endpoint, reversing the requested center as well
as the window order on the return pass. Overlapping windows contain the
preceding center, so existing exterior-canonicalization logic skips preparatory
QR. The same schedule persists through rank growth and 3→2→1 transitions,
without changing budgets or convergence guards. Branched regions use DFS.

Compressed guesses end at the first requested FIT endpoint. SRC/SDC build
complementary environments in the opposite direction to projection; their
bounded immutable geometry cache and per-call numerical caches are unchanged.
No mixed fitted-bra/target boundary is replaced by identity. FIT invalidates
dependent messages before canonical moves can rename live bonds; off-path
messages remain reusable. Only requested/resolved traversal, endpoints and
the actual final center are added to diagnostics.

Direct/DM prepare the complete operator losslessly, gauge its center to the
opposite endpoint, then compress each path edge once toward the requested
terminal endpoint. Exact TreeMPO preparation retains peeling from both ends:
forcing strictly one-ended QR increased QR time from 0.015 to 0.241 seconds
in a 16-qubit profile. It saved compression work but lost overall. Preparing
from both sides avoids that intermediate-rank inflation while retaining the
directional truncation sweep. The terminal center is retained
without a return QR walk. The layered TreeMPO path contracts each node only
after its incoming message arrives, avoiding state/operator outer products
on unvisited nodes. This is exact QR routing followed by canonical compression;
zipup still truncates outgoing messages before the operator is complete.
Message dependencies use incoming counts and a heap of ready edge indices
instead of rescanning the remaining edge list. This also covers two-ended
paths, preserves serial and parallel wave ordering, and does not allocate a
thread pool when there is no independent work.
Explicit subtree and sub-MPO paths reuse directional compression. Existing
low-level two-qubit threading and sibling shortcuts remain in place.

Finite-cap approximations and seeded SRC output can change with order.
The parent RNG retains its single compressed-guess child-seed draw. A fixed
state, seed, and options remain reproducible. The representation retains all
exterior dimensions; path geometry alone does not make large multi-node
factorizations cheap.

## Upstream audit

Checked the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray repository](https://github.com/jcmgray/symmray). The requested
[Abelian-array HTML page](https://symmray.readthedocs.io/en/latest/abelian_arrays.html)
again returned an internal fetch error; actual installed Symmray APIs were
inspected locally.

Installed in the activated genpy environment: Quimb
`1.15.1.dev39+g369d09b9d`, Autoray `0.11.1.dev1+gc56f64427`, Cotengra
`0.8.3.dev6+g08fe1a3a1`, Symmray `0.3.2.dev6+ga17699db6`.

Local probes confirmed `tensor_contract(output_inds, optimize, preserve_tensor,
drop_tags, **contract_opts)`, `tensor_split(left_inds, method, absorb, max_bond,
cutoff, cutoff_mode, right_inds, **kwargs)`, `compress_between(canonize_distance=0,
**compress_opts)`, and `canonize_between(absorb="right", **canonize_opts)`.
Quimb's QR/direct/DM dispatch remains `qr_stabilized`, `svd_truncated`, and
`svd_via_eig_truncated`. Autoray QR/SVD/tensordot dispatch was inspected for
NumPy, Torch, JAX and Symmray. Cotengra still accepts per-contraction
`implementation`; Symmray Abelian `tensordot` accepts `mode`, and native QR
continues through the shared Pepsy policy.

Classification: **adopt** existing public contractions/splits for the new
traversal; **defer** unrelated current Quimb changes to HilbertSpace ordering,
fermion register defaults and additional compression variants. Existing SRC
backend-random and native blockwise capability paths remain in use. No new
compatibility shim, global dispatch override, installed-package edit, or
upstream algorithm prototype is needed.

The final native rerun prompted a second check of all the upstream sources
and the actual `AbelianArray.svd_truncated` signature. Its default
`max_bond_mode="global"` explicitly preserves a degenerate multiplet and can
exceed `max_bond`; `"eager"` changes sector allocation and supports only
absolute/relative cutoffs without renormalization. A deterministic native
identity spectrum with five equal values and a requested cap four returns
rank five and reconstructs exactly. This explains the occasional zipup bond
five at chi four; it is upstream numerical policy, not new QR rank growth.
**Adopt** documentation and spectrum-based regression of the existing global
policy; **defer** any eager or strict-cap policy change. Native numerical
dispatch remains unchanged, and dense caps remain hard.

## Initial validation and performance

Final affected-domain run: **782 passed, 4 skipped, 1 deselected** in 50.94 s.
Coverage includes path execution, FIT messages/priorities, successive
compression, the complete TreeOptimizer and zipup suites, shared TreePeps,
tree/MPS parity, sampling, trajectories, public API and package layout.
The deselected NumPy complex64 DM test has the previously reproduced upstream
Quimb `svd:eig`/Numba typing failure recorded in the
[successive-compression audit](tree_successive_environments.md). Validation
was scoped to these affected domains and shared callers; unrelated repository
subsystems were not changed or rerun. Ruff (`src tests`), skill/catalog
validation and `git diff --check` pass.

Regressions cover induced geometry with exterior branches, permuted labels,
physical-root and collinear three-site gates, both sweep orders, advancing
one-/two-/three-node centers, 3→2→1 and short-path schedules, actual tensor
isometries, mixed exterior messages against independent full-branch
contractions, rank changes, zero/weak targets, represented scale, frozen
orientation before unknown-center recovery, and failure/retry isolation.
NumPy, Torch, JAX and even-parity native Symmray paths are exercised. The
existing incompatible-charge zipup rejection test uses a conservatively
full-support branched TTNO with an injected empty final hub. This exercises
the guard deterministically instead of depending on roundoff-sensitive
choices at a degenerate singular-value boundary. Native zipup tests check
any over-cap split against the full boundary spectrum, and verify canonical
recovery does not grow beyond the dimensions produced by those splits.

Benchmark comparison uses the pre-change `HEAD` source in a temporary checkout
against the final implementation, Torch CPU complex64, one Torch thread,
cutoff zero, identical gates/state seeds and caps, one warmup and median of
three runs. The old DMRG policy is explicit depth-first; the new policy is
auto. Four-iteration budgets, automatic tolerance and growth/refinement rules
are the same. State construction and independent fidelity evaluation are
outside timing. Small states use dense complex128 reference evaluation.
No GPU or live multi-rank MPI benchmark was run. Harnesses and raw output
remain under `/tmp`; none are added to the package.

| 16 qubits, entangled, chi 32, 16 gates | Before (s) | After (s) | Fidelity before | Fidelity after |
| --- | ---: | ---: | ---: | ---: |
| direct | 0.2192 | 0.1652 | 0.863011153 | 0.863011156 |
| SRC | 0.0958 | 0.1112 | 0.587236207 | 0.585152650 |
| DMRG1 | 0.8280 | 0.7118 | 0.860337677 | 0.860254659 |
| DMRG2 | 1.2757 | 1.0990 | 0.863011164 | 0.863011179 |
| DMRG3 | 2.2603 | 1.9244 | 0.863011080 | 0.863011142 |

| 24 qubits, product start, chi 64, 48 gates | Before (s) | After (s) | Fidelity before | Fidelity after |
| --- | ---: | ---: | ---: | ---: |
| direct | 19.6396 | 16.6768 | 0.242875562 | 0.243239421 |
| DMRG1 | 16.4392 | 13.5153 | 0.242752243 | 0.242167152 |

The 24-qubit DMRG1 runs both use thirteen `(2, 2, 1, 1)` growth windows and
thirty-five `(1, 1, 1, 1)` windows. They do the same budgeted work, but the
finite-cap fidelity changes with order. Direct improves 25% on the small
case and 15% on the product replay; DMRG1/2/3 improve about 14–15% on the small
case and DMRG1 improves 18% on the product replay. Standalone SRC is 16%
slower on the small case despite using endpoint environments. These are
workload-specific measurements, not a universal speed or fidelity guarantee.
The large 24-qubit baseline was stopped after its completed direct/DMRG1
measurements; the larger local-block algorithms are compared on the smaller
case and tested separately at bond 200 with bounded runtime.

On the 32-qubit entangled single-gate case with chi 200, direct improves from
21.2243 to 18.8205 seconds (11%). Independent layered-tree overlaps evaluated
in complex128 give fidelities 0.999581592 before and after. Each value is the
median of three replays after warmup. Those combined benchmark processes hit
their 100-second limit after completing direct, before recording DMRG1;
DMRG1 is measured in separate processes below. No incomplete timing is used
as a replay measurement.

Separate DMRG1 runs at bond 200 give 6.5364 seconds with depth-first before
the change and 6.5734 seconds with auto after it: effectively unchanged.
Both stop after three one-site iterations. Fidelities are 0.999570105 and
0.999568561, respectively. One-site refinement remains considerably cheaper
than direct on this prepared state, with a small fidelity tradeoff.

An instrumented comparison on the updated code holds the SRC guess and budget
fixed, disables tolerance stopping, and performs 80 one-site local updates
(four complete iterations) with either traversal. Edge-canonicalization
calls fall from 113 to 81, and environment misses from 245 to 213. These are
API-call counts, not claims that every call performs a numerical QR.
Canonical preparation takes 4.639 versus 4.502 seconds; effective tensors
take 2.950 versus 2.917 seconds. Total instrumented time is 8.582 versus
8.348 seconds. Fewer small moves do not remove the expensive large-edge QR
and contraction work in this case.

The 16-qubit branched three-site replay (eight gates, chi 32) gives identical
fidelity 0.844884393 and norm 0.844673749 for explicit depth-first and auto.
Medians are 0.7328 and 0.7369 seconds respectively; their block-size traces
also agree. Auto's branch fallback thus retains the same finite-cap result
on this workload.

The bond-200 DMRG2 and DMRG3 processes each reached their 60-second limit
without a completed replay, including process setup. No fidelity or complete
timing is claimed for those runs. Endpoint traversal removes avoidable center
movement; large multi-node effective tensors and splits remain a cost that
this change does not eliminate. Prefer direct or one-site refinement when
that cost dominates.

## Follow-up efficiency review — 2026-09-09

The follow-up review found avoidable work in the initial implementation:

- Direct preparation moved the canonical center through active tensors that
  its exact routing immediately refactored. Direct/DM and SRC/SDC now reuse a
  contained canonical region. A known exterior center stops at the first
  active node; unknown gauges still use complete exterior preparation. Zipup
  retains its preparation gauge because it truncates before completing the
  operator. Numerical isometry and independent dense-state regressions check
  contained centers, exterior centers, multi-node regions and unknown gauges.
- Two-ended QR routing repeatedly scanned and copied its pending edge list.
  Incoming counts and ready edge indices now preserve the same deterministic
  serial order and parallel waves without those scans. Scheduling costs
  O(L log W), with L edges and at most W simultaneously ready messages; a
  one-ended or two-ended path has W at most two. Native routing remains serial.
  Tests compare finite-cap serial and parallel outputs on paths and branches.
- SRC/SDC and zipup built physical-index maps for all physical sites on every
  local update. Those maps now include only physical nodes in the active
  region, including a physical root. Duplicate layered-absorption branches
  and the per-wave split-function definition were also removed.
- The TreeMPO `metadata_path` timer started after planning had finished. It
  now encloses planning. Layered `subtree_hub_merge` events explicitly mark
  message queuing with `deferred=True`; the subsequent node contraction is
  timed as `tensor_absorption`. A controlled-clock regression checks this.

The installed stack changed between reviews. All six upstream sources listed
above were checked again, with the same fetch failure for the Symmray Abelian
HTML page. Installed versions are now Quimb `1.15.1.dev51+g2e99c793e`, Autoray
`0.11.1.dev3+g1b476b305`, Cotengra `0.8.3.dev7+g1d7fd333f`, and Symmray
`0.3.2.dev8+g6c6dd34b5`. Local signature and dispatch probes covered public
tensor contraction/split, network compression/canonicalization, Cotengra
contractors, native tensordot/global SVD, and NumPy/Torch/JAX/Symmray dispatch.
Quimb's current split signature defaults to `method="auto", absorb="auto"`;
Pepsy's routing still selects its numerical methods and absorption explicitly.
**Adopt** the existing public interfaces and retain shared native QR policy;
**defer** unrelated upstream ordering/compression changes. No compatibility
shim or installed-package change was needed. The formerly excluded NumPy
complex64 DM regression now passes and is included in the reruns.

For this review, the baseline is a snapshot of the uncommitted implementation
immediately before these cleanups, using the same updated dependencies as the
final code. Both use Torch CPU complex64 with one thread, identical seeds,
cutoff zero, one warmup and three measured replays. Preparation and independent
fidelity evaluation remain outside timing. The 32-qubit reference uses layered
complex128 overlaps without constructing its dense statevector.

| Workload | Before (s) | After (s) |
| --- | ---: | ---: |
| 16 qubits, chi 32, 16 gates: direct | 0.1650 | 0.1666 |
| same: SRC | 0.1126 | 0.1094 |
| same: DMRG1 | 0.7178 | 0.7143 |
| same: DMRG2 | 1.1198 | 1.1217 |
| same: DMRG3 | 1.8646 | 1.8768 |
| 32 qubits, chi 128, one long gate: direct | 4.4445 | 4.3046 |
| same: DMRG1 | 2.6420 | 2.6908 |

These are modest improvements (about 3% for the sampled SRC and larger direct
case); the other timings are effectively unchanged. SRC and DMRG outputs,
norms and iteration traces agree exactly in these seeded replays. Direct
fidelity differs by 2.5e-8 at 16 qubits and 1.4e-11 at 32 qubits; both retain
the requested cap. This is consistent with changed floating-point QR gauges,
not evidence of a universal speedup or an optimal solver. The large DMRG2/3
factorization bottleneck identified above remains. No GPU or live MPI test was
performed during this review.

Final affected-domain validation: **793 passed, 4 skipped**, no deselections,
in 52.18 s. This includes the tree suites, shared TreePeps FIT, tree/MPS parity,
sampling, trajectories, public API and package layout. Ruff (`src tests`), the
skill catalog validator, tree-skill validation, and `git diff --check` pass.

Full-repository verification did not pass. Its first attempt aborted in the
macOS Matplotlib GUI backend during
`test_ham.py::test_build_itf_lattice_show_returns_schematic_drawing`; that test
passes with `MPLBACKEND=Agg`. A headless full-suite run with fail-fast enabled
then stopped at **182 passed, 1 failed**:
`test_bp_symmray.py::test_native_reduced_loop_compression_uses_graded_svd_adapter`.
The failure is `ValueError: vector size does not match Symmray bond
'b(1, 0)-(1, 1)'` in `pepsy.bp.gauges._symmray_block_vector`. It reproduces in
isolated processes against both the pre-review snapshot and the pre-path
`HEAD` source snapshot using the current dependencies. The BP/gate code is
unchanged. Upstream sources were rechecked; this separate BP regression was
not patched as part of the TreeOptimizer path review. The full suite remains
unverified beyond that failure.

Two existing performance candidates remain outside this cleanup. Generic
`TreeTensorNetwork._recover_center_from_region` rescans remaining leaves, so
very long regions can incur quadratic metadata work even though the new
operator-message scheduler avoids it. Shared TreeFIT one-site center movement
also still uses ordinary QR carry absorption into the next tensor before its
local fit replaces that tensor. A dense, rank-preserving shortcut may avoid
that multiplication; it needs separate validation for rank changes and the
shared TreePeps/native paths. Neither candidate was benchmarked independently
in this review or presented as a completed optimization.
