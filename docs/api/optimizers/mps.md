# `pepsy.optimizers.mps.optimizer`

## Torch SVD policy

Torch/Autoray SVD dispatch is process-global, so configure it once at
application startup or use a scoped policy for an experiment:

```python
import pepsy

svd_policy = pepsy.TorchLinalgConfig(
    mode="complex",
    stabilized=False,       # native Torch forward and backward
    svd_driver="auto",     # CUDA: let Torch select its default driver
    cpu_svd="torch",       # CPU: Torch's native LAPACK path
    svd_fallback="auto",   # no fallback for native mode
)
svd_policy.register()
print(svd_policy.describe())
```

`svd_driver` applies only to CUDA and accepts `"auto"`, `"gesvdj"`,
`"gesvda"`, or `"gesvd"`. `gesvda` is approximate and requires
`allow_approximate=True`. `cpu_svd` accepts `"torch"`, `"scipy_gesdd"`, or
`"scipy_gesvd"`; the SciPy choices are intended for explicit forward-only
CPU experiments when `stabilized=False`, or for stabilized autodiff when
`stabilized=True`. `svd_fallback="auto"` means no fallback for native mode
and SciPy `gesvd` for stabilized mode.

The non-approximate choices are CUDA `gesvdj` and `gesvd`, plus CPU Torch,
SciPy `gesdd`, and SciPy `gesvd`. For `complex64`, try CUDA `gesvdj` first;
on CPU, benchmark `scipy_gesdd` against the native Torch path. `gesvd` is a
robust fallback rather than the speed choice. The approximate CUDA `gesvda`
driver is never selected unless `allow_approximate=True` is passed, and the
policy exposes this decision as `policy.exact` and `policy.approximate`.

For example, an exact complex64-oriented CPU experiment is:

```python
pepsy.TorchLinalgConfig(
    mode="complex",
    stabilized=False,
    cpu_svd="scipy_gesdd",
).register()
```

On CUDA, select the exact Jacobi driver explicitly with
`svd_driver="gesvdj"`. These settings do not change the tensor dtype; they
change only the underlying SVD implementation.

For ordinary MPS simulation, native mode is the recommended default. The
regularized mode exists for finite SVD gradients and difficult autodiff
inputs, not as a faster forward SVD. A temporary policy restores the previous
one when the block exits:

```python
with pepsy.TorchLinalgConfig(
    stabilized=True,
    svd_fallback="scipy_gesvd",
).activated():
    run_differentiable_workflow()
```

Use `pepsy.get_torch_linalg_config()` to inspect the last Pepsy-installed
policy. `pepsy.reset_linalg_registrations(backend="torch")` restores native
Torch and Quimb split registrations.

`MpsOptimizer` consumes canonical bundled gate streams of the form
`[(gate, where), ...]`. In `mode="mpo"` the stream can also contain explicit
sub-MPO events for already-factorized nonlocal operators:

```python
event = ("submpo", mpo, where)
# or
event = {"kind": "submpo", "mpo": mpo, "where": where}
```

`where` is a non-empty tuple/list of unique 1D MPS sites. The convenience
helper `MpsOptimizer.submpo_event(mpo, where)` builds the tuple form. These
events are applied with `gate_with_submpo_` and compressed to `chi`; they are
only accepted in `mode="mpo"`.

Modes that use canonical MPS metadata require an open-boundary MPS. A cyclic
MPS has a nontrivial loop environment, so no single tensor norm can equal its
global norm under the open-chain mixed-canonical identity. Such inputs are
rejected before optimizer state is mutated. `mode="exact"` does not consume
canonical metadata and can first contract a cyclic input; switching its
contracted result back to an MPS mode rebuilds an open MPS.

`MpsOptimizer.backend_info()` reports the backend, dtype, and device inferred
from every live MPS tensor; the same values are also available as the
state-derived `backend`, `backend_dtype`, and `backend_device` attributes.
Every gate and every tensor in a sub-MPO is checked against that signature
before replay. Explicit mismatches are converted on an execution copy with a
`UserWarning`; the queued payloads remain unchanged. Native Symmray MPS data
reports `backend="symmray"` and includes `array_backend` for the underlying
NumPy, Torch, or CuPy charge-sector blocks. Dense payloads cannot be promoted
to native Symmray gates because that would lose charge and fermionic metadata;
construct those gates with the matching Symmray convention instead.

Streams may also include control events. `("measure", pauli, where[, outcome])`
collapses onto a Pauli eigenvalue and records `(pauli, where, outcome, prob)`.
`("reset", where[, basis])` resets each target to the `+1` eigenstate of
`basis` (`"Z"` by default, so the legacy form resets to `|0>`); the internal
measurement is not recorded. `("measure_reset", basis, where[, outcome])`
measures each target in `basis`, records the result, then resets it to the
`+1` eigenstate. The aliases `("mrx", where[, outcome])`, `("mry", ...)`, and
`("mrz", ...)` are accepted. `("cap", where, vec[, absorb])` contracts one
physical leg with `vec`, absorbs it into the selected neighbour, and shortens
the MPS by one site.

`mode="fit"` is a clear alias for the historical `mode="dmrg"`. The
convenience modes share the DMRG backend but have distinct schedules:
`"dmrg1"` uses at most two two-site growth sweeps and then fixed-rank one-site
FIT. If every active bond is already at its attainable ceiling before the fit,
`"dmrg1"` starts directly with one-site FIT. Once every full-chain bond
reaches its physical/`chi` ceiling, the optimizer latches one-site updates for
later windows in the same replay. `"dmrg2"` uses two-site FIT for the required
warm-up (two sweeps by default) and then one-site FIT; `"dmrg3"` follows the
same fixed warm-up schedule with three-site FIT and then one-site FIT.
`mode="dmrg"` remains the generic spelling and keeps the adaptive two-site
schedule. `mode="mix"` is the transactional unitary variant.
With `fit_block_size=2`, FIT grows only bonds visited by the gate interval, up
to `chi`, through the middle-bond SVD; it does not pad the whole MPS and does
not need an MPO rank warm-up. `fit_block_size=3` uses a three-site effective
wavefunction and two direction-aware native SVD splits, and is useful when a
larger local window is worth the extra decomposition cost. An adjacent
two-site gate span automatically falls back to `fit_block_size=2`. Both block
sizes preserve native dense and Symmray backends. For block sizes 2 and 3, the
optimizer passes the current MPS directly to FIT without pre-padding bonds;
only bonds visited by the native splits can grow. `fit_block_size=1` retains
the fixed-rank compatibility algorithm, for which mixed mode still warms short
active bonds through MPO. Standalone one-site gates use the exact direct/MPO
path; ordinary DMRG target blocks can absorb intervening one-site gates before
the block's shared compression. Generic `mode="dmrg"` remains rank-adaptive,
but named `"dmrg1"` bounds its two-site warm-up at two sweeps and then uses
one-site FIT for the remaining requested sweeps. The named mode does not
extend the two-site phase because of rank stagnation. Once all full-chain
ceilings are reached, it latches one-site updates for later gate windows.
This keeps the bond spaces opened by the SVD warm-up while avoiding repeated
GPU SVD truncations. The dense open-chain ceilings are
`2, 4, 8, ..., chi, ..., 8, 4, 2` (also limited by the current outside-window
bonds); FIT never pads a bond merely to make it equal to `chi`. Set
`fit_adaptive_sweeps` to configure the named `dmrg2`/`dmrg3` warm-up; `dmrg1`
keeps its two-sweep policy. A generic `mode="dmrg"` with `fit_block_size=1`
remains the fixed-rank one-site compatibility path. `fit_layer_size` is the
clear name for
`k_2q_batch`; it counts two-site gates in a contiguous paper-style target
block. For `fit_block_size=2`, an active window spanning at least three sites
uses the same adaptive-to-one-site schedule; an ordinary two-site gate window
uses exactly one two-site update because that effective tensor already solves
the complete local problem. In particular, `dmrg1`, `dmrg2`, and `dmrg3`
immediately advance to the next gate after that update: they do not repeat
their warm-up or enter one-site refinement, regardless of `n_iter` or
`fit_rtol`.
`fit_three_site_sweeps` remains a deprecated alias for
`fit_adaptive_sweeps`.
`fit_max_span="auto"` also limits the spatial width of a batched
target, splitting disjoint gates before they create an unnecessarily wide FIT
window. Set `fit_max_span=None` to restore unrestricted gate-count batching.
If a DMRG/FIT batch
raises, produces non-finite data, or exceeds `chi`, the optimizer restores the
complete pre-batch state (including canonical and infidelity metadata) and
replays the batch through MPO. Interrupts restore the trial state and are
re-raised.

For ordinary DMRG and mixed DMRG, `n_iter` is a maximum rather than an
unconditional sweep count. `fit_min_iter`, `fit_rtol`, and `fit_patience`
control adaptive stopping from FIT's final local-norm change. The public
`MpsOptimizer.run` default is `fit_rtol=1e-8`, which compares the retained
canonical-center norm (A) between sweeps. Canonical FIT is an orthogonal
projection, so for target norm (T) the true normalized fidelity is
\((A/T)^2\); when `p_target` is normalized, true infidelity is exactly
\(1-A^2\). The stopping test compares changes in (A), rather than applying
an absolute infidelity threshold. `fit_rtol="auto"` remains an explicit
dtype-aware option selecting `1e-3`, `1e-5`, or `1e-8` for 16-,
32-/complex64-, or higher-precision data. Pass `fit_rtol=None` for fixed
iterations. `fit_patience` counts same-phase sweep-norm samples in the
convergence window, so the default `fit_patience=2` stops after one stable
comparison between two one-site samples; `fit_min_iter` still sets the
minimum completed-sweep count. The old
`mix_fit_min_iter`, `mix_fit_rtol`, and `mix_fit_patience` spellings remain as
deprecated aliases. A legacy value replaces the canonical default for old
call sites; a conflicting non-default canonical value fails instead of
silently choosing a policy. FIT computes only the terminal canonical-center
norm once per sweep. Its native finite checks reduce active tensor blocks and
transfer those flags together with the optional tolerance norm as one compact
vector. Adaptive rank-growing windows require `n_iter >= 2`; a
shorter request raises before fitting, except for the adjacent two-site exact
fast path. An under-capacity, non-adjacent `mode="dmrg1"` window requires
`n_iter >= 3`, reserving its first two sweeps for two-site rank growth and at
least one later sweep for one-site refinement. An already-capped `dmrg1`
window has no growth reservation and uses all requested sweeps as one-site
updates.
At least two adaptive block sweeps are required whenever the active window
needs rank growth, regardless of `fit_rtol`; an adjacent two-site interval is
a structural special case whose only pair is
the complete variational problem, so the default
`fit_single_pair_fast_path=True` stops after one effective-tensor SVD even when
`fit_rtol=None`. Set it to `False` only when intentionally benchmarking
repeated identical sweeps.
It does not allocate or scan a second MPS. Ordinary DMRG raises on a detected
non-finite sweep; for compatibility, non-unitary DMRG retains fixed sweeps
when `fit_rtol="auto"`, while an explicit numeric tolerance enables
adaptive stopping there too. Mixed DMRG additionally performs one full tensor
check before committing a trial, while consecutive MPO warm-up steps share one
full check at the next DMRG handoff or at the end of the segment. A
transactional MPO fallback is checked before commit. Torch and CuPy full
checks process one tensor at a time, combine scalar results on the device, and
transfer one Boolean to the host.

The DMRG/FIT update follows the variational update described in
the [Ayral *et al.* PRX Quantum paper](https://doi.org/10.1103/PRXQuantum.4.020304):
the effective tensor is built from cached contractions on the left and right,
then the MPS is swept repeatedly. Recommended `fit_block_size=2` forms a
local wavefunction with the two outer virtual legs and both sites' physical
groups, then splits its middle bond with `Tensor.split`. `fit_block_size=3`
forms the analogous three-site tensor and splits it twice, absorbing singular
values toward the sweep direction. Both dispatch to configured dense SVD
drivers and, crucially, Symmray's native block SVD for U1, U1xU1, and fermionic
tensors. `fit_sweep_sequence="RL"` alternates canonical directions; `"R"`
preserves a one-way sweep for dense and native fermionic arrays. Fermionic FIT
keeps a conjugated native working MPS across the sweep sequence, includes the
actual outside graded overlap environments, applies dual-leg phase corrections
before each split, and restores the physical ket afterward. Thus `R`, `L`, and
`RL` are honored exactly without dense conversion or Jordan-Wigner
bosonization.

The named `dmrg1`, `dmrg2`, and `dmrg3` schedules are backend-independent:
native U1, U1xU1, and Z2 fermionic states use the same schedules as ordinary
arrays. `dmrg1` uses its bounded two-sweep warm-up and sticky one-site phase,
while `dmrg2` and `dmrg3` perform their fixed block warm-up before one-site
refinement. A native nonlocal gate
still receives its chi-capped graded auto-swap warm start before FIT. For a
direct full-chain FIT whose arbitrary MPS guess lacks target virtual charge
sectors, FIT instead uses a target-informed native compressed initialization;
partial gate windows retain their fixed outside-state contract, grow sectors
through native local blocks, and reject only a genuinely empty effective
problem.

In this optimizer the fit is intentionally
restricted to the interval `[xmin, xmax]` touched by the current two-site gate
or batch. This is implemented by `FIT.run_gate`, the gate-window version of
`FIT.run_eff`; `run_eff` remains the default one-site full-chain boundary
solver but also has opt-in native block-2/3 updates, while PEPS
`fit_mode="two-site"` uses `run_gate` over the full boundary.
Using `run_eff` for each gate would refit unrelated sites and would no longer
be local DMRG compression. `fit_layer_size=N` explicitly forms the paper's
multi-gate/layer target before each restricted fit. With the default
`fit_target_strategy="auto"`, ordinary NumPy/Torch/CuPy gates remain as exact
spatially split operator layers: FIT contracts them lazily instead of growing,
copying, and repeatedly decomposing an intermediate target MPS. The gate SVD
has only the operator-Schmidt rank and does not apply the output `chi` limit.
`fit_target_strategy="mps"` selects the traditional materialized target;
`"auto"` also chooses that native routed representation for Symmray U1/U1xU1
and fermionic data. `target_cutoff=0.0` keeps either representation exact while
ordinary `cutoff` controls only the two-site output split, so target-
construction loss is not reported as FIT loss.

For a non-adjacent native fermionic gate, MPS DMRG first replays the target
gate through Quimb's chi-capped graded auto-swap path and uses that native MPS
as the FIT starting point. This is the deterministic counterpart of
SymDMRG2's sector enrichment: it opens the gate-generated virtual charge
sectors before alternating least squares can project them out. After that
warm start, fermionic FIT follows the selected DMRG schedule normally:
`dmrg2`, for example, can switch from its two block warm-up sweeps to native
one-site refinement. The uncapped target remains separate. At
`target_cutoff=0.0`, routed target splits use the smallest
representable positive absolute cutoff, which removes structural zero singular
directions while retaining every representable nonzero value. This prevents
invalid duplicate dummy modes without introducing target truncation.

All unitary compressed modes (`dmrg*`, `mix`, `mpo`, `swap`, `perm`, and
`svd`) default to `stabilize_unitary=True`. After each compression Pepsy first
records retained norm loss in log-fidelity space when infidelity tracking is
enabled, then restores the raw working MPS to its pre-compression norm without
accumulating that approximation loss in `p.exponent`. FIT and the direct
compression modes reuse the final canonical center and retained norm rather
than sweeping or contracting the state again. Stabilization is independent of
`track_infidelity`: disabling samples removes diagnostic work, but an enabled
stabilizer still reads the one center norm it needs. This prevents deep
complex64 streams from underflowing while keeping cumulative infidelity
meaningful. Pass `non_unitary=True` for filters/Kraus/sub-MPO streams so this
unitary rescaling is disabled. Set `stabilize_unitary=False` only to reproduce
historical norm-decay behavior. The old `fit_stabilize_unitary` spelling
remains as a deprecated alias.

`cutoff="auto"` selects `1e-3` for 16-bit data, `1e-6` for 32-bit/complex64
data, and `1e-12` for 64-bit data. Explicit numeric cutoffs are unchanged.
Set `quality_check_every=N` to record finite-data and canonical-gauge health in
`opt.get_quality_checks()`. Checks are disabled by default; when enabled,
`quality_check_repair=True` re-canonicalizes if canonical coverage is lost.

Mixed-mode DMRG trials isolate only the active FIT window and the canonicalization
path leading to it. Untouched MPS tensors are shared until a successful trial is
committed, avoiding a full deep copy for every transaction while preserving
rollback safety for the active update. After a non-finite DMRG result,
`mix_sticky_nonfinite=True` keeps the remainder
of the current `run()` call on MPO rather than retrying an unhealthy FIT for
every gate. An ordinary exception still falls back only for its transaction.
The initial MPS must satisfy `p.max_bond() <= chi`. The mixed replay history is
stored in `opt.mix_history` and summarized in `opt.last_mix_summary`; entries
include logical `where`, execution `execution_where`, FIT iterations and
convergence, target bond, fallback sweep, and sticky-disable diagnostics. With
`progbar=True`, the progress bar shows the current backend, cumulative
MPO/DMRG/fallback counts, and `bond=current/chi`.

Replay timing is opt-in and does not print by itself:

```python
opt.run(timing=True)
print(opt.get_run_timing())
```

The copy-safe record contains replay wall time, event count, final bond,
backend signature, and—when using `mode="mix"`—a copy of
`last_mix_summary`, including its elapsed time and backend decision counts.
Mixed runs leave `last_mix_summary["elapsed_seconds"]` as `None` when replay
timing is disabled, so the normal mixed path performs no profiling clock
reads. The measured replay interval begins after argument validation and any
temporary layout setup; it ends before temporary layout restoration and
before `get_run_timing()` makes its defensive result copy.
It also contains inclusive `stages` totals for `gate_stream.prepare`, the
active mode replay, `canonicalize`, `gate.apply`, `dmrg.target`, `dmrg.fit`,
`normalization`, `control.<event>`, `infidelity.target_norm`,
`infidelity.retained_norm`, and `infidelity.compute`. The three infidelity
stages distinguish construction or contraction of a non-unitary target norm,
the retained one-center tensor norm, and the final scalar log-fidelity update.
Stage totals can overlap with the mode replay total; use them to identify the
dominant work, not to add into a second total. DMRG and mixed-mode timing also
expose
`fit_steps`: one record per completed or failed FIT sweep, including its FIT
call index, global record index, direction, block size, active interval, sweep
time, per-site/block update times, and phase-level sweep overhead. Timing
schema 3 reports `canonicalization_seconds` (including sweep gauge/QR
preparation), `fixed_environment_seconds`, `effective_seconds`,
`svd_seconds`, `writeback_seconds`, `moving_environment_seconds`, and
`sweep_overhead_seconds`. Per-site records additionally expose
`canonicalization_seconds` and `moving_environment_seconds`; the legacy
`environment_seconds` field remains the complete post-writeback environment
phase. `sweep_preparation_canonicalization_seconds` separates the preparation
part of the canonicalization total, while
`moving_canonicalization_seconds` identifies one-site gauge moves inside a
sweep. `MpsOptimizer.get_run_timing()["fit_totals"]` provides the same phase
totals across all FIT calls in the replay, while `fit_steps` retains the
per-sweep and per-site records. FIT phase fields are not one flat additive
list: `canonicalization_seconds` contains both sweep preparation and moving
canonicalization, while legacy `environment_seconds` contains the complete
post-writeback phase. An additive decomposition uses preparation
canonicalization, fixed environments, effective contraction, SVD, writeback,
moving canonicalization, moving environments, and sweep overhead exactly
once. Timing also remains independent of `collect_split_diagnostics`;
profiling an MPS run does not allocate per-SVD truncation dictionaries.

Ordinary runs retain no per-gate timer or timing-record overhead. Enabled
profiling moves its internally owned FIT records into the replay result and
copies them only when `get_run_timing()` is called. These are host wall-clock
measurements by default. Use
`run(timing=True, timing_sync_device=True)` for kernel-complete Torch CUDA,
CuPy, or JAX timings; the added barriers intentionally make profiling slower
and are recorded as `timing_sync_device=True` in both replay and FIT records.
The accelerator backend is detected once per timing session, so CPU timing
does not repeatedly scan the MPS. JAX barriers wait on each newly returned
stage result rather than an unrelated previously ready MPS leaf.

`mode="su"` uses simple-update evolution for imaginary-time or other
non-unitary gate streams. It keeps `opt.p` as the simple-update core and
stores the external bond factors in `opt.gauges`. After every run,
`opt.p_ungauged` is refreshed as a physical copy with those gauges inserted.
If the supplied dictionary
does not contain the current bond gauges, the optimizer initializes it with
`opt.p.gauge_all_simple_(gauges=opt.gauges, progbar=False)`, then applies each
gate through `pepsy.gate_simple(..., renorm=True)`. This mode does not
canonicalize the MPS and does not report compression infidelity. Use
`opt.p_ungauged` for the physical state and `opt.p` for continued SU updates.
If an independent physical copy is needed, use:

```python
physical = opt.p_ungauged.copy()
```

For Symmray block-sparse MPS data, `gate_simple` automatically uses Quimb's
full two-site `split` path so symmetry and fermionic fusion metadata are
preserved. Dense MPS data keeps the faster `reduce-split` path by default.

`mode="swap"` applies non-local two-site gates through a swap-and-split path
and swaps the sites back after each gate. `mode="perm"` uses the same
swap-and-split path but leaves the swaps in place, tracking the current
physical-site-to-logical-site ordering in `opt.qubits`. This is useful for
streams with little expected locality. The returned `opt.p` remains an MPS in
physical order; call `opt.restore_qubit_order()` when a conventional logical
site order is needed.

For repeated evolution, use `opt.apply_layout("quality")` once. This installs
the selected position-to-logical mapping in `opt.logical_order` and keeps the
MPS in that order across subsequent `run()` calls. A bond-one initial MPS is
relabelled without SVD swaps; an initially entangled MPS raises by default, or
can pay one explicit lossy reorder with `allow_lossy_reorder=True` and a caller
provided `cutoff`. The old `run(use_layout_finder=True)` path is retained only
for compatibility and performs the deprecated temporary reorder and swap-back.
Use `opt.logical_site(position)`, `opt.position(site)`,
`opt.remap_sample(config)`, and `opt.to_dense()` for logical readout.

Pauli control events use Quimb's `local_expectation_canonical` when available.
The optimizer passes its `info_c` dictionary, so canonicalization starts from
the tracked `info_c["cur_orthog"]` range, moves only as needed around the
observable support, and records the new range. A concrete tracked range avoids
an orthogonality-center scan. Older Quimb versions without the local evaluator
use a compatibility overlap contraction instead.

Normalization and infidelity use the same canonical-center contract. For a
non-unitary run with `normalize_every` enabled, after every replay step the
optimizer reuses an authoritative one-site center already inside the active
span, normalizes that tensor, and stores the removed scale in `p.exponent`.
Only a genuinely broad tracked center is collapsed to one site. Thus
`p.norm()` restores the represented norm, while a copy with `exponent=0`
exposes the normalized working data. For DMRG, a multi-gate batch is one replay
step for this purpose.
For unitary streams, `get_infidelities()` uses the running product of local
retained fidelities. The cumulative state is preserved across repeated
`run()` calls, so it can be used directly as the simulation-fidelity trace;
call `reset_infidelity_tracking()` when starting an independent accounting
interval. Local products and norm-ratio evaluations are accumulated in the log
domain and exponentiated only for readout, so long streams and very small
retained norms do not lose fidelity to underflow. Local ratios remain available
in detailed samples for diagnostics. Replacing the state with `set_p()` starts
a new accounting interval. Manual `normalize()` preserves represented scale in
`p.exponent`, reuses the tracked singleton center when no insertion site is
requested, and otherwise canonicalizes directly to that site. It rebases the
raw unitary stabilization norm without erasing prior cumulative fidelity. Layout
changes likewise rebase the representation-dependent raw baseline. The trace
is populated by default. Set
``track_infidelity=False`` in the constructor, or pass
``track_infidelity=False`` to ``run()``, to skip target-norm construction,
samples, and progress-bar infidelity fields. When unitary stabilization remains
enabled, its single retained-center norm read is still required. For
dense two-site non-unitary gates, the target norm is obtained from the local
expectation of `G†G`, so no copied target MPS is needed. Symmray uses an
equivalent native contraction containing only the canonical active span, the
gate, and its conjugate; it preserves sectors and graded phases without a
dense reduced density matrix or routed target copy. Unsupported gates and
general sub-MPO backends retain the materialized target-norm fallback. DMRG
still materializes its target because FIT needs it, but a single-gate DMRG
diagnostic first uses the same dense/native local expectation and contracts the
existing FIT target only as a fallback. Batch DMRG contracts its already-owned
multi-gate target once. Target-norm work is reported under
`infidelity.target_norm`; FIT's terminal center norm supplies the retained norm.
For `mpo`, `swap`, `perm`, and `svd`, Quimb's post-compression one-site center
is reused directly, wherever it lies, so infidelity reads one tensor norm and
does not sweep the center across the gate interval merely for diagnostics.
Unitary one-site gates preserve the previous center and do not overwrite the
cache with their support site. In `svd` mode the non-unitary target norm is
measured before both the routed gate split and the final chi compression, so
loss from either truncating stage is included rather than silently becoming
the diagnostic baseline.
Temporary fallback targets never modify the live `info_c` cache.
When tracking is enabled, the `mpo`, `swap`, and `svd` progress bars show the
same cumulative `infidelity` field, starting at zero before the first
compressed gate.
`mode="exact"` and `mode="su"` deliberately skip canonical metadata; switching back to an MPS
mode rebuilds and canonicalizes the contracted state.

The result API is intentionally small:

```python
opt.run(non_unitary=True, normalize_every=True, track_infidelity=False)

opt.get_infidelities()        # [0.0] when tracking is disabled
opt.get_infidelity_samples()  # [] when tracking is disabled
opt.get_normalizations()      # scale events and accumulated exponents
opt.reset_infidelity_tracking()  # start a new fidelity accounting interval
```

When enabled, infidelity is recorded automatically whenever a compressed gate
update occurs. `get_infidelities()` is the cheap cumulative trace for
progress and stopping criteria. Use
`get_infidelity_samples()` when the target norm, retained norm, local ratio, or
step metadata is needed. Each sample's `target_norm_source` identifies whether
the norm came from previous unitary retention, a dense or native local
contraction, a FIT/sub-MPO target, or the conservative materialized fallback.
Samples expose both the clipped reporting values and the unclipped
`raw_norm_ratio`/`raw_local_fidelity`. A retained norm significantly above its
target raises `FloatingPointError` with a dtype-aware rounding allowance,
rather than being silently clipped to perfect fidelity.

For a logical gate stream whose site order has not been chosen yet,
`MpsOptimizer.LayoutFinder(gates, L=...)` or
`MpsOptimizer.gate_stream_layout(gates, L=...)` returns a 1D layout plan with
the optimized site order, old-to-new site map, internal mapped locations, and
span statistics. The finder implementation lives in
`pepsy.optimizers.mps.layout.MpsGateStreamLayoutFinder`, while
`MpsOptimizer.LayoutFinder` keeps the attached optimizer-facing API. The finder
builds a weighted interaction graph from gate and
sub-MPO supports, scores layouts with a Tensy-like scalar objective
`weighted_total_span + weighted_cut_congestion_l2 + tail_span_penalty`, and
uses degree/BFS/spectral/recursive candidates plus adjacent-swap refinement. If
available, the refinement uses numba; `order="quality"` also tries optional
nevergrad candidates, and optional KaHyPar recursive bisection when a config is
provided with `kahypar_config_path=...` or `PEPSY_KAHYPAR_CONFIG`. Event weights
default to `weight_mode="auto"`: angle metadata when present, otherwise a cheap
operator-Schmidt proxy for small dense two-site gates, falling back to count
weights. Pass `weight_fn(payload, support, event_type)` for explicit weights.

For a prescribed baseline rather than a searched order, pass an explicit site
permutation as `order`. The returned plan is marked `selected_order="fixed"`
and keeps the original gate stream unchanged:

```python
zigzag = py.square_lattice_zigzag(6, 6)
fixed_plan = finder.run(order=zigzag)
```

`square_lattice_zigzag` scans x across each row and reverses direction on
successive rows. It is a deterministic comparison layout; it performs no
refinement or tensor work.

For compression-oriented selection, pass `objective="compression"`. This
uses operator-Schmidt load over every MPS cut crossed by each support, with
support span retained as a replay-cost tie-breaker. Exact small dense ranks
are used when available; opaque, native, and wide operators use a conservative
operator-space rank bound and are marked in `rank_bound_reasons` rather than
silently being treated as rank two. The default `objective="locality"` keeps
the faster span/congestion heuristic for backwards compatibility.

The layout score depends on gate supports and optional gate/event weights, not
on the initial MPS tensor values. The plan does not rewrite the gate stream. To
use a layout during replay, call `opt.run(use_layout_finder=True)` or pass a
layout order such as `opt.run(use_layout_finder="quality")`; the optimizer
temporarily permutes the working MPS and restores the returned MPS to the
original site order. Layout-aware replay prints a concise report by default;
pass `layout_report=False` to silence it.

When the current state matters, use the explicit pilot selector:

```python
plan = opt.select_layout_for_compression(
    pilot_candidates=4,
    pilot_steps=64,
)
opt.apply_layout(plan, layout_report=False)
```

The selector replays the best static candidates on independent copies using
the real MPS mode, `chi`, cutoff, backend, and dtype. It enables the
infidelity trace for the pilot and chooses by measured compression
infidelity, final bond dimension, and elapsed time, and returns
per-candidate records under `plan["pilot"]`. The original state, queue, and
layout are unchanged. Perform this before installing a persistent layout;
reordering an already-entangled MPS remains explicitly guarded because the
reorder itself can be lossy or expensive.

The layout can be inspected graphically without changing the optimizer. The
finder returns a Matplotlib `(fig, ax)` pair. The original lattice and gate
connectivity remain a light grey background, while the colored arrow chain
shows the selected MPS permutation directly. The default plot is axis-free and
does not number the background lattice; use `show_site_labels=True` and
`show_axes=True` when those annotations are useful:

```python
finder = opt.layout_finder()
plan = finder.run(order="quality")
fig, ax = finder.plot(
    plan,
    site_coords={q: (q % 4, q // 4) for q in range(opt.p.nsites)},
)
```

`opt.plot_layout(plan, site_coords=...)` is the equivalent convenience wrapper.
Coordinates are optional; tuple-valued site labels are interpreted as `(x, y)`
automatically, and ordinary labels fall back to a 1D line. Install the
optional `viz` profile to enable plotting. A stream-order colorbar is not shown
by default; pass `colorbar=True` only when the MPS-position scale is useful.
The default plot contains visible `0` through `last` order labels but no title,
chain sentence, or other text. The styling follows Quimb's axis-free schematic
drawings while retaining Pepsy's ordinary `(fig, ax)` return value.
Pass `show_order_labels=False` to hide the position labels, or use
`show_chain_label=True` and `show_title=True` for additional annotations.


> API details are maintained as handwritten Markdown in this page.
