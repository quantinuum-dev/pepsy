# MpsOptimizer GPU/backend audit — 2026-09-23

## Assessment

Ordinary dense MPS replay uses Quimb/Autoray tensor operations and preserves
the supplied backend in the tested paths. It is not host-transfer-free.
The initial review found two backend correctness defects, plus synchronous
scalar bookkeeping and host-created control operators. The follow-ups below
fix the correctness defects, repeated control uploads, and per-compression
unitary bookkeeping reads. FIT convergence and physical control decisions
retain their host boundaries.

The local environment has no usable accelerator: Torch reports CUDA and MPS
unavailable, JAX exposes only a CPU device, and CuPy is not installed. Probes
therefore establish dispatch, backend preservation, and host-read call counts,
not CUDA correctness, GPU synchronization timings, or GPU throughput.

## Confirmed findings

### High: exact-to-MPS reconstruction forces the full state onto NumPy

`MpsOptimizer._ensure_mps_state` explicitly calls `np.asarray(ar.to_numpy(...))`
on the contracted state before `MatrixProductState.from_dense`. Both
`set_mode` when leaving exact mode and exact-mode control events reach this
path. A four-site Torch complex128 state becomes NumPy after an exact-mode Z
measurement or `set_mode("direct")`. Appending a Torch Hadamard after the
measurement fails with `tensordot(): argument 'other' (position 2) must be
Tensor, not numpy.ndarray`.

On a GPU input this path necessarily downloads the entire dense state, whose
size is exponential in site count. Reconstruction should pass the backend
array directly to Quimb, preserve represented scale and index ordering, and
keep state/backend metadata consistent. Add coverage for both mode switches
and a control event followed by another gate.

### High: explicit random FIT initialization is NumPy-generator-specific

`_build_randomized_fit_guess` constructs `np.random.default_rng(int(seed))`.
`_fit_random_data` forwards it through `backend_random_array` to Autoray's
`random.array`. The installed Torch RNG adapter expects an integer seed or a
compatible generator; it does not accept a NumPy Generator.

Reproduced through the public API with a four-site Torch complex64 product
state, CNOT on `(0, 3)`, `mode="dmrg2"`, `chi=4`, and:

```python
opt.run(
    progbar=False,
    n_iter=3,
    cutoff=0.0,
    fit_init_strategy="random_expand",  # "random" also fails
    fit_init_rand_strength=1e-3,
)
```

Both strategies raise `RuntimeError: manual_seed expected a long, but got
numpy.random._generator.Generator`. Zero perturbation bypasses the bug;
the default `guess-src` path also passed. Use a backend/device-compatible
Autoray generator, preserving deterministic seed advancement across tensors.

### Medium: every retained unitary compression synchronously reads a norm

`_start_unitary_norm_tracking` reads an initial scalar. Each call to
`_stabilize_unitary_compression_state` then reads the retained norm through
`to_float` and the backend scalar's `.item()` method. This occurs even with
`stabilize_unitary=False`, `finite_check=False`, `timing=False`, and
`progbar=False`: norm-survival bookkeeping is mandatory in the current API.

A Torch CPU probe containing H on site 0 followed by four CNOTs on `(0, 3)`
counted five Torch `.item()` calls in each of `direct`, `svd`, `swap`, and
`perm`, with no calls converting Torch arrays via `ar.to_numpy`. The same
probe counted five `.item()` calls plus 12 scalar `ar.to_numpy` reads in
`dmrg2`, and five plus eight in `mix` (`n_iter=3`, `cutoff=0`). These counts
exclude construction/readout and do not count all possible upstream scalar
protocols. They are not GPU profiler measurements.

The FIT reads come from sweep convergence checks in
`FIT._sweep_diagnostics_to_host`; they transfer scalars, not full tensors.
Replacing `.item()` with Autoray conversion would still require a host read.
Reducing this cost requires deferred/batched norm-ledger evaluation and/or a
fixed-sweep convergence policy while preserving existing diagnostics and
failure semantics. Do not silently remove bookkeeping or misuse
`non_unitary=True` as a performance switch.

### Medium: repeated control operators are built on the host

`_pauli_projector_submpo` builds each projector tensor with NumPy and calls
`_to_state_backend` for each site. Pauli probability preparation also creates
small rotations/projectors on the host. Thus GPU control-heavy runs require
repeated host-to-device copies even though their state contractions remain
on the backend. Cache immutable small constants by backend/device/dtype or
construct projector tensors using Autoray with an explicit device-bearing
`like` array. Preserve native Symmray metadata handling separately.

## Other boundaries and positive checks

- Public numeric gates/sub-MPOs are checked for backend, device, and dtype
  compatibility once at stream installation. Ordinary replay does not
  silently reconvert user gates every segment.
- Finite scans and synchronized timing are opt-in; neither explains away the
  mandatory norm reads above.
- Layout scoring can explicitly download gate arrays and use NumPy SVD.
  This is planning work, not ordinary replay. `remap_sample` intentionally
  returns NumPy; distinguish that public readout boundary from state evolution.
- Configured SciPy SVD fallback explicitly downloads matrices and uploads
  factors. Native ordinary Torch SVD has no automatic SciPy fallback by
  default; stabilized/raw-block policies need separate inspection when used.
- CuPy is excluded from `_fit_window_copy_supported`, retaining conservative
  copies. This is an unbenchmarked device-memory optimization opportunity,
  not evidence of a CPU transfer.

## Dependency audit

Installed in `/Users/rezah/envs/genpy`: Quimb
`1.15.1.dev55+gd0591eb70`, Autoray `0.11.1.dev3+g1b476b305`, Cotengra
`0.8.3.dev7+g1d7fd333f`, Symmray `0.3.2.dev8+g6c6dd34b5`, Torch `2.9.1`,
and JAX `0.8.2`.

Inspected installed signatures for `gate_nonlocal_`, `gate_with_auto_swap_`,
`Tensor.split`, `MatrixProductState.from_dense`, `canonicalize`, `Tensor.norm`,
`ar.do`, and `ar.to_numpy`; resolved NumPy/Torch/JAX SVD, QR, random-array,
and NumPy-conversion dispatch. Read the installed `random.array` and
Torch/JAX `random.default_rng` implementations. In particular, array `like`
dispatch does not make a foreign RNG compatible.

Reviewed the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra docs](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray repository](https://github.com/jcmgray/symmray). The requested
[Symmray Abelian-array page](https://symmray.readthedocs.io/en/latest/abelian_arrays.html)
failed to load. Public development docs can differ from installed builds.
Quimb documents backend/device-aware random arrays for SRC/SRCMPS/FIT; the
confirmed RNG issue is in Pepsy's separate random perturbation path.

Disposition: **adopt/retain** public Quimb/Autoray backend operations;
**prototype** deferred norm reads and cached control constants in a future
performance change; **defer** dependency upgrades, new compressors, and GPU
throughput conclusions. No compatibility shim or dependency edit was made.

## Validation

- Targeted optimizer backend/Torch/finite-check selection plus the same
  selection on entropy tests: **25 passed, 642 deselected**.
- Full `test_mps_entropy.py`, `test_mps_fit_performance.py`, and
  `test_mps_replay_metadata.py`: **22 passed, 2 failed**. Counts across these
  commands can overlap.
- Existing failures in the unchanged checkout:
  `test_mixed_maximum_is_refreshed_after_quality_repair` expects an MPO
  decision but receives DMRG;
  `test_layered_fit_builds_only_visited_tag_selections_and_can_run_globally`
  expects an empty lazy tag cache but finds it populated. Their root causes
  were not established by this backend audit.
- Independent public-API probes reproduced both correctness defects above.
  No live GPU, CuPy, or native Symmray GPU validation was possible.

## Implementation follow-up — 2026-09-23

- Exact reconstruction now passes the backend array directly to
  `MatrixProductState.from_dense`, preserving physical index names and
  refreshing backend metadata. No dense state is downloaded to NumPy.
- Random FIT perturbations create one generator through
  `ar.do("random.default_rng", seed, like=state_array)` and advance it across
  tensors through the existing `random.array` helper. Autoray's installed
  Torch creation dispatch injects both device and dtype for the generator.
  The old-Autoray fallback remains capability-gated, matching the existing
  random-array fallback. The NumPy RNG used for Python trajectory branch
  selection is unrelated and remains unchanged.
- Dense measurement/reset replay caches only seven fixed Pauli/Clifford
  constants per active backend/device/dtype. Each consumer receives an owned
  backend copy. Projectors and MPO tensors are constructed using backend
  arithmetic, `zeros_like`, and `stack`; no host projector/MPO arrays are
  uploaded on warm replay. Native Symmray still exits the dense MPO path
  before constructing constants. Cache signatures are derived from the live
  array, including after state replacement.
- Norm bookkeeping remains synchronous. Autoray cannot remove a host read
  required by a Python fidelity record or stopping decision. Reducing these
  waits requires a separate deferred/batched ledger design, not substituting
  one scalar-conversion function for another.

Rechecked upstream sources and installed versions (unchanged). Inspected
`from_dense` and the NumPy/Torch/JAX `random.default_rng`, `random.array`,
`copy`, `stack`, and `zeros_like` dispatch. The public
[Autoray random-array documentation](https://autoray.readthedocs.io/en/latest/autoapi/autoray/autoray/index.html#autoray.autoray.random_array)
specifies backend-specific generators: supporting `random.array` alone does
not accept a NumPy Generator as a Torch seed.

Disposition: **adopt** existing public backend-array splitting, RNG, and
array-assembly APIs; **defer** asynchronous ledger redesign and unrelated
native randomized-cutoff compatibility. No upstream package was patched.

Validation of the implementation:

- New backend regressions plus MPS audit, dynamic controls, nonfinite policy,
  and Quimb compatibility: **110 passed, 12 skipped**. The new regression file
  was then rerun with custom-index/scale assertions: **14 passed, 12 skipped**.
  Counts overlap. CUDA/CuPy cases are present but skipped on this machine;
  Torch CPU and JAX CPU execute.
- Broader optimizer selection: **81 passed, 22 failed, 596 deselected**.
  All 22 failures are native randomized fermionic guesses with cumulative
  cutoffs rejected by installed Symmray (`max_bond_mode='eager'`). Replayed
  both failing test groups against the unmodified HEAD optimizer loaded from
  a temporary file: **the same 22 cases fail with the same cutoff error**.
  These failures are independent of this change. No native cutoff semantics
  were altered to make those tests pass.
- Default smoke: **130 passed**. Repository Ruff and whitespace checks passed.
  No GPU wall-time benchmark was possible.

## Autoray norm ledger follow-up — 2026-09-23

The user requested retaining norm/infidelity diagnostics on-device rather than
removing them. Ordinary unitary replay now keeps its previous norm, local
fidelity, and cumulative log-fidelity as backend scalars. Autoray `log`,
`exp`, `expm1`, `clip`, `where`, and logical operations replace the Python
scalar arithmetic in the accelerator path. Diagnostic values use
`stop_gradient` so history does not retain complete simulation gradient
graphs. The original norm remains in the differentiable normalization path.
Torch/CuPy diagnostic arithmetic uses float64 without promoting MPS tensor
data, except Torch on Metal uses float32; JAX respects its configured precision.
NumPy histories remain directly serializable Python values.

`norm_events` retains backend values during replay. `get_norm_events()` and
`norm_diagnostics()` materialize independent Python-valued outputs only at
readout. Mixed rollback and trajectory copies preserve the device ledger and
its pending validation flag. Invalid entries do not change accumulated
fidelity, complete loss remains absorbing, and unchecked NaNs retain the
existing propagation policy. Physical branch probabilities remain separate
from compression loss.

The existing unconditional zero-norm guard now accumulates a device Boolean,
read once before ordinary replay returns (also checked at diagnostic readout).
Thus zero norms still raise from `run()` and timed failures remain marked as
failed. With `finite_check=True`, validation remains immediate. Ordinary
deferred errors are reported at the replay boundary rather than the first
affected gate; normalization uses a safe denominator until that error is
raised. FIT adaptive stopping, measurement selection, nonunitary exponent
normalization, progress displays, and upstream truncation can still require
host results. This is not a claim of completely asynchronous simulation.

Rechecked the upstream sources and installed versions listed above. Probed
NumPy/Torch/JAX dispatch/signatures for `stop_gradient`, `log`, `exp`, `expm1`,
`clip`, `where`, `logical_or`, `logical_not`, `full_like`, and `astype`.
**Adopt:** existing public Autoray scalar operations. **Defer:** changing FIT
stopping rules, physical branch selection, or upstream decomposition policy.
No new backend registration or dependency patch was introduced.

Tests forbid Tensor `.item()` numeric reads, Python float/bool conversion, and
Autoray host-array conversion during direct/SVD/swap/perm/DMRG2/mix replay
with convergence disabled. Each tested replay performs exactly one Boolean
read for zero-norm validation, versus the earlier one initial plus one norm
read per compression. This is a call-count test on Torch CPU, not a measured
CUDA speedup. Additional coverage compares NumPy and Torch/JAX results with
normalization both enabled and disabled; tests state dtype, physical controls,
native Torch Symmray compression, detached histories, copying/rollback, zero
errors, and invalid/complete-loss/NaN ledger behavior.

Validation:

- Backend regressions, MPS audit, dynamic controls, nonfinite policy,
  trajectory noise, and Quimb compatibility: **221 passed, 16 skipped**.
  Torch CPU and JAX CPU execute; CUDA/CuPy cases skip on this machine.
- Targeted optimizer norm, fidelity, stabilization, timing, finite-check,
  rollback, and copy regressions: **97 passed, 566 deselected**.
- Default smoke: **130 passed**. Repository Ruff and whitespace checks passed.
  These suites overlap; no GPU wall-time benchmark was possible. The earlier
  independently reproduced native randomized-cutoff failures remain outside
  this change.

## Second review — 2026-09-23

Rechecked the upstream sources and installed versions above; the installed
versions are unchanged. Inspected the actual Torch dispatch functions for
`where`, `stop_gradient`, `logical_or`, `log`, `full_like`, and `astype`.
The review found and corrected two missed cases:

- A NaN followed by a zero-survival event left cumulative fidelity NaN. The
  former Python ledger treated a zero-survival event as unconditional complete
  loss. The Autoray accumulator now preserves that policy in both orders,
  including when a host-computed physical event follows a backend ledger.
- Diagnostic float64 promotion was unconditional for Torch. Apple's Metal
  device rejects float64, as confirmed in the
  [PyTorch 2.9.1 implementation](https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/native/mps/OperationUtils.mm).
  Diagnostic scalars on Metal now use float32 without moving data to the CPU.

**Adopt:** existing Autoray logical operations and dtype conversion, with
device metadata selecting supported diagnostic precision. No upstream patch
or new dependency is required. Regression tests failed before the fixes on
NumPy/Torch NaN ordering and emulated Metal metadata. The metadata test runs
on CPU and proves precision selection only; an additional actual Metal test
is skipped here because no device is available.

Validation: backend, control, trajectory, compatibility, public API, package
layout, and sampler checks: **371 passed, 23 skipped**. The guarded six-mode
unitary tests still perform exactly one Boolean read per replay, with no
numeric norm readbacks. Accelerator performance remains unmeasured.
Targeted norm/fidelity, stabilization, timing, finite-check, rollback, and copy
checks: **97 passed, 566 deselected**. Default smoke: **130 passed**.
The new edge cases also passed after strengthening mixed host/backend coverage
(**4 passed, 1 Metal skip**). Counts overlap. Repository Ruff and whitespace
checks passed.

## Disabled diagnostics and timing defaults — 2026-09-23

Confirmed ordinary replay defaults `finite_check`, `fit_overlap_diagnostics`,
`quality_check_every`, `timing`, and `timing_sync_device` to `False`. Strengthened
the existing five-mode untimed test to omit those options and fail on profiling
clock reads, synchronizer construction, finite-data scans, overlap contractions,
or quality scans. Required norm/fidelity accounting and zero-divisor protection
remain independent of optional diagnostics.

The separate MPI `collect_diagnostics` option was still `True` by default in
the MPS facade. It now defaults to `False`, including its private shot entry
point and non-MPI option validation. The shared MPI runner previously read
profiling clocks even when collection was disabled; ordinary, streaming, and
retained-checkpoint paths now skip those reads. The runner's own default is
unchanged; MpsOptimizer explicitly passes its disabled default. MPI summaries
can still be enabled explicitly.

Validation: **169 passed** across MPI orchestration, nonfinite policy, and
trajectory checks; **15 passed, 648 deselected** for optional-overlap and
disabled-clock checks; **130 passed** in the default smoke suite. Ruff and
whitespace checks passed. MPI orchestration tests use a fake communicator;
no multi-process MPI or accelerator performance claim is made.
