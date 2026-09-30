# 2026-09-30 — Autoray and MPS DMRG GPU consistency

## Scope and environment

Read-only runtime audit of the working-tree `MpsOptimizer` modes `dmrg`,
`dmrg2`, and `dmrg3`, after the [mode cleanup](2026-09-30-mps-dmrg-modes.md).
No runtime, dependency, notebook, or default-setting changes were made by
this audit. GPU: NVIDIA RTX A5000; shared Python 3.12 environment, local
Pepsy source first. JAX probes disabled memory preallocation.

Installed versions: Autoray 0.11.1.dev3+g1b476b305,
Quimb 1.15.1.dev66+ge927f06e1, Cotengra 0.8.3.dev7+g1d7fd333f,
Cotengrust 0.2.1, Symmray 0.4.1.dev7+g83fb22865,
Torch 2.6.0+cu124, JAX 0.10.2. The distribution lookup for `cupy` found
no package with that name; this does not establish CuPy module availability
because CUDA-specific distributions use other names. CuPy was not probed.
Torch defaults observed: float32 matmul precision `highest`, TF32 matmul
allowed False. JAX defaults: matmul precision None, x64 False.

## Findings

- **Adopt / already implemented:** shared conversion helpers use Autoray
  namespaces; FIT uses backend contractions and Quimb tensor splits. Dense
  random guesses use `random.array` and a backend/device-specific generator.
  Gate inputs are checked against state backend, device and dtype.
- On the installed Torch backend, `get_namespace(cuda_complex128_array).zeros`
  and `do("zeros", ..., like=array)` both preserve CUDA complex128. In contrast,
  `like="torch"` alone creates CPU float32. This demonstrates why creation
  routines should receive the array template or explicit device/dtype.
- **Measured precision gap:** JAX GPU default arithmetic failed 7 of 8 existing
  backend regressions despite compatible device/dtype. The initial five-site
  one-site DMRG probe had relative state error 7.4317e-4. Repeating with
  `JAX_DEFAULT_MATMUL_PRECISION=highest` passed all eight regressions and all
  six all-mode probes. The controlled change and JAX documentation point to
  reduced-precision matrix multiplication, not missing Autoray dispatch.
  This is a backend arithmetic policy, separate from storage dtype.
- **Adopt recommendation, not implemented:** make the JAX precision choice
  explicit for accuracy-sensitive GPU runs, e.g. scope state construction,
  optimizer execution and reference/readout contractions with
  `jax.default_matmul_precision("highest")`. A process environment setting
  also works. Do not silently alter global precision at Pepsy import.
- **Compatibility shim / retain:** the declared Autoray minimum is 0.9;
  random-array support used by Pepsy is newer. The existing capability-gated
  fallback generates NumPy noise then uploads it. It preserves correctness
  policy but has a different performance/RNG route. Existing namespace
  compatibility handling also needs to retain device objects on older releases.
- **Prototype only:** newer `ar.to`, `ar.to_device`, and `ar.from_numpy` may
  simplify explicit conversion boundaries. They must preserve Pepsy's complex
  promotion, input validation, native metadata and supported older-version
  behavior before replacing existing helpers. They are not a demonstrated
  improvement to the current DMRG inner loop.
- **Defer pending coverage:** the one-site QR shortcut includes NumPy, Torch
  and JAX; CuPy uses the complete gauge move. This is a possible performance
  disparity, not an established correctness defect. CuPy device synchronization
  also uses the current stream; multi-device/stream timing needs its own test.
- Autoray cannot remove convergence scalar reads or adaptive-SVD rank-selection
  synchronization. All modes can split during guess/target preparation;
  `dmrg` has only one-site FIT updates, while `dmrg2`/`dmrg3` additionally split
  blocks. Preserve Quimb's split/truncation semantics and explicit
  `TorchLinalgConfig` policy. Native Symmray raw-block drivers have separate
  registration paths; a generic Autoray rewrite cannot replace them.

## Measured validation

Small probe: five-site random MPS (seed 51, bond dimension 2), nonlocal CNOT
on (0,4), H on site 2, CNOT on (1,3), chi 8. Compare to NumPy exact replay.
Use default optimizer run settings, separately with default initialization
and `random_expand` at strength 1e-3. This is an exact-capacity fixture, not
an ill-conditioned or severe-truncation benchmark.

- Torch: 24 runs covering CPU/CUDA, complex64/complex128, three modes and
  two initialization strategies. All retained device/dtype and matched the
  reference. Maximum CUDA relative errors: 2.24e-6 (complex64), 3.63e-15
  (complex128). CPU/CUDA comparisons also passed.
- During these Torch runs, instrumented `ar.to_numpy` saw only scalar CUDA
  transfers, not full states. This is not a complete device-transfer profiler;
  it does not count direct `.item()` calls or upstream internal transfers.
- `pytest -q -ra -o addopts='' tests/test_mps_gpu_backend.py -k cuda`:
  **8 passed, 43 deselected** in 2.48 s.
- Same suite with `-k jax`, default JAX GPU precision:
  **7 failed, 1 passed, 43 deselected** in 37.93 s. Failures: exact replay
  followed by direct/perm, random/random_expand FIT, warm controls, and both
  backend-ledger restoration cases. All were numerical comparison failures.
- Same JAX suite with `JAX_DEFAULT_MATMUL_PRECISION=highest`:
  **8 passed, 43 deselected** in 38.83 s.
- JAX GPU complex64 at highest precision: all six mode/initialization probes
  pass, retain GPU complex64, maximum relative state error 1.77e-6.
- A separate all-mode measurement at default JAX precision (recording errors
  instead of stopping at the first failed tolerance) retained GPU complex64
  but had relative state errors from 8.70e-4 to 1.62e-3 across the six cases.
  Log: `/tmp/pepsy-autoray-dmrg-jax-default-all.log`. The difference from the
  initial single-case probe also cautions against claiming bitwise replay.

Temporary scripts/logs: `/tmp/pepsy_autoray_dmrg_gpu_probe.py`,
`/tmp/pepsy_autoray_dmrg_jax_probe.py`, and
`/tmp/pepsy-autoray-dmrg-{cuda-tests,gpu-probe,jax-tests,jax-highest-tests,jax-highest-probe}.log`.
No speed benchmark, full suite, native Symmray GPU, multi-GPU, CuPy,
JAX complex128, or autodiff validation was performed in this audit.
Seeds are repeatable within a backend route, not guaranteed identical between
backends, versions or precision policies.

## Upstream evidence

Reviewed official [Autoray automatic dispatch and namespaces](https://autoray.readthedocs.io/en/latest/automatic_dispatch.html),
[Autoray source](https://github.com/jcmgray/autoray),
[Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray source](https://github.com/jcmgray/symmray).
The requested Autoray index was accessible, but documentation snapshots had
inconsistent version labels; installed source/signatures were authoritative
for probes. Autoray random/changelog and Symmray abelian-array pages were
unavailable through the browser; installed source and official repositories
were used instead. Inspected namespace, Torch random creation, conversion
and Quimb SRC signatures. No upstream library was edited.

JAX documents the arithmetic distinction in
[matmul precision configuration](https://docs.jax.dev/en/latest/_autosummary/jax.default_matmul_precision.html)
and [lax precision levels](https://docs.jax.dev/en/latest/jax.lax.html):
`highest` uses float32 GPU arithmetic, while default can use TF32.
