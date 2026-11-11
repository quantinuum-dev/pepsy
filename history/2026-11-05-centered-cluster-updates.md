# 2026-11-05 — Centered cluster update readiness

- Scope: finish the direct 3×2-cluster improvement over SU inside Pepsy,
  preserving local diagonal gauges without global SU regauging.
- Branch / baseline: `develop`, `ee22198`, plus the preceding
  [local-gauge working-tree changes](2026-11-05-local-bond-svd-gauges.md).
- Status: uncommitted; no staging, publication or production launch.
  Existing November 4 FU/boundary edits and running campaigns were preserved.

## Implementation

- `cluster_mode="loopunion", base_radius=4, fillin=True` reaches the reduced
  gate stream, single gate, compression and problem builders. It delegates
  to Quimb `select_local(..., grow_from="all")` on the original site graph,
  selecting the two plaquettes sharing an interior bond. Vertical bonds
  rotate the patch; edge bonds keep the available plaquette. Defaults retain
  graph-distance selection. With `max_loop_size=0`, one region is contracted
  directly, without additive environment corrections.
- Cotengra optimizers from `build_optimizer` and `ContractionPlanCache` flow
  through environment construction. Cached trees are reused while tensors,
  gauges and environment values are recomputed. An independent single-layer
  SU-weighted cluster contraction validates the reduced metric's scale and
  positivity, including fresh tensor values after a cache hit.
- Dense PEPO upper/lower gates act on the selected physical indices without
  forming a lifted gate. One-site actions use direct physical-leg gating;
  native PEPO simple-update sandwich behavior is avoided. Nonsymmetric complex
  gates check lower-leg transpose/conjugation conventions against independent
  gate-decorated operator networks. Operator legs are restored after reduction.
- Streams beginning with physical inputs switch to the returned SU-core
  representation for later pair updates. Only active gauges and endpoint
  core tensors change; exterior gauges and spectator arrays remain fixed.
- Torch/CuPy reduced ALS chooses native dense local solves by default,
  avoiding Quimb's SciPy iterative dispatch for accelerator dtypes.
- Native Symmray contractions preserve empty physical charge sectors in
  reduced targets. The previously reported native reduced-compression shape
  failure now passes; this does not implement weighted blockwise ALS or
  one-sided native fermionic PEPO updates.

The [BP API](../docs/api/bp.md#centered-six-site-gate-updates) contains the
ready-to-use call and distinguishes base-patch selection from loop-series
degree, additional loop corrections and retained χ.

## Validation

- Complete selected reduced-update, local-gauge, patch and compression
  suites, native BP tests excluding the baseline readout cases below, two
  SymPEPS wrapper checks and CPU/CUDA χ6 FU solve regressions:
  **173 passed, 6 deselected**, 26.99 s. CUDA cases executed.
- The wider selection first reported **133 passed / 5 failed**. All five
  failures reproduced with a clean `git archive ee22198 src/pepsy` package
  under `/tmp`, before these changes: three long-range fermionic exact-readout
  failures (spinless U1 and spinful U1/U1U1), and two open-rho sweep failures
  from empty native charge spaces (U1/U1U1). Their Z2 parameter passes.
  These readout paths were not changed. The final six-case deselection also
  excludes that passing Z2 parameter; it is not an xfail or weakened test.
- Full `ruff check src tests` and `git diff --check` passed. The whole
  repository suite was not run. Existing NumPy and intentional legacy
  `regauge_opts` deprecation warnings remain visible.

The earlier upstream audit in the linked local-gauge record was reused with
unchanged installed dependencies. Additional installed signatures checked:
`TensorNetwork.select_local`, `get_local_patch`, `get_loop_union`,
`tensor_network_fit_als` and `build_optimizer`. Public Quimb selection,
Cotengra paths and native dense local solves were adopted; no installed
library was modified.

## Bounded χ6 CPU/CUDA pilot

One upper-leg RZZ action (diagonal phases ±0.13) on the center bond of a
5×4 random complex128 PEPO, all bonds initially six. Each initial site tensor
was Frobenius normalized; exterior SU weights were `linspace(.8, 1.1, 6)`.
The retained environment patch has six interior sites, including nontrivial
exterior SU closures. Both devices used one numerical thread, native Quimb
ALS with three iterations, tolerance 1e-8, monitored complete-sweep stopping,
and `TorchLinalgConfig(stabilized=False, svd_driver="gesvd")`.
Cotengra used `build_optimizer(parallel=False, max_repeats=8, max_time=.5)`;
the CPU evaluation reused the CUDA contraction tree with new tensors.

| Diagnostic | CUDA | CPU |
| --- | ---: | ---: |
| Wall seconds, including cold setup | 1.036 | 0.438 |
| Initial weighted squared error | 1.2129844609209078e-8 | 1.2129844609209085e-8 |
| Final weighted squared error | 1.2123052973411853e-8 | 1.2123052973411850e-8 |
| Target norm squared in the local metric | 7.365620533787153e-7 | 7.365620533787156e-7 |
| Relative weighted squared error | 0.01645897031730277 | 0.01645897031730276 |
| Raw minimum environment eigenvalue | 5.022095993472113e-6 | 5.022095993472108e-6 |
| Clipped eigenvalues | 0 | 0 |

Both stopped by cost tolerance after three sweeps; every core tensor and
singular value was finite, with retained bond six. The reduced environment
shape was `(24,24,24,24)`. Maximum relative difference between reconstructed
physical site tensors was **2.03e-10**. CUDA peak allocated memory was
111,816,192 bytes (about 106.6 MiB). These are small local-workload timings,
not a device-speed recommendation or a 10×10 convergence result.

Machine-local scripts/results are `/tmp/pepsy_interior_3x2_chi6_readiness.py`
and `/tmp/pepsy-interior-3x2-chi6-readiness.json`; the JSON retains executed
source hashes. The prior whole-3×2 pilot remains separately in
`/tmp/pepsy_3x2_chi6_readiness.py` and `/tmp/pepsy-3x2-chi6-readiness.json`.
The measurements above are retained here because `/tmp` is ephemeral.

The direct cluster route is ready for a downstream χ6 experiment. Long-time
stability, PEPO/boundary convergence and 10×10 global costs remain unmeasured
for this method. Explicit cut-edge loop-series compression remains a separate
API from this direct base-cluster gate path.
