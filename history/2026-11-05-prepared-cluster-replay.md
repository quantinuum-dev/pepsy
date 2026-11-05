# 2026-11-05 — Prepared middle-out cluster replay

- Scope: implement the prepared execution plan proposed for the existing 3×2
  PEPO benchmark, preserving the middle-out schedule and numerical policy.
- Branch / baseline: `develop`, `60407db`; examples `main`, `9c0ab0b`.
- Publication: included in the prepared-replay publication commit under the
  existing session authorization. Saved runs, checkpoints and unrelated edits
  are preserved.

## Implementation

`pepsy.bp.PreparedLoopClusterUpdate` binds gate selectors, single-site axis
permutations, direct six-site loop-union geometry and shape-dependent Cotengra
array contraction programs. Each pair still prepares fresh QR/LQ factors and
recontracts current spectator tensors and SU weights. Programs/trees have a
bounded cache (default 512), with rank-growth rebuilds and explicit hit/miss/
eviction counters. One combined boundary-validity transfer retains the
existing finite/real/nonnegative checks. No numerical environment is cached.

ALS, its conditioning/spectral fallback, rank caps, stopping and local final
gauge extraction retain their existing implementations. RMS scales go to the
network exponent after each pair. Gate order and every truncation are retained;
there is no within-PEPO reordering, fusion or independent-network batching.
`[start:stop]` and callbacks retain source barriers and original gate indices.
Native symmetry, messages and other cluster families use the ordinary route.

The example's `--loop-cluster-execution auto` selects prepared Torch execution
or the explicit Quimb reference route. `reference` also permits matched eager
Torch comparisons. Repeated lower-leg transposes are prepared once per gate
object, retaining native $O G^T$ semantics and original action counts.

## Gradient defect found by the new checks

Full-operator finite differences exposed a pre-existing loss of singular-value
derivatives after local gauge extraction. The reduced ALS tensor and exact
pair reconstruction had correct derivatives, while the returned SU operator
did not. Installed Autoray's Torch `copy` is `detach().clone()`; reduced-update
target/gauge copies now use Torch `clone()` explicitly. Other backend copies
retain their existing dispatch. Forward values and non-autograd simulations
are unchanged. Multi-gate physical-operator derivatives are checked against
finite differences with native/stabilized Torch linalg and eager/AOT ALS.

## Measured workload

Read-only replay on the saved corrected 23-spin homogeneous D20 χ6 final
residual. Apply the first 96 actual actions from the middle-out schedule as
an additional bounded diagnostic: 60 one-site actions and 36 pair truncations.
This is not a new T=6 trajectory or a replacement for the saved χ6 result.

Complex128 CUDA, one numerical host thread, exact `gesvd`, ALS cap10/tol1e-8,
Hermitian-only six-site environments and measured local fidelity. Alternate
the two modes over four warm repetitions; include state/gauge copies and
per-pair diagnostic work, exclude plan/gate-bank construction. Median times:
**2395.279 ms reference** and **2287.879 ms prepared**, a **1.04694×** ratio.
All 36 iteration counts and normalized local errors were identical. Boundary
MPS cap18 diagnostic readouts were also identical. Timings are from a shared
workstation; this modest measured gain is not a universal throughput claim.

Initial isolated-pair pilots showed small gains and timing scatter. Optional
AOT did not consistently improve them. Final AOT capture remains restricted
to shared ALS kernels; capturing an environment graph for every lattice bond
would churn its bounded chart cache. The prepared environment instead uses
Cotengra's generated Torch program directly. No Inductor or whole-step CUDA
graph is enabled. Independent-network batching remains a separate next stage.

Temporary scripts/results: `/tmp/pepsy-prepared-cluster-20261105/`, especially
`stream_bench.py` and `stream-results.json`. No production job was launched.

## Upstream evidence and validation

Unchanged installed versions: Quimb `1.15.1.dev101+g1b64353d9`, Autoray
`0.11.1.dev14+g014a3f69a`, Cotengra `0.8.3`, Symmray
`0.4.1.dev25+gdc377f033`, Torch `2.14.1+cu126`. Reused the active task's
official-source audit; inspected installed `array_contract_expression`,
`ContractionTree.get_contractor`, `gauge_simple_insert` and Torch copy dispatch.
Classification: **adopt** public Cotengra array programs; **compatibility shim**
for differentiable Torch copies; **defer** independent batching and full CUDA
graph capture. No dependency, global registration or installed library changed.

An initial affected BP/API/PEPS/PEPO selection passed **515 tests**, including
the new prepared path and native graded reduced compression. Downstream
cluster/global-cost/FU-resolution/relocation selection passed **109 tests**.
After restricting AOT to shared ALS kernels and extending payload diagnostics,
the focused solver/API selection passed **125 tests** and example selection
passed **18 tests**. Additional input-isolation and native-gradient cases are
included in the final affected selection: **520 passed** in 83.44 s, including
all **20** prepared-replay cases. Final downstream selection: **110 passed**
in 81.28 s. The standalone prepared suite also passed 20 cases; final Ruff,
whitespace and local documentation targets pass.

The previously reproduced full-repository native fermionic readout failure
remains outside this change; a passing selection does not make the full suite
green. Existing optional-mode audit findings retain their separate scope.
Fresh full-suite fail-fast stopped at
`test_bp_symmray.py::test_fermionic_long_range_hopping_sign_survives_su_and_bp_gauges`:
**284 passed, 2 skipped, 1 failed** in 46.16 s, matching the earlier baseline
failure. Publication outcome is reported at handoff after checking both
remote refs; Git identifies the publication commits.
