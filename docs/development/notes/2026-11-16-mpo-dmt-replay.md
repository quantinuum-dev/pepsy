# Ordinary MPO DMT: gate-stream replay and 2D measurements

**Implemented:** `MpoOptimizer(mode="dmt")` takes the existing MPO gate
stream, applies each gate without target truncation, then protects local reduced
operators during SVD compression. See the
[API guide](../../api/optimizers/mpo.md#dmt-gate-stream-replay) and
[reproducible CPU benchmark](../../../examples/mpo_dmt_benchmark.py).
The initial full-sweep implementation and measurements are recorded first;
the automatic-cutoff/local-update follow-up below supersedes that replay policy.

## Correctness contract and implementation

For each requested region $R$,

$$
\operatorname{Tr}_{\bar R}{\color{purple}\rho_{j+1}}
=\operatorname{Tr}_{\bar R}{\color{blue}\mathcal G_j(\rho_j)}.
$$

The reference uses the current compressed $\rho_j$. Discarded information
can affect later local observables, so this does not make the entire circuit
trajectory exact. The 3×3 measurement below explicitly quantifies that error.

- Reuse the existing MPO ket/bra payload convention and target builder.
  A matrix payload `G` acts as `G.T`; pass `U.T` to evolve by `U rho U†`.
  Distant/reversed supports, explicit ket/bra pairs and nonunitary actions
  are checked against independent dense binary-basis embeddings.
- Set **both** dense-gate factorization cutoff and target-compression cutoff
  to zero, with no intermediate bond cap. The gate builder previously
  inherited Quimb's `1e-10` default even for a zero-cutoff output request.
  The new optional `gate_cutoff` keyword changes only explicit requests;
  existing replay paths retain their defaults.
- One protected full-chain sweep follows each target. A candidate commits
  only after marginal and optional finite checks; per-step rollback is
  inherent, while `atomic=True` also restores earlier completed gates.
  In-place mode preserves the original MPO object. Rollback restores backend
  dtype metadata as well as arrays, including real-to-complex promotion.
- Preserve canonical-center and tensor `left_inds` metadata after compression.
  `compress_dmt()` operates on the live state without replay; optional reports
  belong to the latest DMT operation. Norm-survival fidelity estimates are
  suppressed until `set_mpo()` because DMT is not an orthogonal projection.
- Reject incompatible ordinary-compressor overrides/fallbacks, nonzero target
  cutoffs, on-run layout installation, native Symmray/cyclic storage, channels
  and gates of arity above two. Fixed layouts and physical region mappings
  remain supported. No preservation guard is disabled.

## Upstream audit and compatibility

Installed: Python 3.12.14, Quimb 1.15.0, Autoray 0.8.4, Cotengra
0.8.3.dev8+g8954240f2, Cotengrust 0.2.1, Symmray 0.4.1.dev15+g0374aaa3c,
JAX 0.8.2, Optax 0.2.6, NumPy 2.5.2. No package was upgraded.

Reviewed the official [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra docs](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray repository](https://github.com/jcmgray/symmray).
The linked Symmray Abelian-array page was unavailable; installed source and
explicit capability rejection define this dense-only path.

**Adopt:** public Quimb gate-MPO construction, zero-cutoff decompositions,
native backend conversion and the existing protected kernel. Installed Quimb
does not provide `gate_sandwich_with_auto_swap`, so this route uses Pepsy's
maintained layer-target construction rather than relying on the latest docs.

**Compatibility shim:** Autoray 0.8.4 keys explicit backend-name namespace
requests by the Python type `str`, so NumPy/Torch/JAX requests collide. This
caused Torch conjugate tensors to reach NumPy during the new mixed-backend
test sequence. The backend-owned shim checks actual namespace identity,
corrects only faulty string cache keys, preserves array dispatch and the
existing unhashable-device policy, and updates imported Quimb/Cotengra aliases.
It is installed at MPO optimizer construction. Correct modern caches remain
untouched; mocked modern/legacy cases and native replay both pass. DMT also
uses a thread-local Quimb contraction-backend context. No installed files or
default SVD/QR registration policy were changed.

**Initially deferred:** native PEPO evolution, channel replay, native symmetry,
and a persistent ordinary-MPO workspace. The follow-up implements the last
item for adjacent updates; the other routes remain deferred.

## CPU speed and accuracy

Three warmed repetitions, median, one BLAS/OMP thread, complex128. Inputs are
seed-321 positive product mixtures with Bloch length 0.5, constructed directly
in the matrix-unit basis. Regions are open plaquettes in `snake-row-major`
order. Replay uses four physical lattice-edge gates, including reversed and
nonadjacent MPO supports. Every output satisfies the stated global bond cap.

| Case | Sites / initial rank / cap | DMT | SVD baseline | DMT local defect | SVD local defect |
| --- | --- | ---: | ---: | ---: | ---: |
| 4×4 compression | 16 / 72 / 64 | 0.193 s | 0.032 s | 3.24e-15 | 7.08e-4 |
| 5×5 compression | 25 / 96 / 72 | 0.775 s | 0.105 s | 5.63e-15 | 2.74e-3 |
| 3×3 replay, four gates | 9 / 64 / 40 | 0.220 s | 0.119 s | 1.57e-15 | 1.03e-3 |
| 4×4 replay, four gates | 16 / 72 / 64 | 2.454 s | 2.129 s | 3.86e-15 | 2.12e-3 |

Compression defects compare with independent product-mixture marginals.
Replay defects are maximum Frobenius errors against each gate's untruncated
target, measured using separate direct partial traces outside timing.
Replay timings include optimizer construction. The SVD replay baseline adds
a global SVD sweep after each gate: ordinary local SVD replay alone leaves
unvisited initial bonds above `chi`, which would make a same-cap comparison
misleading. These are different algorithms/accuracy contracts, not equal-error
speed comparisons. No general speedup or GPU scaling is claimed.

**Accumulated trajectory error:** the 3×3 final local defect versus the fully
untruncated four-gate circuit is `1.15e-3` for DMT and `1.43e-3` for the SVD
baseline. Machine-precision preservation at each truncation does not remove
this accumulated approximation error. Full dense trajectory comparison is
restricted to this small case; larger checks remain local.

Complex64 repetitions also passed independent validation. Maximum DMT local
defects over these cases ranged from `2.63e-7` to `7.07e-7`; maximum trace
defect was `2.31e-6`. DMT medians were 0.172/0.680 s for the two compression
cases and 0.173/1.667 s for replay. Working-precision limits remain explicit.

## Bottlenecks and bounded optimization

The plain 4×4 instrumented replay spent about 0.966 s constructing targets and
1.464 s in protected compression, roughly 40%/60%. A separate CPU profile
spent 1.128 of 2.663 s in Quimb's NumPy stabilized QR. Exact target
materialization, canonicalization and protected sweeps dominate. Oracle
contractions are excluded from these stage timings.

Avoiding `astype` on already matching dtypes removes about **56.8 MiB** of
redundant array copying per four-gate 4×4 run. A matched alternating comparison
gave 2.447 s without those copies and 2.444 s with them: no measurable wall-time
gain on this CPU. The improvement is allocation reduction, not a claimed
speedup. Reducing exact-target/canonicalization work is the main next route;
approximate targets or fewer preservation checks were not adopted.

## Validation scope

- MPO/backend/gate selection: **364 passed, 3 skipped**, including the legacy
  cache shim and existing MPO modes. The subsequently expanded final replay
  file passes **35 cases**, including all NumPy/Torch/JAX complex64/128
  combinations, nonunitary trace changes, physical layouts, atomic promotion
  rollback, full Torch replay derivatives and a stationary 16-site identity
  density under nonlocal unitary gates.
- Five Pauli DMT files plus the earlier replay selection: **252 passed,
  one existing expected failure**. The full Pauli selection remains 225 passes;
  the expected failure is unresolved float32 auto trace classification.
- Final native-backend/replay rerun after tightening the legacy-cache probe:
  **106 passed, one skipped**, including the expanded 35-case replay file.
- API/layout: **54 passed**, one existing installed-version mismatch
  (`0.5.0` installed versus checkout `0.6.0`). Full-suite attempt stopped at
  the pre-existing BP namespace-dispatch failure, **104 passed, 7 skipped**.
  The MPO-entry shim does not globally initialize unrelated BP execution.
- Ruff, local Markdown links and diff whitespace checks pass. CPU only;
  no GPU or native-Symmray DMT numerical support is claimed.

## Follow-up: automatic cutoffs and local replay

**Implemented:** ordinary `compress_mpo_dmt` and `MpoOptimizer.compress_dmt`
now default to `cutoff="auto", cutoff_mode="auto"`, consistent with DMT
`run`. The shared dtype policy resolves to `1e-12` in double precision and
`1e-6` in single precision. DMT auto mode is `rel`, acting on the free
residual's largest singular value; protected directions are independent of
that threshold. Explicit zero and all explicit cutoff modes remain supported.
These are truncation rules, not bounds on accumulated physical-observable error.

After the first full protected sweep, adjacent gates reuse canonical cores,
trace environments and region plans, and split only their active bond. All
other bonds already obey `chi` and cannot grow under that physical action.
The workspace is copied structurally before each step; array operations
replace cores, so failed updates cannot corrupt the committed state or its
environment caches. Marginal, protected-budget, trace and optional finite
checks remain active. The actual mixed canonical center and tensor isometry
hints are recorded. Local reports contain one split and pre-gate source bond
dimensions. Distant/one-site gates and dtype promotion rebuild the exact full
target and refresh the workspace. It is reused within each replay call.

The local sweep schedule can produce a different approximation trajectory
from resweeping every bond. It preserves the same per-step regions; neither
schedule solves the globally optimal constrained MPO problem. The initial
full sweep is performed **after** applying the first gate, avoiding loss of
information that could enter that gate's protected target.

**Audit reused:** dependencies are unchanged from the audit above. This adopts
the existing generic workspace transfer, native Quimb QR/SVD, and thread-local
backend context; no new dispatch shim or optional package is required. Installed
NumPy QR allocates its factors rather than overwriting source cores.

### Matched CPU comparison

Three alternating warmed repetitions in one process, one BLAS/OMP thread,
complex128, identical four-edge inputs and **auto cutoff in both versions**.
The comparison loaded the pre-follow-up replay helper saved before editing;
both versions used the same protected kernel and dtype policy.

| Mapped square | Full sweeps per gate | Local workspace reuse | Time reduction |
| --- | ---: | ---: | ---: |
| 3×3 | 0.1795 s | 0.1470 s | 18.1% |
| 4×4 | 1.7504 s | 1.5992 s | 8.6% |

Absolute times varied across separate runs; these matched medians support the
speed comparison. A separate instrumented 4×4 run replaced two full targets
and sweeps with two local updates: target 0.909 s, compression 1.124 s, local
updates 0.076 s. Distant gates remain the dominant cost. No claim of general
optimality, GPU scaling, or speedup for every ordering is made.

The benchmark defaults to auto and accepts `--cutoff 0` and
`--cutoff-mode ...` for explicit policies. Its untimed oracle now constructs
an independent full target even for local production updates. Thus no gate
escapes validation merely because it skips the full-target builder.

### Square-layer and precision checks

`--case 3x3-layer --repeats 3` exercises all **12 nearest-neighbor edges**:
five full targets/sweeps and seven local updates. Complex128 DMT median was
0.764 s, maximum protected-region defect `2.09e-15` and trace defect `4.89e-15`.
Final local error versus the entirely untruncated circuit was `1.63e-3`
(ordinary SVD baseline `2.08e-3`). Per-step conservation still differs from
full-trajectory accuracy.

All compression/replay/layer cases passed in complex64 and complex128.
Largest independent local defects were `5.98e-15` in double precision and
`4.28e-6` in single precision; largest single-precision trace defect was
`2.21e-6`. The latter precision defects are larger than the earlier zero-cutoff
four-gate measurements, and remain within the existing working-precision
checks. Use double precision when tighter marginal accuracy is required.

### Follow-up validation

- Ordinary MPO/replay/gate and all five Pauli DMT suites: **536 passed,
  2 skipped, one existing expected failure**. This includes nine-gate mapped
  2D sequences, dense ket/bra references, actual left/right canonical gauges,
  dtype-auto defaults, native backends, stabilized Torch derivatives and
  rollback after local updates. Final incremental checks are recorded in the
  [handoff](../../../history/2026-11-16-mpo-dmt-replay.md).
