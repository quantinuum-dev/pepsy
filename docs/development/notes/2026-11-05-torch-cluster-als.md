# 2026-11-05 — Guarded Torch cluster ALS and local fidelity

Implemented an opt-in dense Torch solver and scalar fidelity tracker in Pepsy.
Public usage belongs to the [BP guide](../../api/bp.md#dense-torch-als-and-local-fidelity).
Branch/baseline: `develop`, `4d50e53`; the changes accompany the Torch ALS
publication commit. See the handoff below for validation and publication scope.

## Numerical policy

The new solver contracts the same retained open environment, reuses the
weighted target, and batches physical indices as RHS columns. It retains
SVD initialization, rank, full-sweep QR/LQ, and normalized stopping. Automatic
solver selection remains Quimb. Native Symmray keeps the graded SVD adapter.

For a Hermitian local matrix $A=LL^\dagger$, certify that spectral clipping is
inactive using

$$
\lambda_{\min}(A)\geq\|L^{-1}\|_F^{-2},\qquad
\lambda_{\max}(A)\leq\operatorname{tr}A,\qquad
{\color{blue}\delta\operatorname{tr}A\|L^{-1}\|_F^2}<\tfrac12,
$$

where $\delta=\max(\mathrm{rcond},10^{-15})$. A finite Cholesky solution must
also pass a backward-residual check. Uncertified, singular or indefinite
matrices retain the reference Hermitian eigensolve and relative eigenvalue
floor. This supersedes the earlier unguarded direct-solve prototype; it does
not silently remove conditioning or loosen ALS tolerance.

Optional local fidelity uses the normalized squared complex overlap in that
same environment. Torch reuses the final weighted residual and weighted
target, so measurement adds inner products and host scalar transfers rather
than another environment contraction. Other solvers perform local metric
contractions when requested. The tracker retains only scalars; its product
is heuristic, not global fidelity or an error bound. Invalid metrics produce
unavailable fidelity with an explanatory status.

The stream can now perform amplitude-preserving RMS extraction after each
pair with `strip_exponent=True`. A 100-gate PEPO regression matches manual
per-pair bookkeeping and remains finite. This addresses the long-stream scale
hazard in the [earlier audit](../../../history/2026-11-05-cluster-postpublication-audit.md).
That audit's tree-reduction, positive message-floor rescaling, and native-U1
sector issues remain unresolved; they are separate from the dense six-site
solver optimization.

## Bounded performance measurement

Read-only replay from the corrected 23-spin homogeneous depth20 χ6 residual;
one additional lower-leg RZZ gate on each listed interior bond. Complex128,
one host numerical thread, native `gesvd`, six-site Hermitian-only environment,
cached Cotengra paths, ALS cap10/tol1e-8. Each warm entry is a median of seven
synchronized complete pair updates, including environment and output work.
The final GPU measurement ran after the test processes finished, on a shared
workstation. No running simulation was modified.

| Bond | Quimb + fidelity | Torch + fidelity | Torch AOT + fidelity | Eager speed ratio |
| --- | ---: | ---: | ---: | ---: |
| (3,1)–(3,2) | 107.8 ms | 63.5 ms | 61.0 ms | 1.70× |
| (2,2)–(3,2) | 103.6 ms | 63.3 ms | 62.2 ms | 1.64× |
| (1,1)–(1,2) | 125.3 ms | 67.0 ms | 68.2 ms | 1.87× |

Both solvers used 4/4/5 sweeps. All 8/8/10 local solves passed the Cholesky
certificate; normalized local errors agreed within 9e-17 absolute. The local
fidelities were approximately 0.99999497345, 0.99999562923 and 0.99999772087.
Independent CPU/CUDA tests verify the spectral fallback separately.

Without fidelity, eager medians were 64.5/61.8/71.6 ms: differences are within
the observed timing scatter, so this experiment does not resolve a precise
measurement overhead. It is small relative to the full update here.
Earlier CPU samples with concurrent test activity showed 1.11–1.30× speed
ratios; these are less controlled and are not a CPU throughput claim.

AOT uses the same optional internal forward/backward capture interface as
prepared SU. The first GPU capture cost about 2.48 s; warm capture did not
consistently improve on eager. Eager therefore remains recommended. Cache
charts are bounded and keyed by shape, layout, dtype, device, autograd /
inference mode and autocast state/dtype. A direct Dynamo prototype hit its shared recompile limit
across shape/device/gradient variants and was replaced; no global Torch limit
was raised. Inductor remains excluded.

The second review reproduced an AOT backward dtype error when a cached
full-precision graph was reused under CPU bfloat16 autocast. A forward-only
check missed it. Adding autocast state/dtype to the chart key fixes the
reproduction; new CPU bfloat16 and CUDA float16 regressions check forward
and gradients while toggling autocast off/on/off. This does not change the
eager solver or establish accuracy of mixed-precision ALS trajectories.

These are isolated pair replays, not a new T=6 trajectory, χ convergence
study, or whole-circuit speed claim. Raw scripts/logs and JSON measurements
are under `/tmp/pepsy-cluster-speed-20261105/`; final GPU data is
`native-cuda-final.json`.

## Upstream audit

The post-publication review found that unconditionally reading an environment's
`.data` silently detached raw Torch environments and exposed a memoryview for
raw NumPy environments. Unwrap labeled tensors only, retain native arrays,
and build a labeled view for Quimb ALS when needed. Eager/AOT CPU/CUDA checks
now compare raw versus labeled environment derivatives and finite differences;
raw NumPy fidelity checks cover Quimb, QR and normal solves. The existing
benchmark supplies labeled environments and avoids the original detach case.

Reused this active task's unchanged installed versions: Quimb
`1.15.1.dev101+g1b64353d9`, Autoray `0.11.1.dev14+g014a3f69a`, Cotengra
`0.8.3`, Symmray `0.4.1.dev25+gdc377f033`, Torch `2.14.1+cu126`.
Inspected installed local ALS dispatch, `cholesky_ex`, triangular solve,
Cholesky solve, eigensolve, and existing prepared-SU AOT capture.

- **Adopt:** native Torch contractions and guarded public linear algebra;
  no installed dependency or global registration changed.
- **Compatibility shim:** optional version-sensitive AOT capture, matching
  the existing prepared-SU interface and checked against eager gradients.
- **Defer:** Inductor, whole-network compilation, independent-update batching,
  and reduced initial-SVD restructuring.

Consulted the official [Torch compilation documentation](https://docs.pytorch.org/docs/stable/torch.compiler.html),
[Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray), and
[Cotengra changelog](https://cotengra.readthedocs.io/en/latest/changelog.html).
The Symmray array documentation was unavailable; installed source and the
unchanged native adapter supplied the local capability evidence.

Validation and publication state are recorded in the
[handoff](../../../history/2026-11-05-torch-cluster-als.md).
