# 2026-09-26 — PEPS sampler rho repair and speed

- Scope: assess faster sampling and implement explicit Hermitian local
  proposals with optional positive spectral repair.
- Branch / baseline: `develop` / `a13031b`, with the earlier uncommitted
  [boundary-scaling fix](peps_sampler_boundary_scaling.md) preserved.
- Status: implemented in the working tree, not committed or published.
  No dependency, production-job, or sibling-repository changes.

## Numerical policy

Every conditional uses the Hermitian part `H = (rho + rho.H) / 2`, with scaling
before addition and normalization. This leaves `real(diag(rho))` unchanged in
exact arithmetic. It does not make a matrix positive semidefinite or recover
NaN/Inf entries.

The new `rho_positivity` option is explicit because spectral repair changes
the finite-cap proposal:

| Policy | Eigenvalue map for the Hermitian part |
| --- | --- |
| `None` (default) | No spectral repair; retain the diagonal roundoff policy |
| `"clip"` | `lambda -> max(lambda, 0)` |
| `"absolute"` | `lambda -> abs(lambda)` |

For `H = V diag(lambda) V.H`, absolute repair is the positive matrix
`sqrt(H.H @ H) = V diag(abs(lambda)) V.H`. Applying the polar absolute value
directly to a non-Hermitian raw rho generally gives a different matrix. The
implemented ordering is **Hermitize, then repair**, as described to the user.
An optional preference question was asked about enabling repair by default;
without a reply, the stated conservative default remains `None`.

Clipping gives the nearest PSD matrix to H in Frobenius norm before trace
normalization. Absolute repair reflects the negative spectrum instead of
removing it. References: [Higham, nearest PSD matrix](https://nhigham.com/2021/01/26/what-is-the-nearest-positive-semidefinite-matrix/)
and [Higham, polar decomposition](https://nhigham.com/2020/07/28/what-is-the-polar-decomposition/).

For qubits the implementation computes eigenvalues and diagonal spectral
projector weights directly, in batches. The small eigenvalue is obtained
from determinant/large-root rather than subtracting nearly equal mean and
radius. The minor projector weight uses the off-diagonal magnitude rather
than subtracting a fraction close to one. Repeated positive/negative spectra
retain the appropriate linear map and gradients. Larger physical dimensions
use native batched `eigh`.

Only `diag(V f(lambda) V.H)` is computed. The code does not form `rho.H @ rho`,
apply an elementwise square root, reconstruct a full positive matrix, or move
rho arrays to NumPy. Invalid groups are masked before a general eigensolver
and rejected by the common validation; successful grouped draws still perform
one explicit validation-scalar read per site. This avoids adding an extra
validation synchronization to the repair path.

All sampling/query routes use the same repaired proposal. Returned amplitudes
still come from the original private ket; log probabilities and weights include
the repair. Raw-rho diagnostics are retained. The new
`relative_positivity_correction` and `max_relative_positivity_correction`
measure the Frobenius correction relative to H. They report zero when repair
is disabled, which does not certify that H was PSD.

A truncated single-layer ket paired with its conjugate bra remains positive
by construction. Unconstrained approximation of double-layer future
environments can break Hermiticity/positivity. Local repair changes the
proposal; it does not establish convergence or restore arbitrary lost support.
Zero repaired trace and non-finite input remain errors.

## Bounded speed measurements

One CPU thread, 4×4 D=4 random PEPS, seed 101, χ=32, χ′=16, Quimb future
boundaries, greedy contraction, default row-cache budget zero. The NumPy state
is complex128; Torch CPU uses the same source tensor data cast to complex64.
Torch runs use `inference_mode`. Construction is excluded because future
boundaries are reused. Each grouped case warms up eight shots and measures
three 32-shot calls with seeds 800–802 in rotating order. Serial medians use
two 32-shot calls. The previous class is the saved source immediately before
this rho-policy task, already including the boundary-scaling fix.

| Mode | NumPy median seconds | Torch CPU median seconds |
| --- | ---: | ---: |
| Previous grouped sampler | 0.6083 | 0.7821 |
| Current Hermitian default, grouped | 0.6157 | 0.7906 |
| `clip`, grouped | 0.6083 | 0.7910 |
| `absolute`, grouped | 0.5735 | 0.7290 |
| Current Hermitian default, serial | 0.8553 | 1.2290 |

In this probe grouping gives approximately **1.39×** (NumPy) and **1.55×**
(Torch) throughput relative to serial draws. Repair cost is small compared
with the contractions for this case. The absolute-policy timings do not
establish an intrinsic speedup: run variability and changed proposals/prefixes
can affect timing. The workstation also runs production calculations, and
numerical validation ran concurrently on CPU. No GPU throughput claim is made.
Temporary detail: `/tmp/pepsy_rho_timing.py` and `/tmp/pepsy_rho_timing.log`.

### Practical choices and further work

- Reuse one sampler for an unchanged PEPS and use `sample_batch`; rebuild only
  with `refresh()` after source changes. Tune batch size against memory and
  effective samples per second.
- Use Torch `inference_mode` when gradients are unnecessary. The new rho repair
  itself stays on the source backend/device.
- Tune χ and χ′ using convergence and importance-weight quality, not only raw
  shots per second. Reduced caps can increase variance or lose support.
- The main remaining algorithmic speed opportunity is reusable, factored
  within-row suffix/prefix contractions. The latest earlier profile attributes
  about 33% to local rho contraction and 35% to conditioned-boundary updates;
  future caching is already implemented. A large dense row-transfer cache is
  not a general substitute: default budget zero avoids the known memory issue.
- Native shape-bucketed boundary contractions and QR/SVD are a separate future
  implementation. The current prefix loop remains Python/Quimb per group.

The latter two items are proposals, not changes implemented in this task.

## Upstream evidence

Installed versions were rechecked: Quimb 1.15.1.dev66+ge927f06e1;
Autoray 0.11.1.dev3+g1b476b305; Cotengra 0.8.3.dev7+g1d7fd333f;
Cotengrust 0.2.1; Symmray 0.4.1.dev7+g83fb22865;
Torch 2.6.0+cu124; JAX 0.10.2. Native batched complex64 `eigh` dispatch was
probed on NumPy, Torch CPU, and JAX CPU. The installed Torch `eigh` docstring
states that CUDA inputs synchronize with CPU; the qubit formula avoids that
call. It does not remove unrelated upstream synchronization.

Official [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra docs](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray repository](https://github.com/jcmgray/symmray) were checked.
The Symmray abelian-array page and Torch 2.6 online `eigh` page failed in the
browser; installed implementation/docstrings supplied capability evidence.
No installed libraries or decomposition registries were edited.

- **Adopt:** public Autoray namespace operations and native batched `eigh`,
  plus the explicitly selected PSD maps and validated qubit formula.
- **Defer:** factored row caches, compiled/native batched boundary compression,
  and GPU throughput measurements on an idle device.

## Validation

Focused checks cover non-Hermitian and indefinite 2×2/3×3 matrices, both complex
precisions, scales from 1e-24 to 1e24, all three CPU backends, representable rare
weights, degenerate-spectrum gradients, NaN/Inf/zero rejection, serial/grouped/
row-cache log-proposal consistency, exact amplitudes, importance weights, and
one explicit scalar validation read per site.

The complete sampler suite passes: **225 tests** in 282.21 seconds, with
36 existing Quimb `mode`/`method` warnings. Ruff and diff checks pass.
API/layout has 58 passes and the known installed-version mismatch; smoke has
162 passes, one skip, and that same mismatch (runtime/installed 0.4.0 versus
checkout metadata 0.5.0). Details are recorded in the
[session handoff](https://github.com/quantinuum-dev/pepsy/blob/develop/history/2026-09-26-peps-sampler-rho-repair.md).
