# 2026-10-09 — Quimb pair autodiff and D/chi scaling audit

Production integration followed this audit; see the
[cached scalar TN implementation note](2026-10-09-cached-pair-tn.md).

This is an investigation, not an implementation change. It corrects the
incomplete upstream assessment in the [pair L-BFGS note](2026-10-09-pair-lbfgs.md).
The existing dense full-tensor implementation is tested as a small-system
reference; its tests do not establish an efficient large-D design.

## Existing Quimb support

Inspected installed Quimb `1.15.1.dev90+g6a3906cbe` and official documentation:

- [`FullUpdate` and `gate_full_update_autodiff_fidelity`](https://quimb.readthedocs.io/en/latest/autoapi/quimb/tensor/tn2d/tebd/index.html):
  `fit_strategy="autodiff-fidelity"`, `autodiff_optimizer="L-BFGS-B"`, and
  `autodiff_backend="torch"` or another supported AD backend. The routine
  leaves `env` as a TN and contracts norm and overlap scalars using the
  supplied contraction optimizer; it optimizes all selected plaquette tensors.
- [`TNOptimizer`](https://quimb.readthedocs.io/en/latest/autoapi/quimb/tensor/optimize/index.html):
  public selective tensor optimization, fixed loss constants, custom loss,
  autodiff, optional Torch tracing, and SciPy L-BFGS-B.
- [`tensor_network_fit_autodiff`](https://quimb.readthedocs.io/en/latest/autoapi/quimb/tensor/fitting/index.html):
  generic fitting through norm/overlap contractions. Explicitly select
  `distance_method="overlap"` to avoid its small-output dense shortcut.
  A double-layer weighted environment needs a custom loss or suitable
  factorization; this generic helper does not accept an arbitrary norm metric.
- Cotengra's [expression interface](https://cotengra.readthedocs.io/en/latest/autoapi/cotengra/interface/index.html)
  caches paths/expressions; fixed-only intermediates can also be precomputed.

The dedicated FU routine uses `-2*abs(overlap) + abs(norm)` (target constant
omitted), rather than Pepsy's target-normalized squared residual with real
cross terms. It does not PSD-project the full environment. Conditioning is
configurable, independent of autodiff. `TNOptimizer.get_tn_opt` returns NumPy
variables in this installed version, so native output restoration is needed.
Its SciPy optimization also exchanges parameters and gradients with the host;
using Quimb alone does not eliminate those transfers.

## Actual local checks

Used the existing activated Python 3.12 environment, one CPU thread,
complex128, deterministic seed 24, a 3x4 PEPS and two adjacent bulk tensors.
Frozen Pepsy boundary strips supplied an environment TN. A custom public
`TNOptimizer` loss evaluated `(real(norm)-2*real(overlap)+target_norm)/target_norm`.
Compared its value and packed complex gradients with the equivalent *raw*
dense environment, not a differently PSD-projected objective.

| D | requested boundary chi | dense kernel value/grad (ms) | eager TNOptimizer (ms) | traced TNOptimizer (ms) | full dense eigh (ms) |
|---|---|---|---|---|---|
| 2 | 4 | 0.285 | 2.38 | 0.804 | 0.833 |
| 2 | 8 | 0.278 | 2.32 | 0.806 | 0.664 |
| 3 | 9 | 1.21 | 13.4 | 6.23 | 170 |
| 3 | 18 | 1.16 | 11.3 | 6.17 | 168 |

Each steady-state evaluation timing is a mean of five calls after warmup.
Tracing/setup cost was 65–113 ms; dense matrix construction cost was 0.49–12 ms.
The dense column is the native analytic kernel, whereas the TN columns
include optimizer parameter/gradient packing; these are microchecks, not
equal-budget end-to-end optimizer benchmarks. All four TN fits lowered loss.
Largest value discrepancy was 1.69e-15; largest gradient discrepancy 5.35e-16.
Traced and eager value/gradient checks agreed within 1e-10 as well.

The tiny strips saturate their required boundary ranks, so increasing the
requested chi in this table does not establish chi scaling. No GPU speed,
large-D accuracy, or general timing winner is claimed.

The dedicated Quimb FU wrapper was also exercised. Its ordinary complex-Torch
simple-guess initialization failed at Autoray's NumPy conversion of a Torch
conjugate view. It ran after materializing the bra conjugate and using
`init_simple_guess=False`. This is a prototype-only workaround; no installed
library was edited. The custom `TNOptimizer` construction worked directly.

## Contraction topology and scaling

Analyzed a standard six-tensor periodic environment around a two-site square
patch. Each environment tensor has shape `(chi, chi, D, D)`. Four active
ket/bra site tensors have virtual dimensions D and physical dimension d=2.
Cotengra's deterministic `optimal` path search was used on dimensions only:
no large dense matrices were allocated. This is a specified topology and
contraction model, not a universal lower bound for arbitrary environments.

For the closed full-pair norm, all probed `(D,chi)` pairs gave a path costing

`4*chi**3*D**4 + 4*d*chi**2*D**6 + d*D**4`

scalar multiplication units, with largest intermediate `d*chi**2*D**4`.
Two of the four `chi**3*D**4` contractions are environment-only and can be
cached. A reverse pass has the same leading powers for this fixed graph,
with additional work and saved intermediates. These sizes exclude total
autodiff memory, constant tensors, optimizer history and workspace.

| D | chi | full dense norm alone, complex128 | largest forward TN intermediate |
|---|---|---|---|
| 4 | 8 | 256 MiB | 0.5 MiB |
| 4 | 16 | 256 MiB | 2 MiB |
| 4 | 32 | 256 MiB | 8 MiB |
| 6 | 36 | 32.4 GiB | 51.3 MiB |
| 8 | 64 | 1 TiB | 512 MiB |

For fixed physical dimension:

- Full dense norm: dimension D^6, D^12 entries, generic eigendecomposition
  O(D^18), and dense metric application O(D^12) per objective/gradient call.
  Environment assembly still depends on chi: the observed open-ring path
  contains a chi^2 D^12 term, plus chi^3 D^6 and chi^3 D^4 terms.
- Full TN contraction: O(chi^3 D^4 + chi^2 D^6) per norm contraction for
  the specified path. If chi is proportional to D^2, this becomes O(D^10)
  work with O(D^8) largest intermediates, before AD overhead.
- Reduced norm: reduced exterior rank r <= dD; matrix storage r^4 and
  eigendecomposition r^6. For fixed d these are O(D^4) and O(D^6).
  Construction from the boundary still has chi-dependent contractions;
  reduction does not eliminate their time or temporary memory. Once built,
  repeated local solves avoid contracting the chi-sized environment again.
- Trainable complex parameter counts are 2dD^4 (full) versus at most
  2d^2D^2 (reduced). L-BFGS history adds storage proportional to this count
  times the chosen history length.

Boundary construction and refresh costs are shared setup costs and must be
included in end-to-end comparisons. Repeated local iterations amortize dense
or reduced metric construction. Larger chi increases work in the retained TN,
so comparisons must specify D, actual boundary ranks, and evaluation count.

## Accuracy and recommendation

For the same fixed environment and objective, dense and TN evaluation are
equivalent up to contraction roundoff; autodiff adds no physical accuracy.
Full tensors enlarge the variational space, but a local optimizer is not
guaranteed to attain its best minimum. Boundary chi controls environment
accuracy independently of that choice. PSD clipping changes the approximate
metric; it can stabilize a solve without improving its approximation to the
true environment. No dense PSD projection is intrinsically required by
L-BFGS or autodiff. Exact environments are PSD by construction; truncated
double-layer environments need validation if used directly.

Keep reduced ALS as the default. For scalable full-tensor refinement, reuse
public Quimb `TNOptimizer` with a custom explicit loss, fixed Pepsy environment
TN, cached Cotengra paths, native output restoration and existing acceptance
checks. **Adopt:** public optimizer infrastructure. **Prototype:** the tested
TN objective. **Defer:** production integration, compatibility shim, GPU and
equal-budget performance/accuracy benchmarks. Keep the dense full-pair route
as a bounded small-D reference; do not present it as the scalable full solver.
