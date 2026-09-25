# TreeOptimizer backend and synchronization audit

Audit date: 2026-09-23. Scope: ordinary tree replay, controls, norm bookkeeping,
and random FIT initialization. No compression algorithms, cutoff defaults,
canonical-region proofs, or native fermionic grading rules are changed.

## Dependency audit

Installed versions: Autoray `0.11.1.dev3+g1b476b305`, Quimb
`1.15.1.dev55+gd0591eb70`, Cotengra `0.8.3.dev7+g1d7fd333f`, Symmray
`0.3.2.dev8+g6c6dd34b5`. Inspected the
[Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra docs](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray repository](https://github.com/jcmgray/symmray). The requested
Symmray `abelian_arrays.html` documentation URL was unavailable; installed
native APIs and regression tests were inspected instead.

Actual installed dispatch/signature probes covered `eye`, `allclose`, `sqrt`,
`minimum`, `logical_and`, `isfinite`, `isnan`, `random.default_rng`, and
`random.array`. `random.default_rng(seed=None, **kwargs)` accepts a seed,
whereas `random.array` exposes `shape`, `dist`, `loc`, `scale`, `dtype`,
`device`, `rng`, and `backend`. Passing a NumPy generator through Torch's
random adapter incorrectly reaches `manual_seed`; instantiate the generator
through Autoray on the selected backend. `eye(..., like=array)` retains
device/dtype. Torch `arange(dtype="int64", like=array)` does not share that
portable contract, so Pauli branch vectors use rows of a backend identity.

- **Adopt:** public Autoray array algebra, backend random generators, and
  Quimb tensor splitting with arrays retained on their original backend.
- **Compatibility shim:** retain the existing NumPy random fallback only
  when the installed Autoray lacks the public random entry points.
- **Defer:** changing native QR numerical safety, FIT convergence/report
  semantics, or upstream SVD rank decisions. These need separate numerical
  validation, not removal as incidental diagnostics.
- **Prototype:** none; no new compression algorithm is selected.

## Findings and changes

1. Cold `SubTreeMPO.from_gate` construction downloaded gate data and split
   factors through NumPy. Dense extraction, reshaping, contractions, exterior
   identities, and factor installation now preserve the backend. Explicit
   `TreeMPO.from_dense` size validation uses Autoray metadata.
2. Random FIT initialization supplied a NumPy generator to accelerator
   backends. It now obtains the matching backend generator; seeded repeat
   runs are covered on Torch and JAX.
3. Update norm bookkeeping materialized Python floats before and after every
   tracked gate. Torch/JAX/CuPy norm, log-fidelity, and cumulative loss scalars
   now stay detached on their backend. Public diagnostic getters materialize
   independent host records. Existing zero, NaN, infinity, complete-loss, and
   extracted-exponent semantics are preserved.
4. `profile=False` still read both update clocks. Those reads are now skipped,
   and untimed update envelopes report `elapsed_seconds=None`. Tree MPI shot
   diagnostics default to false, using the existing clock-free MPI route.
5. One-site unitarity certification downloaded the full matrix. Its algebra
   now runs on the backend with only the final Boolean read on the host.
   Certification remains necessary to preserve canonical-region proofs.
6. Fixed control tensors were rebuilt/transferred repeatedly. They are cached
   per backend/device/dtype and returned as owned copies. Projectors and
   Pauli-sum tensors are constructed with backend operations; a public
   `like=` argument exposes that construction route. Compatible payload
   conversion returns before scanning the full tree's backend contract.

## Remaining explicit host boundaries

- Public norm, dense readout, diagnostics, and progress display.
- Nonzero `tn.exponent` bookkeeping uses host doubles to preserve range on
  JAX without x64 and Metal. Non-unitary normalization already reads scalars.
  An operator may introduce or cancel that exponent during an update; mixed
  host/device norm pairs also use host doubles so neither direction converts
  a large or tiny represented scale to a float32 device scalar.
- FIT target/local norm reports and convergence decisions, including some
  final scalar reporting when `fit_rtol=None`.
- Measurement probabilities, random branch selection, and one-site unitarity
  decisions; native complex64 QR finite/fallback checks protect stability.
- Upstream rank selection in operator SVD preparation and truncation. Warm
  replay can reuse prepared factors, but cold construction is not claimed
  to be entirely free of scalar synchronization.
- Explicit one-/two-site combined-Hamiltonian automata still use host
  construction. They are outside ordinary compact gate replay. Higher-order
  dense direct-sum assembly now preserves the term backend (see follow-up below).
  Default state/random operator constructors remain NumPy unless explicitly
  converted through the public backend setup API.

Optional `fit_finite_check`, profile, full overlap, spectrum,
and bond diagnostics remain off by default. Required norm/infidelity
bookkeeping is not removed. This audit does not claim all TreeOptimizer modes
are free of synchronization or that more backend scalar kernels always run
faster on small trees.

## Validation

`tests/test_tree_gpu_backend.py` checks Torch/JAX numerical agreement,
unchanged device/dtype, cold operator construction without matrix downloads,
seeded random FIT, native U1U1 Torch replay, cached control replay, zero and
nonfinite norm policies, explicit exponent range, and ledger copy isolation.
The five direct compression routes (`direct`, `dm`, `src`, `sdc`, `zipup`)
also run after factor preparation with Torch scalar reads, NumPy downloads,
and profiling clocks forbidden.

The tree domain suite covers canonical regions, operators, replay, FIT,
path routing, and compression. MPI, trajectories, Quimb compatibility,
public API, and package-layout checks cover integration boundaries.
Hardware here provides Torch 2.9.1 and JAX 0.8.2 CPU execution only; CUDA and
CuPy cases skip when unavailable. No GPU timing or speedup is claimed.

Validation results: the combined tree/integration selection passed 1,048
tests (20 skipped), and the default smoke suite passed 130 tests. After the
final `like=` dtype adjustment, the backend/operator selection passed 87
tests (15 skipped). Repository Ruff checks and `git diff --check` passed.

The follow-up review caught and fixed mixed host/device norm transitions when
an operator introduces or cancels an extracted exponent. Tests cover both
directions with norms of `1e-100` and `1e100`, including actual replay that
cancels the state exponent. The updated combined selection passed 1,057
tests (28 skipped), the smoke suite passed 130 tests, and Ruff passed.

## Autoray namespace and direct-sum follow-up (2026-09-23)

Rechecked the upstream sources above and
[Autoray automatic dispatch](https://autoray.readthedocs.io/en/latest/automatic_dispatch.html).
Installed versions are unchanged. Probed `get_namespace(like=None,
device=None, dtype=None, submodule=None)`, NumPy/Torch/JAX `pad`, `zeros`,
`concatenate`, and `stop_gradient` dispatch, and inspected Quimb's public
`Tensor.direct_product(other, sum_inds=(), inplace=False)` implementation.
Its array operation uses Autoray padding and addition, including JAX's
immutable-array route; no Pepsy backend-specific indexed assignment is needed.

- **Adopt:** `Tensor.direct_product_` for higher-order dense term TTNO sums,
  retaining physical sum indices, virtual block order, tags, and explicit
  dtype semantics. Backend/device mismatches now raise instead of silently
  downloading mixed inputs. No native Symmray direct-sum code changes.
- **Adopt:** cached `get_namespace(like=...)` dispatch in ordinary MPS and
  tree scalar norm-ledger helpers. Re-select dispatch when a host contribution
  joins an existing device ledger. Keep precision, detachment, exponent,
  invalid-value, and diagnostic-default policies unchanged.
- **Compatibility shim:** a private shared backend helper falls back to
  `ar.numpy` when `get_namespace` is unavailable; regression coverage removes
  the factory and exercises both ledgers, including a host/device transition.
- **Defer:** compact one-/two-site Hamiltonian automata, whole-optimizer JIT,
  FIT convergence policy, and upstream rank decisions. Namespace caching
  neither fuses scalar kernels nor independently removes device synchronization.

Tests compare a three-term backend sum with independently embedded dense
operators before and after compression, guard tensor downloads, and check
backend/device/dtype preservation. Torch and JAX CPU execution is available;
CUDA/CuPy variants skip here. No GPU speedup has been measured.

Follow-up validation: 70 backend tests passed (43 skipped), 754 tree/operator,
compatibility, stabilizer-backend, and MPS-audit tests passed (27 skipped),
88 MPS norm/fidelity/finite/timing tests passed, and 130 smoke tests passed.
Repository Ruff and `git diff --check` passed. The full repository suite was
not repeated; its previously reproduced unrelated dependency failures remain
documented in `stabilizer_gpu_backend_audit.md`.
