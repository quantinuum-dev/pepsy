# 2026-09-28 — Complex-to-real correctness review

Baseline: `develop` at `bc77796`. This resolves the five cast warnings
identified in the [maintenance triage](2026-09-28-maintenance-triage.md).

## Findings and changes

| Cause | Evidence | Fix |
| --- | --- | --- |
| Ordinary Haar PEPS cast to real | Four explicit `(theta, phi) = (pi/2, pi/2)` sites produced squared norm `0.0625`, instead of `1`. MPS and TTN constructors had the same cast. | All ordinary Haar constructors require complex storage. Seeded complex values and native fermion branches retain their existing construction. Real product states use `ps_to_*`. |
| Stabilizer cached identity cast | Pauli I had complex storage with exactly zero imaginary entries. | Inspect small cached constants; remove exactly zero imaginary components before real conversion, and reject nonzero ones. |
| Stabilizer Y combination cast | `i * coef` was mathematically real for a Y rotation, but retained a complex scalar type during backend assembly. | Validate scalar coefficients, then assemble with real coefficients and cached `-iY`. Local X/Z rotations or Y projectors requiring complex amplitudes now raise for a real coefficient dtype. |
| Two Quimb BP suppression casts | The failing regression supplied `-0.07395391444512747 + 1.3877787807814457e-17j` as a D2 norm weight. Quimb uses real `math.exp`. | Validate finite real D2 edge-loop norm weights, accepting imaginary residue up to `64 * eps * max(1, abs(real_weight))`, then pass native real scalars. Larger residues raise; explicit unsuppressed expansion is retained. |

The Haar fix is a documented input-contract tightening: even explicit ordinary
`haar_params` require complex storage. It follows the existing dense
`haar_random_state` rule and avoids inventing a different random ensemble for
real dtypes. The former real SWAP fixture now constructs a valid real product
state directly. No random samples were changed for valid complex inputs.

BP validation applies only to the norm weights consumed by the D2 edge-loop
suppression solver. It leaves observable numerators, cached weights, returned
complex expectations, and native graded tensor contractions intact. This is
not a general rule to discard imaginary components from BP. Quimb's scalar
free-energy solver already makes host scalar conversions; this change does
not establish an autodiff contract for that solver. A Torch scalar probe
still reports Quimb's warning when a gradient-carrying value reaches
`math.exp`.

## Upstream audit

Installed versions: NumPy 2.5.2, Torch 2.9.1, JAX 0.8.2,
Quimb `1.15.1.dev66+ge927f06e1`, Autoray `0.11.1.dev3+g1b476b305`,
Cotengra `0.8.3.dev7+g1d7fd333f`, Symmray `0.4.1.dev8+gc45f91457`.
Inspected installed `process_loop_series_expansion_weights` and Pepsy's
existing signature adapter. Its `return_all=True` suppression factors use
real exponentials. Native `real` and `astype` remain Autoray operations.

Reviewed the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray source](https://github.com/jcmgray/symmray).
The Symmray array-documentation page again failed to load; installed source
and the repository provided the available evidence.

Classification: **adopt** explicit input validation and native real arithmetic
at Pepsy-owned boundaries. No new upstream shim, vendored solver, optional
dependency requirement, warning suppression, or default approximation.

## Validation

- Focused cast regressions: **14 passed, no warnings**.
- Constructors, gates, tree state, BP, and stabilizer domain suites:
  **751 passed, 24 skipped, 22 unrelated warnings in 19.91s**.
- Real NumPy stabilizer Y rotations match dense product references with
  `chi=None` and `chi=2`; real X/Z projector assembly also passes without
  warnings. The Torch regression verifies Y phases, dtype, rejected X
  rotations without coefficient-state mutation, and rejected Y projectors.
- BP tests cover both complex precisions, roundoff acceptance, significant
  imaginary/nonfinite rejection, preserved input weights, and explicit
  suppression disablement. Native Symmray BP domain tests pass.
- Constructor tests cover MPS/PEPS/TTN complex phases, unit norm, dtype,
  per-site seeded reproducibility, and real/integer rejection.
- The constructor documentation example and its norms pass.
- All three ordinary constructor paths also pass an isolated probe with
  Torch, JAX, NetKet, Flax, Stim, Symmray, CuPy, Nevergrad, NLOpt, and MPI
  imports/discovery blocked. The new smoke cases need only the core profile.
- Ruff and focused CI mypy pass.
- Default smoke: **92 passed, 2 compatibility warnings in 21.29s**.
  Three small constructor cases were added to the existing smoke module;
  test selection and optional dependency profiles are unchanged.
- Full suite: **5,175 passed, 129 skipped, 682 warnings in 516.88s**.
  All five targeted complex-to-real warnings are absent. The warning summary
  still includes 601 upstream Quimb/NumPy shape-assignment deprecations,
  numerical/compatibility notices, two Loky worker-stop notices, and one
  Torch gradient-to-scalar conversion warning. These are not hidden or
  claimed resolved by the cast fixes.
- All 19 local Markdown links in changed documentation resolve, and
  `git diff --check` passes.
