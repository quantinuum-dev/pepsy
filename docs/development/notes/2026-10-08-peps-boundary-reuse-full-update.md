# PEPS boundary reuse and reduced two-site updates

Implemented in the Pepsy working tree on develop, baseline 0ba2a05. This
extends the October 7 gate-by-gate review; earlier notes describing discarded
environments are superseded for the supported dense Torch path. No launcher
changes, commits, or pushes in this implementation session.

## Design and upstream reuse

- `PepsOptimizer(mode="sweep", update_style="row-column" | "row" | "column" |
  "two-site")` selects the fitting scope. Two-site is optional; the historical
  default remains row/column. `mode="full-update"` is a two-site alias.
- `k_2q_batch=1` selects individual two-site gates. Two-site ALS also maps auto
  to one gate and rejects larger explicit batches.
- `gate_order="column" | "row"` sorts contiguous fixed diagonal two-qubit
  gates into strip traversal. Every unsupported/non-diagonal/single-site gate
  is a barrier. Default input order is unchanged. This preserves the exact
  circuit, not equivalence of approximations taken in different orders.
- Adopt existing Pepsy `prepare_reduced_bond_pair(..., run_bp=False)`, PSD
  projection, and `solve_reduced_als`. The new externally supplied reduced
  problem does not claim an exact or BP environment. Quimb public ALS is the
  automatic route, with the existing weighted-QR fallback. Pepsy's Cotengra
  builder supplies all new contraction policies.
- Follow the QR/LQ reduction, positive environment, and environment-gauge
  construction of [Lubasch et al., III B](https://arxiv.org/pdf/1405.3259).
  Full-update currently supports dense Torch complex64/complex128,
  nearest-neighbor coordinate pairs, and no differentiation. Native symmetry,
  JAX/CuPy full-update, and separate one-site ALS are deferred.
- Quimb/Autoray/Cotengra/Symmray upstream audit was reused from this active
  maintenance task. Installed versions inspected: Quimb
  1.15.1.dev75+g4112e304a, Autoray 0.11.1.dev9+g1291702f9, Cotengra
  0.8.3.dev7+g1d7fd333f. No installed libraries were modified.

## Cache correctness

Owned lazy boundary maps retain one validated entry per cut. Source array
identities/version counters, indices, shapes, tags, predecessor boundaries,
and compression policy determine validity. Cache entries retain references
to prevent recycled identities. In-place mutations and altered boundary
arrays invalidate their dependents. Both directions are covered.

Local topology changes refresh the owning network and replace incompatible
cut guesses, preserving unaffected cuts. Independent fresh convergence checks
remain fresh. Calibration overlap boundaries now use the same bra/ket
orientation as fitting; public scalar diagnostics preserve their prior
overlap convention. Norm and overlap both transfer into sweep initialization.
No final scalar contraction is cached. Network exponents are reapplied.
Supported normalization stores magnitude in the exponent, preserving unchanged
site arrays. Explicit custom normalization/balancing retains the ordinary path.

ALS returns the best of its candidate and the two SVD baselines in the same
positive environment. Diagnostics distinguish local positive-environment
fidelity from true global fidelity. Quimb does not expose actual iteration
counts through this adapter, so count/convergence are not invented.

## Validation and measured limits

- Final focused checks: 91 passed, including both solver routes, exact-vector
  references, both complex dtypes, gate-order barriers, axis routing, local
  bond growth, normalization, and norm/overlap cache handoff.
- Broader PEPS/API/layout selection: 338 passed; one existing failure:
  `test_package_version_matches_installed_distribution` (installed metadata
  0.4.0 versus checkout 0.5.0). Subsequent small traversal record/guard changes
  are covered by the final focused checks.
- Full repository suite was interrupted after 403 passed and 23 skipped;
  do not describe the full suite as passing. Ruff is unavailable in this environment.
  `git diff --check` and compilation passed.
- CUDA:0, 3×3 D=2 complex128, two gates: uncached sweep 12.00 s, cached sweep
  10.99 s, two-site ALS 4.88 s. Exact fidelities were respectively
  0.977720828905484, 0.977720828905495, and 0.967319641616237; output norms
  were one within roundoff. This single small probe is not a production
  speedup claim; two-site and row/column optimize different freedoms.
- CUDA:0, random 5×6 D=4 complex128, two ordered gates: 26.64 s, including
  25.15 s chi calibration, 0.114 s reduced environments, 0.201 s ALS. Both
  completed with finite tensors and D=4. Local positive-environment estimates
  were 0.990674 and 0.990712. At max chi=64 and four boundary iterations,
  neither calibration converged; warnings and `converged=False` were retained.
  Thus this demonstrates execution, not certified 5×6 accuracy.

Production runs never imported these edits. On the user's later request the
remaining CUDA:1 PEPS launcher/worker were stopped; saved output was preserved.
No PEPS production run is active and no restart is authorized yet.

## Follow-up review

The user requested another review and test pass. Two regressions were first
reproduced, then fixed:

- Custom output-normalization settings could carry the initial chi into the
  full-update fallback, overriding the cap selected by calibration. The
  normalization call now carries the selected cap in both override channels.
- Zero-norm probes could leave missing or stale handles in the final cache
  handoff. Failed probes now remove their handle records; only successful
  records at the selected cap can be handed to the fitter. The convergence
  failure remains visible rather than becoming an internal KeyError.

Expanded focused checks: 99 passed. New checks include asymmetric gates in
both site orders and cached versus fresh three-gate updates that turn between
row and column environments, with both direct and DMRG boundary compression.
The bounded 3×3 D2 CUDA retest again produced normalized finite outputs.
Cached and uncached sweep fidelities differed by about 4e-14; two-site exact
fidelity remained 0.967319641616237. These tests do not establish accuracy for
the earlier unconverged 5×6 cap64 example. Ruff and Pyflakes are unavailable;
no packages were installed in the shared environment.

The broader PEPS/API/layout selection was rerun after these fixes: 338 passed,
with the same single installed-version metadata mismatch (0.4.0 versus 0.5.0).
Compilation and `git diff --check` passed. The whole repository suite was not
repeated in this follow-up review. Production PEPS remains stopped.
