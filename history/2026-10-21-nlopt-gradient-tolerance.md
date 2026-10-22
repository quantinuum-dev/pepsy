# NLopt explicit gradient stopping threshold

## Initial implementation

The authorized T=2 Gaugy campaign requests ftol=1e-12 and gtol=1e-10 for
SciPy L-BFGS-B followed by NLopt LD_VAR2. NLopt's historical `gtol` alias
means parameter-step tolerance. The new optional `gradient_tol` checks the
maximum absolute unmodified derivative of a finite valid objective trial;
it defaults to disabled. A deliberate force-stop reports `gradient_tol`
without a numerical-failure warning. Existing alias semantics are unchanged.

Working tree on `develop`, baseline `2aacd8749a5398621ec64d040d1c97a2fbcd1f67`.
Changes are uncommitted and unpublished. The downstream runner explicitly
requests this option and records it alongside NLopt's native tolerances.

Validation: **196 passed** across `test_gradient_solver.py`,
`test_gradient_solver_jax.py`, and `test_gradient_solver_regressions.py` on
CPU. New Torch/JAX regressions disable function/step stops and verify actual
gradient stopping. `python -m ruff check src tests` passed. A downstream
CUDA/JAX-JIT BCH4 p=3 LD_VAR2 smoke fit completed with finite outputs; this
does not establish convergence or finite-cluster accuracy. Full suite not run.

## Review and local commit

User-authorized scope: carefully review and commit the pending Pepsy changes.
Review baseline: `develop` at `3921bd3`. The implementation and this completed
handoff are committed together locally; no push was performed. The earlier
uncommitted status above describes the initial implementation.

The review reproduced a returned-state bug on both Torch and JAX:
`restore_best=False` returned the initial point after an intentional gradient
stop. New regressions failed before the fix. The runner now retains the
qualifying trial for this case while preserving default best-state restoration.
Only the expected NLopt `ForcedStop` suppresses the numerical-failure warning.
The API guide now explains raw packed-coordinate derivatives, finite thresholds,
minimum evaluations, constraints, and returned-state selection.

Fresh validation with NLopt 2.11.0, Torch 2.14.1+cu126 and JAX 0.11.2,
using `JAX_PLATFORMS=cpu`:

- `python -m pytest -q -o addopts='' tests/test_gradient_solver.py tests/test_gradient_solver_jax.py tests/test_gradient_solver_regressions.py tests/test_public_api.py tests/test_package_layout.py`:
  **264 passed**, with two existing compatibility-alias deprecation warnings.
  Added checks cover returned states with both restoration settings, input
  preservation, evaluation counts, invalid thresholds and evaluations, clipping,
  minimum evaluations, disabled stopping, and maximum-coordinate semantics.
- `python -m ruff check src tests` and `git diff --check`: passed.
- Local links in the changed Markdown files: checked.

Full-suite and fresh GPU validation were not run for this focused solver change.
