# 2026-10-06 — Gradient optimizer and integration review

Reviewed `develop` at `2abb1e8`, with the existing unrelated working edits
preserved. Scope: correctness, backend handling, integration, and opportunities
to use Autoray. This is a report, not an implemented solver refactor.

## Confirmed correctness findings

### 1. P1 — Native JAX complex gradients have the wrong update direction

Location: `src/pepsy/solvers/gradient.py:1922`.
`jax.value_and_grad` is passed directly to Optax. For real objectives of
complex parameters, JAX's convention requires conjugating the gradient before
a descent update. The host JAX SciPy/NLopt adapter avoids this problem by
packing real and imaginary coordinates separately.

Reproduction: minimize `sum(abs(x)**2)` from `x=[1j]`, one update,
`lr=.1`, `restore_best=False`. SGD returns `1.2j`, loss **1.44**, instead of
`0.8j`, loss **0.64**. Adam, AdamW, and RMSprop also increase this objective
in the same probe. Default best-state restoration hides the wrong update by
returning the starting state. This affects native `jax-*` solvers with complex
parameters, not all JAX solves or real gate-angle parameters.

Proposed fix: conjugate complex JAX gradient leaves before Optax or use a
consistent real-coordinate representation. Validate real/complex mixtures,
all native algorithms, and finite-difference directional derivatives.
See the official [JAX complex differentiation convention](https://docs.jax.dev/en/latest/complex-differentiation.html).

### 2. P1 — Torch and finite-difference Adam miss the final evaluated state

Locations: `gradient.py:1013`, `finite_difference.py:343`.
Torch updates parameters after measuring the loss, but never scores the last
update. With restoration enabled, a one-step solve always restores the
initial point. Without restoration, `final_loss` describes the previous
point. Finite-difference Adam repeats this error. The recently corrected
native JAX bookkeeping has not been carried across these implementations.

Reproduction: minimize `sum(x**2)` from `x=[1.]` with Adam, `lr=.1`, one step.
Both Torch Adam and FD Adam return `x=1` with default restoration. With
`restore_best=False`, both return approximately `x=.9` but report loss **1**
instead of **.81**. qMERA directly installs `result.params`, so one-step
Torch energy solves can make no progress. Sweep re-evaluation protects its
reported applied loss but cannot recover a discarded final update.

Proposed fix: evaluate the terminal update, include it in best selection,
and account for the extra evaluation. Keep loss/state pairing consistent
across all solver families and error exits.

### 3. P2 — Invalid SciPy objectives can be reported as converged

Locations: `gradient.py:1295`, `gradient.py:1353`, `gradient.py:1380`.
Objective exceptions/nonfinite values become a large penalty and zero
gradient. SciPy can immediately declare projected-gradient convergence before
Pepsy's callback checks `bad_max`. The public result then reports convergence
despite having no valid evaluation.

Reproduction on both Torch and JAX: an everywhere-NaN scalar objective returns
`best_loss=inf`, `final_loss=nan`, and
`CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL`, after one tracked
evaluation and without a warning. The history fallback is another evaluation
not included in that count.

Proposed fix: require a finite valid evaluation before reporting convergence;
retain the underlying failure and distinguish invalid objective/gradient from
ordinary solver convergence. Do not indiscriminately hide structural loss
errors under a numerical penalty.

### 4. P2 — SciPy budgets are inconsistent, including actual PEPS callers

Locations: `gradient.py:1167`, `gradient.py:1220`,
`src/pepsy/optimizers/sweep/optimizer.py:2080`.

- `options={'maxiter':1}` is consumed by common controls but replaced with
  the separate `n_steps` value. Rosenbrock with `n_steps=50` took **36**
  iterations and **44** evaluations despite the explicit one-iteration option.
- TNC receives unsupported `maxiter` and `maxls`; its actual limit is
  `maxfun`. With `n_steps=1`, Rosenbrock took **22** iterations and **76**
  evaluations, emitting an unknown-options warning.
- `PepsOptimizer(optimizer='scipy', optimizer_options={'maxeval':1})`
  forwards through SweepOptimizer, but the gradient runner discards maxeval
  and uses SweepOptimizer's default `n_steps=30`. A real 2x2 Torch PEPS gate
  run passed **maxiter=30** to every observed local SciPy solve. The examples
  CLI's maxeval flag feeds this same option, so its documented SciPy override
  does not enforce the requested budget.

Proposed fix: normalize explicit budgets once at the public boundary,
distinguish iteration limits from function-evaluation limits, and pass each
solver its supported options. Test the actual caller-to-SciPy arguments.
See [SciPy minimize's method-specific budget contract](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html).

### 5. P2 — The advertised trust-constr solver crashes in its callback

Location: `gradient.py:1328`.
Pepsy supplies `callback(xk)` for every method; installed SciPy calls
`callback(xk, result)` for trust-constr. A one-variable quadratic raises
`TypeError: callback() takes 1 positional argument but 2 were given` on both
Torch and JAX. Confirmed against installed SciPy's `_wrap_callback`.

Proposed fix: adapt the callback signature/result handling by method, including
early-stop semantics; test an actual trust-constr solve.

### 6. P2 — Native JAX silently discards requested constraints

Location: `gradient.py:1853`.
The native JAX runner removes bounds and maximum-step options without applying
or rejecting them. `angle_wrap` is also parsed into common controls but never
used in the native JAX loop.

Reproduction: JAX SGD minimizing `(x-2)**2` from zero with bounds `[-.25,.25]`,
`lr=.1`, five steps, returns **1.34464**, outside the explicit bounds, without
a warning. A solver switch must not silently change the feasible domain.

Proposed fix: implement supported constraints on-device or reject unsupported
options before evaluating the loss. Do not silently strip correctness controls.

### 7. P2 — qMERA's NLopt default is unsuitable for signed energy

Locations: `gradient.py:1450`, `src/pepsy/optimizers/qmera/parametric.py:275`.
NLopt defaults to `assume_nonnegative=True`; qMERA forwards options unchanged
despite minimizing signed energy. Valid negative energies cannot enter the
true-best tracker, undermining best restoration and result metadata.

Reproduction: `QMeraBuilder(shape=3, seed=4, param_scale=.1)` with Hamiltonian
`{(0,1): -diag([1,-1,-1,1])}`, `energy_per_site=False`, five `LD_LBFGS`
evaluations. Energy decreases from **-.9943197** to **-.9999999**, but
`best_loss=inf`. Explicit `assume_nonnegative=False` yields a finite negative
best energy. This is an integration policy mismatch; it does not justify
changing an infidelity caller's policy globally.

Proposed fix: default signed energy callers to `assume_nonnegative=False`
while respecting an explicit override, and test negative-energy restoration.

### 8. P2 — Native JAX silently removes a significant imaginary loss

Location: `gradient.py:1911`.
The native JAX runner accepts a documented real or near-real scalar but takes
the real part without checking the imaginary magnitude. Torch and the JAX
host adapter validate it.

Reproduction: objective `sum(x**2)+1j` runs without error under JAX SGD and
reports a real final loss; its actual imaginary part remains **1**.
This can conceal a broken contraction or non-Hermitian energy calculation.

Proposed fix: return the imaginary residual as JIT-compatible auxiliary data
and validate the scalar contract outside tracing, using a consistent tolerance.

## Autoray assessment

There is a worthwhile, bounded cleanup after the correctness fixes:

- **Adopt:** reuse `pepsy.backends.convert._array_namespace` and existing
  backend inference rather than growing another solver-specific array layer.
  Autoray covers real/imag/conjugate, reshape, concatenation, norms, clipping,
  finite checks, dtype queries, and array copying. Small installed-dispatch
  probes preserved backend and dtype for Torch/JAX conj, real, isfinite, copy.
- **Adopt:** keep native optimizer snapshots and bound projection on the
  device. Even unbounded Torch Adam calls `_flatten_params_real_numpy` to
  count variables and copy best states. A four-step CPU instrumentation probe
  counted **six full host-packer calls**; on CUDA the helper explicitly moves
  the packed values to CPU. No GPU speedup was measured. Compute flat sizes
  from metadata, clone best states on their backend, and transfer only needed
  scalar diagnostics. Existing Torch clone/restore helpers already exist.
- **Retain explicit backend hooks:** Torch detach/owned-storage handling,
  autograd, optimizer state, JAX value-and-grad/JIT and its conjugation
  convention. `ar.do('copy', x)` is not a replacement for an explicit gradient
  ownership policy. Autoray is array dispatch, not a unified optimizer or
  autodiff semantics layer.
- **Retain explicit host adapters:** SciPy/NLopt require real host-coordinate
  packing. Centralize that packing policy without pretending those optimizers
  run entirely on a GPU.
- **Defer:** a wholesale module rewrite, new backend promises, dependency
  upgrades, and automatic solver choices without a defined policy.

See the [Autoray public API](https://autoray.readthedocs.io/en/latest/autoapi/autoray/index.html).

## Integration boundary and validation

GradientOptimizer is directly used by SweepOptimizer and QMeraEnergyOptimizer.
PEPS sweep mode reaches it through SweepOptimizer. PEPS global mode instead
uses Pepsy GlobalOptimizer wrapping Quimb TNOptimizer; findings about this
gradient module should not be attributed to that separate solver wholesale.
Read-only searches located downstream Gaugy callers too; no sibling edits or
claim of complete downstream validation is made.

Local environment: Torch 2.6.0+cu124, JAX 0.10.2, Optax 0.2.8, SciPy 1.17.1,
NLopt 2.11.0, Autoray 0.11.1.dev9+g1291702f9. No dependency changes.
Inspected installed SciPy callback dispatch and Autoray signatures. Consulted
official JAX, SciPy, Optax, and Autoray documentation. The older JAX
advanced-autodiff URL was unavailable; the current complex-differentiation
page supplied the relevant contract.

Existing checks: 118 solver tests, 45 PEPS safeguard/performance tests, and
four selected qMERA optimizer tests pass. These tests do not cover the cases
above. Standalone probes reproduce the failures; the final solver probes
explicitly set `JAX_PLATFORMS=cpu` and disable JAX GPU preallocation. An earlier
array-dispatch probe triggered default JAX GPU preallocation OOM messages but
completed; that process exited, and no GPU performance claim is made.
The solver suite was also rechecked with explicit CPU selection.
Temporary diagnostic script/logs are `/tmp/pepsy_gradient_review_cpu.py`,
`/tmp/pepsy_gradient_review_cpu.log`, and
`/tmp/pepsy_gradient_review_integrations.log`; this note retains the durable
reproductions and numerical outcomes. No production simulation was launched.

No solver implementation or tests were changed, and nothing was committed
or published. Full-suite and production GPU validation remain unperformed.
