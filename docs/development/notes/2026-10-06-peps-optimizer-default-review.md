# 2026-10-06 — PepsOptimizer defaults and correctness review

Scope: explain and review the current class, without implementation fixes.
Reviewed `develop` at `8f7c896`; findings below describe that baseline.
This report and its session handoff are uncommitted additions.

## Implemented default behavior

The public entry point is `from pepsy.optimizers import PepsOptimizer`.
Construction copies the input by default, stores the gate queue, and requires
the retained PEPS bond cap `chi=D`. Evolution starts with `run()`.

1. Normalize the initial state once, on the first run.
2. Apply and normalize standalone one-site gates directly.
3. For each two-site gate, copy the current state and construct the post-gate
   target with `reduce-split`, `cutoff=0`, `max_bond=None`, `chi=None`, and
   `path_compress=False`. Nonlocal coordinate pairs use the gate router.
4. Normalize the target. If its maximum bond is already at most `D`, keep it
   and record zero truncation infidelity without an overlap measurement.
5. Otherwise compress a copy with `compress_all(max_bond=D)`, normalize it,
   and estimate its infidelity against the target.
6. If the estimate exceeds the tolerance, use `SweepOptimizer` to refine
   PEPS row/column slices against the fixed target. Normalize and remeasure
   the candidate; retain the saved warm start unless the candidate is better.

| Control | Default |
| --- | --- |
| PEPS update | `mode="sweep"`, `k_2q_batch=1` |
| Dense boundary provider | Pepsy `BdyMPS` / `CompBdy` (`boundary_engine="auto"`) |
| Symmray boundary provider | Quimb MPS; Torch-backed Symmray required |
| Boundary compression | `fit_mode="eff"`, joint layers, block size 1 |
| Boundary initialization | `guess-src`, seed 0 |
| Boundary iteration budget | 10; no default adaptive FIT tolerance |
| Environment caps | `(4*D, 5*D)` for norm and overlap |
| Normalization caps | Independently `(4*D, 5*D)`; first entry used |
| Evaluation caps | Independently `(4*D, 5*D)`; see finding 1 |
| Sweep solver | NLopt `LD_LBFGS` |
| Sweep schedule | One global cycle, axes `y,x`, four round trips per axis |
| Local solver inherited controls | `maxeval=100`, `ftol_rel=xtol_rel=1e-9`, restore best |
| Contraction path optimizer | `auto-hq` |
| Normalization | Initial, target, warm start, candidate enabled |
| Acceptance | Strictly smaller measured infidelity, `improvement_tol=0` |
| Progress bars | Disabled |

The NumPy local sweep path uses finite differences; Torch parameters use
autograd. The PEPS sweep does not globally optimize all tensors together.
`mode="global"` is a separate opt-in route, with default NLopt `LD_VAR2`,
budget 1200 and an NLopt-runtime-error fallback of one LBFGS step.

Automatic warm-start cutoffs / entry tolerances are respectively
`1e-12 / 1e-9` for float64/complex128, `1e-6 / 1e-5` for
float32/complex64, and `1e-3 / 1e-3` for 16-bit data. Cutoff mode is `rsum2`.
These are refinement-entry tolerances, not achieved-error guarantees.

`get_fidelities()` is the running geometric mean of local fidelities;
`get_infidelities()` is one minus their product. Both start with an initial
sentinel, and the product uses a `1e-15` local-fidelity floor. They are local
compression diagnostics, not fidelity against the ideal full circuit.
Batching makes a recorded update a batch, not an individual gate. A second
`run()` reapplies the stored queue to the current state and resets traces by
default; it does not restart the original input.

## Confirmed findings

### 1. Evaluation assumes exact unit norms after approximate normalization

`PepsOptimizer.estimate_infidelity` sets `norm=norm_target=1` by default
(`optimizer.py:1084`). Consequently evaluation does not recompute either
denominator, and the first component of `evaluation_chi` has no effect on
those norms by default. Raising only evaluation accuracy cannot repair
normalization error. This qualifies the API guide's claim that evaluation
uses its norm cap for both state norms.

An otherwise-default run fails on an identity gate:

```python
import numpy as np
import quimb.tensor as qtn
from pepsy.optimizers import PepsOptimizer

state = qtn.PEPS.rand(4, 4, bond_dim=2, dtype="complex128", seed=17)
opt = PepsOptimizer(
    state, [(np.eye(4, dtype=complex), ((1, 1), (1, 2)))], chi=2,
)
opt.run()  # ValueError: PEPS infidelity is substantially negative.
```

Measured pre-infidelity was `-0.0003046157519661641`. The same run with
`normalize_chi=32, evaluation_chi=32` succeeds, with residual
`-1.64e-14` cleaned to zero and `reason="below_tol"`.

A separate normalized-state self-comparison (same seed/shape, greedy paths)
had dense norm squared `1.0003189591862283`. Increasing evaluation alone to
`(32,32)` still raised; explicitly recomputing both norms with
`norm=None, norm_target=None` at that cap gave zero self-infidelity.
Recomputing norms at the original small, unequal caps did not remove the
contraction inconsistency. Finite-cap contraction itself is approximate;
simply clipping all negative errors would hide that problem.

Proposed follow-up: define consistent evaluation normalization and an explicit
policy for inadequate metric accuracy, preserving evidence of invalid
contractions. Higher caps fixed this probe, not a universal accuracy guarantee.

### 2. `bond_dim` bypasses exact-target protection

`_target_gate_options` protects four options but leaves `bond_dim` intact
(`optimizer.py:754`). The gate implementation uses `bond_dim` when
`max_bond is None` (`operators/gates.py:2889`). Both inherited
`gate_kwargs={"bond_dim": 1}` and explicit
`target_gate_kwargs={"bond_dim": 1}` therefore truncate the supposedly
exact target.

Measured on a 2x2 all-zero product PEPS, `D=2`, and
`exp(-0.3j * X tensor X)` on `(0,0)/(0,1)`:

| Options | Dense infidelity | Reported final infidelity | Target bond |
| --- | --- | --- | --- |
| Default | `1.11e-16` | 0 | 2 |
| Either mapping with `bond_dim=1` | `0.08733219254516089` | 0 | 1 |

The corrupted target enters `within_chi` and skips refinement. This is silent
state error, unlike an explicitly reported inaccurate contraction.
Proposed follow-up: include aliases in exact-target protection and validate
actual dense reconstruction, not just the forwarded `max_bond` value.

### 3. Explicit physical-index pairs receive coordinate-routing options

For the same gate, coordinate sites `((0,0),(0,1))` work. Physical indices
`("k0,0", "k0,1")` fail with
`TypeError: svd_truncated_numpy() got an unexpected keyword argument 'sequence'`.
The class recognizes string locations, but `_base_gate_options` injects
`sequence` and path controls which reach Quimb's direct index split path.
Proposed follow-up: separate coordinate routing options from index-gate split
options. Coordinate tuples are the currently working route in this probe.

### 4. Sweep FIT diagnostics are silently discarded by the outer collector

`_optimize_with_sweep` passes the list `sweeper.fit_diagnostics` to
`_append_fit_diagnostics` (`optimizer.py:1373`). The collector accepts
structured result objects or mappings and returns immediately for lists.
On a real 2x2 `D=1` sweep with `fit_timing=True`, the sweep generated 46
records; the public getter returned only six metric records and none of the
46 sweep records. Existing collector coverage exercises metrics separately.
Proposed follow-up: accept sequences or forward the structured sweep result.

### 5. Documented temporary `run(mode=...)` persists

`run` describes `mode` as a temporary override, but calls `set_mode` and
never restores the previous mode (`optimizer.py:1843`). An empty-queue
`run(mode="global")` leaves `opt.mode == "global"`; the next plain run uses
global cleanup. Resolve the API contract or keep the override local.

### Documentation discrepancy

The API guide's caution at `docs/api/optimizers/peps.md:245` still says
Torch-backed Symmray defaults to Adam. Current `_apply_sweep_optimizer_options`
and the class docstring select NLopt `LD_LBFGS` for both dense and Symmray
sweep inputs. Torch autograd supplies derivatives; it is not the optimizer
selection. No API prose or implementation was changed in this review.

## Validation and limits

Activated the shared Python 3.12 environment in each Python shell.
Numerical commands set `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`.

- `python -m pytest -q -ra -o addopts='' tests/test_optimize_peps.py
  tests/test_optimize_global.py tests/test_prepare_boundary_inputs.py`:
  **329 passed**, 15 warnings, no skips, 36.29 seconds.
- The first invocation included nonexistent `tests/test_optimize_sweep.py`
  and ran no tests; the corrected selection above completed.
- Fresh small NumPy complex128/complex64 default runs preserved input state,
  output dtype and requested bond cap. At `D=2` the 2x2 entangling gate agreed
  with the dense target; at `D=1` it had the expected truncation loss.
- A real Torch CPU complex128 sweep on a random 2x2 input improved dense
  infidelity from `0.09879138028406464` to `0.09827656102760363`. Both boundary
  estimates agreed with dense values within `4e-15`; input, dtype and device
  were preserved.
- Six small complex64 rotation probes across NumPy/Torch passed. No separate
  low-precision negative-roundoff regression is claimed from these probes.
- NumPy complex64 full sweeps emitted NLopt runtime termination warnings and
  returned their best parameters. Default sweeps also warn about unspecified
  stopping controls even though inherited solver defaults include them.
- `python -m ruff check src tests`: passed. `git diff --check`: passed.
- No full package suite, GPU, broad PEPO/cyclic matrix, or long circuit run.
  Passing selections do not establish correctness of every mode/backend.

Probe scripts and logs were written under `/tmp/pepsy_peps_review_*.py` and
`/tmp/pepsy-peps-review-*.log`; essential evidence is retained above because
temporary files may disappear. Metric instrumentation only observed results;
it did not substitute numerical answers or change optimizer decisions.

## Upstream audit

Installed: Pepsy 0.5.0; Quimb `1.15.1.dev79+gb5e316200`; Autoray
`0.11.1.dev9+g1291702f9`; Cotengra `0.8.3.dev7+g1d7fd333f`; Cotengrust
0.2.1; Symmray `0.4.1.dev11+g1a3481803`; Torch `2.6.0+cu124`;
JAX 0.10.2; NLopt 2.11.0.

Inspected installed `TensorNetwork.compress_all`, `PEPS.gate`, metric, and
sweep signatures and Pepsy backend dispatch. Checked the official
[Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra docs](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray repository](https://github.com/jcmgray/symmray).
The requested [Symmray array page](https://symmray.readthedocs.io/en/latest/abelian_arrays.html)
returned an internal retrieval error; repository and installed implementation
were used instead. Online latest documentation is not the installed version.

Classification: **defer** upstream adoption or new compatibility shims; none
is needed to establish these package-local findings. No environment or
dependency changes were made. The October 1 cap-precedence and final-`chi`
target issues are fixed in current code and covered by the passing suite;
the `bond_dim` finding is a separate remaining alias route.
