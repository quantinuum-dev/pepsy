# 2026-10-07 — PEPS clipping records and local Cotengra policy

- Scope: fix missing above-one clipping diagnostics, clarify the continuation
  policy, and use Pepsy's contraction helper for row/column objectives.
- Branch / baseline: `develop`, `9f4bbcd`.
- Publication: user subsequently requested commit and push to `develop`;
  these changes are included in the commit containing this handoff.

## Changes

Compact sweep records now retain initial/final local losses outside either
end of [0,1], including their bounded counterparts. A streamed JSON regression
reproduced the original omission: lower-bound cases passed, upper-bound cases
failed before the fix.

API docs explicitly explain that the 1e-3 allowance permits continuation;
clipping a negative precheck to zero can skip refinement or satisfy sweep
convergence bookkeeping. It is not an accuracy bound. Boundary-cap convergence
and small exact references remain necessary for accurate comparisons. The
requested continuation behavior and refinement thresholds are unchanged.

`SweepOptimizer.local_contraction_opt=None` now uses Pepsy's reusable Cotengra
`build_optimizer` helper, the canonical name of the `build_contraction` alias.
The same object serves local norm/overlap objectives and optional cost metrics.
The PEPS driver shares it across gate batches and run calls. Explicit local
optimizer/tree overrides are supported; boundary `contraction_opt` remains
separate. Custom tree objects keep their existing sliced-contraction handling;
only string presets use the separate per-slice path cache. No tensor values
or autograd graphs are cached.

The follow-up request expands the helper policy to the complete PEPS optimizer
default path. `PepsOptimizer` and standalone `SweepOptimizer` now default
`contraction_opt=None` to the Pepsy builder and share it with local objectives.
Normalization, overlap checks, and global refinement inherit that object.
`CompBdy` likewise builds the helper by default, and its local FIT optimizer
inherits the selected contraction optimizer unless explicitly overridden.
Unrelated MPS/tree defaults and explicit caller choices are unchanged.

## Compatibility evidence

Classified as **adopt**: use the existing Pepsy factory and public Quimb
contraction APIs, without a compatibility shim or dependency changes.
Installed cloudspace versions: Quimb `1.15.1.dev75+g4112e304a`, Cotengra
`0.8.3.dev7+g1d7fd333f`, Cotengrust `0.2.1`, Autoray
`0.11.1.dev9+g1291702f9`, Symmray `0.4.1.dev11+g1a3481803`.
Inspected `TensorNetwork.contraction_tree(optimize=None, output_inds=None,
**kwargs)`, `ReusableHyperOptimizer`, and the installed worker-selection code.

Rechecked the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray source](https://github.com/jcmgray/autoray),
[Cotengra docs](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray source](https://github.com/jcmgray/symmray). The Symmray abelian-array
documentation page was unavailable; installed source and native-block tests
provide the relevant evidence. Online development banners do not identify
the installed versions.

## Validation

Focused suites (`test_peps_sweep_performance`, `test_peps_optimizer_batching`,
`test_peps_sweep_safeguards`, `test_peps_timing`, `test_optimize_peps`):
**261 passed**, 5 warnings, 58.07 seconds.
This result predates the follow-up expansion of boundary/diagnostic defaults.
Fresh validation adds `test_prepare_boundary_inputs` and
`test_peps_global_jax` to that selection: **446 passed**, 12 warnings,
154.23 seconds. Default smoke selection after the expansion: **93 passed,
1 failed**, 52.12 seconds, with the same baseline metadata mismatch below.

Default smoke selection: **93 passed, 1 failed**, 48.64 seconds. The failure
is `test_package_version_matches_installed_distribution`: installed Pepsy
metadata is 0.4.0 while the unchanged project version is 0.5.0. No shared
environment metadata was modified. The same failure reproduced against the
archived baseline test and project metadata. `git diff --check` passed.

New numerical checks compare local NumPy/Torch
objectives and Torch gradients for both axes against an explicit reference
path policy, using the real Pepsy factory with a small serial search budget.
Driver checks cover optimizer reuse across batches and repeated runs.

The first broader test attempt hit the sandbox's psutil process-ID visibility
failure in Loky workers. Validation was rerun with normal host visibility,
one CPU thread and one Cotengra worker. These limits apply only to test
subprocesses. Ruff is unavailable (`No module named ruff`).

The active production processes were not restarted. Their currently loaded
code is unchanged; new angle workers import the updated working tree. No GPU
speedup is claimed from these correctness and policy checks.
