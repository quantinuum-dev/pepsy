# 2026-09-28 — Pylint design and documentation review

- Scope: user's request to check overall design and comments with Pylint.
- Branch / baseline: `develop` / `f410fa0`, including the existing uncommitted
  readability changes. No source, tests, dependencies, or CI settings changed
  during this audit. This handoff is uncommitted.

## Package-wide result

Pylint 4.0.9 / Astroid 4.0.2, Python 3.12.14, existing development environment:

```bash
python -m pylint --rcfile=/dev/null --persistent=n --output-format=json2 src/pepsy
```

Standard rules were used without a new project configuration. Existing inline
Pylint directives still apply. Result: **8.986599/10**, **7,343 diagnostics**,
exit **30** (findings, not a clean pass). There are 212 source Python files;
Pylint's statistics report 215 modules linted.

| Category | Count |
| --- | ---: |
| Fatal | 0 |
| Error | 721 |
| Warning | 1,523 |
| Refactor | 3,626 |
| Convention | 1,470 |
| Informational | 3 |

Design/documentation findings include 900 `too-many-locals`, 892
`too-many-arguments`, 383 `too-many-branches`, 278 `too-many-statements`,
679 `duplicate-code`, 178 missing function docstrings, and two missing class
docstrings. These overlap and are heuristic findings, not counts of distinct
bugs or missing public API guides. A short existing docstring can pass while
still omitting shapes, mutation, and return contracts.

Raw output is in `/tmp/pepsy-pylint-overall.json`; temporary files are not
durable artifacts. The principal results and reviewed examples follow.

## Confirmed defect — first fix priority

`TreePepsOptimizer._apply_operator` computes `uncompressed_bonds` on the
two-layer path but does not assign `transient_max_bond`. With
`track_bond_diagnostics=True`, report assembly reads the missing local and
raises `UnboundLocalError` at
[`optimizer.py:2040`](../src/pepsy/optimizers/tree_peps/optimizer.py).
The state has already been replaced with the updated state when it raises.

Reproduced with each of `sdc`, `src`, and `zipup`, using the existing
two-layer regression setup with diagnostics enabled:

- `TreePepsPlan.from_shape((1, 5), topology="path")`;
- `TreePeps.rand(plan, bond_dim=2, seed=31)`;
- `TreeSubPepo.from_operator(plan, cnot, support=(1, 4))`;
- optimizer options `chi=4`, `cutoff=0`, `compression_seed=29`,
  `compression_layout="two_layer"`, `run=False`, `track_infidelity=False`;
- `optimizer.apply(operator)` succeeds with bond diagnostics disabled and
  fails with them enabled, for all three compression modes;
- after the exception, the live dense state already matches the exact CNOT
  update at `atol=rtol=1e-10`.

This file has no working-tree diff, so the defect predates the pending
readability changes. The existing two-layer test does not enable bond
diagnostics. The earlier full-suite pass therefore missed this combination.
**Reproduced, not fixed in this review.** The follow-up should initialize the
diagnostic consistently across application paths and cover this combination.

## Findings that require interpretation

- There are 386 `undefined-all-variable`, 95 `no-name-in-module`, 49
  `import-error`, and 135 `no-member` messages. These counts are not confirmed
  defects. For example, `pepsy.backends.infer_backend_signature` is explicitly
  resolved by the namespace's `__getattr__`, and the reported missing CuPy
  import is an optional backend. Do not make optional imports eager to
  satisfy these rules.
- PEPO's `left`/`right` possible-uninitialized warnings are guarded: the
  symmetric branch returns before those SVD factors are used.
- Tree fitting's missing `node0`/`node1` arguments warning occurs at
  `_path_of(state, *endpoints)`, after checking that there are two endpoints.
- Boundary FIT's possible-uninitialized diagnostic locals have a restricted
  private caller contract: the two current call sites supply `"failed"` or
  `"complete"`, and both initialize the values. An explicit invalid-status
  error could make that contract clearer; no failure was reproduced there.
- The remaining error and warning messages have not all been individually
  classified. In particular, no claim is made that all 721 error-labelled
  findings are either real defects or false positives.

## Readability and comment review

Current AST measurements, excluding leading docstrings, identify these
concrete next review targets:

| Function | Body lines | Finding |
| --- | ---: | --- |
| MPS `_run_dmrg` | 843 | One-line method docstring; substantial mode scheduling and state handling |
| MPO `_run_dmrg` | 553 | Better descriptive docstring, but a large orchestration body remains |
| Torch boundary `connected_amplitudes` | 421 | One-line docstring omits configuration/amplitude shapes, connection ordering, return shape, and diagnostic mutation |

These are review priorities, not instructions to split mathematical kernels
mechanically or change public signatures to meet a lint threshold. Existing
MPS comments explaining native-sector preservation and warm-up schedules,
and Torch comments explaining cache-statistics reset, communicate useful
reasoning. The latest random helper and JAX dispatch docstrings also explain
ownership and tracing constraints. This was a spot review, not a semantic
audit of every comment in the package.

Recommended order: fix the reproduced diagnostics defect; classify remaining
potential runtime errors; improve the public connected-amplitude contract;
then simplify the large orchestration methods with their existing domain
checks. Treat naming, formatting, and duplicate-code hints as lower priority.

## Validation and limits

- New checks: package-wide Pylint, AST size inventory, targeted source review,
  and six temporary Tree-PEPS probe cases (three successful controls and three
  reproduced failures). No tracked tests were added or changed.
- Earlier numerical validation remains the
  [previous pass's result](2026-09-28-random-jax-readability.md): 5,168 passed,
  129 skipped; smoke 89 passed. Those are not fresh results for this audit
  and do not cover the newly reproduced option combination.
- Pylint cannot establish mathematical correctness, comment accuracy, or
  readiness by score alone. No hosted CI result is claimed.
