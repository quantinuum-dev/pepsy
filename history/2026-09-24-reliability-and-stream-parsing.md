# 2026-09-24 — Reliability fixes and shared stream parsing

- Scope: resolve the recorded failures, then separate shared event parsing
  from MPS replay as requested.
- Branch / baseline commit: `develop` / `28d9b6c`.
- Commit status: uncommitted working-tree changes; no dependency installation,
  upstream edits, staging, or publication in this session.

## Findings and changes

- The full baseline reproduced the same 13 failure identifiers recorded in
  the [previous handoff](2026-09-24-optimizer-import-boundaries.md):
  **4593 passed, 121 skipped, 13 failed**. Only six failed in isolation.
- [FIT](../src/pepsy/fitting/local.py) had eagerly filled all layered-target
  tag and boundary caches to construct an optional immutable snapshot. Local
  fits now cache only visited entries; `prepared_target` builds and caches the
  full snapshot on explicit access. Existing snapshot behavior is retained.
- Native [MPO-product DMRG](../src/pepsy/operators/mpo_product.py) now chooses
  direct SVD for its automatic warm start. SDC's Gram decomposition divided
  by zero on an empty charge block. Dense automatic guesses remain SDC;
  explicit method selections and the exact FIT target are unchanged. The
  regression now also compares the result with the exact dense product.
- [Backend signatures](../src/pepsy/backends/convert.py) compare concrete
  single-device JAX placement consistently across ordinary and named
  shardings, including Symmray blocks. Abstract tracers remain abstract;
  multi-device placement retains its sharding representation.
- [JAX SVD](../src/pepsy/backends/linalg_jax.py) accepts explicit standard
  options. The thin general-matrix case retains its truncation-safe VJP;
  other options delegate to native JAX. Tests check reconstruction and JIT
  gradients. The JAX registration idempotency test no longer leaks a real
  registration while restoring only Pepsy's bookkeeping.
- Corrected stale tests against existing documented contracts: mixed replay
  keeps using FIT after rank repair; MPI diagnostics default to disabled;
  TreePEPO compression validates structure once and does not request an
  expensive numerical isometry scan. The TreePEPO regression now explicitly
  verifies the actual canonical result. JAX device checks compare physical
  device sets rather than different sharding-wrapper objects.
- Torch batching tests activate and restore the native split policy explicitly.
  Earlier PEPS autodiff tests can leave stabilized custom split drivers active;
  those drivers need not support vmap. The tests still require batching and
  matching parameter gradients under the native policy, without changing the
  production fallback behavior or replacing stabilized derivatives.
- Extracted 25 unchanged parser functions into
  [optimizers/_stream_events.py](../src/pepsy/optimizers/_stream_events.py).
  Tree layout, tree replay, stabilizer consumers, and trajectory condition
  resolution use the shared owner. Existing imports through `mps.optimizer`
  remain available; numerical gate construction stays with its owner. Tree
  layout no longer imports MPS replay or FIT just to parse a control event.

## Validation

- Original failing selection after fixes: **13 passed**.
- Broader optimizer/backend/trajectory selection: **1542 passed, 53 skipped**.
- Initial core/API/layout selection: 359 passed, with one error in the newly
  added exact-reference assertion (used `to_dense` on a semantic MPO instead
  of `to_mpo().to_dense()`). Corrected it; the strengthened regression passed.
- Explicit order-sensitivity probe: **12 passed** after importing NetKet and
  enabling stabilized JAX and raw-block Torch registrations before pytest.
- Compared all 25 extracted parser functions and all 22 retained MPS
  functions/classes against baseline ASTs: identical.
  Fresh-process import regression covers tree layout and shared event parsing.
- Final full suite: **4609 passed, 121 skipped, zero failures** in 320 seconds.
  Command: `MPLBACKEND=Agg python -m pytest -q -o addopts='' -o
  faulthandler_timeout=90 --tb=short`. The 13 original failures are resolved
  in the complete run as well as the focused selection. Optional/hardware
  skips remain; warnings are not suppressed.
- Ruff, whitespace checks, and all 11 relevant local documentation links passed.

## Upstream evidence and limits

Installed versions: Quimb `1.15.1.dev66+ge927f06e1`, Cotengra
`0.8.3.dev7+g1d7fd333f`, Autoray `0.11.1.dev3+g1b476b305`, Symmray
`0.4.1.dev8+gc45f91457`, JAX `0.8.2`, Torch `2.9.1`.
Checked the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Cotengra changelog](https://cotengra.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray), and
[Symmray repository](https://github.com/jcmgray/symmray). The Symmray array
documentation returned an error; the installed implementation supplied the
failing split path. Inspected the installed Quimb compression signature.
Classification: narrow compatibility fixes and internal structural cleanup;
no upstream implementation was copied or modified.

This pass does not remove public aliases or claim an installation-size or
simulation-speed reduction. Remaining compatibility-import migrations and
larger module extractions are separate work. Hosted CI has not run these
uncommitted changes.
