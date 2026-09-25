# 2026-09-25 — qMERA Torch full-graph energy

- Scope: fix qMERA Torch full-graph energy capture and improve repeated
  local-term energy/gradient evaluation.
- Branch / baseline commit: `develop` / `4f62cb0`.
- Commit status: uncommitted and unstaged; earlier qMERA 2D and unrelated
  Tree working-tree edits remain present.

## What changed

- `QMeraBuilder.compiled_parametric_loss_fn(torch_fullgraph=True)` prepares
  a Torch-only dense-spin cost with frozen Cotengra paths and reusable gate
  tensors. Built-in pair gates and RXX/RYY/RZZ provide explicit Pauli metadata.
- Added full-graph, gradient, odd-size, and unsupported-gate regressions;
  updated the qMERA API page, changelog, and sibling TFIM notebook/README.
- Detailed rationale, versions, benchmark method, and limits are in
  `docs/development/notes/qmera_torch_fullgraph_2026_09.md`.

## Validation

- qMERA suite: 98 passed, including the explicit XLA-compiled TFIM check
  and optimizer-level Torch full-graph energy, gradient, and Adam runs
  from both Torch-configured and default NumPy builders.
- TFIM notebook executed twice into `/tmp/pepsy_examples_executed`; the
  fresh run completed 25 JAX Adam steps, matched its local-term sum, and
  reported full-graph Torch AOT energy/gradient agreement.
- Public API/layout: 57 passed, 1 failed on the pre-existing installed
  distribution version mismatch (`0.4.1` metadata versus checkout `0.5.0`).
- Ruff and `git diff --check` passed. Full repository suite not run.
- Inductor remains unverified: even a scalar graph fails because Python 3.12
  headers are absent. No shared environment change was made.

## End-to-end API review

- Fixed `QMeraEnergyOptimizer.compiled_loss`, `compiled_loss_fn`, and
  `run(compiled=True)` routing for `torch_fullgraph=True`; incompatible run
  modes now fail before backend conversion.
- Fixed a Torch Dynamo failure that appeared only when compiler tests ran
  together by assigning permanent static slots to contraction intermediates.
- The qMERA API example now defines the Torch backend and parameters it uses.
  A fresh notebook run retained the JAX energy decrease and Torch AOT check.
- Repeated CPU timing after the change: 49.9 ms standard, 21.5 ms Torch eager,
  77.8 ms AOT eager per warmed energy-plus-gradient evaluation (12 calls).
- Public API/layout: 57 passed, one known installed-version assertion
  deselected. Ruff and `git diff --check` passed. No commit or push.
