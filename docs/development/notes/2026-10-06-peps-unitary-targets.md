# 2026-10-06 — Gate-dependent target budgets and unitary normalization

This follow-up supersedes the 2D-only batching budget, 100-evaluation limit,
and default target normalization described in the earlier
[batching implementation record](2026-10-06-peps-auto-batching.md).
The earlier validation remains evidence for that earlier working tree.

## Implemented

- Automatic batching uses 2D for dense diagonal two-qubit gates such as RZZ,
  expanding to 4D when another two-site gate enters the batch. Shared-site
  stopping, ordered one-site absorption, integer counts, and the exact
  singleton exception remain. These are ceilings, not bond-padding requests.
- Nearest-neighbor coordinate diagonal gates on dense states use an exact
  split rank ceiling of twice the current bond. A diagonal qubit operator is
  a sum of two product operators, so this ceiling removes redundant singular
  directions without introducing an approximation cutoff. The installed
  unrestricted split otherwise produced dimension 8 for RZZ on a 3x3 D=2
  input, despite its exact rank bound of 4.
- Structural detection requires exactly zero off-diagonal entries; no
  numerical rank tolerance is introduced. NumPy, Torch and CuPy dispatch are
  supported. Native block gates, JAX/traced gates and trainable gate arrays
  conservatively receive the 4D budget. Native states, routed gates and
  physical-index gate pairs keep unrestricted exact splits. No symmetry
  arrays are densified and no gate is detached from an autograd graph.
- NLopt local sweep defaults now explicitly use `LD_LBFGS`, `maxeval=50`,
  objective/parameter tolerances 1e-9 and best-iterate restoration. Explicit
  constructor and run overrides continue to merge.
- `normalize_target=None` follows `non_unitary=False`: unitary targets are
  not rescaled. `non_unitary=True` enables target normalization; explicit
  `normalize_target` overrides remain. Initial normalization is unchanged.
- Retained targets, warm starts, optimized candidates and final standalone
  one-site outputs are normalized with `normalize_chi`. Rejection restores
  the normalized warm start. `normalize_final=False` remains an opt-out for
  candidate/direct-output normalization; warm starts are always normalized.
- Fidelity still measures both norms without modifying the target, retaining
  the finite-cap accuracy/retry safeguards from the earlier change.

## Validation

Reused the unchanged environment and official upstream audit recorded in the
[default review](2026-10-06-peps-optimizer-default-review.md#upstream-audit).
Inspected installed `PEPS.bond_size(self, coo1, coo2)` and Autoray diagonal /
count-nonzero dispatch. Classification: **adopt** public exact gate/split
APIs and analytic rank bounds; **defer** specialized native/traced RZZ rank
detection. No dependency or installed-source changes.

Activated the shared Python 3.12 environment, with BLAS/OpenMP threads set to
one. Fresh checks:

```text
python -m pytest -q -ra -o addopts='' \
  tests/test_peps_optimizer_batching.py tests/test_optimize_peps.py \
  tests/test_optimize_global.py tests/test_prepare_boundary_inputs.py \
  tests/test_public_api.py tests/test_package_layout.py
```

**410 passed, 15 warnings**, no skips, 23.70 seconds. Ruff and
`git diff --check` passed. New checks reconstruct two-gate 3x3 RZZ and general
unitary targets densely, verifying 2D/4D stored dimensions. Torch RZZ checks
verify complex128 reconstruction and conservative handling of trainable
gates. Output norm checks cover direct targets, warm starts, accepted and
rejected candidates, and standalone one-site streams. Existing selected
tests retain native fermionic and actual local/global optimization coverage.

The first follow-up run had two orchestration-test failures: a per-gate
fidelity trace test needed explicit `k_2q_batch=1`, and a one-site routing
mock needed to mock the newly required output normalization. These were
updated without weakening numerical reconstruction tolerances.

Normalization is unit norm according to the selected finite boundary
estimate, not a guarantee that the exact dense norm is one at insufficient
`normalize_chi`. No full suite, GPU/CuPy run, broad PEPO/cyclic matrix, or
throughput benchmark was performed. Trial target allocation can exceed the
batching threshold; it is not a hard peak-memory bound.

Changes are uncommitted. Concurrent gradient-solver edits and their separate
documentation were preserved; validation ran in the shared working tree.
