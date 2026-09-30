# 2026-09-28 — Joint cluster MPO/PEPO implementation audit

- Scope: verify that symmetry reuse, fixed construction, lower-support caching
  and complete collection assembly apply correctly to ordered joint MPO/PEPO
  cluster products; fix concrete implementation waste.
- Branch / baseline: `develop`, `4e398e4`.
- Commit status: working-tree edits only; nothing staged, committed or
  published. Existing cluster and unrelated user changes were preserved.

## Changes and findings

- The shared structural symmetry plan respects factor order, operator
  labels and coefficient identity. Located joint PEPO evaluation uses it
  for both exact targets and frozen lower-support contractions. Recursive
  graph MPO assembly uses the ordered connected residuals and retains all
  compatible disjoint collections when no assembly rank cap is requested.
- Located joint PEPO evaluation now skips unused homogeneous
  onsite/edge component tensors. It computes only the located component
  maps read by that route; homogeneous products retain their existing
  path. Public API behavior and rank policy are unchanged.
- The [audit note](../docs/development/notes/2026-09-28-joint-cluster-audit.md)
  records the independent reference method, exact scope and limitations.
  The owning [PEPO guide](../docs/api/operators/cluster_expansion.md),
  [MPO guide](../docs/api/operators/mpo_cluster.md), module map,
  changelog and [status ledger](../docs/development/cluster_optimization_status.md)
  were updated.

## Validation

- Independent 2×2 square set-partition reference: two noncommuting ordered
  factors, `p=2,3,4`, both MPO and PEPO, reuse off/on. Maximum
  standalone matrix-entry errors were `1.56e-15` PEPO and
  `4.45e-16` MPO. The order-four located PEPO lower contractions fell
  from nine to three under joint reuse.
- Complete fixed two-site joint MPO and located PEPO values and
  coefficient/step gradients passed JAX `jit(value_and_grad)` against
  dense ordered exponentials, including zero inputs. A separate order-four
  joint PEPO Torch regression matches exact dense values and
  coefficient/time gradients with reuse on/off at zero and nonzero inputs.
- Affected CPU domain/API/layout gate: **391 passed**, two existing
  deprecation warnings (136.56 s), followed by **19 passed** in the
  complete correctness-review file after the final Torch regression.
  Full `python -m ruff check src tests`,
  whitespace and affected relative link checks passed. The previous
  full-suite count predates this review; no new full-suite or GPU claim.

## Remaining limits

- The previously measured 5×6 `p=4` recursive MPO is numerically
  inaccurate at assembly bond caps one and two. A converged cap and
  global operator error remain unverified. Collection completeness and
  bond-truncation accuracy are separate.
- Full Torch builder graph capture, arbitrary-geometry JAX compilation,
  native sector/fermionic recursive assembly and large-system memory
  guarantees remain outside the validated scope.
