# 2026-09-28 — Follow-up review of joint cluster implementations

- Scope: recheck the joint MPO/PEPO implementation for missed numerical
  or symmetry-reuse errors after the earlier ordered-product audit.
- Branch / baseline: `develop`, `4e398e4`.
- Commit status: working-tree test and evidence only; nothing staged,
  committed or published. Existing changes were preserved.

## Findings and changes

- No additional numerical implementation defect was reproduced.
- A new independent periodic 2×2 order-four joint PEPO regression covers
  directed XY/YX terms, length-two parallel bonds, automatic reuse and a
  validated translation declaration. The dense ordered-product error was
  at most `4.45e-16`; target evaluations fell from 13 to five.
- Separate CPU probes checked mixed uniform/located factor and term
  bindings with Torch values/gradients for both representations, and
  streaming/recursive MPO assembly against the selected-order
  set-partition reference. Detailed inputs and observed errors are in
  the [follow-up evidence](../docs/development/notes/2026-09-28-joint-cluster-followup.md).

## Validation

- The full correctness-review file passed **20 tests** after adding the
  periodic regression.
- The fixed/symmetry/recursive/JIT selection passed **93 tests** before
  that final test addition. Ruff, `git diff --check` and affected relative
  documentation-link checks passed.
- No new full-suite or GPU result is claimed.

## Remaining limits

- The measured 5×6, order-four recursive MPO is still inaccurate at
  assembly bond caps one and two; a converged cap and global operator
  error remain unverified.
- Native sector/fermionic recursive assembly, full Torch builder capture
  and arbitrary large-system performance are outside this review.
