# 2026-10-01 — Resume TreeSampler correctness work

User scope: inspect the unfinished TreeSampler work and resume it. Baseline:
`develop` at `5558857`. The vector/factor implementation and its earlier
finalization were committed in `3bfaaa9` and included in this baseline; the
old handoffs' working-tree statements describe their earlier sessions.

The [September 30 correctness review](2026-09-30-tree-sampler-correctness-review.md)
still applied to the current code, except that the recursive scoring closure
had already been fixed. This session implements the remaining reviewed fixes
in [tree.py](../../../src/pepsy/sampling/tree.py), adds independent
[regressions](../../../tests/test_tree_sampler_validity.py), and updates the
[public contract](../../api/sampling/tree.md). The subsequent local commit
request is recorded in the [handoff](../../../history/2026-10-01-tree-sampler-correctness-resume.md#local-commit-follow-up).
Existing MPI, trajectory and PEPS test edits and their concurrent handoff were
preserved and are outside this task.

## Implemented

- Dense normalization scales the canonical root before squaring, with a
  componentwise complex magnitude and a minimum normal divisor. Its numerical
  divisors remain in the native Torch graph. Zero/non-finite roots are rejected;
  finite float64 scales of `1e200`, `1e-200` and complex128 scales of `1e-320`
  and `8e307` are checked against the analytic distribution `[0.2, 0.8]`.
- Both dense strategies remove positive powers-of-two scales from incoming
  vectors, factors/densities and collapsed subtree messages. Conditional
  probabilities are invariant under those row scales; upward messages remain
  functions only of their subtree codes, preserving factor grouping/cache
  validity. Individual tensor contractions keep their working dtype. Divisors
  have finite reciprocals, avoiding overflow on subnormal NumPy/Torch inputs.
- Dense scoring shares its contraction traversal with amplitude scoring,
  rescales messages and accumulates removed scales in float64 logarithms.
  `probabilities()` now returns float64 on the captured backend/device, rather
  than squaring a complex64 amplitude into float32. `amplitudes()` retains its
  existing dtype and representable range. Probabilities below float64's range
  remain unrepresentable.
- Invalid conditional norms raise after clearing the recursive visitor. A
  device-side validity flag avoids a host synchronization at every site.
  Uniform zero draws skip zero-weight CDF entries rather than selecting an
  impossible branch. Ordinary seeded draws retain their ordering.
- Configuration shape, real/integral values and source-site ranges are checked
  before int64 coercion/indexing, including unsigned overflow. Integral floats
  and empty batches remain valid. Absent canonical Symmray sectors remain
  valid source codes with zero amplitude.
- Native Symmray binary flip ratios compute independent source-code flips,
  reuse the denominator and stack `(batch, nqubits)` on the block backend.
  Single flips of a fixed-charge U1 state independently give zero ratios.
- Clarified that captured off-root arrays can share source storage: raw in-place
  edits require refresh and are not isolated snapshots.

Classification: **adopt** focused correctness changes. No compatibility shim,
dependency upgrade, dtype reduction, truncation, installed-library edit or
default-strategy promotion. **Defer** production-checkpoint validation and
default promotion; no production checkpoint was available.

## Upstream and environment audit

Activated the existing Python 3.12 development environment. Installed versions:
Quimb `1.15.1.dev66+ge927f06e1`, Autoray `0.11.1.dev3+g1b476b305`, Cotengra
`0.8.3.dev7+g1d7fd333f`, Symmray `0.4.1.dev7+g83fb22865`, NumPy `2.5.2`,
Torch `2.6.0+cu124`, cupy-cuda12x `14.1.1`. One CUDA device is available.
Inspected `TensorNetwork.copy(virtual=False, deep=False)`, `tensor_contract`,
`TreeTensorNetwork.canonize_around_node_(nid)`, reduction and `frexp`/`ldexp`
capabilities; actual NumPy/Torch CPU/CUDA/CuPy dispatch is exercised below.

Checked the official [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray repository](https://github.com/jcmgray/symmray). The requested
Symmray Abelian-array documentation page returned an internal error; used the
official repository and installed implementation. No upstream API replacement
or decomposition-policy change is needed for these fixes.

Expanded boundary checks exposed an upstream limitation: installed CuPy
compiles arithmetic with `-ftz=true`, also documented by
[CuPy](https://docs.cupy.dev/en/stable/reference/generated/cupy.nextafter.html)
and its [versioned compiler source](https://github.com/cupy/cupy/blob/v14.1.1/cupy/cuda/compiler.py).
The stored complex64 component `1e-40` is flushed by arithmetic, including a
unit multiply or float64 cast. A capability probe explicitly skips that one
subnormal-source test on this backend. No tolerance was weakened; the NumPy
and Torch cases and all normal-component long-tree CuPy cases remain checked.
**Defer** recovery of already flushed CuPy source components; source dtype and
upstream compilation settings remain unchanged.

## Validation

- Frozen baseline loaded from `git show HEAD:src/pepsy/sampling/tree.py`:
  **22 failed, two passed, 69 deselected** in the NumPy, basic Torch gradient
  and native flip selection. Failures reproduce the original review findings
  plus zero-draw/invalid-conditional boundaries; this is intentional baseline
  regression evidence, not a failure of the changed implementation.
- Initial expanded sampler/canonical/entropy/unitary/API/layout run:
  **501 passed, one skipped**, four compatibility warnings. Subsequent checks
  identified and scoped the CuPy subnormal-source limitation above.
- Final selection: `python -m pytest -q -ra -o addopts=''` on
  `test_tree_sampler_validity.py`, `test_tree_factor_sampler.py`,
  `test_tree_sampler.py`, `test_tree_canonical_regions.py`, `test_tree_entropy.py`,
  `test_tree_unitary_stability.py`, `test_public_api.py` and
  `test_package_layout.py`: **512 passed, two skipped**, four compatibility
  warnings, 96.96 s. Skips: CuPy float32 subnormal-source arithmetic capability
  and two-GPU device coverage. NumPy, Torch CPU/CUDA, CuPy, native Abelian and
  fermionic Symmray paths are exercised separately.
- Includes analytic fair-bit draws and `2**-n` scores for 160/320-site
  complex64 product trees, both strategies/chunk policies and four dense
  backends. NumPy/Torch subnormal scores pass; all backends pass the extreme
  complex128 normalization cases. General trainable Torch CPU/CUDA trees are
  checked against weighted probabilities and every free tensor's derivative
  from an independently contracted and normalized dense state.
- Ruff (`python -m ruff check src tests`), `git diff --check`, and relevant
  local documentation-link checks pass. No full-package run or production
  checkpoint measurement was performed here.

## Timing comparison

The frozen baseline is the committed sampler at `5558857`. Source
construction and sampler capture are excluded, while draws, rescaling,
factor cache lifetime and the complete sampling call are timed. Calls run
serially, with one numerical thread and GPU synchronization before/after.

Balanced 30-site complex128 trees use three root children, state seed 19 and
sampling seed 2. NumPy uses chi=32, 2,048 shots/chunk 256; CuPy uses chi=256,
8,192 shots/chunk 2,048 on an NVIDIA RTX A5000. Each row has two interleaved
before/after repetitions.

| Backend / strategy | Before times (s) | After times (s) | Median time change |
| --- | --- | --- | --- |
| NumPy / standard | 6.266, 5.634 | 5.604, 6.231 | -0.5% |
| NumPy / factor | 3.123, 3.296 | 3.464, 3.556 | +9.4% |
| CuPy / standard | 19.811, 20.430 | 20.540, 20.711 | +2.5% |
| CuPy / factor | 3.421, 3.387 | 3.519, 3.515 | +3.3% |

All 16 timed calls preserve configurations exactly, agree on probabilities
at `rtol=2e-10`, `atol=1e-22`, and pass separate bottom-up scoring for 16
configurations. These measurements quantify the correctness pass's added
scaling cost on this synthetic workload, rather than a universal performance
claim. NumPy standard timings show substantial variability. The corrected
CuPy factor path remains about 5.87x faster than corrected standard in this
case. No allocator peak was measured here; the rescaling temporaries are
additional live buffers and existing workspace controls are not global caps.

Runner, complete timing log and raw JSON:
`/tmp/pepsy_tree_correctness_benchmark.py`,
`/tmp/pepsy_tree_correctness_benchmark.log`,
`/tmp/pepsy_tree_correctness_benchmark.json`. All checks/runners are complete.

Logs and temporary runners are under `/tmp/pepsy_tree_correctness_*` and
`/tmp/pepsy_tree_subnormal_checks.log`.
