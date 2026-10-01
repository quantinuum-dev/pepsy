# 2026-10-01 — Native Symmray factors and dense cache improvements

Scope: the user authorized all three proposed TreeSampler improvements and
continued the existing commit/push request. Baseline: `develop` at published
`ab9a445`. The earlier [factor-default decision](2026-10-01-tree-factor-default.md)
describes the native algorithm before this implementation, rather than its
current behavior. No dependency or sibling project was changed.

## Implemented

- **Adopt:** native canonical factors with measurement-prefix sharing.
  `TreeSampler(tree, backend="native")` now uses factor for Abelian and
  fermionic Symmray trees. Each private conditional tree moves its centre to
  the next physical site using the existing lossless, sector-preserving tree
  QR helper. One-tensor **network** norms provide the graded identity of the
  isometric exterior; plain tensor conjugation would miss outer-leg phases.
  Source physical codes are remapped by charge and within-sector offset after
  unreachable sectors disappear. Selected prefixes are normalized in the
  source dtype; their probabilities accumulate separately in native float64.
- **Adopt:** exact prefix groups within native chunks, plus a call-local cache
  shared across chunks. Payload accounting excludes captured source blocks
  and conservatively counts newly retained blocks, scalars and prefix keys.
  Full/exhausted caches stop admission scans early. A one-chunk native call
  retains no cache payload because it cannot reuse a prefix. Cache-disabled sampling
  still groups shots within a chunk. Tuple prefix keys cannot overflow a
  machine integer. Success, failure and reentrant calls own separate caches.
- **Adopt:** dense virtual-root first-child density reuse within the existing
  byte budget. A physical root remains conditioned on its sampled value.
- **Adopt:** sorted density-cache merging by scattering old/new entries
  directly into merged positions, avoiding full-value concatenation followed
  by a permutation copy. Exact misses and cache admission remain unchanged.
- **Compatibility correction:** scalar extraction unwraps only Quimb Tensor
  containers. Generic `.data` access previously detached Torch scalar graphs
  and replaced CuPy scalars with device pointers. Native normalization,
  amplitudes and fixed-configuration scores now retain the backend graph.
- **Adopt:** normalize a captured native canonical tree from its graded root
  norm and clear the private copy's positive global exponent. The source
  exponent is preserved. This avoids an unnecessary doubled-tree contraction
  and a pathological high-rank optimizer intermediate on a long product tree.

Discrete codes are not differentiable. The native standard reference remains
available with `strategy="standard"`; it still uses full projected-tree norms.
Native draws retain its shot-major random ordering, including persistent seeds.
Final CDF normalization follows `Generator.choice`, so rounding below one
cannot expose a trailing zero-weight sector. `backend="auto"` still selects
dense NumPy compatibility for Symmray input: native factors require explicit
`native` or `symmray`. Dense defaults and all budgets remain unchanged.

The [API guide](../../api/sampling/tree.md) and
[implementation map](../modules/sampling.md) describe these contracts.
`workspace_bytes` controls dense tiles, not native unbatched QR. Cache payload
accounting is not a bound on Python metadata, QR scratch, autograd graphs or
total memory. No rank truncation, charge mixing, densification or dtype
reduction is introduced on the native path.

## Upstream and environment audit

Installed versions rechecked in the selected Python 3.12 environment:
Quimb `1.15.1.dev66+ge927f06e1`, Autoray `0.11.1.dev3+g1b476b305`, Cotengra
`0.8.3.dev7+g1d7fd333f`, Symmray `0.4.1.dev7+g83fb22865`, NumPy `2.5.2`,
Torch `2.6.0+cu124`, cupy-cuda12x `14.1.1`. These match this active task's
[earlier audit](2026-10-01-tree-sampler-correctness-resume.md#upstream-and-environment-audit).

Reviewed the official [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/),
[Cotengra changelog](https://cotengra.readthedocs.io/en/latest/changelog.html)
and [Symmray repository](https://github.com/jcmgray/symmray).
Symmray array, index and changelog documentation pages were unavailable through
the browser; used official repository and installed source instead. Inspected
the public signatures of `Tensor.isel(selectors, inplace=False)`,
`TensorNetwork.copy(virtual=False, deep=False)`, `contract`, `contract_tags`,
`canonize_between`, the existing tree centre movement, and native array norm,
conjugation, copy and `apply_to_arrays`. Native `apply_to_arrays` is in-place;
block backend conversions must operate on block leaves. Reused the existing
native lossless QR dispatch rather than adding a new decomposition or shim.
**Defer:** upstream upgrades, new upstream algorithms and installed-library edits.

## Measured dense improvement

Used the [previous synthetic construction and allocator method](2026-10-01-tree-factor-small-chunks.md#method-and-environment):
30-site balanced binary tree with three root children, actual maximum bond
256, CuPy complex128, source seed 19 on the shared RTX A5000. Compared factor
at frozen baseline `ab9a445` with the changed factor implementation, default
cache/workspace and `threads=1`. State creation and capture are excluded.
Warmed each fresh sampler with two samples at seed 11; timed 1,024 samples
at seed 17 with CUDA synchronization. Two repeats interleaved before/after.
Shell thread limits: `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`.
Separate allocator profiles measure additional live CuPy pool allocations,
excluding source/capture, non-pool allocations and other processes.

| Chunk | Before times (s) | After times (s) | Median speedup | Before peak (MiB) | After peak (MiB) |
| --- | --- | --- | --- | --- | --- |
| 16 | 11.683, 11.721 | 6.636, 6.801 | 1.74× | 775.89 | 770.24 |
| 128 | 2.179, 2.163 | 1.591, 1.615 | 1.35× | 970.13 | 970.13 |
| 1000 | 1.111, 1.070 | 0.987, 0.987 | 1.10× | 2071.73 | 2071.73 |

All runs/profile calls produced identical configurations and probabilities
within `rtol=2e-10, atol=1e-22`. Each timing also checked its first eight
probabilities through the independent bottom-up scorer. Root reuse improved
small-chunk throughput. The cache merge removes intermediate copies but did
not lower the overall peak at chunks 128 or 1,000; other simultaneous buffers
dominated. These results do not establish a universal speed or memory gain,
an isolated GPU timing, a production-checkpoint result or Torch graph memory.

Temporary harness/raw results: `/tmp/pepsy_factor_improvements_benchmark.py`,
`/tmp/pepsy_factor_dense_improvements.{json,log}`. Frozen baseline source was
loaded under temporary package names, including its original factor context,
so the before case did not accidentally use the changed kernels.

## Measured native comparison

Built a 12-site balanced binary tree with three root children using
`hrs_to_ttn`, spinful `Fermion(symmetry="U1")`, half-filled occupations, bond
cap 32 and seed 19. Its **actual maximum bond was nine**, not 32. Blocks were
NumPy complex128. Timed 64 samples, chunk 64, sampling seed 17 after a two-shot
seed-11 warmup, with the same thread limits and capture exclusions as above.
An initial two-repeat interleaved comparison gave standard times 47.250 and
36.470 seconds (median 41.860), and candidate factor times 1.203 and 0.756
seconds (median 0.979). Configurations matched exactly and probabilities
matched the projected-norm reference and independent scores.

The initial factor candidate retained unusable prefix-cache entries during a
one-chunk call. Separate `tracemalloc` profiles measured 3,928,173 bytes
(3.746 MiB) for standard and 14,628,438 bytes (13.951 MiB) for that candidate.
This finding prompted the single-chunk retention correction above. It did not
change the standard algorithm or memory profile.

Reran the **final implementation**, including the CDF boundary correction,
against a fresh full standard 64-shot reference, then timed/profiled both
factor budgets. Final standard reference timing was 33.494 seconds; all final
factor configurations matched it exactly and probabilities matched at
`rtol=2e-10, atol=1e-22`. Every final timing also checked eight separate scores.

| Final factor cache setting | Two times (s) | Median (s) | Tracked peak (bytes) | Tracked peak (MiB) |
| --- | --- | --- | --- | --- |
| 0 | 0.734, 0.741 | 0.737 | 658,309 | 0.628 |
| default 128 MiB | 0.752, 0.740 | 0.746 | 655,280 | 0.625 |

Both settings retain no cross-chunk payloads in this one-chunk request; the
small timing/profile variation does not establish a budget advantage. Compared
with the final reference timing, default factor was approximately 45 times
faster on this small synthetic case. The final tracked peak was approximately
83% below the unchanged earlier standard profile. These native memory profiles
track new Python/NumPy allocations with `tracemalloc`, started after sampler
capture and warmup, with collection beforehand. They exclude resident source
and capture, untracked BLAS/library allocations and process RSS. Profiling was
disabled for timing. They are **not comparable to the dense GPU allocator
numbers** and do not establish large-bond native or Torch graph memory.

Temporary harness/raw results: `/tmp/pepsy_factor_improvements_benchmark.py`,
`/tmp/pepsy_factor_native_improvements.{json,log}`,
`/tmp/pepsy_factor_native_final.py`, `/tmp/pepsy_factor_native_final.{json,log}`.
The tables above preserve the essential evidence if temporary files disappear.

## Validation

Before the final native cache admission refinement, the complete focused
selection passed **582 tests, with two skips and four existing warnings** in
178.80 seconds. After that refinement, **final-code validation passed 585
tests, with two skips and four existing warnings**, in 184.55 seconds. Ran in
the activated selected environment:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest -q -ra -o addopts='' \
  tests/test_tree_sampler_validity.py tests/test_tree_factor_sampler.py \
  tests/test_tree_symmray_factor.py tests/test_tree_sampler.py \
  tests/test_tree_canonical_regions.py tests/test_tree_entropy.py \
  tests/test_tree_unitary_stability.py tests/test_public_api.py \
  tests/test_package_layout.py
python -m ruff check src tests
git diff --check
git diff --cached --check
```

Ruff, relative documentation links and whitespace checks pass. Skips are the
known CuPy float32 subnormal-source limitation and a check requiring two CUDA
devices. Torch CPU/CUDA, CuPy and native Symmray checks otherwise ran. The
four warnings are existing tree/qMERA/stabilizer compatibility warnings.
Final log: `/tmp/pepsy_tree_native_factor_final_validation.log`.

The suite covers native Abelian U1, fermionic spinless Z2 and
spinful U1/U1U1/Z2Z2, physical/virtual roots, NumPy, Torch CPU/CUDA and CuPy.
Independent dense probability oracles account for removed physical sectors;
native source projection checks relative amplitude signs. Tests prohibit
native `to_dense`, check source/canonical metadata preservation, compare
sample and score gradients to independent dense gradients, verify seeded
draws and persistent generators, exercise zero/insufficient/default caches,
and check success/failure cleanup and reentrant calls. A 160-site complex64
product test verifies nonzero float64 sample probabilities and exact draws.
Global exponents, single sites and endpoint uniforms have explicit regressions.

Production large-bond native and Torch graph peak-memory measurements remain
**unverified**; the user has not supplied that checkpoint. `workspace_bytes`
does not cap native QR; float64 probabilities still underflow below their
representable range. Full-package validation was not run. Publication and
exact final check scope are recorded in the
[handoff](../../../history/2026-10-01-tree-native-factor-improvements.md).
