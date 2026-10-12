# 2026-09-30 — Committed TreeSampler correctness review

Scope: the user requested another careful review after publishing the
shared-density optimization. Reviewed `develop` at `7773d2e`; the sampler
implementation is unchanged from `623ccc3`. Working tree was clean at the
start. No library code, tests, dependencies, or installed packages changed
during this review. The findings below are reproducible existing defects,
not newly integrated fixes.

## P1: conditional underflow corrupts complex64 sampling

Locations: `TreeSampler._sample_arrays`, especially unscaled density/message
propagation at lines 1308–1313 and the zero-total fallback at lines 1261–1268
of `src/pepsy/sampling/tree.py` in this commit.

A normalized 160-site balanced product state `|+>**160`, with actual bond
dimension one, should produce independent fair bits and probability
`2**-160 = 6.842277657836021e-49` for every configuration. That probability
fits in the sampler's float64 returned probability array. With complex64
node arrays, the unnormalized conditional densities lose their magnitude
before the output accumulator does. Once both physical weights underflow to
zero, `safe_total=1` leaves an all-zero CDF, and sampling deterministically
selects the final physical value rather than a fair bit.

For eight shots with seed 11, both unchunked and chunk_size=3 calls returned
eight zero probabilities. Compared to the exact fair-bit draw schedule,
NumPy and Torch CPU had 26 wrong bits; CuPy had 133 wrong bits. The differing
underflow behavior makes the failure onset backend dependent. Independent
amplitude scoring also squared into float32 zero, so comparing sampled
probabilities only to that scorer would fail to expose this error.

Controls: 30-site complex64 configurations matched the reference; the largest
relative probability deviation there was about 1.20e-6 on Torch CPU.
Complex128 cases at 30 and 160 sites matched configurations and returned the
exact product probabilities in this probe. The pre-optimization `414798e`
sampler snapshot reproduced the 160-site NumPy failure (eight zero
probabilities, 26 wrong bits).

Proposed remedy: rescale conditional densities and upward messages while
preserving their conditional ratios and the separate probability accumulator.
Do not substitute an arbitrary outcome when a conditional has no finite
positive total. A maintained implementation needs long-tree scale-invariance
checks; simply changing the output dtype cannot fix intermediate underflow.

## P1: native Symmray single-site flip ratios flip all sites together

Location: `single_site_flip_amplitude_ratios`, lines 961–978.

The Symmray branch mutates every column of one `flipped` configuration array,
then evaluates one numerator after the loop. Consequently it returns a
`(batch,)` ratio for simultaneously flipping every site, instead of the
documented `(batch, nqubits)` single-site ratios.

Reproduced with a four-site nonfermionic native U1 tree, physical sectors
`{0:1, 1:1}`, total charge two, bond_dim=4, state seed 3, and three sampled
configurations. The independent per-site reference gives a `(3,4)` zero
matrix: every individual flip changes the conserved charge. The method
instead returns three nonzero complex numbers, exactly equal to the
all-sites-flipped reference. For example one returned ratio was
`0.1008104773 - 0.1696183532j`.

Proposed remedy: evaluate each site's flip independently, retain source-code
maps, and return one column per site on the original backend. Add a native
fixed-charge reference test; the current flip-ratio regression covers dense
trees only.

## P2: root normalization detaches Torch scoring gradients

Location: `_extract_arrays`, lines 807–813.

The root norm is converted to host NumPy and a Python float before division.
Forward amplitudes are normalized, but Torch differentiates the denominator
as a constant. For a one-qubit trainable real tensor `[1,2]`, the returned
probability of zero is correctly 0.2, while its gradient is `[0.4,0]` instead
of `[0.32,-0.16]`. The gradient of the sum of both normalized probabilities is
`[0.4,0.8]` rather than zero.

For the same sampler, `sample_batch(1, seed=2)` selects one. Differentiating
its returned conditional probability gives `[-0.32,0.16]`, but differentiating
`probabilities([[1]], to_numpy=False)` gives `[0,0.8]`. Local conditional
normalization cancels the detached scale in sampling; scoring does not.
This matters for general trainable input tensors. It is not a claim that
every exactly normalized unitary parameterization has a wrong gradient.

The existing gradient regression checks a density-transfer kernel with
already supplied tensors, so it never exercises root normalization.
Proposed remedy: keep the normalization factor in the native differentiation
graph; any host scalar used to validate the state must not replace it in
the numerical expression.

## P2: invalid norms and extreme finite scales produce invalid samples

Location: `_extract_arrays`, lines 807–813; the dense conditional fallback
then turns the failed normalization into apparent sample results.

Across NumPy, Torch CPU, and CuPy float64 one-qubit inputs:

- `[0,0]` is accepted and returns only one bits with zero probability.
- `[NaN,1]` is accepted and returns zero bits with NaN probabilities.
- `[1e200,2e200]` overflows when squaring the root to find its norm, then
  normalization divides by infinity and destroys the state.
- `[1e-200,2e-200]` underflows during the norm calculation and is not
  normalized at all.

Both finite scaled inputs represent the same normalized distribution
`[0.2,0.8]` as `[1,2]`, but return zero probabilities for both configurations.
Native Symmray already rejects a zero/non-finite norm; the dense path lacks
the equivalent validation. Proposed remedy: use a scale-safe native norm,
retain its differentiation graph, and reject genuinely invalid states.
Rejecting every overflow of the current naive squared norm would still
reject mathematically valid finite scaled inputs.

## P2: scoring silently changes invalid configurations

Location: `_check_configs`, lines 1428–1448, followed by advanced indexing
in `_amplitudes` at line 1462.

Inputs are cast to int64 before validating anything except shape. For the
one-qubit `[1,2]` state, code -1 returns the probability of code 1 on all
three dense backends; fractional code 0.8 is silently truncated to zero.
On CuPy, positive out-of-range codes also wrap: 2 returns probability 0.2
and 3 returns 0.8. NumPy/Torch CPU raise IndexError for those positive codes.
Thus the same invalid configuration can either fail or acquire a plausible
probability depending on backend.

Proposed remedy: validate integral values before coercion and validate codes
against each site's physical space before indexing. For native symmetry,
retain the distinction between valid source sectors absent from a canonical
tensor (zero amplitude) and invalid physical codes.

## Existing lifecycle finding and snapshot caveat

The `_amplitudes` recursive visitor still retains its self-reference at
line 1487. Its implementation is unchanged from the previously measured
[GPU snapshot-retention finding](2026-09-30-tree-sampling-rereview.md).
That earlier probe disabled cyclic GC and measured another 32 MiB retained
per subsequent chi=128 scoring/refresh iteration; isolated visitor cleanup
eliminated it. This review confirms unchanged code, not a new memory timing.

Also reproduced shallow snapshot ownership: for an already root-canonical
three-site product tree, a cached non-root NumPy array shares memory with the
source. Applying an in-place physical X to that source leaf changes cached
scoring from `000` to `100` without refresh. The API instructs callers to
refresh after source changes, so this is recorded as an ownership/documentation
caveat rather than a new failure of correctly refreshed use. The class's
claim that it always retains previously captured tensor data is stronger
than the shallow-copy behavior. Ordinary out-of-place tensor replacement
is a different case; it was not shown to change an existing snapshot.

## Validation and scope

New focused repository run:

```text
python -m pytest -q -ra -o addopts='' tests/test_tree_sampler.py \
  tests/test_tree_entropy.py tests/test_tree_unitary_stability.py
152 passed, 1 skipped, 2 warnings in 12.50s
```

The skip requires two CUDA devices. The warnings are the existing deprecated
`dmrg1` alias in unitary-stability tests. This passing selection does not
cover the newly reproduced failures and is not a full-package result.

Additional temporary probes:

- `/tmp/pepsy_tree_committed_review.py`: 27 dense normalization/configuration
  cases plus one gradient and one ownership probe, saved JSON/log.
- `/tmp/pepsy_tree_long_product_review.py`: 24 sampling calls across
  NumPy/Torch CPU/CuPy, complex64/128, 30/160 sites, chunk None/3, saved JSON/log.
- `/tmp/pepsy_tree_native_ratio_review.py`: independent single-site native
  Symmray reference and AST comparison to the saved pre-optimization sampler.
- `/tmp/pepsy_tree_review_baseline.json`: old-code underflow confirmation
  and installed versions.

AST comparisons confirm `_extract_arrays`, `_check_configs`, `_amplitudes`,
and `single_site_flip_amplitude_ratios` are unchanged from the saved
`414798e` sampler. The underflow reproducer separately confirms that failure
on old code. These findings therefore do not establish a regression from
the recent shared-density commit.

Installed versions still match the active task's upstream audit: Quimb
1.15.1.dev66+ge927f06e1, Autoray 0.11.1.dev3+g1b476b305, Cotengra
0.8.3.dev7+g1d7fd333f, Symmray 0.4.1.dev7+g83fb22865, NumPy 2.5.2,
Torch 2.6.0+cu124, cupy-cuda12x 14.1.1. Reused that audit; no compatibility
shim or installed-library change is proposed.

Classification: **adopt next** for focused correctness/lifetime fixes with
regressions. **Defer** broader performance integration until these contracts
are accounted for. The earlier vector/factor/prefix speedups remain isolated
prototypes, not part of this committed implementation. The user's actual
5x6 evolved checkpoint is unavailable; no new failure was found in the
checked 30-site complex128 controls. Canonical-center handling still requires
no redesign.

Only this evidence note and its handoff were added. Local links and whitespace
were checked. Nothing was staged, committed, or pushed during this review.
