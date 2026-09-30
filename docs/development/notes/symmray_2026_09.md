# Symmray 0.4 adoption audit

Audit date: **2026-09-24**. The initial assessment below preceded the authorized
implementation. See the implementation follow-up for adopted changes, measured
results, and remaining limitations. No installed packages were modified by
this work.

The requested [ReadTheDocs changelog](https://symmray.readthedocs.io/en/latest/changelog.html)
could not be retrieved. The assessment used the official
[release notes](https://github.com/jcmgray/symmray/releases),
[decomposition PR](https://github.com/jcmgray/symmray/pull/51), installed source,
and the published 0.4.0 wheel extracted under `/tmp/pepsy-symmray-audit/v040`.
The wheel was downloaded without dependencies; the active environment was not
modified. Isolated processes selected the wheel through `PYTHONPATH`.

## Initial assessment: installed capabilities

At the initial assessment, the genpy environment contained:

- Symmray `0.3.2.dev8+g6c6dd34b5`;
- Quimb `1.15.1.dev55+gd0591eb70`;
- Autoray `0.11.1.dev3+g1b476b305`;
- Cotengra `0.8.3.dev7+g1d7fd333f`.

The installed Symmray commit already includes PR #51. Concrete array methods
accept `max_bond_mode`, truncated decompositions default to `cutoff_mode="rel"`,
and randomized SVD defaults to eager sector allocation. These are opportunities
available now, not benefits that require upgrading this checkout.
`conj_project(axes=...)`, conditional `phase_global(parity=...)`, `dummy_parity`,
`.to(...)`, and `to_dense(index_maps=...)` are also present. Installed Quimb's
TN1D projector helper already calls `conj_project(axes=...)`.

## Recommendations

1. **Adopt after broader compatibility validation: fermionic phase fixes.**
   A scalar containing raw value 2 with a pending global minus sign gives
   `to_dense() == -2` on both builds. Installed `.item()` and `.sum()` incorrectly
   return +2; 0.4.0 returns -2 for both. This is a reproducible upstream bug,
   not proof that a particular Pepsy energy is currently wrong: some paths
   already synchronize phases or convert through `to_dense`. Prioritize scalar
   readout and reduction regressions before upgrading. The release also changes
   conjugate dummy-mode cancellation, so exercise odd operators and mixed
   bra/operator/ket contractions. Preserve Pepsy's native signs and metadata.

2. **Prototype: broader flat-Z2 Torch batching.** A fused rank-four flat array,
   converted to Torch, fails charge selection under `torch.vmap` on the installed
   build with a Tensor `.item()` error. Under 0.4.0 the batched result equals
   stacked serial results, shape `(2, 1, 8, 8)`. This is directly relevant to
   `TorchPEPSAmplitude` in `src/pepsy/vmc/torch/amplitude.py` and truncated
   boundary contractions. A small Pepsy 2x3, D=2, chi=4 boundary-MPS batch already
   works on both versions, including nonzero amplitudes, so the fix does not
   establish a universal speedup. Benchmark configurations that actually hit
   fused-index selection, checking gradients and executed batching. Keep the
   serial fallback. General sparse U1/U1U1 arrays and compiled boundary export
   have additional dynamic-shape constraints.

3. **Prototype, opt-in: eager/randomized sparse SVD.** This is useful for large
   charge blocks in MPS/PEPS compression and native tree operators. A two-sector
   spectrum `[10, 9]` and `[1, 0.5]`, with `max_bond=2`, keeps `[10, 9]` under
   global allocation but `[10, 1]` under eager allocation on both builds.
   Size-based allocation can therefore retain substantially less spectral
   weight. Randomized SVD can benefit from requesting fewer vectors per sector;
   ordinary full SVD is not guaranteed to become faster merely by using eager
   allocation. Positive cumulative-error cutoffs and renormalization are
   incompatible with eager mode in the probed sparse implementation. Keep
   Pepsy's explicit `rsum2` paths and existing defaults; benchmark error as well
   as runtime. Do not globally substitute `method="svd:rand"`.

4. **Prototype: native positive Gram operators.** `gram(axes=-1)` is absent
   locally and present in 0.4.0. On the two-sector example its native `eigh`
   returns `[81, 100]` and `[0.25, 1]`. Candidate applications are local metrics
   and projector construction in boundary compression or future native BP
   work. Fermionic Gram currently permits one open axis. Positivity uses the
   fermionic operator convention; a plain dense PSD check is not equivalent.
   Do not replace arbitrary environment contractions or existing BP messages
   without proving the orientation, dual-leg, and phase conventions agree.

5. **Defer broad rewrites and minimum-version increases.** Pepsy already uses
   explicit `to_dense(index_maps=...)` in its MPO basis restoration. Native
   conversion/local operator helpers need selective review, not blanket
   replacement. Preserve the structural-zero QR safeguard and Torch linalg
   policy; this release does not establish that either is obsolete. No shim
   is recommended by this assessment; any older-version scalar fix should be
   narrow and driven by an affected Pepsy call site.

## Validation and limits

Temporary scripts `/tmp/pepsy-symmray-audit/probe.py` and `boundary_probe.py`
compare the installed build and isolated release. They probe scalar phase
handling, global/eager SVD, Gram eigenvalues, fused-charge vmap, and serial vs
vmapped boundary amplitudes. These are correctness probes, not timing results.

With isolated 0.4.0 and the existing Quimb/Autoray/Cotengra stack, **7 focused
Pepsy tests passed**: two parametrized spinful helper cases, native compiled
contraction phase/dummy-mode preservation, boundary energy, onsite gate phase,
backend scalar conversion, and flat-Z2 Torch export/compile. The native suite
selection deselected 197 other cases. No full-suite, GPU, or new boundary
gradient validation was performed. This evidence supports a targeted upgrade
trial; it is not a complete compatibility certification.

## Implementation follow-up

During implementation, the shared environment was found to have independently
advanced to Symmray `0.4.1.dev7+g83fb22865` and Quimb
`1.15.1.dev66+ge927f06e1`. Autoray and Cotengra retained the revisions above.
The upstream audit and actual API probes were repeated against this stack.
Compatibility checks also used the isolated Symmray 0.4.0 wheel with the
previous Quimb source revision `d0591eb70`, selected through `PYTHONPATH`.

### Adopted changes

- **Adopt: corrected scalar phases.** Symmetry-bearing extras now require
  `symmray>=0.4.0`. These extras require Python 3.11 or newer through Symmray;
  Pepsy's core Python floor is unchanged. Native scalar readout calls `item()`
  directly, avoiding a NumPy object scalar that returns the original wrapper.
  A regression verifies pending global phases in `to_dense`, `item`, `sum`,
  and Pepsy's scalar helper.
- **Adopt and validate: flat-Z2 Torch batching.** A 3x3, D=4, chi=4 truncated
  boundary regression verifies that vmap actually executes, amplitudes match
  serial execution, and every parameter gradient is finite and agrees. The
  existing serial fallback remains. This does not establish general sparse
  U1/U1U1 or compiled boundary support.
- **Compatibility shim: native MPS guesses.** Active cumulative-error cutoffs
  (`sum1`, `rsum1`, `sum2`, `rsum2`, or their numeric equivalents) select
  deterministic native SVD for the disposable FIT guess. Randomized eager
  SVD cannot honor those cutoffs. Diagnostics report `symmray-svd` and
  `fallback_reason="cumulative_cutoff"`. Relative/absolute cutoffs retain the
  existing randomized path. Exact targets and final truncation are unchanged.
- **Compatibility shim: fermionic BP conventions.** A small native probe
  compares actual D2BP messages against `gram()` for both bond dualities.
  Positive-operator messages require a dual-leg parity metric before the
  ordinary block-matrix BP-to-SU eigensolve. Pair normalization is checked on
  a private solver snapshot and repaired only on affected instances; neither
  the class nor installed Quimb is patched. Region bras are conjugated jointly.
  Explicit loop projectors pair matrix-message conversion with physical-leg
  bra conjugation. Reduced-update warm gauges use the union of endpoint charge
  sectors. Legacy Quimb retains its previous conventions.
- **Defer: broad Gram and approximate-compression rewrites.** The measurements
  below do not justify changing global compression defaults. Existing QR
  safeguards, backend policy, and exact FIT targets remain applicable.

### CPU measurements

Illustrative float64 CPU probes used one Torch/BLAS thread, a warmup, and the
median of three measurements. These are small local measurements; some other
checks ran concurrently, and no GPU measurements were made.

For 16 configurations with D=4, chi=4, and zero boundary cutoff, the current
stack produced matching serial/vmap amplitudes and parameter gradients:

| PEPS size | Serial batch | vmap batch | Ratio |
| --- | ---: | ---: | ---: |
| 2x3 | 60.0 ms | 5.49 ms | 10.9x |
| 3x3 | 107.8 ms | 10.16 ms | 10.6x |

This compares execution modes on the new stack, **not** performance before
and after upgrading Symmray. Simple batches already vmapped on the old build.
The isolated 0.4.0 wheel gave similar ratios of about 10x.

Native sparse compression probes used four 256x256 U1 fermionic sectors,
bond cap 64, zero cutoff, and either fast or slow geometric spectral decay.
Randomized SVD used seed 42, oversampling 10, and two iterations. Error is
relative Frobenius reconstruction error:

| Spectrum | Method | Time | Error |
| --- | --- | ---: | ---: |
| Fast decay | Global SVD | 27.35 ms | 0.2292 |
| Fast decay | Eager SVD | 27.15 ms | 0.4203 |
| Fast decay | Randomized SVD | 2.24 ms | 0.4204 |
| Slow decay | Global SVD | 29.37 ms | 0.7526 |
| Slow decay | Eager SVD | 28.78 ms | 0.8640 |
| Slow decay | Randomized SVD | 2.39 ms | 0.8716 |

Randomization was about 12x faster in this synthetic case, with worse accuracy
at the same bond cap. Eager ordinary SVD provided little timing benefit and
also retained less spectral weight. Approximation therefore remains an
explicit method choice, with a deterministic fallback for cumulative cutoffs.

### Final validation and limitations

- Current stack: `test_bp_symmray.py`, `test_bp_reduced_update.py`, and
  `test_bp_compression.py`: **121 passed**, including explicit loop projectors.
- Previous Quimb plus Symmray 0.4.0: `test_bp_symmray.py`: **48 passed,
  1 skipped** (the new-convention-only normalization probe).
- Previous Quimb plus Symmray 0.4.0: symmetric tensors, Torch VMC compilation,
  Quimb compatibility, and contraction dependencies: **225 passed, 1 skipped**.
- Current stack: Torch VMC compilation, fermionic boundary, Quimb compatibility,
  contraction dependencies, public API, and package layout: **71 passed**.
- Current focused scalar and native MPS guess regressions: **8 passed**.

A full-suite attempt stopped after **105 passed, 1 skipped, 1 failed**.
`test_open_rho_series_keeps_the_long_range_path_and_is_exact_on_a_tree` fails
inside its direct Quimb reference: `PEPS.partial_trace` on a 1x4 lattice indexes
`sweep[1]` when the environment sweep has length one. The test passes with the
previous Quimb revision and Symmray 0.4.0. The installed upstream code and the
reference test were left unchanged.

An earlier combined current-stack run also had one compiled-boundary
environment-reuse assertion failure (267 passed, 1 skipped); it passed both
alone and in the final 71-test integration selection. Test-order interaction
with global Torch linalg registration remains unconfirmed. These results are
focused compatibility evidence, not a claim that the entire suite passes.

### Second verification pass

The dependency revisions were unchanged. A combined run of the current-stack
BP symmetry, reduced-update, compression, open-series, symmetric-tensor, Torch
compilation, fermionic-boundary, Quimb compatibility, contraction dependency,
public API, and package-layout suites completed with **420 passed, 1 skipped,
2 failed**. Both failures are direct Quimb 1x4 partial-trace references (the
two-site and multi-site open-series tree tests). A separate process reproduced
the `IndexError` using only Quimb, without importing Pepsy. Both reference
tests pass with the previous Quimb revision and Symmray 0.4.0; that selection
plus the native BP suite completed with **50 passed, 1 skipped**.

The compiled-boundary reuse test passed in this combined run, so its earlier
failure remains unreproduced. Review also strengthened the normalization
regression to start from a fresh upstream D2BP instance: the original test
used Pepsy's initialized solver and thus checked message preservation only
after the probe had already run. The strengthened first-run check passes.
Repository Ruff and whitespace checks passed. No production changes or
installed-package modifications were needed in this verification pass.
