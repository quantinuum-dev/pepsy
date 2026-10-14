# 2026-10-14 — Review and correction of functional FIT

- Scope: review the newly added FIT/JAX and native boundary implementation,
  reproduce numerical/API issues, and fix them within that scope.
- Branch / baseline: `develop` / `8f8695b`, with the preceding implementation
  and concurrent NetKet/TDVP work already uncommitted.
- Commit status: no staging, commit or publication by this session.

## Reproduced issues and corrections

1. The wrapper's closures read the caller's live MPS tags/length. Retagging
   that MPS after construction raised `KeyError` on the next call. Geometry
   and numeric arrays are now snapshots.
2. Changing the real/complex dtype chart could silently drop imaginary
   coordinates. Replacement trees, shapes and dtypes are checked; mixed
   implicit guess charts are rejected.
3. Sixteen Newton steps did not guarantee a unitary polar gauge transform
   for scaled inputs. A composed square-polar primitive uses a scaled native
   SVD forward and a singular-value-sum differential, retaining a unitary
   forward at repeated spectra. Singular overlaps are explicitly invalid.
4. Projector split diagnostics lacked the discarded norm, silently disabling
   adaptive bond growth. A reproduced rank-one fit stayed rank one while
   the ordinary SVD fit grew to three. Discarded Frobenius diagnostics restore
   growth on NumPy and Torch; no extra decomposition is used.
5. Different stored target/guess exponents lost physical scale: exponents
   2 and -1 returned a state 1,000 times too small. The full-chain target is
   now distributed into the guess's exponent frame before invoking FIT.
6. Packed reconstruction reused the initial right-isometry annotations after
   a left-canonical RR fit. Private input skeletons and reconstructions clear
   those annotations while retaining geometry and native array metadata.

Native sparse U1 bosonic and Z2 fermionic boundary norm probes matched exact
contraction for direct/zipup/dm and now have independent regressions. The
native implicit and general raw-factor derivative limitations remain unchanged.
No simulation, dependency change, or GPU benchmark was launched.

## Validation

Existing Python 3.12 environment; JAX numerical checks use CPU. The preceding
[upstream audit](../docs/development/notes/2026-10-14-functional-fit-and-native-compression.md#upstream-audit-and-decisions)
applies to the unchanged installed stack. Inspected Quimb's exponent and split
diagnostic contracts when applying these corrections.

- Projector/JAX derivative and FIT kernel selection: **161 passed**, 18.73 s.
- Functional FIT, existing FIT AD, QR/SVD stability and native flat boundary:
  **87 passed**, 174.07 s, before the final stale-isometry correction.
- Final polar/native boundary/public API/package-layout selection:
  **105 passed**, 90.47 s.
- One-/two-site implicit finite differences after the polar update:
  **2 passed**, 36.36 s, before the final stale-isometry correction.
- Final complete functional FIT selection: **20 passed**, 62.33 s, including
  the stale-isometry, ownership, exponent and dtype-chart regressions.
- Whole-source/test Ruff, whitespace checks and six local Markdown link
  targets passed. Counts overlap and must not be added.
- No full numerical suite was run; unrelated concurrent NetKet edits were
  preserved, and testing remained scoped to this review.
