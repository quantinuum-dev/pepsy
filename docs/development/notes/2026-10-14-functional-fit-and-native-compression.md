# Functional JAX FIT and native boundary compression — 2026-10-14

## Implemented scope

`pepsy.fitting.make_jax_fit` wraps the existing `FIT.run_eff` in a pure
packed-parameter function. Fresh networks and environments are reconstructed
inside the trace; repeated JIT calls reuse compiled kernels without retaining
mutable FIT instances. Finite-sweep evaluation optionally uses JAX checkpointing
and differentiates target and guess arrays. Dense block updates use the
composed projector splitter. Flat bosonic FIT contractions use native `auto`
products instead of the sparse-only `blockwise` mode.

Dense implicit mode converges complete RL cycles and aligns right-canonical
bond gauges through polar overlap factors. Its custom VJP saves only the
accepted parameters and input trees, then uses one aligned cycle and public
JAX GMRES for the adjoint. GMRES uses `lax.custom_linear_solve`. Forward and
adjoint residuals are checked, including under JIT. The guess is a warm start,
so its implicit derivative is zero. Native Symmray implicit mode remains
explicitly unsupported: an isolated blockwise alignment probe encountered
an upstream transpose limitation in the bosonic path and an unavailable
native identity operation in the fermionic path. No dense fallback was added.

The QR backward rule now chooses its chart from primal pivots. The earlier
cotangent-dependent finite/zero checks made the pullback non-transposable
inside JAX's implicit linear solver. Resolved QR finite differences, singular
reconstruction, pullback linearity and transposition have separate regressions.

`pepsy.backends.svd_reconstruct_jax` supplies a composed physical
reconstruction derivative. JAX's `pepsy:projector` split uses horizontal paired
factor differentials; internal retained degeneracies do not enter the formula.
Both JVP and VJP reconstruction checks cover real/complex rectangular matrices
and scales 1e-20 through 1e20. The previous expected-failure regression now
passes through this operation. The raw factor-wise SVD rule is unchanged:
arbitrary singular-vector gauges still have no unique derivative at degeneracy.
The new rule is not a blanket correction for native Symmray SVD/eigh factors.
Closed rank cuts produce nonfinite derivatives; untruncated reconstruction
is the identity even at zero or rank-deficient inputs.

BdyMPS direct compression now accepts native Symmray targets for `direct`,
`zipup`, and `dm`. Existing native Quimb compressors retain charges, phases,
cutoffs and caps. Other direct modes keep their capability guard. Exact norm
checks cover both axes and flat bosonic/fermionic networks; JIT directional
checks cover genuinely truncated native fermionic contractions.

## Upstream audit and decisions

Installed stack: JAX/JAXlib 0.11.2, Quimb 1.15.1.dev91+gb3bf45a50,
Symmray 0.4.1.dev18+g44fc5ae53, Autoray 0.11.1.dev14+g014a3f69a,
Cotengra 0.8.3.dev8+g8954240f2. Inspected FIT, native contraction/fusion,
Quimb compressor/split-driver dispatch, JAX custom-root/linear-solve signatures,
and GMRES implementation. No installed libraries or dependencies were changed.

Checked primary references:
[JAX custom root](https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_root.html),
[custom linear solve](https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_linear_solve.html),
[Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray source](https://github.com/jcmgray/autoray),
[Cotengra changelog](https://cotengra.readthedocs.io/en/latest/changelog.html),
[Symmray array documentation](https://symmray.readthedocs.io/en/latest/abelian_arrays.html)
and [source](https://github.com/jcmgray/symmray).
The Symmray array-documentation page was unavailable to the web reader;
installed public signatures/implementation and official source supplied the
capability check instead.

**Adopt:** public JAX transforms/GMRES and native Quimb compressors.
**Compatibility shim:** choose flat versus sparse native contraction modes.
**Scoped implementation:** dense implicit FIT and composed JAX reconstruction.
**Defer:** native implicit charts, adaptive-rank differentiation, general
singular-vector derivatives, higher-order implicit derivatives and performance
or physical-trajectory claims. `custom_root` is not used because the sweep's
existing custom-VJP QR/SVD route does not provide forward-mode derivatives.

API: [functional FIT](../../api/fitting/local.md#functional-jax-fitting),
[boundary compression](../../api/boundary/compression.md).
Final checks and working-tree status belong to the
[handoff](../../../history/2026-10-14-functional-fit-and-native-compression.md).

## Follow-up implementation review

The [review handoff](../../../history/2026-10-14-functional-fit-review.md)
records ownership, parameter-chart, scale and isometry-annotation fixes.
The fixed-count Newton alignment was replaced by a scale-safe unitary polar
primitive with a first-order composed JVP; its differential depends on sums
of singular values and is smooth at repeated positive spectra. Singular
overlaps return nonfinite factors and are rejected by the outer solve.
Projector diagnostics now report the discarded Frobenius norm without another
decomposition, restoring explicit adaptive bond growth in the imperative FIT
class. Native sparse boundary checks supplement the original flat coverage.
