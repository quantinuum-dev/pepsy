# JAX traced MPS backend validation — 2026-10-08

Updating Gaugy Examples' 2D cluster notebook to `MPSEngine.prepare(jit=False)`
exposed a Pepsy stream-validation failure. Eager `vmap` produces state tracers
with signature `(jax, complex128, None)` while the closed-over gates retain
`(jax, complex128, cpu:0)`. Eager differentiation can trace the opposite side.
Equal device strings cannot be required when one placement is unknown.

The fix is scoped to `MpsOptimizer._backend_signatures_compatible`: dense JAX
inputs must still agree in backend and dtype; known placements must agree.
No arrays are cast, copied or transferred. Shared backend signature inference,
NumPy/Torch/CuPy/native Symmray policies, gate application, compression and
autodiff algorithms are unchanged. Classification: narrow compatibility shim
for mixed concrete/traced public JAX execution.

## Environment and upstream audit

Used the activated shared Python 3.12 environment without changing packages:
Pepsy 0.5.0, Gaugy 0.1.0, JAX 0.10.2, Torch 2.6.0+cu124,
Quimb 1.15.1.dev90+g6a3906cbe, Autoray 0.11.1.dev14+g014a3f69a,
Cotengra 0.8.3.dev8+g8954240f2 and Symmray 0.4.1.dev15+g0374aaa3c.
Inspected installed public `jax.vmap`, Quimb `Tensor`, Autoray `infer_backend`,
and Pepsy signature helpers. Existing backend tests already establish that
abstract JAX tracers lack concrete placement metadata.

Checked the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html),
[Symmray repository](https://github.com/jcmgray/symmray), and
[JAX tracing FAQ](https://docs.jax.dev/en/latest/faq.html#different-kinds-of-jax-values).
The Symmray abelian-array documentation returned an internal fetch error;
the repository and installed signature implementation were available.
No upstream algorithm adoption, dependency update or vendoring was needed.

## Validation

- Existing MPS optimizer, exact-batch and gate-precision suites: 167 passed.
- Seven new public replay regressions passed, using two logical CPU devices.
  Exact/exact-batch batched states agree with independent dense products;
  eager gradients agree with an analytic rotation derivative. Wrong backend,
  wrong dtype and different known CPU devices are rejected. The first test
  attempt omitted the required `chi` constructor argument; corrected only
  the new test fixture, then all seven passed.
- Five downstream notebook MPS checks passed, including 2D eager/compiled
  JAX and Torch optimization, cached targets, untouched cluster parameters
  and exact global scoring. The previously failing eager JAX case passes.
- Gaugy prepared-MPS/direct-target regression suites: 47 passed, including
  compressed replay; ten expected adaptive-QR warnings on singular fixtures.
- Full `src tests` Ruff and whitespace checks passed. No full-suite or GPU
  validation is claimed for this fix.

Changes are local and uncommitted, based on Pepsy `develop` at `ad8ed05`.
